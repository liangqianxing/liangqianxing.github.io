---
title: 'Stanford CS336 学习记录 02：RMSNorm 的广播与 SwiGLU 的门控'
date: '2026-10-16'
description: '用一个 token 向量理解均方根归一化，拆开 dim=-1 与 keepdim=True 的作用，再沿着两条分支读懂 SiLU、逐元素乘法和 SwiGLU。'
series: Stanford CS336
seriesOrder: 2
cover: /images/posts/cs336-rmsnorm-swiglu/cs336-rmsnorm-swiglu-cover.webp
coverAlt: 特征尺度整理和双分支门控的 AI 辅助概念插画
categories:
  - 学习笔记
tags:
  - CS336
  - PyTorch
  - Transformer
  - RMSNorm
  - SwiGLU
draft: false
hidden: false
haloSync: false
haloPublished: false
codeUrl: https://github.com/liangqianxing/cs336-labs/tree/a39eb684710817f7f389626423fcb26ca0f6d629
attachmentUrl: /materials/cs336/experiments/cs336-02-rmsnorm-swiglu-20261016.zip
---

[上一篇](/posts/cs336-linear-embedding-shapes/)把 Linear 和 Embedding 的参数形状理顺了。继续往下，最先遇到的是 RMSNorm：公式不算长，但把它落到一批 token 上时，`dim=-1`、`keepdim=True` 和广播必须一起理解。

然后是 SiLU 和 SwiGLU。这里开始有了两条并行分支。看代码时最容易忽略的是，同样写着“乘”，`@` 与 `*` 做的事情完全不同。

## 先算一个向量的均方根

对一个 token 的特征向量 $x=[3,4]$，均方根 RMS 的计算就是三个动作：先平方，再平均，最后开根号。

$$
\operatorname{RMS}(x)
=\sqrt{\frac{3^2+4^2}{2}}
=\sqrt{12.5}
\approx3.536.
$$

让两个分量都除以这个共同的数：

$$
\frac{x}{\operatorname{RMS}(x)}
\approx[0.849,1.131].
$$

归一化调整的是向量的整体数值规模。第二个分量仍大于 1，所以不能把“归一化”理解为把每个数都压到 0 到 1 之间。

这一步也没有减均值。RMSNorm 与 LayerNorm 的计算方式不同，不能只因为都带 Norm 就互换理解。

## eps 和可训练权重分别解决什么问题

本次 RMSNorm 的公式是：

$$
y_i=
\frac{x_i}
{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\varepsilon}}
\,g_i.
$$

$d$ 是特征维度；$\varepsilon$ 是一个很小的正数；$g_i$ 是第 $i$ 个特征对应的可训练缩放参数，代码里叫 `weight`。

如果一个 token 的所有特征都是 0，没有 $\varepsilon$ 就会除以 0。代码保存的 `eps=1e-5`，即 0.00001，它加在开根号之前。

缩放权重的形状是 `(d_model,)`，初始值全为 1。这样开始时它不额外改变归一化后的结果，训练之后再学习每个特征需要怎样缩放。每个 token 都有自己的 RMS，但同一个 RMSNorm 层的这组特征权重会在所有 token 之间共享。

## keepdim=True 留下的是哪一维

假设输入形状为 `(2, 5, 64)`，分别表示 batch、序列长度和特征维度。我们需要对每个 token 的 64 个特征求平方均值，不能把所有 batch 或所有 token 一起平均。

已有实现中，对应的是：

```python
mean_square=(x**2).mean(dim=-1,keepdim=True)
```

`dim=-1` 指最后一维。`keepdim=True` 保留这根轴，只把长度从 64 改成 1。

| 张量 | 形状 | 每个位置放什么 |
| --- | --- | --- |
| 输入 `x` | `(2, 5, 64)` | 每个 token 的 64 个特征 |
| `mean_square` | `(2, 5, 1)` | 每个 token 的平方均值 |
| `rms` | `(2, 5, 1)` | 每个 token 对应的除数 |
| `weight` | `(64,)` | 每个特征共享的缩放参数 |
| 输出 | `(2, 5, 64)` | 归一化并缩放后的特征 |

广播从尾部对齐维度。`(2, 5, 1)` 的最后一维可以扩展到 64，于是一个 token 的 64 个特征都除以这个 token 自己的 RMS。`(64,)` 也能沿前面两维广播，让各个 token 的第 $i$ 个特征都乘以同一个 $g_i$。

如果把 `keepdim` 去掉，平方均值会变成 `(2, 5)`。拿它与 `(2, 5, 64)` 直接做除法，尾部就成了 5 与 64 对齐，形状无法匹配。保留长度为 1 的轴，既告诉 PyTorch 怎样广播，也保留了“这是特征轴上的一个统计量”这层含义。

## 为什么计算前后还要转换 dtype

这份实现会先记住输入类型，将输入转为 float32，完成平方、平均与归一化后，再把结果转回输入类型。

原因在数值范围和精度。低精度输入的平方与求和可能更容易累积误差，float32 中间计算能降低这部分风险。但这不等于“转成 float32 后所有数值问题都会消失”，更不等于已经做过混合精度训练测试。

这一轮验证的是官方 RMSNorm 前向用例。理解类型转换的目的之后，仍要把结论限制在实际测试覆盖的范围里。

## SiLU 为什么要放在线性层后面

只把多个无偏置 Linear 接起来，整体仍然能合并成一个线性变换。中间维度再大，也不能仅靠这种堆叠获得非线性。

SiLU 对每个数做：

$$
\operatorname{SiLU}(x)=x\,\sigma(x)
=\frac{x}{1+e^{-x}}.
$$

这里的 $\sigma(x)$ 是 sigmoid。它的值在 0 到 1 之间，但 SiLU 的输出本身不受这个区间限制。

| 输入 $x$ | sigmoid 系数 | SiLU 输出 |
| ---: | ---: | ---: |
| -2 | 约 0.119 | 约 -0.238 |
| 0 | 0.5 | 0 |
| 2 | 约 0.881 | 约 1.762 |

已有 `activations.py` 中的 SiLU 是一个普通函数。它对张量逐元素计算，不保存可训练权重，输入和输出形状相同。与 Linear 不同，这里不需要创建一张参数表。

这个朴素表达式已经通过对应官方测试；文章记录的是当前学习版本，没有据此声称它在所有极端输入或低精度场景下都达到最优的数值表现。

## SwiGLU 的三个 Linear 怎样连接

SwiGLU 复用前面写好的 Linear 和 SiLU。同一份输入先进入两条分支，一条经过线性变换再经过 SiLU，另一条只经过线性变换。两条结果逐元素相乘，再投影回原来的维度。

当前代码中，三个层的名字是 `w1`、`w3`、`w2`。这个顺序很容易读错：最后工作的反而是 `w2`。

| 层 | 维度变化 | 所在位置 |
| --- | --- | --- |
| `w1` | `d_model → d_ff` | 第一条分支，后接 SiLU |
| `w3` | `d_model → d_ff` | 第二条分支 |
| `w2` | `d_ff → d_model` | 两条分支相乘后 |

考虑权重按 `(输出维度, 输入维度)` 保存，用行向量约定写出的公式为：

$$
\operatorname{SwiGLU}(X)
=\left[\operatorname{SiLU}(XW_1^\top)
\odot(XW_3^\top)\right]W_2^\top.
$$

$\odot$ 是逐元素乘法。已有 `forward` 中最值得单独看的两行是：

```python
combined=gate*values
return self.w2(combined)
```

假设输入为 `(2, 5, 64)`，`d_ff=128`，那么两条分支都是 `(2, 5, 128)`，相乘后仍是这个形状，最后通过 `w2` 得到 `(2, 5, 64)`。

这里的 `*` 让对应位置相乘，例如 `[2, 3]` 与 `[4, 5]` 得到 `[8, 15]`。它没有像矩阵乘法那样沿某个维度求和。前面 Linear 中的 `@` 和这里的 `*`，不能因为都叫乘法就混用。

## 门控不是一个 0/1 开关

代码把 SiLU 分支叫 `gate`，于是容易想成每个位置只有“放行”和“关闭”两种状态。实际这里是连续的逐元素调节。

还需要区分 sigmoid 与 SiLU：虽然 sigmoid 系数在 0 到 1 之间，经过 SiLU 之后的 `gate` 可以大于 1，也可以小于 0。因此它既可能减弱某个分量，也可能放大或改变符号。把门控理解成“输入相关的调节系数”，比把它想成一排布尔开关更贴近这段计算。

这几个组件目前都在每个 token 内处理特征。它们不会因为输入带着序列维度，就自动让 token 之间交换信息。跨位置的信息汇总，要留给后面的 Attention。

## 从导入失败到测试通过

RMSNorm 最初写过 `import RMSNorm(nn.Module):`。这里是在定义新类，应该使用 `class`。之后测试停在导入阶段，错误信息是：

```text
ModuleNotFoundError: No module named 'cs336_basics.RMSNorm'
```

文件名 `rmsnorm.py` 是小写，里面的类名 `RMSNorm` 保留大小写。导入路径的模块名和导入的类名是两个不同部分。

到 SwiGLU，又遇到了 `linear` 与 `Linear`、`super.__init__()` 与 `super().__init__()`、`Values` 与 `values` 的区别。都是很短的符号，却会让程序在真正执行公式以前就停下。

修正以后，历史记录中 `test_rmsnorm`、`test_silu_matches_pytorch`、`test_swiglu` 分别得到 `1 passed`。这些结果对应组件级的前向测试；当前还没有把它们组装成完整 Transformer block。

下一篇继续记录 [Softmax 和 Attention Mask](/posts/cs336-softmax-attention-mask/)。这一次，前面反复遇到的“沿哪一维”和“保留哪一维”会再次出现。

## 发布时的复核

发布前对[独立代码快照](https://github.com/liangqianxing/cs336-labs/tree/a39eb684710817f7f389626423fcb26ca0f6d629)重新运行了 8 项官方组件测试，结果为 `8 passed`。完整清单和运行环境见[进度记录](https://github.com/liangqianxing/cs336-labs/blob/a39eb684710817f7f389626423fcb26ca0f6d629/docs/progress.md)。

*封面使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 辅助生成，属于概念插画；准确的形状、公式与运算关系以正文为准。*

## 参考资料

- [Stanford CS336 官方课程页面](https://cs336.stanford.edu/)
- [Assignment 1 官方仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)
- [PyTorch：广播语义](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [PyTorch：torch.mean](https://docs.pytorch.org/docs/stable/generated/torch.mean.html)
- [PyTorch：nn.RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
- [PyTorch：nn.SiLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
