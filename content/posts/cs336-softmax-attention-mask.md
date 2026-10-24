---
title: 'Stanford CS336 学习记录 03：Softmax 的数值稳定与 Attention Mask'
date: '2026-10-18'
description: '从减去最大值为何不改变 Softmax 开始，理清函数与有参数层的区别，再沿着 Q、K、V 的形状理解缩放点积注意力，以及 mask 为什么必须填负无穷。'
series: Stanford CS336
seriesOrder: 3
cover: /images/posts/cs336-softmax-attention-mask/cs336-softmax-attention-mask-cover.webp
coverAlt: 注意力分数、权重与因果遮罩的 AI 辅助概念插画
categories:
  - 学习笔记
tags:
  - CS336
  - PyTorch
  - Transformer
  - Softmax
  - Attention
draft: false
hidden: false
haloSync: false
haloPublished: false
codeUrl: https://github.com/liangqianxing/cs336-labs/tree/a39eb684710817f7f389626423fcb26ca0f6d629
---

[上一篇](/posts/cs336-rmsnorm-swiglu/)写到 SwiGLU，接下来进入 Attention。真正开始相乘之前，先做了 Softmax：它把一组分数变成总和为 1 的权重，后面用这些权重汇总不同 token 提供的信息。

这段学习里，有两个很具体的问题值得记下来。一个是“前面都要创建 layer，为什么 Softmax 直接传输入就行”；另一个是 mask 中少写了一个负号。前者关系到代码组织，后者会直接改变被关注的位置。

## 从一组分数到一组权重

对分数 $[1,2,3]$，Softmax 的结果大约是 $[0.090,0.245,0.665]$。

$$
p_i=\frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

先把每个分数取指数，再除以这些指数的总和。对于有限输入，得到的权重为正，总和为 1；分数越高，分到的权重越大。

“除以总和”之前为什么要取指数？这里先记住 Softmax 本身的定义：它不是把原始分数直接除以原始分数之和。分数可以是负数，而指数把它们映射为正值；共同平移所有分数，也不会改变最终分配。

最后这个性质，正好用于改善数值稳定性。

## 减去最大值为什么不改变结果

直接计算很大的指数容易溢出。例如分数已经上千，$e^{x_i}$ 远超常见浮点类型的范围。即使分数之间的差别很普通，中间计算仍可能先变成无穷大。

令 $m=\max_j x_j$，则：

$$
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
=\frac{e^{-m}e^{x_i}}{e^{-m}\sum_j e^{x_j}}
=\frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

分子与分母同时乘上 $e^{-m}$，比值不变。对最初那组分数：

```text
原始分数： [1, 2, 3]
减最大值： [-2, -1, 0]
取指数：   [0.135, 0.368, 1]
归一化：   [0.090, 0.245, 0.665]
```

减去最大值后，最大的指数项是 $e^0=1$，其他项都不大于 1。这样避开了由很大的正指数带来的溢出。

已有 `activations.py` 中，对应的关键步骤是：

```python
max_value=torch.amax(x,dim=dim,keepdim=True)
shifted=x-max_value
exp_x=torch.exp(shifted)
```

再沿相同的 `dim` 求指数之和并相除。这里又出现了 `keepdim=True`：例如分数形状为 `(batch, queries, keys)`，沿最后一维取最大值后得到 `(batch, queries, 1)`，每个 query 都减去自己这一行的最大值。

这个技巧改善的是有限输入下的指数计算。它不能自动解决所有特殊值问题；例如一整行全为负无穷时，减最大值会涉及负无穷减负无穷。当前实现没有特殊处理全屏蔽行，这种情况下会产生 NaN；本次官方用例没有覆盖这个边界。

## 为什么 Softmax 不需要先创建 layer

前面 Linear、Embedding、RMSNorm、SwiGLU 都先创建对象，再把输入交给它。到了 Softmax，适配器直接调用函数。我当时问，这两种写法为什么不一样。

在当前实现里，前面几个组件用类来保存参数。Linear 有权重矩阵，RMSNorm 有可训练缩放参数，SwiGLU 内部还保存三个 Linear 子层。对象创建后，这些参数会被反复使用。

Softmax 只依赖这次传入的张量和计算维度，没有需要保存的可训练权重，所以写成普通函数就够了。SiLU 也是同样的情况。

| 当前组件 | 当前实现形式 | 需要保存的可训练参数 |
| --- | --- | --- |
| Linear、Embedding | 类 | 权重矩阵或向量表 |
| RMSNorm | 类 | 每个特征的缩放权重 |
| SwiGLU | 类，内部有子层 | 三组线性层权重 |
| SiLU、Softmax | 普通函数 | 无 |
| 缩放点积 Attention | 普通函数 | 当前函数没有额外权重，接收外部 Q、K、V |

这是这次代码的组织选择，不是“没有参数就不许写类”的规则。PyTorch 同时提供 `nn.Softmax` 与函数接口。一个无参数计算也可以用模块形式封装，取决于组合方式和接口需要。

## Q、K、V 的三个角色

Softmax 之后，开始看缩放点积注意力：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

三个输入承担不同角色：Q 是 query，表示当前位置用什么向量查询；K 是 key，参与计算匹配程度；V 是 value，是实际被加权汇总的内容。

“我想找什么”“我能怎样被找到”“我能提供什么信息”有助于初次理解，但计算里它们都是向量。匹配分数来自 Q 与 K 的点积，汇总对象是 V，不能把 K 与 V 的用途混在一起。

例如某个 query 最后得到的权重为 $[0.2,0.3,0.5]$，它的输出就是：

$$
o=0.2V_1+0.3V_2+0.5V_3.
$$

Linear 和 SwiGLU 都在一个 token 内变换特征；这里才明确出现了按照权重，从多个位置汇总信息的计算。

## 先把两次矩阵乘法的形状对上

可以先忽略 batch，只看三个位置、每个向量两个特征的例子。Q、K、V 都是 `(3, 2)`：

```text
匹配分数： (3, 2) @ (2, 3) → (3, 3)
注意力权重：                  (3, 3)
加权汇总： (3, 3) @ (3, 2) → (3, 2)
```

分数矩阵的一行对应一个 query，一列对应一个 key。Softmax 沿最后一维计算，意味着对每个 query，在所有 key 之间分配权重。随后用这一行权重汇总相应的 V。

query 数量和 key 数量也不必相同。更一般地：

| 张量 | 最后两维 |
| --- | --- |
| Q | `(queries, d_k)` |
| K | `(keys, d_k)` |
| V | `(keys, d_v)` |
| 分数与权重 | `(queries, keys)` |
| 输出 | `(queries, d_v)` |

代码使用 `K.transpose(-2, -1)`，只交换最后两维，前面的批量维度可以保留。这与对一个已知二维权重使用 `.T` 的场景不同；进入带 batch 或 head 维度的张量时，需要明确到底交换哪两根轴。

## 分数为什么除以平方根

这里的 $d_k$ 是 Q 与 K 的最后一维长度。随着参与点积的特征数增加，分数的波动规模也容易增加，直接送进 Softmax 时可能让权重变得很集中。

在各分量近似独立、均值为 0、方差为 1 的简化假设下，$d_k$ 项乘积之和的方差约为 $d_k$。除以 $\sqrt{d_k}$ 后，方差回到约 1 的量级。这说明了缩放因子的动机，实际向量分布当然不必严格满足这些假设。

因此，缩放针对的是点积分数的规模。它与 Softmax 的“减最大值”解决不同问题，两步不能相互替代。

## Mask：True 允许，False 屏蔽

本次接口接收布尔 mask，约定是 `True` 允许关注，`False` 屏蔽。`mask=None` 时不会自动启用因果约束，需要调用者显式传入 mask。以三个位置的因果 mask 为例：

| 当前 query | 看位置 1 | 看位置 2 | 看位置 3 |
| --- | --- | --- | --- |
| 位置 1 | True | False | False |
| 位置 2 | True | True | False |
| 位置 3 | True | True | True |

每个位置只能看自己和更早的位置，不能利用后面的 token。这张表控制的是 query-key 配对，不是在删除某个 token 的特征维度。

屏蔽发生在 Softmax 之前：将不允许的位置填为负无穷，因为

$$
e^{-\infty}=0.
$$

在这一行仍有有效 key 的情况下，被屏蔽位置就会获得 0 权重，其余允许的位置重新归一化。

已有实现里，关键的一行是：

```python
scores=scores.masked_fill(~mask,float("-inf"))
```

`~mask` 将 True、False 取反，从而选中需要填充的位置。`masked_fill` 填的是分数，不是最终输出。

如果把屏蔽分数写成 0，Softmax 中对应项仍是 $e^0=1$，这个位置还会获得权重。只有放在正确的计算位置、使用符合约定的数值，mask 才能起到屏蔽作用。

## 少一个负号，会把方向整个写反

这次确实把 `float("-inf")` 写成过 `float("inf")`。公式里只差一个字符，含义却完全相反。正无穷不能让概率归零，还可能在稳定 Softmax 的平移过程中产生无效值。

Softmax 里也出现过 `torch,exp`，应该表达的却是对 `torch.exp` 的调用；Attention 的导入语句还混淆过 `import` 与 `from ... import ...`。这些都是可以定位到具体行的错误，不必一看到 Attention 输出异常就先怀疑整套数学公式。

## 两项测试通过，还差一个多头注意力层

历史运行记录里，`test_softmax_matches_pytorch` 得到 `1 passed`。缩放点积注意力运行了两项官方测试：`test_scaled_dot_product_attention` 与 `test_4d_scaled_dot_product_attention`，结果是 `2 passed`。

第二项把输入整理成 `(batch, heads, seq_len, d_k)`，确认当前注意力函数能在这类四维输入下给出参考结果。它没有验证完整的多头自注意力层：Q、K、V 的投影、拆头、合并和输出投影，都是下一步才要继续学习的内容。

到这篇为止，已经验证的是若干基础组件和缩放点积注意力。多头注意力刚进入形状理解，RoPE、完整 Transformer、优化器及训练流程还不在本轮完成范围内。

同日后续完成的多头自注意力另记在[学习记录 04：拆头、合头与因果 Mask](/posts/cs336-multihead-attention-shapes/)，保留这里对应的阶段性代码和验证结果。

## 发布时的复核

发布前对[独立代码快照](https://github.com/liangqianxing/cs336-labs/tree/a39eb684710817f7f389626423fcb26ca0f6d629)重新运行了 8 项官方组件测试，结果为 `8 passed`。完整清单和运行环境见[进度记录](https://github.com/liangqianxing/cs336-labs/blob/a39eb684710817f7f389626423fcb26ca0f6d629/docs/progress.md)。

*封面使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 辅助生成，属于概念插画；准确的形状、公式与运算关系以正文为准。*

## 参考资料

- [Stanford CS336 官方课程页面](https://cs336.stanford.edu/)
- [Assignment 1 官方仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)
- [Assignment 1 官方模型测试](https://github.com/stanford-cs336/assignment1-basics/blob/main/tests/test_model.py)
- [PyTorch：nn.Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
- [PyTorch：torch.amax](https://docs.pytorch.org/docs/stable/generated/torch.amax.html)
- [PyTorch：torch.transpose](https://docs.pytorch.org/docs/stable/generated/torch.transpose.html)
- [PyTorch：Tensor.masked_fill](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
- [PyTorch：scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
