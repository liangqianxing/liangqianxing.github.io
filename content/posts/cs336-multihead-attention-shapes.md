---
title: 'Stanford CS336 学习记录 04：多头注意力怎样拆头、计算和合头'
date: '2026-10-20'
description: '沿着 (2, 5, 64) 的输入追踪 QKV 投影、reshape、transpose、共享因果 mask 和合头，理解每个头仍处理完整序列，以及最后为什么还需要输出 Linear。'
series: Stanford CS336
seriesOrder: 4
cover: /images/posts/cs336-multihead-attention-shapes/cs336-multihead-attention-shapes-cover.webp
coverAlt: 宽向量束分成多条并行特征轨道后重新合并的 AI 辅助概念插画
categories:
  - 学习笔记
tags:
  - CS336
  - PyTorch
  - Transformer
  - Attention
  - 张量形状
draft: false
hidden: false
haloSync: false
haloPublished: false
codeUrl: https://github.com/liangqianxing/cs336-labs/tree/31363223498fafd1528c550a51efbe0aed231736
---

[上一篇](/posts/cs336-softmax-attention-mask/)完成了缩放点积注意力。那个函数接收已经准备好的 Q、K、V，计算匹配分数，应用 mask，再用权重汇总 V。这一篇继续把它放进一个带参数的多头自注意力模块。

这次最需要看清的是“拆头”。代码里只是几行 `reshape` 和 `transpose`，但如果不知道每根轴代表什么，很容易做到形状能运行，却把不同 token 的数据拼到一起。

我沿用学习过程中反复使用的例子：输入形状为 `(2, 5, 64)`，一批有两条序列，每条序列五个 token，每个 token 有 64 个特征，设置四个头。最后，这个不含 RoPE 的多头自注意力模块通过了对应官方测试。

## 先生成 Q、K、V，再拆头

“自注意力”中的“自”，在这里指 Q、K、V 都来自同一份输入 `x`。但它们并不是把 `x` 换三个名字，而是分别经过三个有独立权重的 Linear：

$$
Q=XW_Q^\top,\qquad
K=XW_K^\top,\qquad
V=XW_V^\top.
$$

这里继续使用前几篇的约定：输入向量按行组织，Linear 权重按 `(输出维度, 输入维度)` 保存。

当前 `q_proj`、`k_proj`、`v_proj` 都是 `64 → 64`，因此三个输出的形状仍然是 `(2, 5, 64)`。形状相同，并不意味着它们的数值相同：三组参数可以学出不同变换。

为什么先生成完整的 64 维，再分头？这样每种投影都可以用一次矩阵乘法同时生成所有头需要的特征。Q、K、V 各做一次投影，一共三次。每个输出特征都能依赖输入的全部 64 个特征，随后才按输出特征分组。

因此，不能把这一过程理解成“把原始输入切成四块，每个头永远只能看到自己那一块原始特征”。拆分发生在可学习投影之后。

## 四个头分的是特征，不是五个 token

当前实现采用等宽的头，满足：

$$
d_{\mathrm{head}}=\frac{d_{\mathrm{model}}}{H},
\qquad
64=4\times16.
$$

$H$ 是头数。对于这个约定，头数必须是正整数，且能整除 `d_model`。代码用 `%` 检查余数，再用 `//` 得到每头的特征维度。这一检查对应的是能否拆成等宽的头，不能只用整除运算默默丢掉余数。

拆头后，每个 token 的 64 个投影特征被组织为四组，每组 16 维。五个 token 仍然都在，每个头也仍然处理完整的五个位置。

| 维度 | 数值 | 含义 |
| --- | ---: | --- |
| batch | 2 | 两条序列 |
| sequence length | 5 | 每条序列五个位置 |
| heads | 4 | 四组注意力计算 |
| head dimension | 16 | 每组中一个 token 的特征数 |

如果把五个 token 分给不同头，就会改变每个头能够参与匹配和汇总的位置集合，不是这里要实现的多头注意力。

## head_shape 里的加号在拼形状

当前 `forward` 中，目标形状由这一行得到：

```python
head_shape=x.shape[:-1]+(self.num_heads,self.d_head)
```

第一次看时，要把张量的数值运算与形状对象的操作分开。`x.shape` 是 `torch.Size`，支持像元组一样切片和拼接。

对于这个例子：

```text
x.shape                    → (2, 5, 64)
x.shape[:-1]               → (2, 5)
(num_heads, d_head)        → (4, 16)
拼接后的 head_shape        → (2, 5, 4, 16)
```

`[:-1]` 取除了最后一维以外的形状。这里的 `+` 把两段形状连接起来，不是在把 2 与 4、5 与 16 分别相加，也没有修改输入张量里的数字。

这样写还保留了原输入的前导维度。虽然解释时用了一个 batch 维，表达式本身没有把这个维度的位置或长度写死。

## reshape 之后为什么还要 transpose

拆 Q 的已有代码是：

```python
q=q.reshape(head_shape).transpose(-3,-2)
```

这行有两个动作，不能只记最后的形状。

第一步将每个 token 的 64 个特征组织成四组 16 维：

```text
(2, 5, 64) → (2, 5, 4, 16)
             B  L  H  d_head
```

第二步交换倒数第三维和倒数第二维，即交换 token 轴与 head 轴：

```text
(2, 5, 4, 16) → (2, 4, 5, 16)
 B  L  H  d     B  H  L  d
```

为什么不保留 `(B, L, H, d_head)`？因为前面写好的注意力函数，把最后两维解释为“位置数、特征数”。整理为 `(B, H, L, d_head)` 后，batch 和 head 都成为前导维度，每个头可以独立处理自己的 `(L, d_head)` 矩阵。

这里的 `transpose` 在交换轴的含义，不是把二维矩阵转置的口诀照搬到四维张量。使用负数轴编号，是为了相对最后几维定位，在有额外前导维度时也能保留同样的关系。

K、V 做相同的整理。每个头独立计算注意力，并不意味着把它们写成四次 Python 循环；批量矩阵乘法可以一起完成这些计算。

## 每个头都会产生一张五乘五的匹配表

拆头后，Q、K、V 都是 `(2, 4, 5, 16)`。把最后两维代入前面的缩放点积注意力：

$$
S=\frac{QK^\top}{\sqrt{d_{\mathrm{head}}}}.
$$

| 步骤 | 形状 |
| --- | --- |
| Q | `(2, 4, 5, 16)` |
| K 交换最后两维 | `(2, 4, 16, 5)` |
| 匹配分数 | `(2, 4, 5, 5)` |
| Softmax 权重 | `(2, 4, 5, 5)` |
| 权重乘 V | `(2, 4, 5, 16)` |

每个 batch、每个头都有自己的五乘五分数矩阵。一行对应一个 query 位置，一列对应一个 key 位置。

缩放因子也跟着拆头后的维度变化。在这个例子中，它是 $\sqrt{16}$，不是 $\sqrt{64}$。已有注意力函数读取 `Q.shape[-1]`，所以整理好形状后，取到的自然就是每头维度。

这也解释了“多头”具体在哪里不同：各头使用不同的投影特征，因此可以产生不同的匹配分数和加权结果。仅仅把一个相同结果复制四份，并不会获得这些差异。

## 同一张因果 mask 可以用于所有头

自回归语言模型不能让当前位置读取未来 token。这一版模块内部根据序列长度构造布尔下三角 mask：包含对角线的下三角为 True，上三角为 False。

以三个位置说明：

| query 位置 | key 1 | key 2 | key 3 |
| --- | --- | --- | --- |
| 1 | True | False | False |
| 2 | True | True | False |
| 3 | True | True | True |

在五个位置的例子中，mask 的形状为 `(5, 5)`，分数的形状为 `(2, 4, 5, 5)`。广播从尾部对齐，相当于 mask 前面有两个长度为 1 的维度，可以应用到所有 batch 和所有头。

这里共享的是“哪些位置允许关注”的规则。每个头的分数仍然不同，通过 Softmax 得到的注意力权重也可以不同。共享 mask 不等于共享注意力结果。

mask 使用输入的 `device`，避免计算设备不一致；使用布尔类型，则能沿用上一节函数中 `True` 允许、`False` 屏蔽的约定。每行至少保留对角线上的自己，因此这张正常的非空因果 mask 不会构造出整行全被屏蔽的情况。

## 合头前，先把同一个 token 的结果放回一起

注意力输出仍然是 `(B, H, L, d_head)`。要回到输入形状，不能只看元素总数相同就直接压成 `(B, L, d_model)`。

当前代码先把轴交换回来，再合并最后两维：

```python
head_outputs=head_outputs.transpose(-3,-2)
merged=head_outputs.reshape(x.shape)
```

对应的形状变化是：

```text
(2, 4, 5, 16) → (2, 5, 4, 16) → (2, 5, 64)
 B  H  L  d     B  L  H  d      B  L  D
```

先交换回来，才能让同一个 token 的四个头在最后两维相邻。然后把 `4 × 16` 个特征连接为 64 维向量。

如果省略这次交换，直接 `reshape` 仍可能得到想要的外观尺寸，但会按照错误的轴顺序重新组织元素，把 head 与 token 的关系混淆。维度乘积一致只是必要条件；每一根轴的语义也必须对。

## reshape、view 和非连续张量

`transpose` 通常通过交换尺寸与步长来创建视图，不必实际移动底层存储中的全部元素。逻辑上的轴顺序改变后，张量可能不再满足通常的连续布局。

这时要谨慎区分 `view` 和 `reshape`：

- `view` 要求新的形状与已有尺寸、步长兼容，不能在任意转置结果上都成功。
- `reshape` 在布局允许时返回视图，否则可以复制数据，得到符合新形状的结果。

所以不能说 `reshape` 永远不复制，也不能把 `view` 的要求简化成“只能处理连续张量”——某些非连续张量也能在满足步长条件时使用特定的 `view`。

当前合头使用 `reshape`，可以处理转置后需要复制的情况。不过它只解决存储布局与目标形状的兼容问题，不会替我们判断轴的语义。前面该做的 `transpose`，仍然不能省略。这里没有做两种写法的性能比较，也不据此推断复制开销可以忽略。

## 已经回到 64 维，为什么还要输出 Linear

合头只是把各头结果连接起来，恢复了 `(2, 5, 64)`。它没有自动学会怎样组合不同头的特征。

最后的 `output_proj` 仍是 `64 → 64`，按每个 token 的完整合并向量做一次可学习线性变换：

$$
Y=\operatorname{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)
W_O^\top.
$$

输入输出宽度相同，不代表这是恒等变换。一个输出特征可以组合多个头提供的分量。与前面的投影一样，它在每个 token 内处理特征；这一步本身没有再次跨 token 汇总信息。

因此，完整路径可以读成：先用三组权重生成 Q、K、V，拆成头，在各头里做跨位置注意力，再合并各头，由第四组权重学习特征组合。

## 这次的真实错误，先发生在 class 上

继续写模块时，我把定义开头写成了 `def MultiHeadSelfAttention(nn.Module):`。这里需要定义类，应使用 `class`；`def` 用于定义方法或函数。

这个错误还没有进入任何注意力运算，先把“类”和“函数”的语法混在一起了。到这一篇，仍然不能因为已经理解了注意力公式，就忽略最基础的 Python 结构。

另一处需要认真对照的是测试适配器。模块里保存了四个 Linear，因此要加载四组测试给定的权重：

| 模块内部参数名 | 测试传入的权重 |
| --- | --- |
| `q_proj.weight` | `q_proj_weight` |
| `k_proj.weight` | `k_proj_weight` |
| `v_proj.weight` | `v_proj_weight` |
| `output_proj.weight` | `o_proj_weight` |

最后一项的名字并不完全相同。`load_state_dict` 的键要对应模块内部的成员名，所以这里是 `output_proj.weight`。固定测试权重加载进去之后，才能用同样的输入与参考输出比较，避免随机初始化干扰结果。

## 测试通过的范围，以及接下来要做什么

原学习过程运行了官方 `tests/test_model.py::test_multihead_self_attention`，结果为：

```text
test_multihead_self_attention PASSED
1 passed in 0.09s
```

这是在上一篇 SDPA 测试之后，新增的一项模块测试。它把投影、拆头、因果 mask、注意力计算、合头与输出投影接在一起，确认当前不含 RoPE 的多头自注意力在对应官方用例中与参考结果一致。

下一步才是 RoPE。到这次记录结束时，只介绍了根据位置旋转 Q、K，以及把每个头的特征两两配对的概念，尚未完成实现，也没有 RoPE 测试通过的记录。完整 Transformer block、语言模型和训练流程，继续留在后面的学习阶段。

## 发布时的复核

发布前对[独立代码快照](https://github.com/liangqianxing/cs336-labs/tree/31363223498fafd1528c550a51efbe0aed231736)重新运行了已完成组件对应的 9 项官方测试，结果为 `9 passed`。这次新增无 RoPE 多头自注意力，累计覆盖 8 类组件。完整清单与运行环境见[进度记录](https://github.com/liangqianxing/cs336-labs/blob/31363223498fafd1528c550a51efbe0aed231736/docs/progress.md)。前三篇仍保留最初 8 项测试对应的历史代码版本。

*封面使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 辅助生成，属于概念插画；轨道数量、块数与连线不代表精确计算，张量形状和运算关系以正文为准。*

## 参考资料

- [Stanford CS336 官方课程页面](https://cs336.stanford.edu/)
- [Assignment 1 官方仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)
- [Assignment 1 官方模型测试](https://github.com/stanford-cs336/assignment1-basics/blob/main/tests/test_model.py)
- [PyTorch：torch.Size](https://docs.pytorch.org/docs/stable/size.html)
- [PyTorch：torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html)
- [PyTorch：torch.transpose](https://docs.pytorch.org/docs/stable/generated/torch.transpose.html)
- [PyTorch：Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html)
- [PyTorch：Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [PyTorch：广播语义](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [PyTorch：torch.tril](https://docs.pytorch.org/docs/stable/generated/torch.tril.html)
