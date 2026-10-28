---
title: 'Stanford CS336 学习记录 05：RoPE 旋转位置编码与带位置的多头注意力'
date: '2026-10-21'
description: '从二维平面正交旋转推导 RoPE 的奇偶特征配对公式，理解为什么内积只依赖相对位置差 (i - j)，并用 register_buffer 预计算 cos/sin 缓存接入因果多头自注意力。'
series: Stanford CS336
seriesOrder: 5
cover: /images/posts/cs336-rope-rotary-position-embedding/cs336-rope-rotary-position-embedding-cover.webp
coverAlt: 二维特征平面中的旋转向量对与预计算余弦正弦网格的 AI 辅助概念插画
codeUrl: https://github.com/liangqianxing/cs336-labs/tree/b5c93ad0cc73178b2b2a62e518a0900da47a0fb7
attachmentUrl: /materials/cs336/experiments/cs336-05-rope-mha-20261021.zip
categories:
  - 学习笔记
tags:
  - CS336
  - PyTorch
  - Transformer
  - RoPE
  - Attention
draft: false
hidden: false
haloSync: false
haloPublished: false
---

[上一篇](/posts/cs336-multihead-attention-shapes/)完成了不含位置编码的因果多头自注意力。如果只看那一套矩阵乘法，把输入序列前两个 token 交换顺序（并在因果 mask 允许的范围内看同一个后缀 query 对它们的点积），点积本身并不知道哪个 key 离当前 query 更近、哪个更远。

现代语言模型（LLaMA、Qwen 等）普遍采用 **RoPE（Rotary Position Embedding，旋转位置编码）** 来注入相对位置信息。这一篇记录 `cs336_basics/rope.py` 与带 RoPE 的 `MultiHeadSelfAttention` 的完整推导、形状对齐与单元测试验证。

[查看代码仓库](https://github.com/liangqianxing/cs336-labs/tree/b5c93ad0cc73178b2b2a62e518a0900da47a0fb7) · [下载本节代码与测试 ZIP](/materials/cs336/experiments/cs336-05-rope-mha-20261021.zip)

## 为什么把 $d_k$ 维向量拆成 $d_k/2$ 个二维平面

在二维平面上，把一个向量 $\begin{bmatrix}x_0 \\ x_1\end{bmatrix}$ 逆时针旋转角度 $\theta$，对应的正交变换矩阵是：

$$
R(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix},
\qquad
R(\theta)
\begin{bmatrix}
x_0 \\
x_1
\end{bmatrix}
=
\begin{bmatrix}
x_0\cos\theta - x_1\sin\theta \\
x_0\sin\theta + x_1\cos\theta
\end{bmatrix}.
$$

如果位置 $i$ 的 query 二维向量旋转 $i\theta$，位置 $j$ 的 key 二维向量旋转 $j\theta$，那么它们旋转后的点积满足正交旋转群的加性性质：

$$
\left(R(i\theta)q\right)^\top \left(R(j\theta)k\right)
= q^\top R(i\theta)^\top R(j\theta) k
= q^\top R\big((j-i)\theta\big) k.
$$

也就是说，虽然旋转操作是按每个 token 自己的绝对位置 $i$ 和 $j$ 独立施加在 $Q$ 和 $K$ 上的，但点积注意力分数只依赖它们的**相对位置差** $i-j$。

推广到每头维度 $d_k$（要求 $d_k$ 为偶数），RoPE 将第 $k \in \{1, 2, \dots, d_k/2\}$ 对相邻分量 $(x_{2k-2}, x_{2k-1})$ 视为一个独立的二维子空间，赋予按几何级数递减的角频率：

$$
\theta_{i, k} = \frac{i}{\Theta^{(2k-2)/d_k}},
\qquad
\Theta = 10000.0.
$$

低维对（$k$ 小）波长短、旋转快，对局部相邻 token 顺序敏感；高维对（$k$ 大）波长长、旋转慢，负责承载长距离相对位置关系。

## 预计算 `cos_cached` 与 `sin_cached`：为什么用 `register_buffer`

因为 $\theta_{i, k}$ 只依赖位置编号 $i \in [0, \text{max\_seq\_len})$、特征对下标 $k$ 和超参数 $\Theta$，完全不包含可训练参数，如果在每次 `forward` 时重新调用 `torch.outer` 和三角函数，会产生不必要的重复开销。

但也不能直接把预计算好的张量挂在普通属性 `self.cos_cached` 上——那样调用 `model.to(device)` 时，普通属性不会跟着模型迁移设备；而如果包进 `nn.Parameter`，优化器又会误以为它是可训练权重，甚至写进 `state_dict` 破坏官方权重加载检查。

PyTorch 的标准解法是 `self.register_buffer(..., persistent=False)`：

```python
pair_dims = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
inv_freq = theta ** (-pair_dims / d_k)
positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
angles = torch.outer(positions, inv_freq)

self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
```

这里有三个关键细节：

| 设计点 | 原因 |
| --- | --- |
| `torch.arange(0, d_k, 2)` | 直接生成偶教学下标 $0, 2, 4, \dots, d_k-2$，恰好对应公式中的 $2k-2$ |
| `torch.outer(positions, inv_freq)` | 一次性生成形状为 `(max_seq_len, d_k // 2)` 的角度表 |
| `persistent=False` | 跟随模块迁移 device，但不写入 `state_dict()`，保证 `load_state_dict` 与官方测试快照严格对齐 |

## 奇偶切片与 `torch.stack` 交错还原

在 `RotaryPositionalEmbedding.forward(self, x, token_positions)` 中，输入 `x` 的形状是 `(..., seq_len, d_k)`，其中前导维度 `...` 可以是 `(batch,)`，也可以是拆头后的 `(batch, num_heads)`。

我们无需显式拼出稀疏的 $d_k \times d_k$ 分块对角矩阵，只需沿最后一维按步长 2 切出偶数位和奇数位：

```python
cos = self.cos_cached[token_positions]
sin = self.sin_cached[token_positions]

x_even = x[..., 0::2]
x_odd = x[..., 1::2]

rotated_even = x_even * cos - x_odd * sin
rotated_odd = x_even * sin + x_odd * cos

rotate = torch.stack((rotated_even, rotated_odd), dim=-1)
return rotate.reshape(x.shape).to(dtype=x.dtype)
```

为什么最后要用 `torch.stack(..., dim=-1)` 再 `.reshape(x.shape)`，而不是直接 `torch.cat((rotated_even, rotated_odd), dim=-1)`？

用一组 4 维向量对照就能一眼看清：

- `rotated_even` 包含新向量的第 `[0, 2]` 号分量，形状为 `(..., seq_len, 2)`；
- `rotated_odd` 包含新向量的第 `[1, 3]` 号分量，形状为 `(..., seq_len, 2)`；
- 如果用 `torch.cat(..., dim=-1)`，排出来的顺序会变成 `[0, 2, 1, 3]`，把原本成对的相邻坐标拆成了前后两半；
- 用 `torch.stack(..., dim=-1)` 先得到 `(..., seq_len, 2, 2)`，其中最后两维分别是 `[[第0分量, 第1分量], [第2分量, 第3分量]]`，紧接着 `.reshape(x.shape)` 按行展平，就精确还原了 `[0, 1, 2, 3]` 的交错顺序。

## 接入多头自注意力：只旋转 Q、K，以及 `token_positions` 的广播轴

把 RoPE 放进 `MultiHeadSelfAttention` 时，有两个极易踩坑的位置：

第一，**RoPE 只能作用在拆头后的 $Q$ 和 $K$ 上，绝对不能作用在 $V$ 上**，而且每个头的旋转维度必须是 `d_head = d_model // num_heads`，而不是 `d_model`。因为每个注意力头各自在一个 `d_head` 维空间内计算点积，所有头共享同一组位置编码规则。

第二，`token_positions` 与四维张量 `(batch, num_heads, seq_len, d_head)` 的广播对齐。在官方测试 `test_multihead_self_attention_with_rope` 中，传入的 `token_positions` 形状是 `(1, seq_len)`（二维），而拆头后的 `q` 形状是 `(batch, num_heads, seq_len, d_head)`（四维）：

```text
q_even 的形状：                      (batch, num_heads, seq_len, d_head // 2)
直接索引 cos_cached[token_positions]：(1, seq_len, d_head // 2)
```

如果直接相乘，PyTorch 从右往左对齐维度时，会把 `1` 对齐到 `num_heads`（4），完全没问题；但如果调用者传入的 `token_positions` 形状是 `(batch, seq_len)`（即 `batch > 1` 且没有 `num_heads` 这一维），那么 `(batch, seq_len, d_head // 2)` 的第 0 维 `batch` 就会错误地和 `q_even` 的第 1 维 `num_heads` 对齐！

为了同时兼容一维 `(seq_len,)`、二维 `(batch, seq_len)` 以及带 head 维度的 `token_positions`，我们在 `MultiHeadSelfAttention.forward` 中加入显式维度对齐：

```python
if self.rope is not None:
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
    elif token_positions.ndim == x.ndim - 1:
        token_positions = token_positions.unsqueeze(-2)
    q = self.rope(q, token_positions)
    k = self.rope(k, token_positions)
```

当输入 `x` 是 `(batch, seq_len, d_model)`（3 维）而 `token_positions` 是 `(batch, seq_len)`（2 维）时，`unsqueeze(-2)` 把它变成 `(batch, 1, seq_len)`。这样索引 `cos_cached` 后得到的 `(batch, 1, seq_len, d_head // 2)` 就能在 `num_heads` 轴上安全广播给所有头。

## 官方测试验证结果

对接 `adapters.py` 中的 `run_rope` 与 `run_multihead_self_attention_with_rope` 后，运行官方测试：

```text
tests/test_model.py::test_rope PASSED
tests/test_model.py::test_multihead_self_attention_with_rope PASSED
```

两项测试均在 `atol=1e-5` 容差下与官方快照完全吻合。下一篇继续把带 RoPE 的多头自注意力、RMSNorm 和 SwiGLU 组装成 [Pre-Norm Transformer Block 与完整 Transformer LM](/posts/cs336-pre-norm-transformer-block-lm/)。

*封面使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 辅助生成，属于概念插画；准确的公式、形状与代码逻辑以正文为准。*

## 参考资料

- [Su et al., 2021: RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Stanford CS336 Assignment 1 官方仓库](https://github.com/stanford-cs336/assignment1-basics)
- [PyTorch：Module.register_buffer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)
