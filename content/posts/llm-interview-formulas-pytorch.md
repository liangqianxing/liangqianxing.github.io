---
title: 大模型面试手写清单：展开公式、PyTorch 代码与易错点
date: 2026-09-30 10:33:33
description: 从 Attention、RoPE 和 SFT 到 LoRA、PPO、DPO、GRPO，逐步展开公式并给出可运行的 PyTorch 实现，补充 KV Cache、采样和训练循环的边界情况。
categories:
  - AI
tags:
  - LLM
  - PyTorch
  - Transformer
  - LoRA
  - RLHF
  - GRPO
  - 面试
hidden: true
haloPublished: true
---

准备大模型面试时，能写出一个 loss 的最终表达式还不够。继续往下写代码，往往就会遇到更具体的问题：logits 和 labels 怎么对齐，prompt 是否参与 loss，advantage 沿哪个维度归一化，KV Cache 中的位置编号从哪里开始。

这篇文章把一组适合白板练习的公式和 PyTorch 实现放在一起。公式按计算步骤展开，代码尽量保留张量形状与梯度关系。范围包括 Transformer、监督微调、参数高效微调、后训练和自回归推理；不涉及分布式训练与 CUDA kernel 的完整实现。

代码块可以按出现顺序放进同一个 Python 文件。Attention 示例不包含 padding、dropout、GQA 或模型特有的 RoPE 扩展；GRPO 使用结果奖励、组内总体标准差和按回答长度归一化，具体约定在对应章节说明。

## 1. 先统一符号和练习范围

| 符号 | 含义 | 常见形状 |
| --- | --- | --- |
| $B$ | batch size | — |
| $T$ | 当前序列长度 | — |
| $C$ | 隐藏维度 | — |
| $H$ | attention head 数量 | — |
| $D=C/H$ | 每个 head 的维度 | — |
| $V$ | 词表大小；与 attention 中的 Value 矩阵按上下文区分 | — |
| $G$ | GRPO 中每个 prompt 的回答数 | — |
| $X$ | 隐藏状态 | `[B, T, C]` |
| $Q$、$K$、$V_{\mathrm{attn}}$ | 分头后的 Query、Key、Value | `[B, H, T, D]` |
| logits | 每个位置对词表的预测 | `[B, T, V]` |
| response mask | 回答中的有效 token 为 `True` | `[B, T]` |

下面是本文的练习清单。它是按实现依赖组织的复习建议，不代表对面试出现频率的统计。

| 模块 | 公式要写到哪里 | 代码要写到哪里 |
| --- | --- | --- |
| Attention | Q/K/V、缩放、causal mask、softmax、合头 | `[B, H, T, D]` 的完整前向 |
| 位置编码 | 正余弦编码、二维旋转、相对位置关系 | RoPE 旋转与位置索引 |
| 归一化与 FFN | LayerNorm、RMSNorm、SwiGLU | 一个 Pre-Norm block |
| 监督训练 | CE、SFT、softmax 梯度 | 标签错位与回答 mask |
| 优化与微调 | AdamW、LoRA | 单步参数更新与低秩层 |
| 后训练 | GAE、PPO、DPO、GRPO | log-prob、clip、detach、归一化 |
| 推理 | KV Cache、temperature、top-k、top-p | 缓存拼接与采样 |

统一使用以下 imports：

```python
import math

import torch
from torch import nn
from torch.nn import functional as F
```

## 2. Causal Self-Attention：从投影到合头

以下公式采用行向量约定。输入是 $X\in\mathbb R^{B\times T\times C}$，先做三个投影 [1]：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V_{\rm attn}=XW_V
$$

拆成 $H$ 个 head 后，每个 head 的维度为 $D=C/H$。对一个 head，位置 $i$ 与 $j$ 的匹配分数是：

$$
S_{ij}=\frac{q_i^\top k_j}{\sqrt D}+M_{ij}
$$

causal mask 限制当前位置只能看到自己和过去：

$$
M_{ij}=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

沿 key 的维度做 softmax：

$$
P_{ij}=\frac{\exp(S_{ij})}{\sum_u\exp(S_{iu})}
$$

用权重汇总 Value：

$$
o_i=\sum_jP_{ij}v_j
$$

最后拼接各 head，并做输出投影：

$$
\operatorname{MHA}(X)=\operatorname{Concat}(O_1,\ldots,O_H)W_O
$$

可以先写独立的 attention 函数，再写投影层。函数中的 `query_start` 默认是 0，第 12 节会用它处理缓存后的绝对位置。

```python
def causal_attention(query, key, value, query_start=0):
    query_len = query.size(-2)
    key_len = key.size(-2)
    head_dim = query.size(-1)

    query_positions = torch.arange(query_len, device=query.device) + query_start
    key_positions = torch.arange(key_len, device=query.device)
    allowed = key_positions[None, :] <= query_positions[:, None]

    scores = query.float() @ key.float().transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)
    scores = scores.masked_fill(~allowed, -float("inf"))
    weights = scores.softmax(dim=-1).to(value.dtype)
    return weights @ value


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, dim = hidden_states.shape
        projected = self.qkv(hidden_states)
        projected = projected.reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = projected.unbind(dim=0)

        attended = causal_attention(query, key, value)
        merged = attended.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.output(merged)
```

需要解释的细节：

- 缩放因子使用每个 head 的 $D$，不是模型隐藏维度 $C$。
- softmax 的最后一维是 key 位置，分数形状为 `[B, H, T_query, T_key]`。
- 这里在 FP32 中计算点积和 softmax，再转回 Value 的 dtype。
- 若加入 padding mask，要保证有效 query 至少有一个可见 key；整行都是 $-\infty$ 会使普通 softmax 产生 NaN。

## 3. 位置编码：正余弦编码与 RoPE

### 3.1 正余弦位置编码

原始 Transformer 将位置编码加到输入 embedding 上 [1]。第 $j$ 对维度为：

$$
\operatorname{PE}(m,2j)=\sin\left(\frac{m}{10000^{2j/C}}\right)
$$

$$
\operatorname{PE}(m,2j+1)=\cos\left(\frac{m}{10000^{2j/C}}\right)
$$

于是输入变成：

$$
X_m=\operatorname{Embedding}(y_m)+\operatorname{PE}(m)
$$

### 3.2 RoPE 的二维旋转

RoPE 对投影后的 Q、K 进行旋转 [2]。令 head dimension 为偶数，第 $j$ 对维度的频率为：

$$
\theta_j=10000^{-2j/D},\qquad j=0,\ldots,D/2-1
$$

在位置 $m$，旋转角度为 $m\theta_j$：

$$
\begin{bmatrix}
x'_{2j}\\x'_{2j+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_j)&-\sin(m\theta_j)\\
\sin(m\theta_j)&\cos(m\theta_j)
\end{bmatrix}
\begin{bmatrix}
x_{2j}\\x_{2j+1}
\end{bmatrix}
$$

也可以直接展开成两行：

$$
x'_{2j}=x_{2j}\cos(m\theta_j)-x_{2j+1}\sin(m\theta_j)
$$

$$
x'_{2j+1}=x_{2j}\sin(m\theta_j)+x_{2j+1}\cos(m\theta_j)
$$

由于旋转矩阵满足 $R_m^\top R_n=R_{n-m}$，有：

$$
(R_mq)^\top(R_nk)=q^\top R_{n-m}k
$$

这解释了相对位置如何进入 Q、K 的内积。

```python
def apply_rope(tensor, positions, base=10000.0):
    head_dim = tensor.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    frequencies = base ** (
        -torch.arange(0, head_dim, 2, device=tensor.device).float() / head_dim
    )
    angles = positions.to(tensor.device).float()[:, None] * frequencies[None, :]
    cosine = angles.cos()[None, None, :, :]
    sine = angles.sin()[None, None, :, :]

    even = tensor.float()[..., 0::2]
    odd = tensor.float()[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    )
    return rotated.flatten(-2).to(tensor.dtype)


class RoPESelfAttention(CausalSelfAttention):
    def forward(self, hidden_states, cache=None):
        batch_size, seq_len, dim = hidden_states.shape
        projected = self.qkv(hidden_states).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = projected.unbind(dim=0)

        cached_len = 0 if cache is None else cache[0].size(-2)
        positions = torch.arange(seq_len, device=hidden_states.device) + cached_len
        query = apply_rope(query, positions)
        key = apply_rope(key, positions)

        if cache is not None:
            key = torch.cat((cache[0], key), dim=-2)
            value = torch.cat((cache[1], value), dim=-2)

        attended = causal_attention(query, key, value, query_start=cached_len)
        merged = attended.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.output(merged), (key, value)
```

代码采用相邻维度配对：`(0, 1)`、`(2, 3)`。也有模型使用前后半区配对；加载已有模型权重时必须保持原模型的布局。缓存中保存的是已经旋转过的 Key，下一步只旋转新 Key。

## 4. LayerNorm 与 RMSNorm

LayerNorm 对每个 token 的隐藏维度计算均值和方差：

$$
\mu=\frac1C\sum_{j=1}^Cx_j
$$

$$
\sigma^2=\frac1C\sum_{j=1}^C(x_j-\mu)^2
$$

$$
\operatorname{LayerNorm}(x)=
\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

RMSNorm 去掉均值中心化，使用均方根进行缩放 [3]：

$$
\operatorname{RMS}(x)=\sqrt{\frac1C\sum_{j=1}^Cx_j^2+\epsilon}
$$

$$
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\operatorname{RMS}(x)}
$$

```python
class ManualLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states):
        states = hidden_states.float()
        mean = states.mean(dim=-1, keepdim=True)
        variance = (states - mean).square().mean(dim=-1, keepdim=True)
        normalized = (states - mean) * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight + self.bias).to(hidden_states.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        states = hidden_states.float()
        mean_square = states.square().mean(dim=-1, keepdim=True)
        normalized = states * torch.rsqrt(mean_square + self.eps)
        return (normalized * self.weight).to(hidden_states.dtype)
```

两者都沿最后的隐藏维度归一化。这里使用总体方差；若手写 LayerNorm 时直接调用带样本修正的 `std()`，会与这个定义不一致。

## 5. FFN、SwiGLU 与 Pre-Norm Block

普通 FFN 包含两个线性层：

$$
\operatorname{FFN}(x)=W_2\phi(W_1x+b_1)+b_2
$$

这一节改用列向量写法，以便直观看到线性层作用顺序。SwiGLU 有三组权重 [4]：

$$
g=W_{\rm gate}x
$$

$$
u=W_{\rm up}x
$$

$$
\operatorname{SiLU}(g)=g\odot\sigma(g)
$$

$$
\operatorname{SwiGLU}(x)=W_{\rm down}\bigl(\operatorname{SiLU}(g)\odot u\bigr)
$$

Pre-Norm block 的两次残差连接为：

$$
X'=X+\operatorname{Attention}(\operatorname{Norm}_1(X))
$$

$$
Y=X'+\operatorname{FFN}(\operatorname{Norm}_2(X'))
$$

```python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states):
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attention = RoPESelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, hidden_states):
        attended, _ = self.attention(self.norm1(hidden_states))
        hidden_states = hidden_states + attended
        return hidden_states + self.ffn(self.norm2(hidden_states))
```

如果忽略 bias，普通 FFN 的参数量约为 $2C F_{\rm plain}$，SwiGLU 约为 $3C F_{\rm gated}$。在比较相同参数预算时，可取 $F_{\rm gated}\approx\frac23F_{\rm plain}$，而不是固定两个版本使用相同中间维度。

## 6. Cross-Entropy、SFT 与 Softmax 梯度

### 6.1 Softmax 与单个 token 的 loss

给定一个位置的 logits $z\in\mathbb R^V$：

$$
p_c=\frac{e^{z_c}}{\sum_{j=1}^Ve^{z_j}}
$$

目标 token 是 $y$，交叉熵为：

$$
L=-\log p_y=-z_y+\log\sum_{j=1}^Ve^{z_j}
$$

对每个 logit 求导：

$$
\frac{\partial L}{\partial z_c}=p_c-\mathbf1[c=y]
$$

注意：这是一个 token 的未加权 loss；若对多个 token 求平均，梯度还会乘上平均的系数。

### 6.2 SFT 的 label shift 与 mask

模型位置 $t$ 的 logits 预测位置 $t+1$ 的 token。令 $m_{b,t+1}=1$ 表示这个目标 token 属于有效回答：

$$
L_{\rm SFT}=
-\frac{\sum_{b,t}m_{b,t+1}\log p_\theta(y_{b,t+1}\mid y_{b,\le t})}
{\sum_{b,t}m_{b,t+1}}
$$

因此要把 `logits[:, :-1]` 与 `input_ids[:, 1:]` 对齐。第一个回答 token 可以由最后一个 prompt 位置的 logits 预测，是否计入 loss 应由**目标 token** 的 mask 决定。

```python
def token_logps(logits, input_ids):
    log_probs = logits[:, :-1].float().log_softmax(dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    return log_probs.gather(dim=-1, index=targets).squeeze(-1)


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def sft_loss(logits, input_ids, response_mask):
    log_probs = token_logps(logits, input_ids)
    target_mask = response_mask[:, 1:]
    return -masked_mean(log_probs, target_mask)


def sft_loss_with_ignore_index(logits, labels):
    shifted_logits = logits[:, :-1].float().reshape(-1, logits.size(-1))
    shifted_labels = labels[:, 1:].reshape(-1)
    valid_count = shifted_labels.ne(-100).sum().clamp_min(1)
    total = F.cross_entropy(
        shifted_logits,
        shifted_labels,
        ignore_index=-100,
        reduction="sum",
    )
    return total / valid_count
```

`labels` 版本用 `-100` 标记 prompt 和 padding；这是 PyTorch `cross_entropy` 的默认忽略值 [11]。`input_ids` 版本的 ID 必须是合法词表索引，不能把 `-100` 交给 `gather`。

用相同 tokenization、mask 和平均方式得到平均负对数似然后，perplexity 为：

$$
\operatorname{PPL}=\exp(L_{\rm CE})
$$

这里得到的是所选 token 集合上的 perplexity，不能直接与使用不同 tokenizer 或统计范围的结果比较。

## 7. AdamW：动量与独立的 Weight Decay

先更新一阶和二阶动量：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

再做偏差修正：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

采用 PyTorch 常见的学习率缩放约定，参数更新为：

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

AdamW 的 weight decay 不加入 $g_t$，不会进入动量估计 [5]。它与“先给 loss 加 L2 项，再使用 Adam”通常不等价。

```python
@torch.no_grad()
def adamw_step(parameter, gradient, first_moment, second_moment, step,
               lr=1e-4, beta1=0.9, beta2=0.999,
               eps=1e-8, weight_decay=0.01):
    first_moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
    second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

    corrected_first = first_moment / (1 - beta1 ** step)
    corrected_second = second_moment / (1 - beta2 ** step)
    parameter.mul_(1 - lr * weight_decay)
    parameter.addcdiv_(
        corrected_first,
        corrected_second.sqrt().add(eps),
        value=-lr,
    )
```

`step` 从 1 开始，两个 moment 在第一次更新前初始化为零，并在之后的更新中持续复用。

## 8. LoRA：冻结原权重，训练低秩增量

原权重与增量的形状为 [6]：

$$
W_0\in\mathbb R^{d_{\rm out}\times d_{\rm in}}
$$

$$
A\in\mathbb R^{r\times d_{\rm in}},\qquad
B\in\mathbb R^{d_{\rm out}\times r}
$$

修改后的权重为：

$$
W'=W_0+\frac\alpha rBA
$$

列向量形式的前向为：

$$
y=W_0x+\frac\alpha rB(Ax)
$$

新增的可训练参数量为：

$$
N_{\rm LoRA}=r(d_{\rm in}+d_{\rm out})
$$

```python
class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank

        self.factor_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.factor_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.normal_(self.factor_a, mean=0.0, std=0.02)

    def forward(self, inputs):
        delta = F.linear(F.linear(inputs, self.factor_a), self.factor_b)
        return self.base(inputs) + self.scale * delta
```

这里随机初始化 $A$、将 $B$ 初始化为零，使初始增量为零。如果两个矩阵都初始化为零，两个分支的梯度都会被另一方的零值挡住。首次反向传播时，由于 $B=0$，$A$ 的梯度为零是正常现象。

## 9. PPO：GAE、概率比与 Clip

### 9.1 GAE advantage

令 $d_t=1$ 表示当前位置之后是真正的终止状态，$V_t$ 是价值估计。TD residual 为：

$$
\delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t
$$

GAE 可以倒序递推 [7]：

$$
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}
$$

对应的 value target 为：

$$
\hat R_t=A_t+V_t
$$

```python
@torch.no_grad()
def compute_gae(rewards, values, terminals, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])

    for index in reversed(range(rewards.size(1))):
        alive = 1.0 - terminals[:, index].float()
        delta = (
            rewards[:, index]
            + gamma * alive * values[:, index + 1]
            - values[:, index]
        )
        running = delta + gamma * lam * alive * running
        advantages[:, index] = running

    returns = advantages + values[:, :-1]
    return advantages, returns
```

输入 `rewards`、`terminals` 为 `[B, T]`，`values` 为 `[B, T+1]`。这段代码假设轨迹已经对齐，没有尾部 padding。达到真正终止状态时关闭 bootstrap；仅因长度上限截断时，是否 bootstrap 应按任务定义处理，不能自动等同于 terminal。

### 9.2 Clipped policy objective

概率比比较的是当前策略与采样时的 old policy [8]：

$$
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\rm old}(a_t\mid s_t)}
$$

用 log-prob 实现：

$$
\rho_t=\exp\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\rm old}(a_t\mid s_t)\right)
$$

最大化的 clipped objective 为：

$$
J_{\rm clip}=\mathbb E_t\left[
\min\left(
\rho_tA_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

优化器最小化 loss，因此：

$$
L_{\rm policy}=-J_{\rm clip}
$$

若加入 value loss 和 entropy bonus，一种常见写法是：

$$
L_{\rm PPO}=L_{\rm policy}
+\frac{c_v}{2}\mathbb E_t[(V_\theta(s_t)-\hat R_t)^2]
-c_e\mathbb E_t[\mathcal H(\pi_\theta(\cdot\mid s_t))]
$$

```python
def ppo_policy_loss(new_logp, old_logp, advantages, mask, clip_eps=0.2):
    ratio = (new_logp.float() - old_logp.detach().float()).exp()
    advantages = advantages.detach().float()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    return -masked_mean(torch.minimum(unclipped, clipped), mask)
```

Clip 不是把所有概率比强行限制在区间内；它改变的是目标函数对过度更新的激励。LLM 的完整 PPO 训练还需要奖励构造、价值模型和参考策略 KL 等环节，这里只实现 policy loss。

## 10. DPO：先汇总回答 Log-Prob，再比较偏好

对于 prompt $x$ 的一条回答 $y$，序列概率分解为：

$$
\log\pi_\theta(y\mid x)=\sum_{t\in\text{回答}}
\log\pi_\theta(y_t\mid x,y_{<t})
$$

令 $y^+$ 是偏好回答，$y^-$ 是拒绝回答，分别计算当前策略与参考策略的差 [9]：

$$
\Delta_\theta=\log\pi_\theta(y^+\mid x)-\log\pi_\theta(y^-\mid x)
$$

$$
\Delta_{\rm ref}=\log\pi_{\rm ref}(y^+\mid x)-\log\pi_{\rm ref}(y^-\mid x)
$$

偏好 margin 为：

$$
z=\beta(\Delta_\theta-\Delta_{\rm ref})
$$

DPO loss 为：

$$
L_{\rm DPO}=-\mathbb E_{(x,y^+,y^-)}[\log\sigma(z)]
$$

```python
def response_logp(logits, input_ids, response_mask):
    log_probs = token_logps(logits, input_ids)
    mask = response_mask[:, 1:].to(log_probs.dtype)
    return (log_probs * mask).sum(dim=-1)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen,
             reference_rejected, beta=0.1):
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen.detach() - reference_rejected.detach()
    return -F.logsigmoid(beta * (policy_margin - reference_margin)).mean()
```

四个输入都是 `[B]` 的回答 log-prob。标准 DPO 在这里使用回答 token 的**和**；改成长度平均会改变目标。参考模型应冻结并在 `torch.no_grad()` 下计算，避免构建不必要的计算图。

## 11. GRPO：组内 Advantage 与 Token Loss

本节讨论每条回答只有一个结果奖励的情形。它对应 DeepSeekMath 中的 outcome supervision；过程奖励下 advantage 的构造不同 [10]。

### 11.1 先在同一个 prompt 内比较回答

对 batch 中第 $b$ 个 prompt，采样 $G$ 条回答，其奖励为 $r_{b,1},\ldots,r_{b,G}$。

组内均值为：

$$
\mu_b=\frac1G\sum_{i=1}^Gr_{b,i}
$$

本文显式采用总体标准差：

$$
\sigma_b=\sqrt{\frac1G\sum_{i=1}^G(r_{b,i}-\mu_b)^2}
$$

每条回答的 advantage 为：

$$
A_{b,i}=\frac{r_{b,i}-\mu_b}{\sigma_b+\epsilon_{\rm norm}}
$$

沿 token 维度广播：

$$
A_{b,i,t}=A_{b,i}
$$

例如奖励是 `[1, 2, 3]`：

$$
\mu=2,\qquad\sigma=\sqrt{2/3}
$$

$$
A\approx[-1.225,\ 0,\ 1.225]
$$

如果同组奖励完全相等，advantage 全为 0，policy 项没有相对优劣信号；若使用非零 KL 系数，KL 项仍可能产生梯度。

### 11.2 再计算概率比和 KL 项

概率比使用 old policy：

$$
\rho_{b,i,t}=\exp(\log\pi_\theta(y_{b,i,t})-\log\pi_{\rm old}(y_{b,i,t}))
$$

参考策略的作用是计算 KL 正则。为简洁起见省略共同的 prompt 与历史 token 条件，令：

$$
d_{b,i,t}=\log\pi_{\rm ref}(y_{b,i,t})-\log\pi_\theta(y_{b,i,t})
$$

常用的逐 token 估计量是 [10]：

$$
\widehat D_{{\rm KL},b,i,t}=\exp(d_{b,i,t})-d_{b,i,t}-1
$$

它在数学上非负。当 token 从当前策略采样时，其期望对应 $D_{\rm KL}(\pi_\theta\Vert\pi_{\rm ref})$；重复使用 old-policy 样本时，不能直接宣称未加修正的样本平均仍是当前策略 KL 的无偏估计。

### 11.3 最后定义归一化方式

令 $m_{b,i,t}$ 表示有效回答 token，$L_{b,i}=\sum_t m_{b,i,t}$。定义 clipped policy 项：

$$
u_{b,i,t}=\min\left(
\rho_{b,i,t}A_{b,i},
\operatorname{clip}(\rho_{b,i,t},1-\epsilon_{\rm clip},1+\epsilon_{\rm clip})A_{b,i}
\right)
$$

先在每条回答内求平均：

$$
\ell_{b,i}=\frac1{L_{b,i}}\sum_tm_{b,i,t}
\left(-u_{b,i,t}+\beta\widehat D_{{\rm KL},b,i,t}\right)
$$

所有回答非空时，整体 loss 为：

$$
L_{\rm GRPO}=\frac1{BG}\sum_{b=1}^B\sum_{i=1}^G\ell_{b,i}
$$

代码对零有效 token 的回答不计入最终平均。正常训练应在采样阶段保证有效回答，不能靠这个保护分支修复采样数据。

```python
def group_advantages(rewards, norm_eps=1e-8):
    rewards = rewards.float()
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return ((rewards - mean) / (std + norm_eps)).detach()


def grpo_loss(new_logp, old_logp, reference_logp, rewards, response_mask,
              clip_eps=0.2, beta=0.01):
    advantage = group_advantages(rewards).unsqueeze(-1)
    new_logp = new_logp.float()
    ratio = (new_logp - old_logp.detach().float()).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage
    token_loss = -torch.minimum(unclipped, clipped)

    if beta != 0:
        log_ratio = reference_logp.detach().float() - new_logp
        kl_estimate = torch.expm1(log_ratio) - log_ratio
        token_loss = token_loss + beta * kl_estimate

    mask = response_mask.to(token_loss.dtype)
    lengths = mask.sum(dim=-1)
    per_answer = (token_loss * mask).sum(dim=-1) / lengths.clamp_min(1)
    return masked_mean(per_answer, lengths > 0)
```

输入约定：

- `rewards` 是 `[B, G]`；多个奖励项先合成为每条回答的标量奖励。
- `new_logp`、`old_logp`、`reference_logp`、`response_mask` 都是 `[B, G, T]`，已对齐同一批回答 token。
- old/reference log-prob 和 advantage 不参与梯度，当前策略的 log-prob 必须保留梯度。
- `unbiased=False` 与本文总体标准差公式一致；使用样本标准差会得到不同的 advantage 尺度。
- 同一 prompt 的 $G$ 条回答不能与其他 prompt 混在一起计算组内均值。

不同 GRPO 实现会调整 reward scaling、标准差和长度归一化。面试手写时，先说明自己选择“组内标准差、每条回答平均”还是“所有有效 token 一起平均”，再写 loss；这两种平均方式对长短回答的权重不同。

## 12. KV Cache：位置偏移与非方形 Causal Mask

生成新 token 时复用历史 Key、Value：

$$
K_{1:t}=\operatorname{Concat}(K_{1:t-1},K_t)
$$

$$
V_{1:t}=\operatorname{Concat}(V_{1:t-1},V_t)
$$

单 token attention 为：

$$
o_t=\operatorname{softmax}\left(\frac{q_tK_{1:t}^\top}{\sqrt D}\right)V_{1:t}
$$

第 3 节的 `RoPESelfAttention` 已经实现 cache。假设有 $P$ 个历史 token、一次处理 $L$ 个新 token，正确的可见关系为：

$$
\operatorname{allowed}(i,j)=\mathbf1[j\le P+i],\qquad i=0,\ldots,L-1
$$

这也是 `query_start=cached_len` 的作用。新 token 的 RoPE 位置必须从 $P$ 开始。

以下代码核对整段前向、逐 token decode、分块 decode 的输出是否一致：

```python
@torch.no_grad()
def check_cache_equivalence():
    torch.manual_seed(7)
    attention = RoPESelfAttention(dim=32, num_heads=4).eval()
    inputs = torch.randn(2, 6, 32)
    full_output, _ = attention(inputs)

    cache = None
    pieces = []
    for index in range(inputs.size(1)):
        output, cache = attention(inputs[:, index:index + 1], cache=cache)
        pieces.append(output)
    decoded_output = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full_output, decoded_output, rtol=1e-5, atol=1e-6)

    prefix_output, cache = attention(inputs[:, :3])
    suffix_output, _ = attention(inputs[:, 3:], cache=cache)
    chunked_output = torch.cat((prefix_output, suffix_output), dim=1)
    torch.testing.assert_close(full_output, chunked_output, rtol=1e-5, atol=1e-6)
```

如果改用 PyTorch SDPA，单 token decode 的 cache 没有未来 token，可以使用 `is_causal=False`。不要在 query 长度为 1、key 长度大于 1 时直接依赖 `is_causal=True`：官方文档说明，非方形 causal mask 使用左上对齐 [12]。分块 decode 应传入按绝对位置构造的 mask。

KV Cache 的元素数量还可以顺便手写。令层数为 $N$、KV head 数为 $H_{\rm KV}$、每个元素占 $s$ 字节，则缓存近似占用：

$$
\operatorname{Bytes}_{\rm KV}=2NBT H_{\rm KV}Ds
$$

系数 2 来自 K 和 V。普通 MHA 中 $H_{\rm KV}=H$；GQA/MQA 改变 KV head 数。这个式子不包括分配器、分页和其他运行时开销。

## 13. Temperature、Top-k、Top-p 与 EOS

Temperature 调整采样分布：

$$
p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)},\qquad\tau>0
$$

当使用 `temperature=0` 时，代码单独走 argmax，避免除零。

Top-k 保留最高的 $k$ 个 logits 对应的集合 $S_k$：

$$
z'_i=
\begin{cases}
z_i,&i\in S_k\\
-\infty,&i\notin S_k
\end{cases}
$$

Top-p 将概率降序排列，取累计概率首次达到 $p$ 的最小前缀：

$$
k^*=\min\left\{k:\sum_{j=1}^kp_{(j)}\ge p\right\}
$$

保留第 $1$ 到第 $k^*$ 个 token，再归一化采样。

```python
def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.float() / temperature
    if top_k is not None:
        count = min(top_k, scores.size(-1))
        indices = scores.topk(count, dim=-1).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(dim=-1, index=indices, value=True)
        scores = scores.masked_fill(~keep, -float("inf"))

    if top_p is not None and top_p < 1:
        sorted_scores, indices = scores.sort(dim=-1, descending=True)
        cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove = torch.zeros_like(cumulative, dtype=torch.bool)
        remove[..., 1:] = cumulative[..., :-1] >= top_p
        sorted_scores = sorted_scores.masked_fill(remove, -float("inf"))
        scores = scores.scatter(dim=-1, index=indices, src=sorted_scores)

    return torch.multinomial(scores.softmax(dim=-1), num_samples=1)
```

这里 `logits` 为 `[B, V]`，输出为 `[B, 1]`。Top-p 保留越过累计阈值的那个 token；否则很小的 $p$ 可能删掉所有候选。同时启用 top-k、top-p 时，这份实现先执行 top-k，再在过滤后的分布上执行 top-p。

生成循环还要维护每条序列是否已经输出 EOS。下面的最小版本假定 prompt 等长且没有 padding，模型接受 `input_ids` 并返回带 `.logits` 的结果；它每轮重算完整序列，适合检查终止逻辑。高效版本再接入 KV Cache。

```python
@torch.no_grad()
def generate_greedy(model, input_ids, eos_token_id, max_new_tokens):
    model.eval()
    generated = input_ids
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

    for _ in range(max_new_tokens):
        logits = model(input_ids=generated).logits[:, -1]
        next_ids = sample_next(logits, temperature=0)
        next_ids = next_ids.masked_fill(finished[:, None], eos_token_id)
        generated = torch.cat((generated, next_ids), dim=1)
        finished = finished | next_ids.squeeze(-1).eq(eos_token_id)
        if finished.all():
            break

    return generated
```

## 14. 一个最小训练循环

手写循环时，先把梯度的生命周期写清楚：

$$
g=\nabla_\theta L
$$

梯度裁剪的一种表达方式为：

$$
\widetilde g=g\cdot\min\left(1,\frac{c}{\lVert g\rVert_2+\epsilon}\right)
$$

优化器再使用 $\widetilde g$ 更新参数。

```python
def train_sft_step(model, optimizer, input_ids, response_mask, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids=input_ids).logits
    loss = sft_loss(logits, input_ids, response_mask)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.detach()
```

这段循环同样假设输入没有 padding。真实 batch 如果包含 padding，既需要传给模型的 attention mask，也需要 loss 的 response mask；两个 mask 的职责不同。

梯度累积时，多次 `backward()` 之间不清零，到累积边界再裁剪和更新。只有各 micro-batch 的有效 token 数相同，简单地把每个 mean loss 除以累积步数才等价于全局 token 平均；token 数不同时，应按有效 token 数加权。

## 15. 写完后，用这些问题检查自己

| 检查点 | 应该能回答的问题 |
| --- | --- |
| Attention | Q/K/V 形状是什么？mask 的 `True` 在当前 API 中表示允许还是屏蔽？ |
| RoPE | head dimension 是否为偶数？cache 后的位置从哪里开始？ |
| SFT | 哪个位置预测第一个回答 token？prompt/padding 是否参与 loss？ |
| LoRA | 原权重冻结了吗？为什么不能让两个低秩矩阵都从零开始？ |
| PPO | advantage 和 old log-prob 是否停止梯度？loss 的负号是否正确？ |
| DPO | sequence log-prob 是求和还是平均？参考策略是否被冻结？ |
| GRPO | reward 是否按 prompt 分组？std 使用哪种定义？长短回答如何加权？ |
| KV Cache | 单 token 和多 token decode 的 mask 是否使用绝对位置？ |
| 采样 | temperature 为 0 怎么处理？top-p 是否保留阈值处的 token？ |

本文的 16 个代码块在 Python 3.14、PyTorch 2.14.0 的 CPU 环境下执行，21 组检查通过。核对范围包括：Attention 与 PyTorch SDPA 的输出和梯度、causal mask 的隔离效果、RoPE 的范数与相对位置性质、LayerNorm 与官方实现的一致性、SFT 标签对齐、AdamW 参数更新、LoRA 冻结、PPO/GRPO clip 与梯度隔离、DPO margin、KV Cache 等价性、采样约束以及最小训练和生成循环。它们验证这些教学实现的局部行为，不代表完成了大模型训练效果或性能评测。

想继续复习概念，可以配合站内的 [Transformer、ViT 与 CLIP 基础](https://diycv.top/archives/transformer-vit-clip-interview-guide) 和 [DPO 精读](https://diycv.top/archives/dpo-direct-preference-optimization)。练习时先从白纸写公式，再标出 shape，最后写 forward 和 loss；遇到边界情况时，解释清楚自己的约定。

## 参考资料

公式依据原始论文和官方文档核对；代码是为本文编写的简化实现。

1. Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，重点对应第 3.2 节 Attention 与第 3.5 节位置编码。
2. Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)，重点对应第 3.2 节的二维旋转与一般形式。
3. Zhang and Sennrich. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)，重点对应第 4 节 RMSNorm。
4. Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)，重点对应第 2 节的 FFN 变体与参数预算。
5. Loshchilov and Hutter. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)，重点对应 Algorithm 2 的动量与解耦 weight decay。
6. Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)，重点对应第 4.1 节的低秩更新与初始化。
7. Schulman et al. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)，重点对应第 3 节的 TD residual 与 GAE。
8. Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，重点对应第 3 节 clipped surrogate objective。
9. Rafailov et al. [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)，重点对应第 4 节、Equation 7。
10. Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)，重点对应第 4.1 节的 GRPO、KL 估计与结果奖励。
11. PyTorch 官方文档：[torch.nn.functional.cross_entropy](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html)。
12. PyTorch 官方文档：[torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。
