---
title: "大模型面试手写题：核心公式与 PyTorch 实现"
date: 2026-09-30 10:03:36
description: "按 Transformer、SFT、参数高效微调、后训练与推理五条线，展开大模型面试常见公式，并给出对应的 PyTorch 手写实现。"
categories:
  - AI
tags:
  - LLM
  - Transformer
  - PyTorch
  - 模型后训练
  - 面试
hidden: true
haloPublished: true
---

大模型面试要求手写代码时，难点通常不在记住一个函数名，而在于把公式、张量形状、时间步对齐和 mask 同时写对。本文把常见题目放在一处：每题先写公式，再给出尽量短、能独立解释的 PyTorch 实现。代码是教学版本，不包含分布式训练、FlashAttention 或批量推理调度。

如果想系统回顾架构背景，可先看[Transformer、ViT 与 CLIP 基础](/archives/transformer-vit-clip-interview-guide)；想单独推导偏好学习，可看[DPO 精读](/archives/dpo-direct-preference-optimization)。本文聚焦面试现场的手写过程。

**统一约定**：$B$ 为 batch size，$T$ 为序列长度，$C$ 为隐藏维度，$H$ 为注意力头数，$D=C/H$，$V$ 为词表大小。所有代码块沿用下面的导入；没有特殊说明时，序列的有效回答 token 至少有一个。

```python
import math
import torch
from torch import nn
from torch.nn import functional as F
```

## 1. Transformer：Attention、RoPE 与前馈层

### 1.1 Causal Multi-Head Attention

给定 $X\in\mathbb{R}^{B\times T\times C}$，每个头的输入投影为：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

分头后，$Q,K,V$ 的形状都是 $[B,H,T,D]$。第 $i$ 个 query 对第 $j$ 个 key 的分数为：

$$
S_{ij}=\frac{q_i^\top k_j}{\sqrt D}+M_{ij}
$$

Decoder-only 模型使用 causal mask：

$$
M_{ij}=\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

沿 key 维度归一化，再加权求和：

$$
P_{ij}=\frac{\exp(S_{ij})}{\sum_{u=1}^{T}\exp(S_{iu})}
$$

$$
o_i=\sum_{j=1}^{T}P_{ij}v_j
$$

多头的输出拼接后再投影：

$$
\operatorname{MHA}(X)=\operatorname{Concat}(O_1,\ldots,O_H)W_O
$$

下面的教学实现只处理没有 padding 的整段序列；有 padding 时，还要在分数矩阵上屏蔽无效 key。

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.d = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(b, t, 3, self.h, self.d)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )  # 每个张量都是 [B, H, T, D]

        positions = torch.arange(t, device=x.device)
        q, k = rope(q, positions), rope(k, positions)

        scores = q.float() @ k.float().transpose(-2, -1)
        scores = scores / math.sqrt(self.d)
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal[None, None], -torch.inf)
        probs = scores.softmax(dim=-1).to(v.dtype)

        y = probs @ v
        return self.out(y.transpose(1, 2).reshape(b, t, c))
```

这里的 `rope` 定义在下一节。面试追问通常是：为什么除以 $\sqrt D$；softmax 沿哪一维；mask 放在 softmax 前还是后；输出如何从 $[B,H,T,D]$ 回到 $[B,T,C]$。

### 1.2 RoPE

RoPE 对每个头的相邻两维做旋转。对第 $j$ 对维度，频率和位置 $m$ 的角度为：

$$
\theta_j=10000^{-2j/D},\qquad \phi_{m,j}=m\theta_j
$$

$$
\begin{bmatrix}x'_{2j}\\x'_{2j+1}\end{bmatrix}
=
\begin{bmatrix}
\cos\phi_{m,j}&-\sin\phi_{m,j}\\
\sin\phi_{m,j}&\cos\phi_{m,j}
\end{bmatrix}
\begin{bmatrix}x_{2j}\\x_{2j+1}\end{bmatrix}
$$

Q 和 K 都旋转。由于 $R_m^\top R_n=R_{n-m}$，内积中自然出现相对位置 $n-m$：

$$
(R_mq)^\top(R_nk)=q^\top R_{n-m}k
$$

```python
def rope(x, positions):
    # x: [B, H, T, D]; positions: [T]，使用绝对位置
    d = x.size(-1)
    assert d % 2 == 0
    inv_freq = 10000 ** (
        -torch.arange(0, d, 2, device=x.device).float() / d
    )
    angles = positions.float()[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, None]
    sin = angles.sin()[None, None]

    even = x.float()[..., 0::2]
    odd = x.float()[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)
    return rotated.to(x.dtype)
```

用 KV cache 逐 token 生成时，新 token 的 `positions` 必须从缓存的已有长度开始，不能每次都传 0。

### 1.3 LayerNorm、RMSNorm 与 SwiGLU

LayerNorm 先减均值，再按方差缩放：

$$
\mu=\frac1C\sum_{j=1}^{C}x_j,
\qquad
\sigma^2=\frac1C\sum_{j=1}^{C}(x_j-\mu)^2
$$

$$
\operatorname{LayerNorm}(x)
=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

RMSNorm 不减均值：

$$
\operatorname{RMSNorm}(x)
=\gamma\odot\frac{x}{\sqrt{\frac1C\sum_{j=1}^{C}x_j^2+\epsilon}}
$$

SwiGLU 使用一个门控分支、一个值分支：

$$
\operatorname{SwiGLU}(x)
=W_{\rm down}\left[
\operatorname{SiLU}(W_{\rm gate}x)\odot W_{\rm up}x
\right]
$$

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        scale = torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)
        return (x32 * scale * self.weight).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

这是 Pre-Norm block：归一化放在每个子层之前。`x + ...` 两处都是残差连接。

## 2. 训练：Cross-Entropy、AdamW 与 LoRA

### 2.1 Next-token Cross-Entropy 与 SFT mask

第 $t$ 个位置的 logits $z_t$ 预测第 $t+1$ 个 token：

$$
p_\theta(y_{t+1}\mid y_{\le t})
=\operatorname{softmax}(z_t)_{y_{t+1}}
$$

令 $m_{b,t+1}=1$ 表示目标 token 参与训练；prompt 和 padding 的目标位置取 0。监督微调的平均损失为：

$$
L_{\rm SFT}
=-
\frac{\sum_{b,t}m_{b,t+1}
\log p_\theta(y_{b,t+1}\mid y_{b,\le t})}
{\sum_{b,t}m_{b,t+1}}
$$

单个有效 token 的 softmax + cross-entropy 对 logit $z_c$ 的梯度是：

$$
\frac{\partial L}{\partial z_c}
=p_c-\mathbf{1}[c=y]
$$

平均交叉熵为 $L_{\rm CE}$ 时，困惑度为 $\operatorname{PPL}=\exp(L_{\rm CE})$。

```python
def token_logps(logits, input_ids):
    # logits: [B, T, V]; input_ids: [B, T]
    logp = logits[:, :-1].float().log_softmax(dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    return logp.gather(-1, targets).squeeze(-1)  # [B, T-1]


def sft_loss(logits, input_ids, response_mask):
    # response_mask: [B, T]，回答 token 为 True
    logp = token_logps(logits, input_ids)
    mask = response_mask[:, 1:].to(logp.dtype)
    return -(logp * mask).sum() / mask.sum().clamp_min(1)
```

`input_ids` 中的 padding 必须是合法 token id；用 `response_mask` 排除它。最常见的错误是忘了 `logits[:, :-1]` 对齐 `input_ids[:, 1:]`，或把 prompt token 也计入 SFT loss。

### 2.2 AdamW

一阶和二阶动量：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

偏差修正：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

将 weight decay 与梯度更新分开：

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

```python
@torch.no_grad()
def adamw_step(param, grad, m, v, step,
               lr=1e-4, beta1=0.9, beta2=0.999,
               eps=1e-8, weight_decay=0.01):
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)

    param.mul_(1 - lr * weight_decay)
    param.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-lr)
```

这是单参数的核心更新。真实训练还要按参数组处理不衰减的偏置/归一化参数，并管理混合精度与优化器状态。

### 2.3 LoRA

冻结原矩阵 $W_0\in\mathbb{R}^{d_{\rm out}\times d_{\rm in}}$，只训练低秩增量：

$$
W'=W_0+\frac{\alpha}{r}BA
$$

$$
A\in\mathbb{R}^{r\times d_{\rm in}},
\qquad
B\in\mathbb{R}^{d_{\rm out}\times r}
$$

因此增加的可训练参数量是 $r(d_{\rm in}+d_{\rm out})$，而原矩阵有 $d_{\rm in}d_{\rm out}$ 个参数。常见初始化是随机初始化 $A$、将 $B$ 置零，让初始模型仍等于 $W_0$。

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        assert rank > 0
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        delta = F.linear(F.linear(x, self.a), self.b)
        return self.base(x) + self.scale * delta
```

## 3. 后训练：PPO、DPO、GRPO

### 3.1 PPO：GAE 与 clipped objective

PPO 通常训练一个 value function $V(s_t)$ 来估计回报。令 $d_t=1$ 表示该步之后 episode 结束，时序差分误差为：

$$
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)
$$

GAE 用从后向前的递推计算 advantage：

$$
A_t^{\rm GAE}=\delta_t+\gamma\lambda(1-d_t)A_{t+1}^{\rm GAE}
$$

新旧策略在已采样 action 上的概率比：

$$
\rho_t=\exp\left[
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\rm old}(a_t\mid s_t)
\right]
$$

要**最大化**的 clipped objective 是：

$$
J_{\rm PPO}=\mathbb{E}_t\left[
\min\left(
\rho_tA_t,
\operatorname{clip}(\rho_t,1-\varepsilon,1+\varepsilon)A_t
\right)
\right]
$$

代码做梯度下降，因此 policy loss 是 $-J_{\rm PPO}$。完整 PPO 还会加入 value loss，常见形式为 $\frac12\mathbb E[(V_t-R_t)^2]$，以及可选的 entropy 项。

```python
def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # rewards、dones: [B, T]; values: [B, T+1]
    # 每条轨迹最后一个有效位置的 dones 必须为 1
    advantage = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])

    for t in reversed(range(rewards.size(1))):
        alive = 1.0 - dones[:, t].float()
        delta = rewards[:, t] + gamma * alive * values[:, t + 1] - values[:, t]
        running = delta + gamma * lam * alive * running
        advantage[:, t] = running
    return advantage


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    mask = mask.bool()
    log_ratio = new_logp.float() - old_logp.detach().float()
    ratio = log_ratio.masked_fill(~mask, 0.0).exp()
    advantage = advantage.detach()
    objective = torch.minimum(
        ratio * advantage,
        ratio.clamp(1 - eps, 1 + eps) * advantage,
    )
    mask = mask.to(objective.dtype)
    return -(objective * mask).sum() / mask.sum().clamp_min(1)
```

### 3.2 DPO：成对偏好的序列 log-prob

对于 prompt $x$ 和回答 $y$，先把**有效回答 token** 的 log-prob 求和：

$$
\log\pi_\theta(y\mid x)
=\sum_{t\in\text{回答}}
\log\pi_\theta(y_t\mid x,y_{<t})
$$

对偏好回答 $y^+$ 与拒绝回答 $y^-$，定义两个 margin：

$$
\Delta_\theta
=\log\pi_\theta(y^+\mid x)
-\log\pi_\theta(y^-\mid x)
$$

$$
\Delta_{\rm ref}
=\log\pi_{\rm ref}(y^+\mid x)
-\log\pi_{\rm ref}(y^-\mid x)
$$

DPO 最小化：

$$
L_{\rm DPO}
=-\mathbb{E}_{(x,y^+,y^-)}
\log\sigma\left[\beta(\Delta_\theta-\Delta_{\rm ref})\right]
$$

```python
def response_logp(logits, input_ids, response_mask):
    logp = token_logps(logits, input_ids)
    mask = response_mask[:, 1:].to(logp.dtype)
    return (logp * mask).sum(dim=-1)  # [B]


def dpo_loss(policy_good, policy_bad, ref_good, ref_bad, beta=0.1):
    # 四个输入均为每条回答的 sequence log-prob，形状 [B]
    policy_margin = policy_good - policy_bad
    ref_margin = ref_good.detach() - ref_bad.detach()
    return -F.logsigmoid(beta * (policy_margin - ref_margin)).mean()
```

参考模型通常冻结并在 `torch.no_grad()` 下计算 log-prob。标准 DPO 在这里对回答 token **求和**，不是按回答长度求均值。

### 3.3 GRPO：组内 advantage 与逐 token loss

同一个 prompt 采样 $G$ 条完整回答，分别得到标量奖励 $r_{b,1},\ldots,r_{b,G}$。先在组内计算均值和标准差：

$$
\mu_b=\frac1G\sum_{i=1}^{G}r_{b,i}
$$

$$
\sigma_b=\sqrt{\frac1G\sum_{i=1}^{G}(r_{b,i}-\mu_b)^2}
$$

每条回答的 advantage：

$$
A_{b,i}=\frac{r_{b,i}-\mu_b}{\sigma_b+\epsilon}
$$

**不是每个 token 单独算 reward 或 advantage**：同一回答的所有有效 token 共用 $A_{b,i}$。下面使用总体标准差，即代码里的 `unbiased=False`。组内奖励全相等时，$A_{b,i}=0$。

每个已采样 token 的新旧策略比值为：

$$
\rho_{b,i,t}
=\exp\left[
\log\pi_\theta(y_{b,i,t})
-\log\pi_{\rm old}(y_{b,i,t})
\right]
$$

令 $d_{b,i,t}=\log\pi_{\rm ref}(y_{b,i,t})-\log\pi_\theta(y_{b,i,t})$，一个常用的非负逐样本 KL 估计是：

$$
\widehat D_{{\rm KL},b,i,t}
=\exp(d_{b,i,t})-d_{b,i,t}-1
$$

一种按**每条回答长度**归一的 GRPO 损失为：

$$
L_{\rm GRPO}
=-\frac1{BG}\sum_{b=1}^{B}\sum_{i=1}^{G}
\frac1{\sum_t m_{b,i,t}}
\sum_t m_{b,i,t}
\left[
\min\left(
\rho_{b,i,t}A_{b,i},
\operatorname{clip}(\rho_{b,i,t},1-\varepsilon,1+\varepsilon)A_{b,i}
\right)
-\beta\widehat D_{{\rm KL},b,i,t}
\right]
$$

```python
def grpo_loss(new_logp, old_logp, ref_logp, rewards, mask,
              eps=0.2, beta=0.01):
    # logp、mask: [B, G, T]; rewards: [B, G]
    # 在计算组统计量之前排除空回答
    if not mask.bool().any(dim=-1).all():
        raise ValueError("each response must contain a valid token")
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    advantage = ((rewards - mean) / (std + 1e-8)).detach()
    advantage = advantage.unsqueeze(-1)  # [B, G, 1]

    mask = mask.bool()
    new = new_logp.float()
    log_ratio = new - old_logp.detach().float()
    ratio = log_ratio.masked_fill(~mask, 0.0).exp()
    clipped = ratio.clamp(1 - eps, 1 + eps)
    policy_term = torch.minimum(ratio * advantage, clipped * advantage)

    d = ref_logp.detach().float() - new
    d = d.masked_fill(~mask, 0.0)
    kl = d.exp() - d - 1.0
    token_loss = -policy_term + beta * kl

    mask = mask.to(token_loss.dtype)
    answer_loss = (token_loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return answer_loss.mean()
```

空回答必须在计算组均值和标准差之前排除；这里直接拒绝包含空回答的 group。不同实现还可能采用样本标准差、batch 级标准差或不同的 token 归一方式。面试时先说清自己写的是哪一种。`old_logp`、`ref_logp`、advantage 停止梯度，当前策略的 `new_logp` 保留梯度。

## 4. 推理：KV cache 与采样

### 4.1 单 token KV cache

已生成 $t-1$ 个 token 时，历史 K/V 可以复用：

$$
K_{1:t}=\operatorname{Concat}(K_{1:t-1},K_t)
$$

$$
V_{1:t}=\operatorname{Concat}(V_{1:t-1},V_t)
$$

当前 token 的 attention 只对历史和自身计算：

$$
o_t=\operatorname{softmax}\left(
\frac{q_tK_{1:t}^\top}{\sqrt D}
\right)V_{1:t}
$$

```python
def decode_one_token_attention(q, k_new, v_new, cache=None):
    # q、k_new、v_new: [B, H, 1, D]
    if cache is None:
        k_all, v_all = k_new, v_new
    else:
        old_k, old_v = cache
        k_all = torch.cat((old_k, k_new), dim=2)
        v_all = torch.cat((old_v, v_new), dim=2)

    out = F.scaled_dot_product_attention(
        q, k_all, v_all, is_causal=False
    )
    return out, (k_all, v_all)
```

这里的 Q 和新 K 应已按**绝对位置**完成 RoPE。`is_causal=False` 只适用于单 token decode，且缓存只含当前及过去 token；一次处理多个新 token 时，必须构造正确的 causal mask。batch 内缓存长度不同也要屏蔽 padding 的缓存位置。

### 4.2 Temperature、Top-k、Top-p

Temperature 对 logits $z_i$ 缩放：

$$
p_i(T)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

Top-k 只保留分数最高的 $k$ 个 token。Top-p 先按概率降序排列，保留累计概率首次达到 $p$ 的最小前缀，再重新归一化采样。$T=0$ 通常单独定义为贪心解码。

```python
def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    # logits: [B, V]
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.float() / temperature

    if top_k is not None:
        if not 1 <= top_k <= scores.size(-1):
            raise ValueError("top_k must be in [1, vocab_size]")
        indices = scores.topk(top_k, dim=-1).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(-1, indices, True)
        scores = scores.masked_fill(~keep, -torch.inf)

    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if top_p < 1:
            sorted_scores, indices = scores.sort(dim=-1, descending=True)
            cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
            remove = torch.zeros_like(cumulative, dtype=torch.bool)
            remove[..., 1:] = cumulative[..., :-1] >= top_p
            sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
            scores = scores.scatter(-1, indices, sorted_scores)

    probs = scores.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

Top-k 使用索引构造 mask，保证即使分数并列也只留下 $k$ 个 token。完整生成循环还要处理 EOS、最大长度和每条序列提前结束。

## 5. 面试现场的检查顺序

| 问题 | 手写时检查什么 |
| --- | --- |
| Attention | $Q,K,V$ 和输出形状；除以 $\sqrt D$；causal/padding mask；softmax 维度 |
| RoPE | $D$ 为偶数；Q/K 同时旋转；decode 使用绝对位置 |
| SFT | logits 与目标错位一位；prompt/padding 不计 loss |
| LoRA | $A,B$ 维度；冻结 $W_0$；初始 $B=0$ |
| PPO | 最大化目标在代码中取负号；old log-prob 和 advantage 停止梯度 |
| DPO | 回答 log-prob 按有效 token 求和；参考模型停止梯度 |
| GRPO | 同组完整回答先算 reward；组内归一；advantage 广播到有效 token |
| KV cache | 只缓存 K/V；单 token 和多 token decode 的 mask 不同 |

如果准备时间有限，优先从 Attention、SFT、LoRA、DPO、GRPO 和 KV cache 六题开始练。每题能写出公式、形状和一段小实现，再解释一个边界条件，比单独背公式有效。

## 参考资料

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290).
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300).
