---
title: Flow Matching：从噪声到数据的连续流生成模型
date: 2026-05-07 00:00:00
tags:
  - 生成模型
  - Diffusion
  - Flow Matching
  - 深度学习
categories:
  - AI
mathjax: true
---

Flow Matching 是近几年生成模型里非常重要的一条路线。它和 Diffusion Model 关系很近，但视角更直接：不再把生成过程理解成“一步步去噪”，而是学习一个连续的速度场，让噪声样本沿着这条流逐渐移动到真实数据分布。

一句话概括：

> Flow Matching 学的是“数据应该怎么流动”，而不是“每一步应该去掉多少噪声”。

这类方法正在被大量图像、视频和 DiT（Diffusion Transformer）相关工作采用，也和 Rectified Flow、Continuous Normalizing Flow、Consistency Model 等方向有很深的联系。

![Diffusion vs Flow Matching overview](/images/flow-matching-overview.svg)



## 从 Diffusion 说起

经典扩散模型包含两个过程。

训练时，从真实样本 $x_0$ 出发，不断加入高斯噪声，最后得到近似纯噪声的 $x_T$。

生成时，从随机噪声开始，模型一步步反向去噪，最终得到图片、音频或视频。

它的核心可以粗略理解为学习：

$$
p(x_{t-1} \mid x_t)
$$

也就是：当前这个带噪样本，下一步应该变得稍微干净一点。

这个框架很成功，但它的理论推导通常会涉及 score function、SDE、reverse process 等概念。对于工程实现来说，扩散模型也常常需要较多采样步数，虽然 DDIM、DPM-Solver、Consistency 等方法已经大幅加速。

Flow Matching 的出发点是：能不能不绕这么多弯，直接学习从噪声到数据的“流动方向”？

## Flow Matching 的核心思想

假设有两个端点：

- $x_0$：真实数据样本
- $x_1$：高斯噪声样本

我们希望在 $t \in [0, 1]$ 上构造一条路径 $x_t$，让样本可以从噪声端连续流向数据端。

Flow Matching 要学习的是一个速度场：

$$
v_\theta(x_t, t)
$$

它表示：在时间 $t$，位于 $x_t$ 的样本应该往哪个方向移动。

这个生成过程可以写成一个常微分方程（ODE）：

$$
\frac{dx_t}{dt} = v_\theta(x_t, t)
$$

如果把样本想象成河里的粒子，那么：

- 噪声分布是河流上游
- 数据分布是河流下游
- 模型学习的是整条河的水流方向
- 生成就是沿着速度场积分

所以，Diffusion 更像“每一步帮你擦掉一点噪声”，Flow Matching 更像“直接告诉你该往哪里走”。

## 最常见的训练形式：Conditional Flow Matching

Flow Matching 的关键是：训练时我们需要知道中间点 $x_t$ 的目标速度。

最常见、也最直观的一种方式是在线性路径上训练。

给定真实样本 $x_0$ 和噪声样本 $x_1$，构造：

![Flow Matching linear interpolation path](/images/flow-matching-linear-path.svg)

$$
x_t = (1 - t)x_0 + tx_1
$$

这是一条从数据到噪声的直线路径。它对时间求导得到：

$$
u_t = \frac{dx_t}{dt} = x_1 - x_0
$$

于是训练目标就非常直接：

$$
\mathcal L(\theta)
= \mathbb E_{x_0, x_1, t}
\Bigl[
  \Vert v_\theta(x_t, t) - (x_1 - x_0) \Vert_2^2
\Bigr]
$$

也就是说，模型输入中间状态 $x_t$ 和时间 $t$，输出这个点应该具有的速度。监督信号不是复杂的后验分布，而是一个明确的向量。

这也是 Flow Matching 很吸引人的地方：训练目标简单、稳定、直观。

## 推理时怎么生成

训练完成后，我们已经得到了一个速度场 $v_\theta(x, t)$。

生成时通常从噪声开始：

$$
x_1 \sim \mathcal{N}(0, I)
$$

然后解 ODE，从 $t=1$ 积分到 $t=0$：

$$
\frac{dx_t}{dt} = v_\theta(x_t, t)
$$

最终得到的 $x_0$ 就是生成样本。

这里要注意一个符号方向问题：如果训练路径定义为 $x_t=(1-t)x_0+tx_1$，那么 $t=0$ 是数据，$t=1$ 是噪声。生成时需要从 $1 \rightarrow 0$ 反向积分。不同论文可能会把时间方向反过来，但本质相同。

## 它和 Diffusion 的区别

Diffusion 和 Flow Matching 并不是完全割裂的两类模型，而是有很强的统一关系。但从建模直觉上看，它们有明显差异。

Diffusion 通常从随机过程角度出发，可以写成 SDE：

$$
dx = f(x,t)dt + g(t)dW_t
$$

这里有随机噪声项 $dW_t$。

Flow Matching 更常用确定性 ODE 表达：

$$
dx = v(x,t)dt
$$

它没有显式随机项，而是通过一个确定性速度场把分布从噪声推向数据。

可以这样理解：

- Diffusion：通过加噪和反向去噪构造生成过程
- Flow Matching：通过学习分布之间的连续传输构造生成过程
- Diffusion 更偏 score / SDE 视角
- Flow Matching 更偏 velocity field / ODE 视角

不过很多现代扩散模型也可以写成 probability flow ODE，所以二者在理论上存在交汇。

## 它和 CNF 的关系

Flow Matching 和 Continuous Normalizing Flow（CNF）关系也很深。

CNF 同样使用连续动力系统：

$$
\frac{dx}{dt} = v_\theta(x,t)
$$

并通过可逆流把简单分布变成复杂数据分布。

传统 CNF 的难点在于似然计算通常需要处理 divergence 或 trace 项，训练成本较高。Flow Matching 则换了一个角度：不直接最大化精确似然，而是构造可监督的路径和速度目标，让模型直接回归速度场。

因此可以把 Flow Matching 理解为一种更实用、更易训练的连续流生成模型训练方式。

## Rectified Flow：把路径拉直

Rectified Flow 是 Flow Matching 相关方向里非常重要的一支。

它的核心思想是：让从噪声到数据的轨迹尽可能直。

为什么“直”很重要？

- 轨迹越弯，模型越难学
- 轨迹越弯，ODE 积分越困难
- 轨迹越弯，采样通常需要更多步
- 轨迹越直，少步采样越容易

所以 Rectified Flow 会通过重新配对、重训练等方式，把原本弯曲复杂的生成路径逐渐拉直。直观地说，它不是只学习“怎么流”，还希望这条流尽可能接近最短路径。

这也是为什么很多人会把 Rectified Flow 和 Optimal Transport（最优传输）联系起来：理想情况下，我们希望噪声分布到数据分布的搬运路径既正确又高效。

## 为什么 Flow Matching 适合 DiT

现在图像和视频生成里，Transformer 架构越来越重要，例如 DiT、视频 DiT、多模态 Transformer 等。

Flow Matching 与这类架构很契合，原因包括：

- 输入输出形式简单：模型预测速度向量，和预测噪声一样容易实现
- 连续时间建模自然：时间 $t$ 可以作为 embedding 输入 Transformer
- 训练目标稳定：MSE 回归速度场，工程上很友好
- 采样步数可控：ODE solver 可以灵活选择步数和精度
- 适合 latent space：可以在 VAE latent 或视频 latent 中建模连续流

在工程上，一个 Flow Matching Transformer 和一个 Diffusion Transformer 往往非常相似：都是输入 noisy latent、time embedding、condition embedding，然后输出一个与 latent 同形状的张量。差别主要在训练目标和采样方程。

## 极简 PyTorch 伪代码

下面是一个非常简化的训练形式，用来体现核心思想：

```python
import torch
import torch.nn.functional as F

def flow_matching_loss(model, x_data):
    batch_size = x_data.shape[0]

    x_noise = torch.randn_like(x_data)
    t = torch.rand(batch_size, device=x_data.device)

    view_shape = [batch_size] + [1] * (x_data.ndim - 1)
    t_view = t.view(*view_shape)

    x_t = (1 - t_view) * x_data + t_view * x_noise
    target_velocity = x_noise - x_data

    pred_velocity = model(x_t, t)
    return F.mse_loss(pred_velocity, target_velocity)
```

推理时则类似：

```python
@torch.no_grad()
def sample(model, shape, steps, device):
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((shape[0],), 1 - i * dt, device=device)
        velocity = model(x, t)
        x = x - dt * velocity

    return x
```

这里用的是最简单的 Euler 积分。真实系统里通常会使用更好的 noise schedule、time sampling、ODE solver、classifier-free guidance，以及在 latent space 中训练。

## Flow Matching 为什么火

它受欢迎的原因可以总结为四点。

第一，理论视角统一。它把扩散模型、连续流、最优传输等概念放到了同一个“速度场”框架里。

第二，训练目标直接。相比复杂的反向扩散推导，Flow Matching 可以直接回归目标 velocity。

第三，采样有潜力更快。因为生成过程是 ODE 积分，可以通过更好的路径设计和求解器减少步数。

第四，工程迁移成本低。对于已经有 Diffusion / DiT 系统的团队来说，把预测噪声改成预测 velocity，并调整采样器，是一条相对自然的演进路线。

## 如何继续深入

如果想系统学习，建议按这个顺序：

1. DDPM：理解扩散模型的基本训练与采样。
2. Score-based Model：理解 score function 和噪声条件分数网络。
3. SDE / ODE：理解随机过程与 probability flow ODE。
4. CNF：理解连续可逆流和分布变换。
5. Flow Matching：理解速度场监督训练。
6. Rectified Flow：理解路径拉直和少步生成。
7. Consistency Model：理解一步或少步生成的另一条路线。

几篇经典论文包括：

- Flow Matching for Generative Modeling
- Rectified Flow: A Marginal Preserving Approach to Optimal Transport
- Consistency Models
- Scalable Diffusion Models with Transformers

## 总结

Flow Matching 的核心不是“去噪”，而是“流动”。

它把生成建模看成一个连续传输问题：从简单噪声分布出发，学习一个速度场，把样本沿着 ODE 推向真实数据分布。

这个视角同时继承了 Diffusion 的生成质量、CNF 的连续优雅和 ODE 的采样灵活性。因此在图像、视频、音频和 latent generative model 中，Flow Matching 很可能会继续成为主流方向之一。

如果用一句更形象的话收尾：

> Diffusion 像是在雾里一点点擦出图像；Flow Matching 像是知道风往哪里吹，让噪声顺着风长成数据。
