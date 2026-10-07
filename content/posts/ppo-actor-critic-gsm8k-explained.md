---
title: 从 Actor、Critic 到 PPO Loss：用 GSM8K 理解语言模型强化学习
date: 2026-10-07 18:00:00
description: 从 GSM8K 和 verl 实验出发，理解 PPO 中的 Actor、Critic、奖励对齐、GAE、概率比与反向传播。
categories:
  - AI
tags:
  - LLM
  - PPO
  - 强化学习
  - GSM8K
  - verl
  - 模型后训练
mathjax: true
hidden: true
haloPublished: true
---

最近用 verl 跑 GSM8K 的 PPO 训练时，我连续遇到几个看起来简单、实际很容易混在一起的问题：Actor 到底是什么？Critic 是不是另一个语言模型？一整段回答只有一个对错奖励，为什么训练张量却要和每个 token 对齐？PPO 的反向传播又是从哪里开始的？

这篇文章把这些问题串成一条完整的链路。先用 GSM8K 说明任务，再从一次回答的生成过程出发，依次解释 Actor、Critic、奖励对齐、GAE、PPO loss 和反向传播。公式会保留，但每个公式都先给直觉。

## GSM8K：一个适合练习奖励模型的数学题数据集

GSM8K 的全名是 **Grade School Math 8K**。它包含约 8.5K 道小学数学应用题，每道题通常需要几步加减乘除或简单的数量关系推理，答案一般是一个数值。数据集常见的划分是约 7.5K 条训练数据和约 1.3K 条测试数据。

它适合做语言模型强化学习，原因是奖励规则比较清楚：模型生成答案后，抽取最终答案，与标准答案比较，答对给正奖励，答错给零奖励。例如可以定义：

```text
回答正确 -> reward = 1
回答错误 -> reward = 0
```

这不是说 GSM8K 只有一种评分实现。实际工程里还可能加入格式奖励、答案抽取失败惩罚，或者对 KL 偏离增加惩罚。但它的核心特点是：奖励往往在整段回答结束后才确定。

此前在一台 RTX 4090 上用 Qwen2.5-0.5B-Instruct 跑过一轮 GSM8K PPO 实验，最终记录是 703/1319，约 **53.30%**。这个数字是一次具体实验的结果，不是 GSM8K 的官方基线，也不代表 PPO 在所有配置下都会达到同样效果。它更适合作为后面理解训练流程的实际背景。

## 一次 PPO 训练到底在循环什么

把一次训练拆开，会得到下面这条链路：

```text
题目 prompt
    |
    v
Actor 逐 token 生成回答
    |
    v
Reward 函数检查答案，得到回报
    |
    +--> Critic 估计每个状态的价值 V(s_t)
    |
    v
GAE 计算每个 token 的 advantage A_t
    |
    +--> PPO actor loss 更新 Actor
    |
    +--> value loss 更新 Critic
    |
    v
重复 rollout 和 update
```

这里的 `rollout` 指用当前策略采样回答，`update` 指使用采样结果更新模型参数。PPO 并不是把一条题目直接喂给模型，然后用最终分数对模型做普通的监督学习；它先采样行为，再估计这次行为相对于预期好多少，最后用这个“好多少”来调整生成概率。

## Actor：真正负责生成回答的策略模型

在强化学习术语里，Actor 也叫 policy，记作 \(\pi_\theta\)。在语言模型场景中，它就是正在被训练的语言模型。给定已经生成的前缀 \(s_t\)，Actor 为下一个 token 产生一个概率分布：

$$
\pi_\theta(a_t\mid s_t)
$$

其中：

- \(s_t\) 是当前状态，通常是题目、已经生成的 token 以及必要的上下文；
- \(a_t\) 是第 \(t\) 步选中的 token；
- \(\pi_\theta(a_t\mid s_t)\) 是 Actor 选择这个 token 的概率；
- \(\theta\) 是语言模型参数。

例如，模型读到：

```text
题目：小明有 3 个苹果，又买了 2 个，他现在有
```

Actor 可能为下一个 token 给出这样的分布：

```text
5      0.62
6      0.08
个      0.05
...    ...
```

模型继续采样，直到生成答案或到达最大长度。PPO 做的事情，是根据这次回答带来的结果，增加有帮助 token 的概率，降低有害 token 的概率。

## Critic：估计“从现在继续下去能得到多少回报”

Critic 也叫 value function，记作 \(V_\phi\)。它回答的问题不是“下一个 token 是什么”，而是：

> 当前已经走到这个状态，按照接下来的策略继续生成，预期能得到多少回报？

数学上可以写成：

$$
V_\phi(s_t)\approx
\mathbb{E}[G_t\mid s_t]
$$

其中 \(G_t\) 是从时间步 \(t\) 开始的折扣回报，\(\phi\) 是 Critic 的参数。

Critic 可以和 Actor 共享 Transformer 主干，再接一个 value head；也可以使用独立模型。无论采用哪种实现，它的输出都是一个标量价值估计，而不是下一 token 的词表概率。

举例来说，在 GSM8K 中，模型已经生成了：

```text
先计算总共有多少个苹果，所以答案是
```

Critic 可能预测这个前缀最终答对的期望回报是 0.7。如果最后模型真的答对了，实际回报接近 1，那么这次结果比 Critic 的预期好；如果最后答错，实际回报接近 0，那么这次结果比预期差。

这就是 Critic 的作用：给 Actor 提供一个动态的“基准线”。没有 Critic 时，也可以直接使用回报做策略梯度，但估计方差通常更大，训练更不稳定。

## “回答级奖励”和“token 对齐”到底是什么意思

一条 GSM8K 回答通常只在结束时得到一个整体奖励。例如：

```text
The answer is 42
```

如果答案正确，奖励可能是 1。可是语言模型的训练张量是按 token 排列的，因此工程代码需要把这段回答的 token、log probability、value、advantage 和 mask 放在同一个时间轴上。一个简化示意如下：

```text
tokens:   [The, answer, is, 42]
reward:   [  0,       0,  0,  1]
mask:      [  1,       1,  1,  1]
```

这就叫奖励与 token 对齐。它不是给每个 token 重新人工标注“正确”或“错误”，而是让每个 token 位置都有一个可计算的奖励槽位，方便后续计算回报和优势。

GAE 会把结尾的奖励向前传播。直观上，如果最后答案正确，那么越靠近正确答案、并且对完成推理有帮助的动作，通常会得到更积极的优势；如果答案错误，相关动作的优势可能为负。实际实现还会使用 `response_mask` 排除 prompt token 和 padding token，只在模型生成的 response 部分计算 policy loss。

很多语言模型 RL 实现还会加入每 token 的 KL 惩罚，让当前 Actor 不要过度偏离 reference policy。此时放进时间轴的“奖励”可能已经是任务奖励与 KL 惩罚组合后的 shaped reward，而不只是最后的 0 或 1。

## GAE：把稀疏回报变成每个 token 的优势

先定义 TD error。这里把环境奖励写成 \(r_t^{\mathrm{env}}\)，因为后面 PPO 还会用 \(r_t(\theta)\) 表示新旧策略的概率比；两者是不同的量。

$$
\delta_t=r_t^{\mathrm{env}}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

这里的 \(r_t^{\mathrm{env}}\) 是第 \(t\) 步奖励，\(\gamma\) 是折扣因子。若最终奖励在最后一步才出现，那么中间很多步的 \(r_t^{\mathrm{env}}\) 都是 0，但价值函数的差异仍然可以帮助估计每一步的改进方向。

广义优势估计（Generalized Advantage Estimation, GAE）把未来多个 TD error 加权求和：

$$
A_t=\delta_t+\gamma\lambda\delta_{t+1}
 +(\gamma\lambda)^2\delta_{t+2}+\cdots
$$

其中 \(\lambda\) 控制偏差与方差的折中。\(\lambda\) 较大时，会看更长的未来，估计通常更充分但方差也可能更高；\(\lambda\) 较小时，更依赖局部价值估计。

优势 \(A_t\) 的含义可以用一句话概括：

> 在状态 \(s_t\) 下采取当前这个 token，比 Critic 原本预期的结果好多少。

因此，奖励是回答级别的，优势最终会变成 response 中每个有效 token 的训练信号。

## PPO 的核心：限制策略一次更新不要走太远

PPO 不直接让新策略无限增大高奖励动作的概率，而是比较新旧策略对同一个 token 的概率：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
     {\pi_{\mathrm{old}}(a_t\mid s_t)}
$$

这个 \(r_t(\theta)\) 叫概率比。它不是 reward，也不是回报，而是“当前策略相对于采样时的旧策略，把这个动作的概率改了多少”。

Actor 的 clipped surrogate objective 通常写成：

$$
L_{\mathrm{actor}}
=-\mathbb{E}_t\left[\min\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)\right]
$$

加负号是因为深度学习框架通常执行最小化，而 PPO 原始目标是最大化。\(\epsilon\) 是裁剪范围，例如 0.2。

Critic 则通过回归目标更新：

$$
L_{\mathrm{critic}}
=\mathbb{E}_t\left[(V_\phi(s_t)-R_t)^2\right]
$$

其中 \(R_t\) 是用于训练价值函数的 return，常见构造是优势和旧价值的和：\(R_t=A_t+V(s_t)\)。完整实现还可能加入熵奖励鼓励探索：

$$
L=L_{\mathrm{actor}}+c_vL_{\mathrm{critic}}
-c_{\mathrm{ent}}H(\pi_\theta)
$$

如果使用 reference model 约束，也可能将 KL 项加入 reward 或总 loss。不同 verl 配置对这些项的具体组合可能不同，需要以配置文件和 trainer 实现为准。

## 如何理解 \(r_tA_t\)

把两个量分开看：

- \(r_t\)：当前策略把这个 token 的概率改了多少；
- \(A_t\)：这个 token 相对 Critic 预期是好还是坏。

两者相乘后，才知道“这次概率变化是否沿着奖励方向”。例如：

| 概率比 \(r_t\) | 优势 \(A_t\) | 直觉 |
| ---: | ---: | --- |
| 1.2 | 2 | 好 token 的概率提高，方向正确 |
| 1.2 | -2 | 坏 token 的概率提高，方向错误 |
| 0.8 | -2 | 坏 token 的概率降低，方向正确 |
| 0.8 | 2 | 好 token 的概率降低，方向错误 |

当概率比偏离 1 太多时，`clip` 会限制目标，避免某个 batch 让策略发生过大的更新。这是 PPO 比普通 REINFORCE 更稳定的关键原因之一。

## PPO 是怎么反向传播的

很多人第一次接触 RL 会问：GSM8K 的“答对/答错”是离散规则，reward 本身不可导，梯度到底从哪里来？答案是：**PPO 不需要穿过 reward 函数反向传播。**

一次更新可以抽象为：

```text
GSM8K 规则评分
      |
      v
reward -> return / advantage（通常视为常量）
      |
PPO loss -> ratio -> 当前 log probability
      |
      v
logits -> Transformer 参数
```

采样阶段先保存旧策略对已生成 token 的 `old_logprob`。更新阶段重新用当前 Actor 计算这些 token 的 `new_logprob`，再得到：

$$
r_t=\exp(\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\mathrm{old}}(a_t\mid s_t))
$$

当前 `new_logprob` 连接着模型的 logits 和参数，所以 PPO loss 可以沿着这条路径求梯度。`old_logprob`、GAE 得到的 advantage，以及通常冻结的 reference model 输出，都不会把梯度传回采样阶段。

实际代码通常会分别做两次更新：Actor optimizer 最小化 policy loss，Critic optimizer 最小化 value loss。若 Actor 和 Critic 共享主干，还需要合理处理梯度累积和参数更新顺序，避免 value loss 意外改变 policy 的计算图。

## 正向传播和反向传播谁更快

对同一个模型和同一个序列长度来说，正向传播通常比反向传播快。粗略理解：

```text
forward                  ≈ 1 份计算
backward                 ≈ 2 份计算
forward + backward       ≈ 3 份计算
```

这是因为反向传播需要使用正向阶段保存的中间激活，并分别计算参数梯度和输入梯度，实际比例会随模型结构、序列长度、实现和显存带宽变化。

PPO 还要区分两类正向计算：rollout 阶段逐 token 生成回答，通常不保存训练梯度，但自回归生成很慢；update 阶段重新计算整段 response 的 log probability，并进行反向传播。于是“生成阶段没有反向传播”并不意味着整个 PPO 训练一定很快，长回答的 rollout 可能反而是主要瓶颈。

## 复现时，算力配置要怎么看

算法能不能理解，和一套配置能不能跑起来，是两个问题。verl 的 Quickstart 适合先熟悉数据流和训练入口；如果想参考完整的数学推理强化学习复现，也可以看 SimpleRL-Zoo v1。该项目 README 给出的建议是：Qwen2.5-0.5B 级别通常从单张 H100 或 A100 80GB 起步，7B/14B 使用 2 个节点、每节点 8 张 H100 80GB，32B 则需要更多节点。这些是项目作者针对其训练规模给出的参考配置，不是所有 PPO 实现的硬性要求。

显存需求会被模型大小、最大 response 长度、rollout batch size、并行方式、是否使用参数或 KV cache 分片等因素共同决定。因此，小模型在消费级 GPU 上可以用于验证流程，但吞吐和稳定训练的配置可能明显高于“能启动一次”。此前用 RTX 4090 跑 Qwen2.5-0.5B-Instruct 的实验，就是一个流程验证规模；如果换成更大的模型或更长的回答，需要重新估算显存和每小时采样量。

## 把几个对象放在一张表里

| 对象 | 解决的问题 | 典型输出 | 是否直接被 PPO 更新 |
| --- | --- | --- | --- |
| Actor / Policy | 下一步生成什么 token | 词表上的概率分布 | 是 |
| Critic / Value | 当前状态值多少回报 | 一个标量 \(V(s_t)\) | 是，通常单独更新 |
| Reward 函数 | 这次回答好不好 | 标量或 shaped reward | 通常不是可训练模型 |
| Reference Policy | 当前策略偏离原模型多少 | log probability / KL | 通常冻结 |
| GAE | 每个 token 应该被鼓励还是抑制多少 | \(A_t\) 序列 | 作为训练信号使用 |

最容易混淆的地方，是把 reward、value 和 advantage 都叫“分数”。它们的角色完全不同：reward 是环境给出的结果，value 是 Critic 的预期，advantage 是实际结果相对于预期的差值。

## 用一句话复述完整过程

给定一条 GSM8K 题目，Actor 逐 token 生成回答；规则奖励在回答结束后判断对错；Critic 估计每个前缀的预期回报；GAE 将最终结果转换成每个 response token 的 advantage；PPO 用概率比和裁剪目标更新 Actor，用 value loss 更新 Critic，并通过重复采样和更新逐步改变模型的生成分布。

这也是理解 verl 训练日志的一个实用顺序：先确认 rollout 生成了什么，再看 reward 是否合理；然后检查 response mask、优势值和 KL；最后再判断 actor loss、critic loss 与评测准确率是否相互支持。只盯着一个 loss 数字，很难判断 PPO 到底有没有学到更好的解题行为。

## 参考资料

1. [verl Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
2. Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
3. Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), 2015.
4. Cobbe et al., [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168), 2021.
5. [SimpleRL-Zoo v1](https://github.com/hkust-nlp/simpleRL-reason/tree/v1)，包含 GSM8K 等数学推理任务的开源强化学习复现配置与硬件说明。
