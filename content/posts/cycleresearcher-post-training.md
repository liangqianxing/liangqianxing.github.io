---
title: 从 DeepReviewer 2.0 到 CycleResearcher：科研 Agent 如何真正做 Post-training
date: 2026-08-07 17:33:05
description: 区分推理时 Agent 编排与模型后训练，并拆解 CycleResearcher 的领域 SFT、生成式 Reward Model、Iterative SimPO 和评测风险。
categories:
  - AI
tags:
  - LLM
  - Post-training
  - Agent
  - SimPO
  - Reward Model
  - ICLR
mathjax: true
hidden: true
haloPublished: true
---

看到一个系统会检索、规划、反思和调用工具，不能据此判断它做过 Post-training。Agent 描述的是模型在运行时如何完成任务；Post-training 描述的是预训练之后，是否继续用监督数据、偏好数据或奖励信号更新模型参数。两者可以结合，也可以完全分离。

这篇文章以西湖大学 NLP 团队的两个科研 Agent 为对照：DeepReviewer 2.0 主要依靠推理时流程控制，CycleResearcher 则包含完整的领域 SFT、生成式 Reward Model 和迭代偏好优化。重点不是复述系统功能，而是回答一个求职和面试中更有价值的问题：**一个项目究竟在哪些环节真正更新了模型，训练信号从哪里来，为什么这样设计？**

## 先给结论：Agent 不等于 Post-training

| 系统 | 底层模型是否针对任务继续训练 | 核心方法 | 更准确的定位 |
| --- | --- | --- | --- |
| DeepReviewer 2.0 | 否 | 分阶段审稿、检索、证据锚定、导出门控 | 推理时 Agent 系统 |
| CycleResearcher | 是 | 领域 SFT、CycleReviewer、Iterative SimPO | Agent + Post-training |
| FPO / Pre-DPO | 是 | DPO 目标与参考约束改进 | 偏好优化算法 |

DeepReviewer 2.0 的论文表格注明其主系统运行在 `Step-3.5-Flash 196B` 上，并强调模型未经审稿任务专项微调。这里的 `196B` 是总参数量口径；StepFun 模型卡给出的单 Token 激活参数约为 `11B`。它的提升来自 claim-evidence-risk ledger、检索议程、段落锚定和 export gate，而不是更新底层模型权重。

CycleResearcher 不同。它先训练科研写作 Policy，再训练能够模拟同行评审的 Reviewer，最后让 Reviewer 对 Policy 的多个输出打分，构造偏好对并继续优化 Policy。论文把这套闭环称为 Research-Review-Refinement。

## Instruct 模型是后训练的起点

CycleResearcher 没有从 Base Model 开始。它使用的底座包括 Mistral-Nemo-Instruct-2407、Qwen2.5-72B-Instruct 和 Mistral-Large-2。

Base Model 主要通过 next-token prediction 学习语言、知识和代码模式，但不一定稳定服从用户指令。Instruct Model 已经过通用指令微调，学会把输入识别为任务并按要求回答。可以把常见路径写成：

```text
Base Model
  -> 通用指令 SFT
  -> Instruct Model
  -> 领域 SFT
  -> 偏好优化
  -> 领域模型或 Agent Policy
```

因此，CycleResearcher 做的是 Instruct 模型之上的二次领域 Post-training，而不是从头教模型如何对话。

## CycleResearcher 的训练闭环

![CycleResearcher Post-training 闭环](/images/posts/cycleresearcher-post-training/cycleresearcher-post-training-loop.svg)

*CycleResearcher 的训练信号流。本文原创重绘，依据 [CycleResearcher 论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/0a48036026dc7946ef6033ae14719cc5-Paper-Conference.pdf) Section 2-3。*

整个 Post-training 可以拆成三个阶段：领域 SFT、生成式 Reward Model 训练、Iterative SimPO。

## 阶段一：用 Research-14K 做领域 SFT

Research-14K 从 2022 至 2024 年 ICLR、NeurIPS、ICML、ACL、EMNLP、CVPR 和 ICCV 等会议论文中构建。作者先收集 14,911 篇论文，再过滤无效内容、补充参考文献摘要，并把正文拆成结构化 outline 与 section。最终得到：

- 12,696 个训练样本；
- 802 个按时间划分的测试样本；
- 输入是参考文献、BibTeX 和摘要；
- 输出是研究动机、主要想法、方法、实验设计、结果分析和完整 LaTeX 正文；
- 平均输出长度约 28K tokens。

这不是普通短指令 SFT，而是长上下文、长输出的领域行为学习。模型需要学习科研写作的内容，还要学习先规划 outline、再生成正文的输出协议。

Policy 训练使用 8 张 H100、DeepSpeed ZeRO-2，12B 模型训练上下文设为 32K，72B 和 123B 设为 24K；学习率为 $4\times10^{-5}$，训练 12,000 steps。该阶段得到初始 CycleResearcher，也就是后续偏好优化的初始策略 $P_1$。

## 阶段二：训练生成式 CycleReviewer

传统 Reward Model 常在 Transformer 顶部增加标量打分头，输入 prompt-response 后直接输出一个 reward。CycleReviewer 更接近 Generative Reward Model：它先生成完整评审，再在评审中给出结构化分数和最终建议。

Review-5K 来自 ICLR 2024 投稿与 OpenReview 评审数据，包含 4,991 篇论文和超过 16,000 条评审意见。其中 4,189 篇用于训练，782 篇用于测试。每个样本保留：

- 多位 Reviewer 的 summary、strengths、weaknesses 和 questions；
- soundness、presentation、contribution 与 overall score；
- Meta Reviewer 的汇总意见和最终决定。

CycleReviewer 使用 Mistral-Large-2 和 LoRA-GA，在 8 张 H100 80GB 上训练 12 epochs，学习率为 $10^{-5}$。推理时，它会模拟多名不同严格程度的 Reviewer，最后由 Senior Reviewer 汇总，得到自然语言反馈与平均分数。

这种设计的优点是 reward 可解释：可以看到模型为什么认为一篇论文更好。代价是它比标量 Reward Head 更慢，而且生成式评分更容易受到提示格式、文本长度和语言风格影响。

## 阶段三：从评分构造偏好对

作者另外收集了 4,152 篇近期 arXiv 机器学习论文的参考文献部分，将其作为知识输入。对于每个输入，当前 CycleResearcher 以 `temperature=0.4` 采样三个候选结果：

```text
x -> M1, M2, M3
```

CycleReviewer 分别模拟多名评审，计算候选结果的平均得分 $r_1,r_2,r_3$。最高分结果作为 chosen $y_w$，最低分结果作为 rejected $y_l$：

```text
(x, y_w, y_l)
```

这一步把连续评分转换成 pairwise preference。偏好来自 AI Reviewer 而不是逐条人工标注，因此也可以视为一种 RLAIF 数据生产方式。

## 为什么使用 SimPO，而不是 PPO

CycleResearcher 没有使用 PPO，也没有直接把 Reviewer 分数接入 policy gradient。它使用 SimPO 对 chosen 与 rejected 的生成概率进行比较。

SimPO 对一个回答定义长度归一化的隐式奖励：

$$
r_{\text{SimPO}}(x,y)=\frac{\beta}{|y|}\log \pi_\theta(y\mid x)
$$

训练目标要求 chosen 的隐式奖励比 rejected 至少高出 margin $\gamma$：

$$
\mathcal{L}_{\text{SimPO}}
=-\mathbb{E}\left[\log\sigma\left(
r(x,y_w)-r(x,y_l)-\gamma
\right)\right]
$$

与标准 DPO 相比，SimPO 不需要在训练时常驻 reference model，并且对序列 log probability 做长度归一化。对于 24K 甚至更长的输出，这能减少显存和计算压力，也避免模型仅通过改变回答长度获得更高序列概率。

## NLL 不是附属项，而是稳定器

论文没有只优化 SimPO，而是为 chosen response 增加 Negative Log-Likelihood：

$$
\mathcal{L}=\mathcal{L}_{\text{SimPO}}
-\lambda\,\mathbb{E}_{(x,y_w)}\log\pi_\theta(y_w\mid x)
$$

偏好损失只要求 chosen 相对 rejected 更好，不保证长文本仍然流畅、完整且不重复。NLL 则继续用标准语言建模目标拟合高质量 chosen，承担行为克隆和分布约束的作用。

消融实验很能说明问题：完整模型平均分为 5.36，去掉偏好训练后降至 5.12，去掉迭代采样后为 5.21；去掉 NLL 后直接降至 4.91，自动接收率从 35.14% 降至 12.03%。论文观察到无 NLL 时会出现重复文本和明显内容错误。

## Iterative SimPO 到底算不算强化学习

论文将方法描述为 iterative reinforcement learning framework，但面试时最好说得更精确。

从优化目标看，它不是 PPO/GRPO 式在线 RL：没有对 Reviewer 标量分数做策略梯度，也不在每个 rollout 后直接更新策略。每一轮仍然先把评分转换成离线偏好对，再优化 SimPO loss。

从数据闭环看，它又不是固定数据集上的一次性离线 DPO。第 $t$ 轮使用当前策略 $P_t$ 采样，Reviewer 构造 $D_t$，训练得到 $P_{t+1}$，下一轮再由新策略产生数据。作者每轮抽取约三分之一数据、训练一个 epoch，RL 阶段学习率为 $5\times10^{-7}$。

所以更准确的表述是：

> CycleResearcher 使用带在线数据刷新的迭代离线偏好优化，形成近似在线的 Policy-Reward 数据闭环。

这个表述既承认论文的 RL 框架定位，也不会把 SimPO 错说成 PPO。

## 结果背后的两个风险

### Reward hacking

Policy 可能学会迎合 CycleReviewer，例如偏好某种结构、措辞或篇幅，而没有真正提升研究质量。论文也明确指出，Policy 与 Reward Model 没有同步更新，长期迭代可能放大 Reviewer 的固定漏洞。

作者用独立 Reward Model 做了补充评估：在该附录实验的评估设置下，原 Reviewer 的平均分为 5.36，独立 Reviewer 下为 5.29；接收率从 31.07% 降到 28.65%。这里的 31.07% 与正文消融表中的 35.14% 来自不同表格和评估设置，不应直接混用。整体下降幅度不算剧烈，但只能缓解疑虑，不能证明不存在 reward exploitation。

### 训练环境不等于真实科研

训练阶段并不执行真实实验，论文中的虚拟实验结果由模型生成。系统主要训练文献理解、研究规划和论文写作，不应被理解成已经完成了从假设到实验验证的全自动科学发现。

因此，自动评分高于某个基线只能说明它更符合当前 Reviewer 的评价分布，不能直接等价为科学创新能力更强。

## 西湖大学近两年的相关方向

如果关注 2024 至 2026 年的 Post-training，可以把相关工作分成三类：

1. **科研 Agent 的领域后训练**：[CycleResearcher](https://arxiv.org/abs/2411.00816) 在 ICLR 2025 发表，将 SFT、生成式 Reviewer 和 Iterative SimPO 放进同一闭环。
2. **偏好优化算法**：[FPO](https://arxiv.org/abs/2411.07618) 使用 Sparse Autoencoder 的特征级约束改进 DPO；[Pre-DPO](https://arxiv.org/abs/2504.15843) 使用 guiding reference model 改善偏好数据利用率。
3. **多模态后训练**：[MergeMix](https://arxiv.org/abs/2510.23479) 用图像混合构造偏好关系，并以 SimPO 训练多模态模型；它属于西湖大学其他团队，而不是 CycleResearcher 所在的 WestlakeNLP 工作线。

[DeepReviewer 2.0](https://arxiv.org/abs/2604.09590) 则适合作为反例：它是很新的科研 Agent，但论文强调底层模型未针对审稿任务微调，主要贡献在推理时流程和可审计输出协议，不应包装成 Post-training 项目。

## 面向 Post-training 岗位，应该讲清什么

如果用 CycleResearcher 准备面试，重点不应停留在“做了一个科研 Agent”，而应能回答下面这些问题：

1. Research-14K 和 Review-5K 分别提供什么训练信号？
2. Generative Reward Model 与标量 Reward Model 有什么差异？
3. Reviewer 分数为什么要转成 chosen-rejected，而不是直接回归分数？
4. SimPO 为什么可以不加载 reference model？
5. 长度归一化对长文本偏好学习有什么作用？
6. 为什么仅使用 preference loss 会导致重复和能力漂移？
7. 每轮重新采样如何改变训练分布？
8. 如何用独立 Judge、规则奖励、Reward Ensemble 和人工评测降低 reward hacking？
9. 如果改成 GRPO/PPO，训练成本、稳定性和 reward 设计会发生什么变化？

一句适合面试的总结是：

> CycleResearcher 以 Instruct 模型为起点，在 Research-14K 上进行长上下文领域 SFT，并在 Review-5K 上训练生成式 Reward Model。随后从当前 Policy 采样多个科研输出，由 Reviewer 评分并构造偏好对，再使用带 NLL 正则的 Iterative SimPO 更新 Policy，每轮重新生成训练数据，形成自动反馈的 Post-training 闭环。

## 参考资料

1. Weng et al., [CycleResearcher: Improving Automated Research via Automated Review](https://proceedings.iclr.cc/paper_files/paper/2025/file/0a48036026dc7946ef6033ae14719cc5-Paper-Conference.pdf), ICLR 2025.
2. [CycleResearcher 项目与模型仓库](https://github.com/zhu-minjun/Researcher).
3. Weng et al., [DeepReviewer 2.0: A Traceable Agentic System for Auditable Scientific Peer Review](https://arxiv.org/abs/2604.09590), 2026.
4. StepFun, [Step-3.5-Flash Model Card](https://huggingface.co/stepfun-ai/Step-3.5-Flash).
5. Yin et al., [Direct Preference Optimization Using Sparse Feature-Level Constraints](https://arxiv.org/abs/2411.07618), 2024.
6. Pan et al., [Pre-DPO: Improving Data Utilization in Direct Preference Optimization Using a Guiding Reference Model](https://arxiv.org/abs/2504.15843), 2025.
7. Jin et al., [MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](https://arxiv.org/abs/2510.23479), 2025.
