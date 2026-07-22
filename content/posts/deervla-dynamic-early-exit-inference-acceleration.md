---
title: "DeeR-VLA 精读：用动态早退加速多模态机器人推理"
date: 2026-07-22 15:18:29
description: 从多出口 MLLM、动作一致性判据与预算约束出发，拆解 NeurIPS 2024 的 DeeR-VLA 如何动态跳过 LLM 层，实现多模态机器人推理加速。
series: 三大会论文精读
seriesOrder: 8
categories:
  - AI
tags:
  - 多模态大模型
  - 推理加速
  - 动态早退
  - VLA
  - 机器人学习
  - NeurIPS
hidden: true
haloPublished: true
---

机器人执行一个长任务时，并不是每一步都同样困难。机械臂朝目标物体移动，往往只需浅层特征；真正抓取、堆叠或操作开关时，才需要更深的视觉语言推理。如果每个控制周期都跑完整个多模态大模型，计算预算会被大量简单步骤消耗。

DeeR-VLA 把这个观察变成动态推理机制：在同一个视觉-语言-动作模型中设置多个出口，逐段执行 LLM；相邻出口给出的动作足够一致时立即停止，否则继续进入更深层。它还把平均计算量、单步峰值计算量和显存上限写进阈值选择问题，使“跑多深”成为可调的系统策略。

这篇论文直接属于本专题的**多模态大模型推理加速**方向，更具体地说，是面向 VLA 机器人策略的动态早退。它不压缩视觉 token，也不改变注意力内核，而是按当前图像、指令与历史状态动态路由到不同深度。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution** |
| 作者 | Yang Yue、Yulin Wang、Bingyi Kang、Yizeng Han、Shenzhi Wang、Shiji Song、Jiashi Feng、Gao Huang |
| 会议 | NeurIPS 2024，Main Conference Track |
| 专题子方向 | 多模态大模型推理加速：动态早退、预算感知路由、机器人服务效率 |
| 官方论文 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html) |
| 作者版本与许可 | [arXiv:2411.02359](https://arxiv.org/abs/2411.02359)，CC BY 4.0 |
| 官方补充材料 | [NeurIPS Supplemental](https://proceedings.neurips.cc/paper_files/paper/2024/file/67b0e7c7c2a5780aeefe3b79caac106e-Supplemental-Conference.zip) |
| 代码与模型 | [yueyang130/DeeR-VLA](https://github.com/yueyang130/DeeR-VLA)，Apache-2.0；[Hugging Face checkpoints](https://huggingface.co/Yang130/DeeR-VLA) |

**选择理由**：昨天的专题文章属于个性化，今天切换到推理加速；此前又已覆盖视觉 token 压缩与注意力内核，因此选择动态早退这一不同细分主题。DeeR-VLA 直接处理多模态机器人模型的计算、峰值延迟和显存约束，方法图、效率曲线、真实 GPU 延迟、消融、完整代码与 checkpoint 均公开，适合从算法和系统两个层面核验。

## 问题背景：控制周期不该一律跑满模型

论文以 RoboFlamingo 为基础。输入是语言指令 $l$ 与时刻 $t$ 的 RGB 观测 $o_t$，视觉编码器先产生视觉 token，再由带交叉注意力的 LLM 融合图文信息，最后由动作头预测 7 自由度动作：前 6 维为末端位姿，第 7 维控制夹爪开合。

作者先做了一个静态深度实验。在 CALVIN 的 D$\rightarrow$D 设置中，RoboFlamingo++ 使用 24、12、6 个 LLM 层时，每个动作的 LLM 计算量分别为 31.2、15.6、7.8 GFLOPs；任务成功率分别为 78.9%、78.0%、75.7%。从 6 层增加到 24 层，计算变为 4 倍，成功率只增加 3.2 个百分点。

这不说明深层模型没用。更合理的解释是：大部分控制时刻较容易，只有少数精细操作需要深层推理。固定小模型会在困难步骤失效，固定大模型又浪费简单步骤；动态深度要解决的正是两者之间的资源分配。

## 核心贡献

论文的贡献可以归纳为四点：

1. **多出口机器人 MLLM**：把 LLM 层分成连续分组，每个分组后都可聚合中间特征并预测动作。
2. **动作一致性早退判据**：不依赖分类 Softmax 置信度，而是比较相邻出口的动作差异；动作趋于稳定时停止计算。
3. **预算化阈值求解**：同时考虑总 FLOPs、单步峰值 FLOPs 和显存上限，可用演示数据估计阈值，也可通过环境交互做贝叶斯优化。
4. **匹配动态推理的训练方法**：随机组合不同出口的时间序列特征，并为每个出口添加仅训练时使用的辅助动作头。

**论文结论**：在 CALVIN 模拟机器人基准上，DeeR 在保持竞争性任务表现的同时，将 LLM 平均计算降低 5.2-6.5 倍；在同一 V100 上，报告的 LLM 推理时间从 55 ms 降到 17.5 ms。

**我的判断**：最有工程价值的不是“早退”三个字，而是论文把平均算力、峰值延迟和驻留显存拆成三个约束。需要警惕的是，这些数字只覆盖 LLM 子模块，不是完整视觉编码器、控制栈和机器人硬件的端到端延迟。

## 方法总览：推理早退，训练覆盖所有出口

![DeeR-VLA 的动态推理与多出口训练](/images/posts/deervla-dynamic-early-exit-inference-acceleration/deervla-figure1-training-inference.png)

*图源：Yue et al., [DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution](https://arxiv.org/abs/2411.02359), Figure 1, NeurIPS 2024；从作者 CC BY 4.0 arXiv 源码中的原始矢量图直接栅格化，结构与标签未修改。原图用于论文解读。*

左图是推理：输入逐段通过 MLLM，每到一个出口便检查是否停止；一旦退出，后续层被跳过，中间特征与历史隐藏状态送入共享动作头。右图是训练：不同时间步从不同出口取特征，让动作头适应真实推理时不断变化的深度，同时以辅助动作头直接监督每个出口。

### 多出口特征与动作头

设融合后的 MLLM 被分成 $N$ 个连续块 $F_\theta^1,\ldots,F_\theta^N$。第 $i$ 个出口产生长度为 $L$ 的隐藏序列 $x_t^i$，经 token 维最大池化得到：

$$
\tilde{x}_t^i=P(x_{t,1}^i,x_{t,2}^i,\ldots,x_{t,L}^i).
$$

动作头是 4 层 LSTM 加两组 MLP。它在历史窗口中维护状态 $h_t$，并将选中出口的特征变成动作：

$$
a_t^*,h_t=\pi_\theta(\tilde{x}_t^{c(t)},h_{t-1}).
$$

LSTM 很关键：机器人处于部分可观测环境，单帧图像无法完整说明速度、先前动作和任务进度。动态出口不仅要在当前帧上正确，还要让不同深度的特征能进入同一个时间模型而不破坏状态连续性。

## 关键模块一：用动作一致性决定是否继续

分类模型可以用最大概率或熵判断置信度，但连续动作没有自然的 Softmax 置信度。DeeR 改为比较相邻出口预测：

$$
\left\|
\pi_\theta(\tilde{x}_t^i,h_{t-1})-
\pi_\theta(\tilde{x}_t^{i-1},h_{t-1})
\right\|_2<\eta_i.
$$

系统选择满足条件的最浅出口。若浅层与更深一层已给出近似动作，论文将其视为计算趋于饱和；若差异仍大，就继续向下。最后一个出口的阈值设为无穷，保证每个时刻一定能产生动作。

这个判据有两个边界。第一，它衡量的是**相邻预测一致**，不是与正确动作一致；两个出口可能稳定地给出同一个错误动作。第二，为比较第 $i-1$ 与第 $i$ 个出口，系统仍要顺序执行到第 $i$ 个出口并运行轻量动作头，因此 FLOPs 节省不会一比一变成墙钟时间节省。

## 关键模块二：把资源约束写进阈值优化

阈值 $\eta_i$ 决定性能与成本。阈值大，模型更容易早退；阈值小，更多样本进入深层。论文把选择写成约束优化：

$$
\max_{\eta_1,\eta_2,\ldots}
\operatorname{Scc}(\mathcal T,\{\eta_i\})
$$

满足：

$$
\begin{aligned}
\operatorname{FLOPs}(\mathcal T,\{\eta_i\})&<B,\\
\operatorname{MFLOPs}(\mathcal T,\{\eta_i\})&<G,\\
\operatorname{Mem}(\mathcal T,\{\eta_i\})&<M.
\end{aligned}
$$

$B$ 限制整批任务的平均计算，$G$ 限制任一控制步的峰值计算，$M$ 限制可加载的最大前缀。三者不能混为一谈：动态早退主要降低平均 FLOPs；峰值 FLOPs 与显存还取决于部署时最多加载多少层。

只有演示数据时，作者假设到达每个出口的样本以固定概率 $q$ 退出，用目标退出比例反推各层阈值。允许在线环境交互时，则以成功率减预算违规罚项为目标，用贝叶斯优化搜索阈值。后者能利用真实反馈，但在实体机器人上会带来额外时间与安全成本。

## 训练与推理流程

### 训练

1. 以 OpenFlamingo 为图文骨干，冻结视觉编码器与原 LLM 自注意力层，只训练 Perceiver Resampler、交叉注意力、共享动作头和辅助动作头。
2. 每两个 LLM 自注意力层设置一个出口。3B 与 9B 实验都取骨干的前 12 层，形成 6 个出口。
3. 对长度为 $H=12$ 的轨迹片段使用两种采样：逐时刻均匀随机出口；或把窗口切成两段，每段共享一个随机出口，以模拟连续若干步停在同一深度。
4. 位姿使用 MSE，夹爪状态使用交叉熵；两个采样策略的序列损失相加。
5. 每个出口额外接一个辅助动作头，确保中间特征能直接支持动作预测；这些头推理时移除。

总损失为：

$$
\mathcal L_{\text{total}}=
\underbrace{\sum_{s\in\{s_1,s_2\}}\sum_{i=0}^{H-1}
\mathcal L(a_{t+i}^{*,s},a_{t+i})}_{\mathcal L^*}
+
\underbrace{\sum_{j=1}^{N}\sum_{i=0}^{H-1}
\mathcal L(a_{t+i}^{j},a_{t+i})}_{\mathcal L_{\text{aux}}}.
$$

### 推理

1. 根据设备显存和峰值 FLOPs 决定最多加载多少层，即 DeeR-S、DeeR-B 或自定义前缀。
2. 当前图像与语言指令进入视觉编码器和第一组 LLM 层。
3. 每个出口用相同历史状态预测候选动作，与前一出口比较。
4. 差异低于阈值就执行动作并更新 LSTM 状态；否则进入下一组层。
5. 通过修改阈值，在不重新训练模型的情况下调整平均计算量。

## 实验设置

CALVIN 将环境分为 A、B、C、D 四种背景与物体配置。每个 split 有超过 200 万条机器人轨迹，但只有约 1%、约 2.4 万条轨迹带语言标注；DeeR 只使用这部分 `LANG` 数据。

论文测试三种设置：D$\rightarrow$D 为同环境训练测试，ABCD$\rightarrow$D 为多环境训练，ABC$\rightarrow$D 为对未见环境 D 的零样本泛化。每次评测包含 1000 条任务链，每条最多 5 个自然语言子任务；指标是平均连续完成子任务数，范围为 0 到 5。

3B 骨干使用 MPT-1B Instruct 与 CLIP ViT-L/14，9B 骨干使用 MPT-7B 与同一视觉编码器。3B 训练使用 8 张 V100 32GB：D、ABC、ABCD 设置分别约 14、24、25 小时；9B 的 D 设置使用 8 张 A100 80GB，约 24 小时。

## 主要结果：平均计算、峰值和真实延迟

![OpenFlamingo 3B 上的成功长度与平均 LLM GFLOPs](/images/posts/deervla-dynamic-early-exit-inference-acceleration/deervla-figure3-flamingo3b-results.png)

*图源：Yue et al., [DeeR-VLA](https://arxiv.org/abs/2411.02359), Figure 3 (upper), NeurIPS 2024；从作者 CC BY 4.0 arXiv 源码中的原始矢量子图直接栅格化，三组坐标轴、图例与数据标记均保留。原图用于论文解读。*

Figure 3 的横轴只计算 **LLM GFLOPs/action**，纵轴为平均成功长度。与固定深度 RoboFlamingo++ 相比，在相近表现点上，DeeR 在 D$\rightarrow$D、ABCD$\rightarrow$D、ABC$\rightarrow$D 三个设置分别标出 5.9 倍、5.2 倍、6.5 倍的平均 LLM 计算缩减。

主结果表给出的饱和点更适合看绝对数值：

| 设置 | RoboFlamingo++：平均成功长度 / LLM GFLOPs | DeeR：平均成功长度 / LLM GFLOPs | DeeR 在线阈值 |
| --- | ---: | ---: | ---: |
| D$\rightarrow$D | 2.71 / 31.2 | **2.83 / 8.6** | **2.92 / 8.5** |
| ABCD$\rightarrow$D | 4.07 / 31.2 | **4.13 / 10.0** | **4.13 / 9.7** |
| ABC$\rightarrow$D | 2.59 / 31.2 | **2.82 / 12.5** | **2.90 / 9.5** |

这里不能把“5.2-6.5 倍 FLOPs 缩减”直接写成同倍数端到端加速。论文在同一 NVIDIA V100 上额外测得：ABCD$\rightarrow$D 中，RoboFlamingo++ 与 DeeR 的平均成功长度分别为 4.07 与 4.08，LLM 计算为 31.2 与 6.0 GFLOPs，LLM 推理时间为 55 ms 与 17.5 ms。真实时间下降 68.1%，约为 3.14 倍加速，低于理论 FLOPs 缩减。

![OpenFlamingo 9B 上的动态早退扩展结果](/images/posts/deervla-dynamic-early-exit-inference-acceleration/deervla-figure4-flamingo9b-results.png)

*图源：Yue et al., [DeeR-VLA](https://arxiv.org/abs/2411.02359), Figure 4, NeurIPS 2024；从作者 CC BY 4.0 arXiv 源码中的原始矢量图直接栅格化，坐标轴、图例与峰值资源条形图均保留。原图用于论文解读。*

扩展到 OpenFlamingo 9B 后，论文报告在相同性能下平均计算降低 1.8-5.7 倍，峰值 FLOPs 与显存降低 2.7-4.0 倍。图中的 DeeR-S/B 都是**固定加载上限**：论文使用的 12 层 DeeR-B 约占 12GB，8 层 DeeR-S 约占 8GB，而完整 32 层 RoboFlamingo++ 约占 32GB。早退阈值改变平均深度，加载层数决定显存上限。

## 消融分析

### 辅助头不是可有可无

在 ABCD$\rightarrow$D 中，4.9 GFLOPs 预算下，带辅助损失的平均成功长度为 3.94，移除后只有 2.64；10.0 GFLOPs 下则是 4.13 对 2.71。中间层原本服务于后续层，不天然适合直接输出动作；逐出口监督是多出口训练成立的关键。

### 动作一致性优于特征相似度与固定时间策略

论文比较三种早退判据：出口特征余弦相似度、随任务进度逐步加深、相邻动作一致性。动作一致性在三种数据设置和两档预算下都取得最高平均成功长度。例如 ABC$\rightarrow$D 的 4.9 GFLOPs 档，三者分别为 2.29、2.46、2.62。

### 早退与量化可以叠加

ABCD$\rightarrow$D 中，DeeR 的 float32、float16、int4 版本分别占 6GB、3GB、1.7GB，平均成功长度为 4.13、4.12、3.91。结果说明动态深度与低比特是互补维度，也提醒我们：int4 显存更低，但在该实验中已经出现 0.22 的任务表现损失。

## 退出行为与局限

![CALVIN 轨迹中的动态出口编号](/images/posts/deervla-dynamic-early-exit-inference-acceleration/deervla-figure5-rollout-exits.png)

*图源：Yue et al., [DeeR-VLA](https://arxiv.org/abs/2411.02359), Figure 5, NeurIPS 2024；从作者 CC BY 4.0 arXiv 源码中的原始矢量图直接栅格化，三条轨迹、帧序列与出口编号均保留。原图用于论文解读。*

Figure 5 中，机械臂直线接近目标时常在第 1 个出口停止；抓起方块、放置方块、拨动开关等精细阶段会进入更深出口。这是符合直觉的定性证据，但论文没有逐类统计“困难动作被错误早退”的失败率，也没有证明出口编号可以被解释成通用任务难度。

论文明确承认两项限制：

1. **只加速 LLM**：视觉编码器的计算仍然显著，所有主要 FLOPs、显存和延迟数字都明确排除了视觉编码器。
2. **只在模拟器验证**：实验局限于 CALVIN，没有实体机器人上的传感延迟、控制抖动、硬件功耗和安全评测。

从工程部署看还应补充四点：

1. **一致不等于正确**：浅层和深一层可能对同一误判达成一致，尤其在分布外图像或遮挡场景。
2. **阈值会随分布漂移**：离线阈值来自环境 D 的验证集；在线贝叶斯优化更有效，却需要真实环境反馈，不能在安全关键机器人上无代价试错。
3. **显存不是逐样本动态释放**：部署先决定最大加载层数，早退主要节省平均计算；不能把平均退出深度直接解释成同等显存缩减。
4. **复现成本较高**：官方 README 提醒代码重组可能引入误差；完整 1000 链评测使用 8 张 V100 仍需约 4-5 小时，训练则需要 8 卡集群。

## 可复现资源

- [NeurIPS 正式页面与 25 页 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/hash/67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html)
- [NeurIPS 官方补充代码包](https://proceedings.neurips.cc/paper_files/paper/2024/file/67b0e7c7c2a5780aeefe3b79caac106e-Supplemental-Conference.zip)
- [arXiv:2411.02359 全文、源码与 CC BY 4.0 许可](https://arxiv.org/abs/2411.02359)
- [官方代码仓库](https://github.com/yueyang130/DeeR-VLA)，Apache-2.0
- [官方 3B/9B checkpoints](https://huggingface.co/Yang130/DeeR-VLA)
- [CALVIN 基准与数据](https://github.com/mees/calvin)

仓库提供 D、ABC、ABCD 三种设置的评测命令、在线阈值、贝叶斯优化脚本和训练命令。复现前需要安装 CALVIN、下载 OpenFlamingo 与 MPT 权重，并按 README 替换为支持中间出口的 MPT 模型定义。官方 checkpoint 降低了从头训练门槛，但上游 OpenFlamingo、MPT、CALVIN 数据各有独立许可，不能只用 DeeR-VLA 的 Apache-2.0 概括整个复现栈。

## 个人判断

DeeR-VLA 展示了一个适合在线多模态系统的思路：不要只问“模型平均多快”，而要同时管理平均成本、最坏单步延迟和可加载模型上限。动作一致性又把早退从分类置信度扩展到了连续控制，为 VLA、视频 Agent 和多模态交互系统提供了可迁移的判据。

但它还不是实体机器人上的完整推理系统论文。视觉编码器未被加速，GPU 测试只覆盖 LLM 部分，在线阈值优化依赖环境交互，失败早退也缺少安全兜底。生产化更可能采用组合方案：轻量视觉编码器或视觉 token 压缩负责前端，动态早退负责 LLM 深度，量化负责驻留显存，再加一个面向异常观测的保守路由器。

因此，我会把 DeeR-VLA 定位为：一篇把“按情境分配模型深度”讲清楚、并用真实 GPU 延迟验证了 FLOPs 收益不会等比例落地的动态推理论文。它证明了多模态机器人控制可以不必每一步跑满 LLM，但距离端到端、真实硬件、带安全约束的加速仍有明显空间。

## 参考资料

1. Yue et al., [DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution](https://proceedings.neurips.cc/paper_files/paper/2024/hash/67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html), NeurIPS 2024.
2. Yue et al., [arXiv:2411.02359](https://arxiv.org/abs/2411.02359), CC BY 4.0.
3. Yue et al., [DeeR-VLA Code and Checkpoints](https://github.com/yueyang130/DeeR-VLA).
4. Li et al., [Vision-Language Foundation Models as Effective Robot Imitators](https://arxiv.org/abs/2311.01378), ICLR 2024.
5. Mees et al., [CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks](https://arxiv.org/abs/2112.03227), 2021.
