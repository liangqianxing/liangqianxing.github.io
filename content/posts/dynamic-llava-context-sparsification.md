---
title: "Dynamic-LLaVA 精读：同时压缩视觉 Token 与生成上下文"
date: 2026-07-28 10:27:02
description: 从视觉 Token 选择、生成上下文稀疏化与在线 KV cache 决策出发，拆解 ICLR 2025 的 Dynamic-LLaVA 如何贯穿预填充和解码阶段加速多模态大模型，属于推理加速（推理侧）方向。
series: 三大会论文精读
seriesOrder: 12
categories:
  - AI
tags:
  - 多模态大模型
  - 推理加速（推理侧）
  - 视觉 Token 压缩
  - KV Cache
  - 动态稀疏化
  - Dynamic-LLaVA
  - ICLR
hidden: true
haloPublished: true
---

很多多模态推理加速方法只盯着输入图像：先把 576 个视觉 token 裁到 144 个，预填充自然更快。但回答越长，已经生成的文字会逐步接管计算和显存开销，最初省下的视觉 token 占比越来越小。

Dynamic-LLaVA 的出发点正是这个阶段错位。它在第 2 个 LLM decoder layer 后用图像预测器选择视觉 token，同时用输出预测器筛选已经生成的文本上下文；有 KV cache 时，后者不回头淘汰历史缓存，而是在每个新 token 到来时决定是否把它的 K/V 激活写入缓存。

这篇论文直接属于本专题的 **推理加速（推理侧）** 方向。它面向 LLaVA 类多模态大模型，在预填充、无 KV cache 解码和有 KV cache 解码三种模式下分别报告 FLOPs、延迟与显存，不是只讨论表征压缩的泛多模态论文。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification** |
| 作者 | Wenxuan Huang、Zijie Zhai、Yunhang Shen、Shaosheng Cao、Fei Zhao、Xiangfeng Xu、Zheyu Ye、Yao Hu、Shaohui Lin |
| 会议 | ICLR 2025 |
| 方法名 | Dynamic-LLaVA |
| 专题子方向 | 推理加速（推理侧）：视觉 token 裁剪、生成上下文稀疏化、KV cache 压缩 |
| 正式论文 | [OpenReview: hzVpZDrW73](https://openreview.net/forum?id=hzVpZDrW73) |
| 作者全文 | [arXiv:2412.00876 v4](https://arxiv.org/abs/2412.00876)，arXiv 非专属发行许可 |
| 官方代码 | [Osilly/dynamic_llava](https://github.com/Osilly/dynamic_llava)，Apache 2.0 |
| 模型权重 | [Dynamic-LLaVA-7B](https://huggingface.co/Osilly/Dynamic-LLaVA-7B)、[Dynamic-LLaVA-13B](https://huggingface.co/Osilly/Dynamic-LLaVA-13B) |

**选择理由**：上一篇专题文章 VideoLISA 属于个性化训练侧，本轮切回推理加速。已有文章中的 M3 解决多档视觉粒度，SparseVLM 做问题引导的视觉 token 裁剪，DeeR-VLA 做机器人策略早退；Dynamic-LLaVA 则进一步把生成中的语言上下文和 KV cache 纳入动态稀疏化。29 页论文与附录、原始配图、代码、7B/13B 权重、硬件实测和消融均公开可核验。

## 问题背景：为什么只裁图像不够

LLaVA-1.5 的推理可分成两个阶段：

1. **预填充**：视觉编码器产生图像 token，和问题文本一起经过 LLM，生成第一个输出 token。
2. **自回归解码**：每一步追加一个输出 token，再预测下一个 token。有 KV cache 时主要增长的是缓存；不用 cache 时则要重复计算不断变长的上下文。

设第 $l$ 层的图像、输入文本和输出文本 token 集合分别为 $\mathcal S_l^I$、$\mathcal S_l^T$ 与 $\mathcal S_l^{OT}$。论文用下面的近似关系说明瓶颈迁移：

$$
\begin{aligned}
C_{\text{prefill}}^{(l)} &\propto |\mathcal S_l^I\cup\mathcal S_l^T|,\\
C_{\text{decode,no-cache}}^{(l)} &\propto |\mathcal S_l^I\cup\mathcal S_l^T\cup\mathcal S_l^{OT}|\approx |\mathcal S_l^{OT}|,\\
M_{\text{decode,cache}}^{(l)} &\propto |\mathcal S_l^I\cup\mathcal S_l^T\cup\mathcal S_l^{OT}|\approx |\mathcal S_l^{OT}|.
\end{aligned}
$$

这里表达的是增长趋势，不是精确的 Transformer 复杂度公式。当输出长度持续增加时，只减少一次图像 token 的方法仍有收益，但在总开销中的占比会下降。

![输出变长后视觉 Token 压缩收益逐渐减弱](/images/posts/dynamic-llava-context-sparsification/dynamic-llava-iclr2025-figure1-inference-cost.png)

*图源：Huang et al., [Dynamic-LLaVA](https://arxiv.org/abs/2412.00876), Figure 1, ICLR 2025；取自作者 arXiv v4 源码中的原始 PNG，仅等比例缩放，坐标轴、图例、标注和数据均未修改。原图用于论文解读，版权归作者。*

Figure 1 把这个现象画得很直观。FastV 在预填充处显著减少视觉 token，但随着生成长度增加，它与原始 LLaVA 的差距缩小；Dynamic-LLaVA 同时压缩输出文本上下文，因此无 cache 时的 FLOPs、带 cache 时的显存增长都更慢。

## 核心贡献

论文的贡献可以归纳为四点：

1. **视觉与语言双预测器**：图像预测器决定保留哪些视觉 token，输出预测器决定哪些已生成 token 继续参与后续推理。
2. **按推理模式设计稀疏化**：预填充裁图像，无 cache 解码裁图像与输出文本，有 cache 解码则控制新 K/V 是否进入缓存。
3. **端到端可训练的离散选择**：用 MaskedSoftmax 隔离被丢 token 的影响，并用 Gumbel-Softmax 与直通估计器绕过 `argmax` 的不可导问题。
4. **批并行实现**：对 batch 内可变长度序列做固定比例 top-k、左侧 padding 和缓存对齐，使稀疏模型仍能利用 GPU 并行。

**论文结论**：在 LLaVA-1.5-7B/13B 上，Dynamic-LLaVA 可把视觉 token 从 576 减到约 115，并保留约 50% 的生成上下文；作者报告预填充计算下降约 75%，无 cache 解码计算和带 cache 显存约下降 50%，多数理解与生成指标只小幅变化。

**我的判断**：这篇论文最有价值的不是再次证明“视觉 token 有冗余”，而是把效率优化从一次性的输入压缩扩展到整个生成生命周期。需要同时看到，它依赖额外训练，而且带 cache 的逐 token 实测延迟只小幅改善；显存收益比速度收益更稳定。

## 方法总览：三种推理模式，三种处理方式

![Dynamic-LLaVA 的三种稀疏推理模式](/images/posts/dynamic-llava-context-sparsification/dynamic-llava-iclr2025-figure2-inference-modes.png)

*图源：Huang et al., [Dynamic-LLaVA](https://arxiv.org/abs/2412.00876), Figure 2, ICLR 2025；取自作者 arXiv v4 源码中的原始 PNG，仅等比例缩放，三种模式、图例与箭头均未修改。原图用于论文解读，版权归作者。*

### 预填充

图像预测器 $P^I$ 接收第 $l$ 层视觉特征，对每个 token 输出“丢弃/保留”两个分数：

$$
\mathcal D^I=P^I(\mathcal S_l^I)\in\mathbb R^{N_l^I\times2},
\qquad
\mathcal M^I=\operatorname*{argmax}_j(\mathcal D^I).
$$

由二值 mask 得到保留集合：

$$
\mathcal S_l^{I*}=\{\mathcal S_{l,i}^I\mid \mathcal M_i^I=1\}.
$$

论文默认在第 2 层后只保留 20% 图像 token，即 576 个 token 约留下 115 个。输入问题文本不裁剪，因为附录实验表明预填充阶段再删 30% 文本会明显伤害 VQAv2、GQA 与 POPE。

### 无 KV cache 解码

每个解码步都要重新处理历史上下文。Dynamic-LLaVA 除了复用已裁剪的视觉 token，还让输出预测器 $P^{OT}$ 对历史输出文本做动态筛选。最新 token 始终保留，因为它负责预测下一 token；较早且被判定不重要的 token 不再进入后续层计算。

默认语言 keep rate 为 50%。与随机丢一半或隔一个丢一个相比，学习到的预测器在相同预算下能维持更好的 PPL 与 METEOR，说明“保留哪些 token”比固定稀疏模式更重要。

### 有 KV cache 解码

带 cache 时，每一步只计算当前 token，但 K/V 缓存会随输出长度增长。输出预测器在当前 token 到来时给出一个二值决定：

$$
\mathcal M_{N_l^{OT}}^{OT}=
\operatorname*{argmax}_j P^{OT}(\mathcal S_{l,N_l^{OT}}^{OT}).
$$

当前 token 始终参与本步注意力；只有当 $\mathcal M_{N_l^{OT}}^{OT}=1$ 时，它的 K/V 才写入后续层的 cache。作者把它称为 **online KV cache compression**：决策依赖当前 token 特征，不需要先读取全部历史 cache 再计算淘汰分数。

![H2O 与 Dynamic-LLaVA 的 KV cache 决策差异](/images/posts/dynamic-llava-context-sparsification/dynamic-llava-iclr2025-figure4-kv-cache-comparison.png)

*图源：Huang et al., [Dynamic-LLaVA](https://arxiv.org/abs/2412.00876), Figure 4, ICLR 2025；取自作者 arXiv v4 源码中的原始 PNG，仅等比例缩放，H2O 与 Dynamic-LLaVA 两条流程均完整保留。原图用于论文解读，版权归作者。*

H2O 需要用当前 query 与历史 cache 计算注意力分数，再从旧缓存中淘汰低分项。Dynamic-LLaVA 则在新激活写入前做门控。前者能根据当前查询重新评价历史信息，后者更容易接入不显式返回 attention score 的高效算子，也避免额外的历史 cache 扫描。

## 两个轻量预测器

两个预测器都先用线性层把 LLM hidden size 压到 512 维。图像预测器再经过两个 Vision Transformer block 和 $512\to256\to128\to2$ 的 MLP，以利用 patch 间关系；输出文本预测器去掉 ViT block，只保留 MLP。

去掉文本侧的全局交互是一个有意约束：带 cache 解码时，当前 token 的写入决策只能依赖当前特征，不能为了做判断再读取完整历史缓存。论文报告预测器计算低于总开销的 1%，附录 FLOPs 表中 7B/13B 均约增加 0.01 TFLOPs。

## 端到端训练：怎样训练离散的丢弃决定

![Dynamic-LLaVA 的 MaskedSoftmax 与预测器训练流程](/images/posts/dynamic-llava-context-sparsification/dynamic-llava-iclr2025-figure3-training-pipeline.png)

*图源：Huang et al., [Dynamic-LLaVA](https://arxiv.org/abs/2412.00876), Figure 3, ICLR 2025；取自作者 arXiv v4 源码中的原始 PNG，仅等比例缩放，mask 矩阵、Gumbel-Softmax 与梯度路径均未修改。原图用于论文解读，版权归作者。*

训练时若真的删除输出 token，就无法并行计算其下一个 token 的语言损失。作者因此保留完整张量，但把预测 mask 展开为注意力 mask 矩阵 $\mathbb G$，并把普通 softmax 替换为：

$$
\operatorname{MaskedSoftmax}(X_{i,j},\mathbb G)
=\frac{e^{X_{i,j}}\mathbb G_{i,j}}
{\sum_{k=1}^{N_l}e^{X_{i,k}}\mathbb G_{i,k}}.
$$

对角线被强制设为 1，使每个 token 仍能产生自己的训练输出；被判定不重要的 token 不再影响其他位置。附录 Table 7 显示，直接把无用 token 置零而不使用 MaskedSoftmax，会让 VQAv2 从 77.8 降到 76.7、GQA 从 61.3 降到 59.8、POPE 从 85.9 降到 84.5。

`argmax` 不可导，因此训练前向使用 Gumbel-Softmax 把决策松弛为连续分布，温度从 1 指数衰减到 0.1；反向则用 Straight-Through Estimator，把二值 mask 的梯度直接传回预测器。

作者还用保留率正则约束资源预算：

$$
\mathcal R=
\left\|\frac{\operatorname{sum}(\mathcal M^I)}{|\mathcal S_l^I|}-r^I\right\|_F
+\mathbf 1[|\mathcal S_l^{OT}|\ge \mathrm{LEN}^{OT}]
\left\|\frac{\operatorname{sum}(\mathcal M^{OT})}{|\mathcal S_l^{OT}|}-r^{OT}\right\|_F.
$$

默认 $r^I=20\%$、$r^{OT}=50\%$、$\mathrm{LEN}^{OT}=50$、正则权重 $\lambda=100$。只在输出长度至少 50 的训练样本上施加文本稀疏约束，是为了避免短答案训练不稳定。

## 训练与推理流程

### 训练

1. 从 LLaVA-1.5-7B/13B 的公开权重开始，使用与 LLaVA-1.5 相同的 656K 图文指令混合数据继续训练 1 个 epoch。
2. 冻结视觉编码器与 projector，只更新 LLM 和两个预测器。
3. 第 2 层产生图像与输出文本 mask，通过 MaskedSoftmax 模拟被裁后的注意力依赖。
4. 用 Gumbel-Softmax 和 STE 优化离散决定，同时用 keep-rate 正则把平均预算拉向 20%/50%。
5. 在 8 张 A100 80GB 上训练，global batch size 为 64；LLM 学习率 $5\times10^{-6}$，预测器学习率 $2\times10^{-4}$。

附录报告 7B 约训练 13 小时，13B 约 24 小时。这不是 training-free 方法；部署前需要为目标骨干做一次稀疏化指令微调。

### 推理

1. 预填充先运行 2 个完整 decoder layer，再按图像预测器分数选出约 115 个 token。
2. 无 cache 解码时，每一步重新筛选历史输出文本，最新 token 强制保留。
3. 有 cache 解码时，输出预测器逐 token 决定新 K/V 是否写入第 2 层之后的 cache。
4. batch 推理时，视觉侧用固定比例 top-k，文本与 cache 侧对不同长度做左 padding，以保持矩阵并行。

预测决定一旦在第 2 层产生，就共享给后续所有层。这样避免了逐层学习不同稀疏率，但也意味着后层不能恢复早期误删的信息。

## 实验设置

视觉理解覆盖 VQAv2、GQA、VizWiz、ScienceQA、TextVQA、POPE、MMBench、SEED、MM-Vet、MMVP、RealWorldQA 与 CVBench-2D。生成质量使用作者构建的三个基准：

- LVIS-VQA single-round：从 LVIS-Instruct4V 选择 1,000 个答案超过 100 词的单轮样本；
- LVIS-VQA multi-round：另选 1,000 个平均答案超过 300 词、对话超过 7 轮的样本；
- ShareGPT4V-VQA：178 个 caption 不少于 300 词、平均输出超过 1,000 token 的样本。

生成实验用 PPL 衡量流畅度、METEOR 衡量与参考答案的相似度。作者声明这些评测图像不在训练集，但 LVIS-VQA 和 ShareGPT4V-VQA 都是论文自行筛选构造，并非已有公共 leaderboard；理解结果与生成结果的证据强度应分开看。

## 主要结果

### 视觉理解：115 个视觉 Token 的代价

以 7B 模型为例，原始 LLaVA-1.5 使用 576 个视觉 token，图像部分计算为 10.1 TFLOPs。Dynamic-LLaVA 同时开启视觉与语言稀疏后使用 115 个视觉 token、2.5 TFLOPs：

| 指标 | LLaVA-1.5-7B | Dynamic-LLaVA-7B | 变化 |
| --- | ---: | ---: | ---: |
| VQAv2 | 78.5 | 77.9 | -0.6 |
| GQA | 62.0 | 61.3 | -0.7 |
| ScienceQA | 66.8 | 68.6 | +1.8 |
| TextVQA | 58.2 | 56.5 | -1.7 |
| POPE | 85.9 | 85.9 | 0.0 |
| MMBench | 64.3 | 64.1 | -0.2 |
| MMVP | 29.3 | 26.3 | -3.0 |

平均变化不大，但 MMVP 与 TextVQA 的下降提醒我们：需要细粒度视觉差异或 OCR 的任务更怕激进裁剪。ScienceQA 的提升不能简单解释为稀疏化增强推理，因为 Dynamic-LLaVA 比基础模型多训练了一个 epoch，训练量并未完全对齐。

### 端到端时间与显存

论文 Table 4 在单张 A100 80GB、batch size 8、LLaVA-1.5-13B 上给出实际测量：

| 方法 | 预填充 | 无 cache 1K | 无 cache 2K | 无 cache 4K | 有 cache 1K 显存 | 2K 显存 | 4K 显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5-13B | 0.83 s | 1453 s | 4117 s | 13368 s | 46 GB | 58 GB | OOM |
| FastV-13B | 0.43 s | 1079 s | 3462 s | OOM | 41 GB | 53 GB | 77 GB |
| Dynamic-LLaVA-13B | **0.37 s** | **838 s** | **2382 s** | **6184 s** | **35 GB** | **42 GB** | **56 GB** |

这组结果支持论文的主张：输出越长，只裁图像的 FastV 越难维持优势；同时稀疏输出上下文才能控制 4K 生成的计算与缓存增长。

但带 cache 的速度结果要单独看。附录 Table 14 在 batch size 1、生成 1,000 token 时，Dynamic-LLaVA 的预填充延迟从 124.52 ms 降到 72.51 ms，逐 token 解码延迟只从 28.42 ms 降到 26.85 ms，约快 5.5%。它显著节省 cache 显存，却没有把同等比例直接转化为 decode latency。

### 生成质量

在 LVIS-VQA single-round 的 7B 实验中，基础模型 PPL/METEOR 为 4.59/0.3103；Dynamic-LLaVA 同时稀疏视觉与语言后为 4.90/0.3108。无 cache 计算从 2.75/2.99 TFLOPs 降到 1.52/1.57，带 cache 的最大缓存从 103/91M 降到 63/46M。

多轮场景中，基础模型为 2.97/0.4227，Dynamic-LLaVA 为 3.17/0.4251；计算和缓存同样明显下降。METEOR 没有下降，但 PPL 上升，说明输出与参考答案仍相似，语言模型对生成序列的置信度却有所变差。

## 消融：预算不能只看平均值

### 视觉与语言 keep rate

视觉 keep rate 从 20% 提到 50% 和 80% 时，VQAv2 从 77.9 升到 78.8/78.9，GQA 从 61.3 升到 62.3/62.5，但视觉 token 也从 115 增到 288/461。视觉预算与理解质量的权衡相对平滑。

语言侧更敏感。keep rate 为 20%/50%/80% 时，LVIS 单轮 PPL 分别为 5.53/4.90/4.76，METEOR 为 0.2592/0.3108/0.3116。20% 虽然更省计算，却让生成质量明显崩塌；论文默认 50% 是一个经验折中，不是可以跨模型直接套用的常数。

### 稀疏层、长度阈值与正则权重

在第 1、2、4 层做选择，三项理解指标差异很小；作者沿用 FastV 的经验选择第 2 层。文本稀疏长度阈值从 0 提到 50 后，VQAv2 从 77.4 升到 77.9、GQA 从 61.0 升到 61.3；继续提到 100 收益有限。正则权重 10、100、1000 的变化也不大。

这说明结果对几个训练超参数较稳，但论文没有测试更晚层、逐层动态稀疏，或让模型为每个请求自动选择整体预算。

## 失败案例与局限

论文没有发布典型的错误回答图，因此不能伪造定性失败案例。现有消融已经暴露出三种可核验的失败模式：

1. **语言上下文删得过多**：$r^{OT}=20\%$ 时，LVIS 单轮 METEOR 从基础模型的 0.3103 降到 0.2592，PPL 升到 5.53。
2. **预填充文本不适合直接照搬视觉裁剪**：额外删除 30% 输入文本后，VQAv2 从 77.8 降到 75.3，GQA 从 61.3 降到 60.2。
3. **早期误删不可恢复**：第 2 层的决定共享给所有后续层，后层即使需要被删细节也无法重新取回。论文没有专门衡量这种错误的输入类型。

还应看到以下证据边界：

1. **需要重新训练**：7B/13B 分别需要约 13/24 小时、8 张 A100 80GB，并非即插即用的 training-free 压缩。
2. **骨干范围有限**：主要实验围绕 LLaVA-1.5 与 TokenPacker，尚未证明同一预测器设计可稳定迁移到 Qwen2-VL、动态分辨率模型或 MoE MLLM。
3. **系统指标不完整**：只报告单张 A100；没有多并发吞吐、能耗、TTFT 分布或实际服务成本。
4. **长文本基准由作者构造**：LVIS-VQA 与 ShareGPT4V-VQA 规模有限，且依赖 PPL/METEOR，缺少人工偏好与事实正确性评估。
5. **理论 FLOPs 口径有限**：预填充表只计算图像 token 在 LLM 内的 FLOPs，不包含视觉编码器、数据搬运、动态选择和服务调度的完整成本。

## 可复现资源

- [ICLR 2025 OpenReview 正式页面、评审与最终论文](https://openreview.net/forum?id=hzVpZDrW73)
- [arXiv v4 全文、HTML 与源码](https://arxiv.org/abs/2412.00876)
- [Apache 2.0 官方代码](https://github.com/Osilly/dynamic_llava)
- [Dynamic-LLaVA-7B 权重](https://huggingface.co/Osilly/Dynamic-LLaVA-7B)
- [Dynamic-LLaVA-13B 权重](https://huggingface.co/Osilly/Dynamic-LLaVA-13B)
- [LLaVA-1.5 训练数据与基础权重](https://github.com/haotian-liu/LLaVA)

官方仓库给出 7B/13B 训练脚本、视觉理解评测脚本和预训练权重。训练命令中的关键超参数与论文附录一致，包括图像/语言 keep rate、Gumbel 温度、预测器学习率和输出长度阈值。

本文使用的 4 张配图均来自作者 arXiv v4 源码中的原始 PNG，只做等比例缩放。arXiv 页面标记为非专属发行许可，并未授予 CC BY；因此这里不把原图声明为开放许可素材，图片版权仍归论文作者，使用范围限于带出处的论文解读。代码则另行采用 Apache 2.0。

## 个人判断

Dynamic-LLaVA 把一个常被忽略的事实讲清楚了：多模态模型的“多模态开销”不只发生在图像进入模型时。回答一旦变长，普通语言生成的历史上下文和 KV cache 会重新成为主瓶颈，视觉 token 裁剪只是第一步。

它的在线 cache 决策很适合工程讨论。相比需要回看历史 attention score 的淘汰法，写入前门控更容易与 fused attention 对接，也能在无 cache 模式复用同一个文本预测器。代价是决策信息更局部，一旦没写入就无法后悔。

我不会把论文中的“约 50%”直接翻译成“线上吞吐翻倍”。带 cache 的单 token 延迟只改善约 5.5%，真正显著的是显存曲线和无 cache 长生成时间。要进入服务系统，还需要在 vLLM/SGLang 一类运行时里验证连续批处理、请求长度混合、padding 浪费和 cache block 回收。

因此，我会把 Dynamic-LLaVA 定位为：**一个贯穿预填充与解码、同时管理视觉 token 和生成上下文的多模态动态稀疏框架**。它提供了扎实的长生成与显存证据，也留下了跨骨干泛化和真实服务吞吐两个关键问题。

## 参考资料

1. Huang et al., [Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification](https://openreview.net/forum?id=hzVpZDrW73), ICLR 2025.
2. Huang et al., [Dynamic-LLaVA, arXiv:2412.00876 v4](https://arxiv.org/abs/2412.00876).
3. Osilly, [Dynamic-LLaVA Official Implementation](https://github.com/Osilly/dynamic_llava), Apache 2.0.
4. Chen et al., [An Image is Worth 1/2 Tokens After Layer 2](https://arxiv.org/abs/2403.06764), ECCV 2024.
5. Zhang et al., [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html), NeurIPS 2023.
6. Zhang et al., [SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](https://proceedings.mlr.press/v267/zhang25s.html), ICML 2025.
