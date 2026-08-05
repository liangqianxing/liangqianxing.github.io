---
title: "MMInference 精读：为百万 Token 多模态预填充重排稀疏注意力"
date: 2026-08-05 20:25:00
description: "拆解 ICML 2025 的 MMInference：以模态感知置换、Grid 稀疏模式与定制 GPU Kernel 加速长视频视觉语言模型的 prefill，属于推理加速（推理侧）方向。"
series: 三大会论文精读
seriesOrder: 18
categories:
  - AI
tags:
  - 多模态大模型
  - 推理加速（推理侧）
  - 稀疏注意力
  - 长上下文
  - GPU Kernel
  - MMInference
  - ICML
hidden: true
haloPublished: true
draft: false
---

给一段两小时视频做问答时，视觉语言模型往往要先把成千上万帧编码成几十万甚至上百万个 token，再完成一次 prefill，最后才开始输出第一个文字。此时瓶颈不一定是视觉编码器，也不只是 token 数量，而是 LLM 中随序列长度平方增长的注意力计算。

ICML 2025 论文《MMInference: Accelerating Pre-filling for Long-Context Visual Language Models via Modality-Aware Permutation Sparse Attention》选择不删除输入 token，也不重新训练模型。它观察长视频注意力中的 Grid 模式与跨模态边界，把原本分散的稀疏区域通过置换聚拢，再交给定制的 block-sparse GPU Kernel 计算。论文在单张 A100、1M token 的 LongVILA prefill 上报告相对 FlashAttention-2 最高 8.3 倍加速。

这篇论文直接属于本专题的 **推理加速（推理侧）**。它面对长上下文视觉语言模型，给出注意力 FLOPs、Kernel 延迟、端到端 prefill 延迟和多个图像/视频任务的质量结果，不是只讨论稀疏表征的泛多模态论文。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **MMInference: Accelerating Pre-filling for Long-Context Visual Language Models via Modality-Aware Permutation Sparse Attention** |
| 作者 | Yucheng Li、Huiqiang Jiang、Chengruidong Zhang、Qianhui Wu、Xufang Luo、Surin Ahn、Amir H. Abdi、Dongsheng Li、Jianfeng Gao、Yuqing Yang、Lili Qiu |
| 会议 | ICML 2025，Proceedings of the 42nd International Conference on Machine Learning |
| 专题子方向 | **推理加速（推理侧）**：动态稀疏注意力、模态感知置换、GPU Kernel 协同优化 |
| 正式论文 | [PMLR 267:34998-35020](https://proceedings.mlr.press/v267/li25aq.html) |
| 正式评审 | [OpenReview: me6PfbATWM](https://openreview.net/forum?id=me6PfbATWM) |
| 作者版本 | [arXiv:2504.16083 v2](https://arxiv.org/abs/2504.16083)，arXiv 非专属发行许可，版权归作者 |
| 官方项目页 | [MMInference Project](https://hqjiang.com/mminference.html) |
| 官方代码 | [microsoft/MInference](https://github.com/microsoft/MInference)，MIT License |

**选择理由**：上一篇专题文章 RePIC 属于训练侧个性化，本轮按计划切回推理侧。现有精读已经覆盖视觉 token 压缩、动态 token/上下文稀疏、KV cache 与早退；MMInference 改从长多模态 prefill 的注意力模式和 GPU Kernel 入手，补上系统—算法协同这一细分主题。论文正式收录、23 页全文与附录、原始矢量图、MIT 代码和硬件实测均可公开核验，仓库与历史记录中也没有同题文章。

## 问题背景：为什么长视频的第一个 Token 要等几分钟

对序列长度 $S$、单头维度 $d_h$，标准因果注意力为：

$$
O=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M_{\mathrm{causal}}\right)V,
\qquad Q,K,V\in\mathbb R^{S\times d_h}.
$$

$QK^\top$ 产生 $S\times S$ 的注意力矩阵，计算量与显存访问随 $S^2$ 增长。短回答的 decode 每次只追加一个 token；但 prefill 必须一次处理整段视频与文字，因此百万 token 会把 Time-to-First-Token 推到分钟级。

![长视频 VLM 的注意力成本、稀疏度与动态性](/images/posts/mminference-modality-aware-sparse-attention/mminference-icml2025-figure2-cost-sparsity-dynamics.png)

*图源：Li et al., [MMInference](https://arxiv.org/abs/2504.16083), Figure 2, ICML 2025；从作者 arXiv v2 的正式排版 PDF 以 300 DPI 裁取，只移除页面正文与留白，三组坐标轴、图例、子图标识和原始图注均完整保留。版权归作者，原图用于论文解读。*

Figure 2 给出三个关键事实：

1. 在 LongVILA-7B-1M 上，帧数从 512 增到 4K 后，prefill 总时间从不足 1 分钟升到约 24 分钟，增长几乎都来自 attention；ViT 与 FFN 仍在 1 分钟附近。
2. 128K 上要召回 95% 注意力质量，VLM 平均只需计算 5.78% 权重，但比文本 LLM 的 1.79% 更稠密；底层尤其不容易裁。
3. 稀疏位置依赖当前请求。把另一条请求的 top-k 索引直接复用，平均注意力召回只有 71.3%，所以静态掩码或跨请求照搬索引并不可靠。

这里的机会不是“注意力永远只看同几个 token”，而是：**稀疏结构有规律，但具体位置必须在线估计。**

## 核心贡献

论文的贡献可以拆成四层：

1. **识别多模态特有的 Grid head**：视频帧中的时空局部性形成等间隔横线与竖线，和文本长上下文常见的 vertical-slash 不同。
2. **显式处理模态边界**：将跨模态注意力分成 No-Boundary、K-Boundary、Q-Boundary 和 2D-Boundary，不假设整段序列只有一种稀疏模式。
3. **用置换把“数学稀疏”变成“硬件可算”**：重排 $Q/K/V$ 或其加载顺序，将散乱格点聚成连续块，再复用 Tensor Core 擅长的稠密块乘。
4. **打通离线搜索、在线估计与定制 Kernel**：每个 head 离线选择模式，推理时只用末尾 64 个 query 动态估计索引，Triton/FlashAttention/PIT Kernel 完成实际加速。

**论文结论**：在 LongVILA、LLaVA-Video、VideoChat-Flash 与 Qwen2.5-VL 上，MMInference 能显著降低 prefill 注意力计算，并在长视频理解与 needle-in-a-haystack 任务中接近全注意力。

**我的判断**：论文真正的价值不在“稀疏注意力”四个字，而在模式、置换和 Kernel 三者同时成立。若只把注意力权重置零却仍执行稠密矩阵乘，FLOPs 账面下降不会自动变成墙钟时间下降。

## 方法总览：先给每个 Head 分类，再动态构造稀疏索引

![MMInference 的跨模态与模态内稀疏模式](/images/posts/mminference-modality-aware-sparse-attention/mminference-icml2025-figure4-framework.png)

*图源：Li et al., [MMInference](https://arxiv.org/abs/2504.16083), Figure 4, ICML 2025；直接从作者 arXiv v2 源码中的原始矢量 PDF 栅格化，仅等比例缩放，模式、编号、箭头和文字未修改。版权归作者，原图用于论文解读。*

框架把模式分成两条轴：

- **模态内（intra-modality）**：A-shape、vertical-slash、Grid；
- **模态间（inter-modality）**：No-Boundary、K-Boundary、Q-Boundary、2D-Boundary。

一个 head 不是在七个模式中简单选一个，而是先判断跨模态边界，再为每个模态区域选择合适的模态内模式。离线搜索得到每层每头的配置；在线输入到来后，再估计当前 stride、phase 与 sparse index。

## 关键模块一：Grid Pattern 从哪里来

视频通常以帧为单位采样，每帧又按二维 patch 展开成 token。相邻 patch、同一空间位置的跨帧 patch，以及固定间隔的局部位置，会在注意力图中形成规则横线和竖线。论文称之为 Grid pattern。

![置换前后的 Grid 与模态边界稀疏模式](/images/posts/mminference-modality-aware-sparse-attention/mminference-icml2025-figure3-patterns-permutation.png)

*图源：Li et al., [MMInference](https://arxiv.org/abs/2504.16083), Figure 3, ICML 2025；从作者 arXiv v2 的正式 PDF 以 300 DPI 裁取，只移除页眉、正文、原 PDF 图注与页面留白，六个子图、标识和标签均完整保留。版权归作者，原图用于论文解读。*

Figure 3(a) 中，固定帧步长与 $14\times14$ patch 网格共同制造等间隔线。问题是这些线在原始矩阵里相隔很远，直接逐点读取会形成碎片化、不连续的显存访问。

MMInference 搜索 Grid 的 stride $s_g$ 与 phase $p_g$。它只取末尾 $q_{\mathrm{last}}=64$ 个 query，计算近似注意力：

$$
\hat A=
\operatorname{Softmax}\left(
\frac{Q_{[-q_{\mathrm{last}}:]}K^\top}{\sqrt{d_h}}+M_{\mathrm{causal}}
\right).
$$

随后在候选 stride 集合 $\Phi_g$ 中寻找使视图内最大响应最高的组合：

$$
(s_g,p_g)=
\arg\max_{s\in\Phi_g,\,p}
\operatorname{score}(\operatorname{view}(\hat A;s,p)).
$$

这不是论文 Algorithm 1 的逐符号复刻，而是其工程含义：用一小段 query 定位当前请求的网格周期和起点，再把同余位置聚在一起。

置换后的 Figure 3(d) 将横线、竖线移动到连续边界。论文实现不先显式生成一份完整重排张量，而是在 Kernel 内按置换地址加载和写回，减少额外 transpose 开销。

## 关键模块二：模态边界不能用同一张 Mask 硬套

混合输入可能是“视频—文字—视频—文字”。文本与视觉、视觉与视觉、文本与文本的注意力分布不同，论文把边界分成四类：

- **No-Boundary**：整张注意力图可共享同一模态内模式；
- **K-Boundary**：key 维有边界，但 query 维连续，仍可整体处理；
- **Q-Boundary**：query 维跨模态出现明显断裂；
- **2D-Boundary**：query 与 key 两个维度都有边界，形成 V2V、V2T、T2V、T2T 四块。

对 Q-Boundary，方法按模态重排 $Q$，让同一模态的 query 连续，再分别运行稀疏注意力。可概括为：

$$
Q'=P_QQ,
\qquad
O'=\bigcup_{m\in\mathcal M}
\operatorname{SparseAttn}(Q'_m,K,V;\pi_m),
$$

其中 $P_Q$ 是按模态排序的置换，$\pi_m$ 是该模态选择的稀疏模式。

对 2D-Boundary，则同时重排 $Q,K,V$：

$$
Q'=P_QQ,\quad K'=P_KK,\quad V'=P_KV,
$$

并在每个模态对 $(m_i,m_j)$ 的子块上运行动态稀疏注意力，最后按逆置换恢复输出顺序。Figure 3(e)(f) 显示，原来被文本打断的视觉稀疏线在重排后重新连续。

## 关键模块三：离线模式搜索为何只用一条样本

MMInference 为每个 attention head 搜索模式及超参数，例如 Grid stride、是否启用横线/竖线/slash、A-shape 的 sink/local 大小、vertical-slash 的线数。

作者没有只用理论 FLOPs 统一预算，而是先按 GPU Kernel 实测构造“计算成本相近”的候选集合。候选配置 $p$ 的选择目标可以写成：

$$
p^*=\arg\min_{p\in\rho}
\left\|
\operatorname{SparseAttn}(Q,K,V;p)
-\operatorname{Attn}(Q,K,V)
\right\|.
$$

这里比较的是最终 attention output，因此把 $V$ 也纳入评分，比只看 $QK^\top$ 上的权重召回更贴近真实误差。

附录给出的校准设置很克制：每个模型只取 EgoSchema 的 **一条不超过 25K token 的样本**，在单张 A100 上搜索约 **15 分钟**。LLaVA-Video-7B、LongVILA-256Frame 与 LongVILA-1M 分别搜索一次；论文称这套配置能跨长度和任务泛化。

这是易部署的优点，也是需要复验的假设：一条样本能否覆盖企业私有视频、OCR 密集画面、音视频交错输入，论文没有给出分布外校准实验。

## 训练与推理流程

MMInference **不训练模型，不修改权重，也不要求稀疏微调**。完整流程分成两个阶段。

### 离线准备

1. 为目标 VLM 取一条校准样本；
2. 逐层逐头比较 Grid、A-shape、vertical-slash 与边界模式；
3. 在实测成本接近的候选中，选择 attention output 误差最小的配置；
4. 保存每个 head 的模式与超参数，不保存某条请求的固定 sparse index。

### 在线 Prefill

1. 根据输入的模态位置构造 modality index；
2. 用最后 64 个 query 估计动态稀疏索引；
3. 对 Grid、Q-Boundary 与 2D-Boundary 在 Kernel 内执行地址置换；
4. 用 FlashAttention/FlashDecoding/PIT 风格的 block-sparse Kernel 完成注意力；
5. 把输出写回原始 token 顺序，继续后续 FFN 与 decoder 层。

论文实现基于 Triton、FlashAttention 与动态稀疏编译器 PIT。Grid Kernel 同时结合 block-sparse FlashDecoding 与 block-sparse FlashAttention-2，分别减少 query 和 key 的加载。

## 实验设置

硬件与数值精度：

- 单张 NVIDIA A100；
- bfloat16；
- greedy decoding，以减少随机波动；
- 延迟实验关注 prefill，Kernel microbenchmark 还计入动态索引估计与稀疏结构构建时间。

模型与长度：

- LLaVA-Video-7B：110 帧，20,240 token；
- LongVILA-7B：256 帧，65,800 token；
- Qwen2.5-VL-7B-Instruct：256 帧，33,950 token；
- LongVILA-Qwen2-7B-1M：V-NIAH 与 MM-NIAH，最长约 1.1M token；
- VideoChat-Flash：512 帧，先从每帧 196 个视觉 token 压到 16 个，再接 MMInference。

视频理解覆盖 VideoDC、ActivityNet-QA、EgoSchema、NExT-QA、Perception Test 与 VideoMME。长上下文检索使用 Video Needle in a Haystack（V-NIAH），论文还构造了 25% 输入为文本片段的 Mixed-Modality NIAH（MM-NIAH）。

对照包括 FlashAttention-2、Sparse Transformer 的 fixed/strided 模式、A-shape、Tri-shape、MInference vertical-slash 与视觉 token 压缩方法 VisionZip。

## 主要结果：质量、FLOPs 与延迟必须放在一起看

### 视频理解主表

| 模型与设置 | 方法 | Attention FLOPs | 平均分 |
| --- | --- | ---: | ---: |
| LLaVA-Video，20,240 token | Full Attention | 100% | 57.6 |
|  | MInference | 78.8% | 57.5 |
|  | **MMInference** | **47.3%** | **57.6** |
| LongVILA，65,800 token | Full Attention | 100% | 55.5 |
|  | MInference | 47.0% | 55.2 |
|  | **MMInference** | **31.8%** | **55.4** |
| Qwen2.5-VL，33,950 token | Full Attention | 100% | 59.5 |
|  | **MMInference** | **41.3%** | **59.4** |

LLaVA-Video 上，MMInference 用 47.3% attention FLOPs 得到与全注意力同为 57.6 的平均分；MInference 需要 78.8% FLOPs。LongVILA 上，MMInference 比全注意力低 0.1 分，同时比 MInference 再少约三分之一的 attention FLOPs。

这些平均分不是“每项都不掉”。例如 LLaVA-Video 的 EgoSchema 从 57.0 升到 57.1，但 Perception Test 从 66.1 到 66.2、NExT-QA 从 81.2 降到 80.1，指标间存在小幅波动。论文支持的是整体接近，而不是逐样本数学等价。

### 百万 Token 的检索质量

LongVILA-1M 的 Figure 5 报告：

- V-NIAH：MMInference 97.7%，全注意力 98.3%；
- MM-NIAH：MMInference 91.3%，全注意力 90.9%；
- MInference 在附录 V-NIAH 为 96.7%，MM-NIAH 为 88.0%。

MM-NIAH 中稀疏方法略高 0.4 个百分点不应解读成“稀疏化提升了模型能力”。热图汇总、离散题目与评测波动都可能产生小差异，更稳妥的结论是没有观察到明显退化。

### 与视觉 Token 压缩叠加

VideoChat-Flash 已在 ViT 阶段把每帧 token 从 196 压到 16。加入 MMInference 后，512 帧视频的平均分从 56.8 到 56.7，说明 token 压缩和 LLM 内稀疏注意力可以叠加：前者缩短序列，后者减少剩余序列的注意力块。

## 延迟结果：8.3× 到底指什么

![MMInference 在不同长上下文上的 prefill 延迟](/images/posts/mminference-modality-aware-sparse-attention/mminference-icml2025-figure7-prefill-latency.png)

*图源：Li et al., [MMInference](https://arxiv.org/abs/2504.16083), Figure 7, ICML 2025；直接从作者 arXiv v2 源码中的原始矢量 PDF 栅格化，仅等比例缩放，坐标轴、图例、数据点与加速比标记均未修改。版权归作者，原图用于论文解读。*

在单张 A100 上，序列越长，稀疏 Kernel 摊薄索引估计等固定开销的能力越强：

- 360K token：相对 FlashAttention-2 约 3.3×，相对 MInference 约 1.5×；
- 720K token：相对 FlashAttention-2 约 6×，相对 MInference 约 1.5×；
- 1M token：相对 FlashAttention-2 **8.3×**，相对 MInference **1.6×**。

附录 Figure 16 的单 attention Kernel microbenchmark 在 1M token 时给出 Grid 为 358 ms，并报告相对 FlashAttention 的 Kernel 级最高 12×。这个数字高于端到端 8.3×，因为完整 prefill 还有 FFN、归一化、投影与其他不可被稀疏 attention 消掉的部分。

必须把标题中的“推理加速”限定准确：论文重点加速 **长多模态输入的 prefill**。它没有证明 decode 每个新 token 也加速 8.3×，更没有证明一个包含视觉编码、排队、网络传输与长答案生成的在线请求端到端加速 8.3×。

## 消融与机制分析

论文没有提供一张传统的“去掉模块”表，但正文与附录给出三组机制对照。

### 1. Grid 比 Vertical-Slash 更适合视频

Grid 与 vertical-slash 在视频理解和 V-NIAH 上都能保持质量，但 vertical-slash 的斜线覆盖块彼此重叠少，实际 Kernel 必须读取更多分散 block。置换后的 Grid 聚合成更连续的块，在 1M token 上还能比 vertical-slash 快 2-3 倍。

### 2. 跨模态边界模块不能省

附录 MM-NIAH 中，MInference 与 “MMInference w/o Inter” 都是 88.0%，完整 MMInference 为 91.3%。这说明只优化模态内 Grid 还不够；在文字插入视频的场景里，Q-Boundary 与 2D-Boundary 处理承担了约 3.3 个百分点的差距。

### 3. 文本索引不能直接外推到视觉

Figure 8 显示，由文本区域建立的 sparse index 在文本上召回高，但进入视觉区域就明显失效；从视觉区域建立的索引则能跨越中间文本边界，外推到另一段视觉区域。这支撑了“先按模态聚合，再在同模态内共享规律”的设计。

## 失败案例与局限

论文没有单列自然语言失败案例，也没有逐样本展示 MMInference 答错而全注意力答对的画面。这本身是阅读时应注意的缺口。结合正文、附录与实验范围，可以确认以下边界。

1. **硬件证据集中在单张 A100**：没有 H100、消费卡、AMD GPU、TPU 或多卡张量并行结果，置换与稀疏块在其他架构上的收益不能直接外推。
2. **只测 BF16，未与量化联合做系统评估**：论文证明能与 token 压缩叠加，但没有给出 INT8/INT4 权重、KV cache 量化或低精度稀疏 Kernel 的结果。
3. **百万 Token 的 8.3× 是 prefill 指标**：decode、吞吐、并发、P95/P99 延迟、调度与 KV cache 占用没有完整报告。
4. **每个模型要离线搜索**：虽然一条样本、15 分钟很便宜，但换 backbone、层数、视觉编码方式或 Kernel 预算后都可能重搜；论文没有证明一个模型的配置可迁移到另一个模型。
5. **校准分布偏窄**：模式搜索只用一条 EgoSchema 样本。OCR 密集图、文档、多图对话、音频—视觉交错和企业私有数据可能形成不同边界。
6. **质量评估以任务汇总为主**：平均分和 NIAH recall 接近全注意力，不等于每个罕见细节都安全。视觉稀疏模式中被低估的小目标仍可能造成单样本失败。
7. **能耗与成本未报告**：更短延迟通常会降低单请求 GPU 时间，但论文没有实测功耗、每请求能量或单位吞吐成本。

这些是基于实验覆盖范围的作者外分析，不是论文声称已经解决的问题。

## 可复现资源与工程落地

- [PMLR 正式页面与 23 页论文](https://proceedings.mlr.press/v267/li25aq.html)
- [OpenReview 评审记录](https://openreview.net/forum?id=me6PfbATWM)
- [arXiv v2 全文、HTML 与 LaTeX 源码](https://arxiv.org/abs/2504.16083)
- [作者项目页](https://hqjiang.com/mminference.html)
- [官方 MInference 仓库](https://github.com/microsoft/MInference)，MIT License
- [LongVILA](https://github.com/NVlabs/VILA)
- [LLaVA-NeXT / LLaVA-Video](https://github.com/LLaVA-VL/LLaVA-NeXT)

复现时建议按下面顺序，而不是一上来跑 1M token：

1. 先在官方支持的 A100 环境验证 dense FlashAttention-2 基线；
2. 固定模型、输入帧数、dtype 与计时边界，复现 120K/360K 的 prefill；
3. 检查每层 head 的离线模式文件是否与模型版本对应；
4. 分开记录 sparse index 构建、Kernel attention 与整层 forward；
5. 同时跑官方任务指标，不能只看 Kernel microbenchmark；
6. 再叠加视觉 token 压缩或量化，逐项确认质量损失是否可加。

## 个人判断

MMInference 最值得工程读者记住的，是它把“稀疏”从数学属性推进到硬件执行计划：

$$
\text{输入的模态结构}
\rightarrow
\text{每头稀疏模式}
\rightarrow
\text{在线索引}
\rightarrow
\text{置换后的连续块}
\rightarrow
\text{可获得墙钟收益的 Kernel}.
$$

这条链路少一环都不完整。只看权重热图会忽略显存访问，只看 FLOPs 会忽略动态索引开销，只看 Kernel 加速又会夸大完整请求收益。

在当前证据下，我会把 MMInference 定位为：**长视频 VLM prefill 的高价值系统基线，尤其适合数十万到百万 token、A100 级 GPU、允许按模型做一次离线校准的场景。** 若输入只有几万 token，或主要瓶颈在视觉编码器、decode 与服务调度，应该先做端到端 profile，再决定是否引入定制稀疏 Kernel。

## 参考资料

1. Li et al., [MMInference: Accelerating Pre-filling for Long-Context Visual Language Models via Modality-Aware Permutation Sparse Attention](https://proceedings.mlr.press/v267/li25aq.html), ICML 2025.
2. Li et al., [arXiv:2504.16083 v2](https://arxiv.org/abs/2504.16083), 2025.
3. Jiang et al., [MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](https://openreview.net/forum?id=fPBACAbqSN), NeurIPS 2024.
4. Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://openreview.net/forum?id=mZn2Xyh9Ec), ICLR 2024.
5. Chen et al., [LongVILA: Scaling Long-Context Visual Language Models for Long Videos](https://openreview.net/forum?id=wCXAlfvCy6), ICLR 2025.
6. Zhang et al., [LLaVA-Video: Video Instruction Tuning With Synthetic Data](https://arxiv.org/abs/2411.14565), 2024.
7. Tu et al., [VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration](https://arxiv.org/abs/2410.23317), ICLR 2025.
