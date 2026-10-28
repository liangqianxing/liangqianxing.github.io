---
title: "SparseVLM 精读：让问题决定保留哪些视觉 Token"
date: 2026-07-23 20:30:00
description: 从文本 rater、逐层视觉 Token 稀疏化与被裁 Token 回收出发，拆解 ICML 2025 的 SparseVLM 如何在不训练新参数的前提下加速多模态大模型推理。
series: 三大会论文精读
seriesOrder: 10
categories:
  - AI
tags:
  - 多模态大模型
  - 推理加速
  - 视觉 Token 裁剪
  - 稀疏推理
  - SparseVLM
  - LLaVA
  - ICML
hidden: true
haloPublished: true
---

同一张图片，问题是“车身写了什么”“有几辆公交车”还是“屋顶是什么颜色”，决定了模型真正需要看的区域。如果裁剪器只按图像本身挑 patch，三种问题会得到相同的视觉子集；这节省了计算，却可能在问题变化时删掉答案证据。

SparseVLM 把问题文本纳入视觉 Token 评分：先选出与图像相关的文本 token 作为 rater，再用语言到视觉的注意力逐层裁掉低相关视觉 token，并把部分被裁 token 聚类回收。它不训练额外网络，直接改造现成 VLM 的推理过程。

这篇论文直接属于本专题的 **推理加速（推理侧）** 方向。它在 LLaVA、Mini-Gemini、Qwen2-VL 和 Video-LLaVA 上报告 FLOPs、CUDA 时间与 KV cache 降幅，并在图像、视频问答中验证质量，不是仅讨论视觉表征压缩的泛多模态论文。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference** |
| 作者 | Yuan Zhang、Chun-Kai Fan、Junpeng Ma、Wenzhao Zheng、Tao Huang、Kuan Cheng、Denis A. Gudovskiy、Tomoyuki Okuno、Yohei Nakata、Kurt Keutzer、Shanghang Zhang |
| 会议 | ICML 2025，Proceedings of the 42nd International Conference on Machine Learning |
| 专题子方向 | 推理加速（推理侧）：视觉 Token 裁剪、动态稀疏化、KV cache 压缩 |
| 正式论文 | [PMLR 267:74840-74857](https://proceedings.mlr.press/v267/zhang25s.html) |
| 正式评审 | [OpenReview: 80faIPZ67S](https://openreview.net/forum?id=80faIPZ67S) |
| 作者版本与许可 | [arXiv:2410.04417v4](https://arxiv.org/abs/2410.04417)，CC BY-NC-SA 4.0 |
| 官方代码 | [Gumpest/SparseVLMs](https://github.com/Gumpest/SparseVLMs)，Apache 2.0 |

**选择理由**：上一篇专题文章 InstructBLIP 属于个性化训练侧，本轮切回推理加速。已有文章已经覆盖 M3 的多档视觉粒度与 DeeR-VLA 的动态早退；SparseVLM 则研究问题引导的推理期 Token 裁剪，机制不同。正式论文、18 页附录、原始矢量图、代码、硬件环境与效率实验均公开可核验。

## 问题背景：视觉 Token 多，但答案证据往往很少

主流 VLM 先把图像编码成视觉 token，再与文本 token 拼接送入 LLM。LLaVA-1.5 的 $336\times336$ 图像产生 576 个视觉 token；更高分辨率模型和视频模型可达到数千个。它们会增加 Transformer attention、FFN、KV cache 和数据搬运成本。

已有方法通常在视觉编码器、projector 或 LLM 内压缩 token。SparseVLM 关注其中一个具体缺口：**如果裁剪发生在多模态推理阶段，问题文本本身就是最直接的选择条件。**

![不同问题需要保留不同视觉区域](/images/posts/sparsevlm-text-guided-visual-token-sparsification/sparsevlm-icml2025-figure1-text-guided-sparsification.png)

*图源：Zhang et al., [SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](https://arxiv.org/abs/2410.04417), Figure 1, ICML 2025；从作者 CC BY-NC-SA 4.0 arXiv 源码中的原始矢量图直接栅格化，仅去除外围页面留白，图内内容未修改。原图用于论文解读。*

Figure 1 中，同一张公交车图片面对三个问题时，SparseVLM 保留的 patch 不同：OCR 问题集中到标牌，计数问题覆盖车身，颜色问题转向屋顶。文本无关方法的保留区域基本不随问题改变。

## 核心贡献

论文的贡献可以归纳为四点：

1. **无需训练的文本引导裁剪**：复用 VLM 内部注意力，不增加可学习参数或微调数据。
2. **先筛文本 rater，再评视觉 token**：避免介词、代词等视觉无关词稀释评分。
3. **逐层控制稀疏度并回收信息**：用注意力矩阵的秩估计冗余，把一部分被裁 token 聚类重建为少量 token。
4. **覆盖图像与视频 VLM**：在 LLaVA、Mini-Gemini、Qwen2-VL、Video-LLaVA 上同时评估质量与效率。

**论文结论**：PMLR 摘要称，LLaVA 加入 SparseVLM 后可减少 54% FLOPs、降低 37% CUDA 时间，并保持 97% 准确率。

**我的判断**：方向成立，Table 1 也显示清楚的质量-效率折中；但摘要把不同近似配置压成一句话。正式表格中，192 token 配置对应 53.7% FLOPs 降幅、36.9% 延迟降幅和 99.1% 相对平均分；128 token 配置对应 62.8%、42.4% 和 96.7%。工程评估应引用具体配置，不应只转述摘要。

## 方法总览

![SparseVLM 架构](/images/posts/sparsevlm-text-guided-visual-token-sparsification/sparsevlm-icml2025-figure2-architecture.png)

*图源：Zhang et al., [SparseVLM](https://arxiv.org/abs/2410.04417), Figure 2, ICML 2025；从作者 CC BY-NC-SA 4.0 arXiv 源码中的原始矢量图直接栅格化，仅去除外围页面留白，结构、标签与数值未修改。原图用于论文解读。*

整个流程分成 LLM 前和 LLM 内两段：

1. LLM 前比较图像嵌入与问题嵌入，选出视觉相关文本 rater；
2. LLM 内从 self-attention 取出“文本 query 到视觉 key”的子矩阵；
3. 对 rater 的注意力求平均，得到每个视觉 token 的重要性；
4. 逐层删掉低分 token，并把删除池中相对重要的一部分聚类回收。

### 1. 从 Self-Attention 得到视觉优先级

对长度为 $L$ 、隐藏维度为 $D$ 的序列，单头注意力为：

$$
A=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt D}\right),
\qquad A\in\mathbb R^{L\times L}.
$$

设文本 token 数为 $L_t$ 、视觉 token 数为 $L_v$ 。从 $A$ 中取文本 query 与视觉 key 的交叉区域，得到：

$$
P=A_{\mathcal L,\mathcal I},
\qquad P\in\mathbb R^{L_t\times L_v}.
$$

若直接对全部文本行求平均，格式词和视觉无关词也会参与评分。SparseVLM 因此先选 rater。

### 2. 选择文本 Rater

设视觉嵌入为 $H_v=Wg(x_v)$ ，问题嵌入为 $H_q$ 。作者先计算跨模态相似度，再沿视觉维度平均：

$$
r=\frac{1}{L_v}\sum_{j=1}^{L_v}
\operatorname{Softmax}(H_vH_q^\top)_j.
$$

只保留高于 $r$ 均值的文本 token：

$$
\mathcal S=\{i\mid r_i\geq \operatorname{mean}(r)\}.
$$

这些 token 成为 rater。视觉 token $j$ 的最终重要性，是 $P$ 在 rater 行上的平均值；值越大越应该保留。rater 选择只在进入 decoder 前做一次，论文估算其主要矩阵乘 FLOPs 为 $2L_tL_vD$。

### 3. 按矩阵秩估计逐层裁剪量

论文把 $P$ 的低秩视为视觉信息冗余的信号。每层删除数量写为：

$$
N=\lambda\left(L_v-\operatorname{rank}(P)\right).
$$

$\lambda$ 是缩放因子；若 $N=0$ ，该层跳过裁剪。随后删除重要性最低的 $N$ 个视觉 token。秩通过 SVD 估计，论文给出的额外复杂度约为：

$$
L_tL_v\min(L_t,L_v).
$$

这里需要谨慎： $P$ 的最大秩受较短维度限制，数值秩又依赖奇异值阈值。论文没有在正文中充分讨论阈值敏感性；公开代码与这套自适应公式也并不完全对应，后文会单独说明。

### 4. 回收被裁 Token

直接丢弃全部低分 token 容易损失小物体、文字或背景证据。SparseVLM 从删除池中取重要性较高的前 $\tau$ 比例，使用 k 近邻密度峰值聚类。

对候选 token $\bar h_v^i$ ，局部密度为：

$$
\rho_i=\exp\left(
-\frac{1}{k}\sum_{\bar h_v^j\in\mathcal K(\bar h_v^i)}
\lVert\bar h_v^i-\bar h_v^j\rVert_2^2
\right).
$$

再计算它到更高密度点的最小距离 $\delta_i$ ，用 $\rho_i\delta_i$ 选择聚类中心。其他 token 按余弦相似度分配到最近中心，同组 token 逐元素求和：

$$
T_k=\sum_{i=1}^{N_k}\mathcal T[i].
$$

这样，一组将被删除的视觉细节被压成一个重建 token，重新送回序列。它比“全部保留”便宜，也比“全部丢弃”更稳，但聚类本身并非免费。

## 与 FlashAttention 的兼容

标准 FlashAttention 不显式保存完整注意力矩阵，不能直接读取 $P$ 。论文附录提出 dual-flash attention：

1. 第一次 FlashAttention 正常计算 hidden states；
2. 第二次使用特殊 $V$ 矩阵，只在 rater 对应行放置 $1/n$ ，其余为 0；
3. 分块 attention 与该 $V$ 相乘后，直接返回 rater 平均注意力；
4. 对结果做 top-k，生成视觉 token mask，再应用到第一次前向的 hidden states。

它避免物化完整 $L\times L$ 注意力，但多出一次专用 attention 前向。论文把这部分包含在 CUDA 时间实测中，因此不能只用删后序列长度估算真实速度。

## 训练与推理流程

SparseVLM 本身不训练。复现实验从已经训练好的多模态模型开始：

- 图像模型：LLaVA-1.5-7B/13B、Mini-Gemini、Qwen2-VL；
- 视频模型：Video-LLaVA；
- 硬件：单张 NVIDIA A100 80GB；
- 软件：Python 3.10、PyTorch 2.1.2、CUDA 11.8、Transformers 4.31.0；
- attention：效率实验使用 FlashAttention；
- CUDA 时间：包括图像编码、适用时的 KV cache 加载和 Transformer forward，不包括模型加载等固定开销。

推理时先选择 rater，再在指定 decoder 层重复评分、裁剪和 token 回收。对视频，作者把相同机制扩展到多帧视觉 token；Video-LLaVA 从 2048 个 token 压到 194 个。

## 实验设置

图像理解覆盖 GQA、MMBench、MME、POPE、ScienceQA、SEED-Bench、TextVQA 和 MM-Vet。视频理解覆盖 TGIF-QA、MSVD-QA、MSRVTT-QA 和 ActivityNet-QA；视频实验按 FastV 设置，每个基准取前 1000 个样本，并使用 Video-ChatGPT 评估工具评分。

主要对照包括：

- Vanilla：不裁剪的原模型；
- ToMe：合并相似 token；
- FastV：推理期视觉 token 裁剪；
- PDrop：PyramidDrop 的 training-free 配置；
- Random Sparse：附录效率曲线中的随机裁剪。

## 主要结果：必须同时看 Token、质量和延迟

LLaVA-1.5-7B 的原始配置为 576 个视觉 token。Table 1 的关键行如下：

| 方法 | 保留 Token | 相对平均分 | FLOPs | CUDA 延迟 |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 576 | 100.0 | 4.62 T | 57.82 ms |
| SparseVLM | 192 | 99.1 | 2.14 T | 36.50 ms |
| SparseVLM | 128 | 96.7 | 1.72 T | 33.28 ms |
| SparseVLM | 64 | 89.3 | 1.30 T | 29.89 ms |
| FastV | 192 | 87.9 | 2.11 T | 34.87 ms |
| PDrop | 192 | 95.9 | 2.03 T | 36.74 ms |

192-token SparseVLM 把视觉序列缩短 66.7%，平均只下降 0.9 个百分点；相同 token 预算下，FastV 下降 12.1 个百分点。它的代价是 36.50 ms 略慢于 FastV 的 34.87 ms，说明文本评分与回收确实有运行时开销。

128-token 配置从 4.62 T 降到 1.72 T，减少 62.8%；CUDA 延迟从 57.82 ms 降到 33.28 ms，减少 42.4%。64-token 配置的 FLOPs 更低，但延迟只进一步减少 3.39 ms，显示固定开销和额外稀疏化操作逐渐主导。

### Mini-Gemini：裁得越狠，差距越明显

![Mini-Gemini 在不同视觉 Token 预算下的性能](/images/posts/sparsevlm-text-guided-visual-token-sparsification/sparsevlm-icml2025-figure4-mgm-token-tradeoff.png)

*图源：Zhang et al., [SparseVLM](https://arxiv.org/abs/2410.04417), Figure 4, ICML 2025；从作者 CC BY-NC-SA 4.0 arXiv 源码中的原始矢量图直接栅格化，坐标轴、图例、数据点与基线均完整保留。原图用于论文解读。*

在 POPE、TextVQA 和 GQA 上，SparseVLM 的曲线在低 token 区间明显高于 ToMe 和 FastV。这个趋势支持作者的机制解释：文本引导在预算紧张时更重要，因为每次误删都更难被剩余 token 弥补。

但三条曲线也说明不存在“统一无损压缩率”。当 token 进一步降到约 64 时，SparseVLM 仍显著下降；TextVQA 的 OCR 需求尤其敏感。

### Qwen2-VL 与视频

Qwen2-VL 的动态分辨率输入在三个基准上平均约 1320 个视觉 token。压到 600 token 后，平均分从 83.7 降到 82.1，约保留 98.1%；压到 400 token 后为 80.7。SparseVLM 因此不只适用于固定 576-token 的 LLaVA。

Video-LLaVA 从 2048 压到 194 token：

| 方法 | 平均原始准确率 | 各基准归一化准确率的均值 | GPT 评分均值 |
| --- | ---: | ---: | ---: |
| Video-LLaVA | 47.9 | 100.0% | 3.44 |
| FastV | 40.5 | 80.3% | 3.27 |
| SparseVLM | 47.0 | 95.0% | 3.40 |

论文所说 SparseVLM 比 FastV 高 14.7 个百分点，来自四个基准各自归一化后再平均的 $95.0-80.3$ ，不是用表中原始准确率均值直接相除。两种汇总方式回答的问题不同，复现时应保持口径一致。

## 消融分析

### 文本 Rater 是否必要

![文本 rater 消融](/images/posts/sparsevlm-text-guided-visual-token-sparsification/sparsevlm-icml2025-figure5-text-rater-ablation.png)

*图源：Zhang et al., [SparseVLM](https://arxiv.org/abs/2410.04417), Figure 5, ICML 2025；从作者 CC BY-NC-SA 4.0 arXiv 源码中的原始矢量图直接栅格化，完整保留两组坐标轴、标签与数值。原图用于论文解读。*

在固定保留 64 个视觉 token 时：

- TextVQA：全部 token 评分为 52.6，只用文本为 53.0，筛选 rater 为 53.4；
- POPE：全部 token 为 66.1，只用文本为 74.8，筛选 rater 为 77.5。

TextVQA 增益较小，POPE 对问题选择明显更敏感。它支持“不要让视觉无关词参与评分”，但只覆盖两个基准、一个 token 预算，不能证明所有任务都同等受益。

### Token 回收是否必要

Table 4 给出的绝对分数为：

| 基准 | Token | 无回收 | 有回收 | 绝对提升 |
| --- | ---: | ---: | ---: | ---: |
| GQA | 64 | 52.2 | 53.8 | +1.6 |
| GQA | 192 | 59.4 | 59.5 | +0.1 |
| POPE | 64 | 72.8 | 77.5 | +4.7 |
| POPE | 192 | 85.2 | 85.3 | +0.1 |

预算越紧，回收越有价值，这个趋势很清楚。不过论文正文称平均提升为 TextVQA 1.2% 和 POPE 7.2%，而 Table 4 的标题与数据是 GQA/POPE，四档绝对均值差分别为 0.8 和 2.6。由于正文与表格不一致，本文采用可直接核验的 Table 4 数值，不转述那组平均增益。

## 失败案例与局限

### 1. 公开实现没有完整落地论文的自适应规则

论文公式使用 $\operatorname{rank}(P)$ 和 $\lambda$ 逐层决定删除量；但官方 `v1.5` 分支在 ICML 版本公开时的 `score.py` 固定在第 2、6、15 层裁剪，并为 192/128/64 三种最终预算写死中间 token 数。例如 192 预算使用 `[300, 200, 110]`。

这不是小的命令行差异，而是复现语义差异：代码更接近“按预设层位和预算 top-k”，论文则描述“按输入注意力秩自适应”。想验证论文的动态性，需要先确认所用分支、配置和代码路径，不能仅运行 README 命令就声称复现了公式。

### 2. 摘要、结论和表格的效率数字不完全一致

PMLR 摘要给出 54% FLOPs、37% CUDA 时间、97% 准确率；Table 1 的 192-token 行更接近前两个数字但质量为 99.1%，128-token 行质量为 96.7%但效率降幅更大。结论又把 77.8% token 压缩、37% 延迟和 97% 组合在一起。

因此，最稳妥的报告方式是逐行引用 Table 1，并说明指标为单张 A100、指定模型和作者计时口径下的结果。

### 3. 注意力相关性不是因果重要性

高 attention 不等于 token 对最终答案有因果贡献。问题可能需要低权重的小字、罕见物体或多步关系；早期层评分也可能与后期推理需求不同。Figure 4 和 64-token 结果已经显示，激进裁剪仍会明显损失质量。

### 4. 动态操作会侵蚀理论收益

SVD 秩估计、第二次 FlashAttention、top-k、kNN 密度聚类和序列重排都增加 kernel 与内存访问。Table 1 中 SparseVLM 在相同 192-token 预算下比 FastV 慢 1.63 ms，就是这类开销的实证。

### 5. 评测范围仍有限

效率主要在单请求、单张 A100 80GB 上报告，没有并发吞吐、batch size、TTFT/TPOT、能耗或消费级 GPU 数据。视频只取每个基准前 1000 个样本，GPT 评分还引入外部评估器误差。部署到长视频、文档 OCR 或高并发服务仍需重新测量。

## 可复现资源

- [PMLR 正式页面与 18 页论文](https://proceedings.mlr.press/v267/zhang25s.html)
- [OpenReview 评审记录](https://openreview.net/forum?id=80faIPZ67S)
- [arXiv v4 全文、附录、源码与 CC BY-NC-SA 4.0 许可](https://arxiv.org/abs/2410.04417)
- [官方代码仓库](https://github.com/Gumpest/SparseVLMs)
- [ICML 版本代码快照](https://github.com/Gumpest/SparseVLMs/tree/b9619e61a6f840d7aa9817eadd68bb5e84ce7b95)
- [Video-LLaVA 稀疏化分支](https://github.com/Gumpest/SparseVLMs/tree/video)

官方 README 提供 192、128、96、64 token 的评测命令，并要求 Python 3.10、Transformers 4.37.0 和 FlashAttention 2.3.3。这里与论文附录记录的 Transformers 4.31.0 存在版本差异，复现报告应同时记录代码 commit、依赖锁定、GPU、输入长度和计时范围。

代码为 Apache 2.0；本文四张论文原图来自 arXiv 作者源码，遵循 CC BY-NC-SA 4.0，均保留原始内容并完整署名。模型权重、LLaVA 数据与底层模型仍受各自许可约束，不能由代码或论文许可替代。

## 个人判断

SparseVLM 最有价值的地方，不只是“视觉 token 可以少”，而是把 **问题文本变成压缩策略的一部分**。M3 提供固定的多档视觉粒度，SparseVLM 则尝试在模型内部按问题选择 patch；两者分别对应预算控制与内容选择，理论上可以组合。

但这篇论文也提醒我们，算法公式、公开代码和系统指标必须分开核验。论文描述的 rank-based 动态裁剪很漂亮，公开实现却主要依赖固定层位与预算表；FLOPs 显著下降，也没有按比例转化成延迟。对工程团队，下一步不是只复现准确率，而是做一张真实 Pareto 曲线：固定输入集与硬件，同时记录任务质量、TTFT、总延迟、峰值显存、吞吐和裁剪操作自身耗时。

我会把 SparseVLM 定位为：**一个有充分实验支持的文本感知视觉 Token 裁剪框架，也是一个公开实现与论文机制仍需对齐的推理加速基线。** 它证明了问题条件能改善激进裁剪下的质量，但还没有证明在所有模型、任务和服务负载上都能自动得到最优 token 预算。

## 参考资料

1. Zhang et al., [SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](https://proceedings.mlr.press/v267/zhang25s.html), ICML 2025.
2. Zhang et al., [arXiv:2410.04417v4](https://arxiv.org/abs/2410.04417), CC BY-NC-SA 4.0.
3. Gumpest, [SparseVLMs Official Implementation](https://github.com/Gumpest/SparseVLMs), Apache 2.0.
4. Chen et al., [An Image is Worth 1/2 Tokens After Layer 2](https://arxiv.org/abs/2403.06764), ECCV 2024.
5. Cai et al., [Matryoshka Multimodal Models](https://openreview.net/forum?id=Uhj5OxAz7I), ICLR 2025.
6. Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html), NeurIPS 2022.
