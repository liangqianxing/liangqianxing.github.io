---
title: "MM-FSS 精读：免费多模态如何提升少样本 3D 分割"
date: 2026-07-29 09:15:00
description: "拆解 ICLR 2025 Spotlight 的 MM-FSS 如何用文本语义、隐式 2D 对齐和测试时校准适配少样本 3D 点云分割，属于个性化（训练侧）方向。"
series: 三大会论文精读
seriesOrder: 13
categories:
  - AI
tags:
  - 多模态大模型
  - 个性化（训练侧）
  - 少样本学习
  - 3D 点云分割
  - 视觉语言模型
  - MM-FSS
  - ICLR
hidden: true
haloPublished: true
---

少样本 3D 分割通常只有点云：给模型几份带掩码的 support 点云，再要求它在 query 点云里找出一个没见过的类别。问题是，点云的几何信息很强，却缺少图像的纹理和语言的语义；而支持样本又少，模型很容易把训练类别的偏见带到新类别上。

《Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation》提出了一个很工程化的答案：类别名几乎不需要额外标注，2D 图像只在预训练时使用，之后的元学习和推理仍然只吃点云和类别名。作者把这套设置称为 **cost-free multimodal FS-PCS**，并用 MM-FSS（MultiModal Few-Shot SegNet）把三种信息接到同一个 support-query 分割流程里。

这篇文章属于本专题的 **个性化（训练侧）**：它不是泛泛讨论多模态，而是把预训练视觉语言特征迁移到少样本 3D 分割任务，通过两阶段训练、跨模态融合和测试时适配，让模型适应新类别和新场景。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation** |
| 作者 | Zhaochong An、Guolei Sun、Yun Liu、Runjia Li、Min Wu、Ming-Ming Cheng、Ender Konukoglu、Serge Belongie |
| 会议 | ICLR 2025 Spotlight |
| 方法 | MultiModal Few-Shot SegNet（MM-FSS） |
| 专题方向 | 个性化（训练侧）：VLM 特征迁移、少样本 3D 分割适配、测试时校准 |
| 正式全文 | [arXiv:2410.22489](https://arxiv.org/abs/2410.22489)，页面标注 Published at ICLR 2025 (Spotlight) |
| 官方代码 | [ZhaochongAn/Multimodality-3D-Few-Shot](https://github.com/ZhaochongAn/Multimodality-3D-Few-Shot) |
| 许可证 | arXiv 页面标注 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

**为什么选它**：上一轮是 Dynamic-LLaVA 的推理加速，今天轮换到训练侧。附件把这篇论文列入 ICLR 2025 进阶文献，它直接研究文本和图像信息如何适配 few-shot 3D segmentation；论文、附录、代码和原始方法图都能从作者 arXiv 页面核验，而且与仓库已有的 VideoLISA、Yo'LLaVA、InstructBLIP 等选题不同。

## 问题背景：少样本分割缺的不是一个分类器

设一个 episode 是 $N$-way $K$-shot。support 集合给出 $N$ 个目标类别的少量点云和点级掩码，query 集合只给点云，模型要输出目标类别和 background。传统 FS-PCS 的核心是从 support 点云采样 prototype，再把 prototype 与 query 点做匹配。

这条路径有三个瓶颈：

1. **几何表征单一**：点云是稀疏、无规则的表面采样，细节和纹理不如 RGB 图像。
2. **新类别监督太少**：support 只覆盖几个点云，prototype 很容易被视角、遮挡或实例形状牵着走。
3. **base bias**：预训练和元学习阶段大量使用 base 类别，测试 novel 类别时，模型会对训练中常见的类别产生错误激活。

作者的关键观察是：类别名在标注 support 时本来就知道；ScanNet 一类数据还天然带 RGB 图像和相机矩阵。与其在推理时强行要求所有模态同时在线，不如把 2D 视觉语言知识蒸馏进 3D 特征，再用文本 embedding 做轻量的语义提示。

![MM-FSS 与传统单模态 few-shot 3D 分割的设置对比](/images/posts/mm-fss-multimodal-few-shot-3d-segmentation/mm-fss-iclr2025-figure1-setup.png)

*图源：An et al., [Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation](https://arxiv.org/abs/2410.22489), Figure 1, ICLR 2025；取自作者 arXiv v4 源码中的原始 PDF 图，仅等比例缩放，箭头、图例、示例和数据均未修改。论文页面标注 CC BY-NC-SA 4.0；原图用于论文解读。*

Figure 1 的绿色箭头只出现在预训练阶段。2D 图像不是元学习和推理的必需输入，因此作者所谓的“免费”并不是没有训练成本，而是**不增加新类别的图像标注成本，也不把相机数据绑定到部署端**。

## 核心贡献

论文的贡献可以归纳为三点：

1. **提出多模态 FS-PCS 设置**：显式使用类别名，隐式利用 2D 图像，把 VLM 的视觉语言空间迁移到 3D 点云。
2. **提出 MM-FSS**：共享 3D backbone，分出 IF（Intermodal Feature）和 UF（Unimodal Feature）两条 head，再用 MCF 和 MSF 融合 support-query 相关性与文本语义。
3. **提出 TACC**：只在测试时根据 support episode 的语义可靠度计算自适应系数，校准 query 预测，缓解 base bias。

论文结论是：在 S3DIS 和 ScanNet 的 1/2-way、1/5-shot 设置中，MM-FSS 的平均 mIoU 全面超过 COSeg 等对照方法。我的判断是：最值得迁移的不是某个固定的 2D 编码器，而是“先把跨模态对齐固化，再在少样本 episode 中动态决定语义提示权重”的训练分层。

## 方法总览：两阶段训练，三种信息

![MM-FSS 的整体架构、MCF、MSF 与 TACC](/images/posts/mm-fss-multimodal-few-shot-3d-segmentation/mm-fss-iclr2025-figure2-architecture.png)

*图源：An et al., [Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation](https://arxiv.org/abs/2410.22489), Figure 2, ICLR 2025；取自作者 arXiv v4 源码中的原始 PDF 图，仅等比例缩放，模块、符号和测试时虚线区域均未修改。论文页面标注 CC BY-NC-SA 4.0；原图用于论文解读。*

### 第一步：让 3D 特征对齐 VLM 的 2D 空间

给定点云中的 3D 点 $p$ 和 RGB 图像中的像素 $u$，论文用相机内参和外参建立对应关系：

$$
\tilde u = M_{\mathrm{int}} M_{\mathrm{ext}} \tilde p.
$$

LSeg 的视觉编码器从图像提取 $F_{2D}$，MM-FSS 的 IF head 从点云提取 $F_{3D}$。对对应的点-像素对使用 cosine similarity loss，让 IF head 学到与 VLM 视觉 embedding 对齐的 3D 特征。这个阶段只训练 backbone 和 IF head。

训练结束后冻结 backbone 和 IF head。这样，IF 特征不仅携带 3D 几何，还处在 LSeg 的视觉语言空间里；即使后续换到没有 RGB 图像的 S3DIS，也能直接复用从 ScanNet 得到的 class-agnostic 权重。

### 第二步：在 episode 中适配新类别

对 support/query 点云 $X_{s/q}$，共享 backbone $\Phi$ 后接两条 head：

$$
F_s^i=H_{IF}(\Phi(X_s)),\quad F_s^u=H_{UF}(\Phi(X_s)),
$$

$$
F_q^i=H_{IF}(\Phi(X_q)),\quad F_q^u=H_{UF}(\Phi(X_q)).
$$

其中 $i$ 是 intermodal，$u$ 是只依赖点云的 unimodal。类别名和 background 经过 LSeg text encoder 得到：

$$
T=\{t_0,\ldots,t_N\}\in\mathbb R^{N_C\times D_t},\quad N_C=N+1.
$$

UF head 保留纯 3D 路径，IF head 提供可与文本相乘的语义路径。后面的 MCF、MSF 和 TACC 都围绕“support 到 query 的知识迁移”展开，而不是把额外模态简单拼在输入末尾。

## MCF：先融合两种 support-query 相关性

### Prototype 与 correlation

作者从 support 的 foreground/background 点中用 farthest point sampling 和 points-to-samples clustering 生成 $N_P$ 个 prototype：

$$
P^i_{fg},P^i_{bg}=F_{proto}(F_s^i,Y_s,L_s),\quad
P^u_{fg},P^u_{bg}=F_{proto}(F_s^u,Y_s,L_s).
$$

把各类别的 foreground/background prototype 拼接后，query 点与 prototype 做归一化点积：

$$
C^i=\frac{F_q^i(P^i_{proto})^\top}{\|F_q^i\|\|P^i_{proto}\|},\quad
C^u=\frac{F_q^u(P^u_{proto})^\top}{\|F_q^u\|\|P^u_{proto}\|}.
$$

$C^i$ 反映跨模态空间里的匹配，$C^u$ 反映纯点云空间里的匹配。两者不能直接相加，因为 prototype 维度和统计分布不同。

### Multimodal Correlation Fusion

MCF 用两个线性层把 prototype 维度投到统一的 $D$，再相加：

$$
C_0=F_{lin}(C^i)+F_{lin}(C^u),\quad
C_0\in\mathbb R^{N_Q\times N_C\times D}.
$$

这一步的直觉是：IF head 负责把 query 点放进有语义的 VLM 空间，UF head 保留纯几何细节；MCF 不要求二者谁更“正确”，而是让后面的 attention 学习如何组合两种关系。

## MSF：让文本语义参与每个点、每个类别的判断

因为 IF 特征已经和 LSeg 的 text encoder 对齐，query 点与类别名的相似度可以直接作为语义引导：

$$
G_q=F_q^iT^\top,\quad G_q\in\mathbb R^{N_Q\times N_C}.
$$

MSF 不是给所有点统一加一个文本 bias。对第 $k$ 个 MSF block，它先把 $G_q$ 扩展到 $D$ 维，与当前 correlation $C_k$ 拼接，再由 MLP 预测每个点-类别对的权重：

$$
W_q=F_{mlp}(F_{expand}(G_q)\oplus C_k),\quad
W_q\in\mathbb R^{N_Q\times N_C\times1}.
$$

随后把带权文本提示注入当前相关性，并用 linear attention 和 MLP 继续细化：

$$
C'_k=G_q\odot W_q+C_k,
$$

$$
C_{k+1}=F_{mlp}(F_{attention}(C'_k)).
$$

这套设计让“bookcase”这种支持/query 外观差异较大的类别更多依赖文本，而外观相近的“table”可以更多依赖几何匹配。最终 $C_K$ 经过 KPConv 和 MLP decoder，得到 query 的逐点分类预测 $P_q$，用交叉熵训练。

## TACC：测试时校准 base bias

少样本模型在 base 类上接受过大量监督，到了 novel 类 episode，UF head 可能会把背景或相似物体错误激活成 base 类。作者没有重新训练一个校准器，而是利用冻结的 IF head 和文本 embedding 计算 support 的语义预测质量。

先在 support 上得到 $G_s=F_s^iT^\top$，用 support 标签计算语义预测的 IoU，作为每个 episode 的自适应指标 $\gamma$：

$$
\gamma=\frac{\sum_i\mathbf 1[P_s(i)=1\land Y_s(i)=1]}
{\sum_i\mathbf 1[P_s(i)=1\lor Y_s(i)=1]},\quad
P_s(i)=\arg\max G_s[i,:].
$$

最终预测为：

$$
\hat P_q=\gamma G_q+P_q.
$$

如果文本语义在 support 上已经很可靠，$\gamma$ 较大，TACC 就多使用 $G_q$；如果 VLM 对这一类不熟，$\gamma$ 较小，模型主要保留元学习得到的 $P_q$。5-shot 时多个 support 样本的 $\gamma$ 默认取 max，避免某个低质量 shot 把整集语义提示压低。

## 训练与推理流程

### 训练

1. 用 ScanNet 的点云、RGB 图像和相机矩阵预训练 backbone 与 IF head 100 个 epoch。通过 3D-2D 对应和 cosine loss，把 3D IF 特征对齐到 LSeg（或 OpenSeg）的视觉空间。
2. 冻结 backbone 和 IF head，进入 40,000 个 episode 的元学习阶段；UF head、MCF、MSF 和 decoder 端到端更新。
3. 每个 episode 随机抽取目标类别和 support/query 点云，使用 1/2-way、1/5-shot 设置训练与评测。
4. 测试时根据 support 的 IoU 计算 $\gamma$，再把 TACC 应用于 query；2D 图像不再是输入依赖。

实现细节：backbone 使用 Stratified Transformer 的前两个 block，IF/UF head 对应第三阶段；特征在 $1/4$ 和 $1/16$ 分辨率提取后插值并拼接。每类抽取 $N_P=100$ 个 prototype，输入包含 XYZ 和 RGB，训练和推理使用 4 张 RTX 3090。预训练 AdamW 的学习率为 0.006，元学习降为 0.0001，weight decay 为 0.01；S3DIS 使用 2 个 MSF block，ScanNet 使用 4 个。

## 实验设置与主要结果

作者在 S3DIS 和 ScanNet 上按标准协议把场景切成 $1\,m\times1\,m$ block，使用 0.02 m voxel grid，并把每个 block 的点数上限设为 20,480。每个 1-way 设置评估 1,000 个 episode，每个 2-way 类别组合评估 100 个 episode，指标是 mean IoU（mIoU）。

### S3DIS

| 设置 | COSeg† | MM-FSS | 提升 |
| --- | ---: | ---: | ---: |
| 1-way 1-shot | 47.77 | **52.09** | +4.3 |
| 1-way 5-shot | 50.41 | **54.21** | +3.8 |
| 2-way 1-shot | 38.07 | **44.30** | +6.2 |
| 2-way 5-shot | 41.49 | **50.16** | +8.7 |

### ScanNet

| 设置 | COSeg† | MM-FSS | 提升 |
| --- | ---: | ---: | ---: |
| 1-way 1-shot | 42.01 | **44.73** | +2.7 |
| 1-way 5-shot | 46.61 | **50.07** | +3.5 |
| 2-way 1-shot | 29.03 | **39.21** | +10.2 |
| 2-way 5-shot | 35.51 | **44.09** | +8.6 |

这里的 COSeg† 是作者用与 MM-FSS 相同的 2D 对齐预训练 backbone 重新训练的版本。它没有明显超过原 COSeg，说明收益不能只归因于更好的初始化，MCF/MSF/TACC 的结构设计确实起了作用。

论文还报告：跨两个数据集，MM-FSS 在 1-way 和 2-way 设置的平均提升分别为 3.97% 和 9.25%。2-way 的增益更大，符合“support 更少、单一几何 prototype 更容易失效”的直觉。

![MM-FSS 与 COSeg 的少样本分割可视化对比](/images/posts/mm-fss-multimodal-few-shot-3d-segmentation/mm-fss-iclr2025-figure3-qualitative.png)

*图源：An et al., [Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation](https://arxiv.org/abs/2410.22489), Figure 3, ICLR 2025；取自作者 arXiv v4 源码中的原始 PDF 图，仅等比例缩放，support/query、ground truth、COSeg 和 MM-FSS 列均完整保留。论文页面标注 CC BY-NC-SA 4.0；原图用于论文解读。*

Figure 3 的重点不是“每个点都完美”，而是错误位置的结构差异：在 sofa 和 window 的 1-way 1-shot episode 中，MM-FSS 更少把相似区域整片涂成目标类。作者用红、绿圆圈标出两种方法的差异区域，读者可以直接对照 support mask 和 query ground truth。

## 消融：多模态的收益来自组合，而不是堆输入

论文在 ScanNet 上汇报 1-way 1/5-shot 的平均 mIoU：

| 组件或设置 | 1-shot | 5-shot |
| --- | ---: | ---: |
| 仅 UF head | 40.69 | 45.51 |
| 加 MCF | 41.45 | 46.38 |
| 加 MSF | 42.21 | 46.46 |
| MCF + MSF | 42.83 | 48.04 |
| IF + UF + TACC | **44.73** | **50.07** |

MCF 和 MSF 单独使用都有效，组合后继续提升；TACC 再把 1-shot/5-shot 推到 44.73/50.07。模块之间是互补关系：MCF 解决视觉相关性如何合并，MSF 解决文本语义如何参与，TACC 解决跨 base/novel 分布的测试偏差。

### 模态增量

3D-only baseline 的 1-shot/5-shot 为 40.69/45.51；加入隐式 image modality 后为 41.45/46.38；再加入 text modality 后达到 44.73/50.07。文本不是简单的类别 one-hot，它来自 VLM 的 embedding 空间，因此能为外观变化大的实例提供额外语义先验。

### MSF 深度与权重

不使用 TACC 时，MSF block 数从 3、4、5 变化，对应 1-shot/5-shot 为 43.33/45.97、42.83/48.04、44.69/48.36。更多 block 并不保证两项指标同步上涨，作者仍按数据集选择 2 或 4 个 block，说明深度需要和数据规模、任务难度一起调。

MSF 的动态权重也有可视化证据：当 support 和 query 的 bookcase 外观差异较大时，$W_q$ 会在文本提示更有帮助的区域提高权重；table 的外观更一致时，权重分布更均匀。

![MSF 中文本语义权重的可视化](/images/posts/mm-fss-multimodal-few-shot-3d-segmentation/mm-fss-iclr2025-figure5-weight-ablation.png)

*图源：An et al., [Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation](https://arxiv.org/abs/2410.22489), Figure 5, ICLR 2025；取自作者 arXiv v4 源码中的原始 PDF 图，仅等比例缩放，文本预测、无权重预测、有权重预测和 $W_q$ 热力图均完整保留。论文页面标注 CC BY-NC-SA 4.0；原图用于论文解读。*

### 复杂度与代价

MM-FSS 的计算量为 29.21 GFLOPs、参数量 10.25M，COSeg 为 27.76 GFLOPs、7.75M。也就是说，论文用约 5% 的 FLOPs 和约 32% 的参数增量换取更高的 few-shot mIoU；它不是推理加速论文，额外的融合层和预训练阶段都是真实成本。

## 失败案例、局限与证据边界

论文没有把所有错误预测单独列成 failure gallery，但附录和消融已经给出几个可核验的边界：

1. **文本语义可能不可靠**：TACC 只使用 $G_q$（系数 1:0）时，ScanNet 1-shot/5-shot 只有 35.10/37.32，说明 VLM 语义不能独立替代 support-query 关系。
2. **固定融合系数不如自适应系数**：1:1 和 1:0.5 比 baseline 有所改善，但都不如 $\gamma:1$；不同 episode 的文本可靠度确实不同。
3. **模态对齐不是免费训练**：2D 图像不进入推理，但需要 ScanNet 的相机矩阵和点-像素对应来预训练 IF head；没有这一步，文本 embedding 与 3D 点云不在同一个空间。
4. **跨数据集迁移有假设**：S3DIS 没有 2D 图像，作者通过复用 ScanNet 预训练权重来启动元学习。这个结果说明权重可迁移，但不能证明所有室内、户外或激光雷达场景都适用。
5. **部署复杂度略增**：MM-FSS 比 COSeg 多 1.45 GFLOPs 和 2.50M 参数，并且测试时要额外计算 support 的语义 IoU 来获得 $\gamma$。
6. **作者明确的限制**：模型可能学习到 S3DIS/ScanNet 的数据集偏置，实际感知系统部署前需要更多场景验证；训练和部署 GPU 也带来碳排放。

因此，作者的“显著提升”应理解为论文定义的两个室内 FS-PCS benchmark 上的提升，而不是对所有 3D 分割任务的普遍保证。

## 可复现资源

- [arXiv 全文与 HTML](https://arxiv.org/abs/2410.22489)
- [作者官方代码](https://github.com/ZhaochongAn/Multimodality-3D-Few-Shot)
- [LSeg 视觉语言模型](https://github.com/isl-org/lang-seg)
- [OpenSeg 预训练模型](https://github.com/isl-org/OpenSeg)
- [S3DIS 数据集](http://buildingparser.stanford.edu/dataset.html)
- [ScanNet 数据集](http://www.scan-net.org/)

复现顺序应保持论文的两阶段结构：先用 ScanNet 的 2D-3D 对齐训练 IF head，再冻结 backbone/IF head 做 episode 元学习，最后在测试时开启 TACC。只运行第二阶段而跳过跨模态预训练，会失去 text embedding 能够指导 3D 点的前提。

## 个人判断

我认为 MM-FSS 最有价值的设计是把“多模态”拆成了不同时间尺度的三种用途：

- **2D 图像**负责训练期的表征迁移，帮助 3D 点云进入 VLM 空间；
- **文本类别名**负责元学习和推理期的语义补充，不需要再采集新图像；
- **support mask**负责 episode 内的实例级几何匹配。

这比把 RGB、点云和文本简单拼接更适合少样本场景，因为每种模态都被放在自己最有优势的位置。另一方面，MM-FSS 也提醒工程实现者：所谓“免费模态”通常只是把成本从在线输入转移到离线预训练；相机标定、点-像素投影、VLM 特征抽取和冻结权重都需要维护。

如果把这篇工作迁移到更大的多模态模型，我会优先保留两个原则：一是为新任务保留一条冻结的通用语义支路，二是让适配器根据 support 质量动态控制语义注入强度，而不是固定加一个 prompt。论文的 TACC 很轻，却直接对应了个性化系统里常见的“这次 support 是否可信”问题。

## 参考资料

1. An et al., [Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation](https://arxiv.org/abs/2410.22489), ICLR 2025 Spotlight. 原文、附录、图表和 CC BY-NC-SA 4.0 许可均来自该页面。
2. [MM-FSS 官方代码仓库](https://github.com/ZhaochongAn/Multimodality-3D-Few-Shot).
3. Lai et al., [Stratified Transformer for 3D Point Cloud Segmentation](https://arxiv.org/abs/2203.14547).
4. Li et al., [Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546)（LSeg）.
5. Ghiasi et al., [OpenSeg](https://arxiv.org/abs/2210.13239).

本文中的论文原图均用于论文解读，未修改实验数据或图中标签；版权和再使用范围以作者 arXiv 页面标注的 CC BY-NC-SA 4.0 为准。
