---
title: "新芽专题介绍：多模态大模型的个性化与推理加速"
date: 2026-08-10 18:30:00
description: 面向新芽学子的专题导览：梳理多模态大模型在个性化（训练侧）与推理加速（推理侧）两条主线上的核心思想、代表方法、主要挑战，并给出带链接的 66 篇文献地图与分阶段学习路径。
categories:
  - AI
tags:
  - 多模态大模型
  - 个性化（训练侧）
  - 推理加速
  - 参数高效微调
  - 模型量化
  - 视觉Token剪枝
  - 新芽专题
hidden: true
haloPublished: true
---

随着人工智能从单模态迈向多模态时代，多模态大模型（Multimodal Large Models, MLMs）成为智能感知与理解的核心引擎：它们同时处理图像、文本、语音、视频等多源信息，在跨模态检索、智能问答、视觉理解、机器人感知等任务上展现出类人般的推理与生成能力。但要把这样的模型真正用起来，绕不开两个问题——**如何让它适应特定任务或个性化需求**，以及**如何在有限算力下高效推理**。

![多模态 AI 核心体概念插画：图像、文本、语音、视频四路数据流汇入统一智能核心（本文原创，AI 辅助生成）](/images/posts/xinya-mllm-personalization-inference-acceleration/aigc-cover-multimodal-core.jpg)

本专题就围绕这两个问题展开。本文是专题的导览篇：先讲清研究背景与两条主线的核心思想，再梳理当前的主要挑战，最后给出一份带链接的文献地图和分阶段学习路径，供新芽学子入门与汇报参考。

![专题全景：多模态大模型的个性化与推理加速（本文原创）](/images/posts/xinya-mllm-personalization-inference-acceleration/topic-overview.svg)

两条主线分别切在模型生命周期的两端：**个性化**是训练侧问题，关注“如何让通用模型适配具体场景”；**推理加速**是推理侧问题，关注“如何降低部署时的计算与存储开销”。

## 多模态大模型是怎么工作的

要理解两条主线切在哪里，先看一个典型的多模态大模型推理流水线：图像或视频经视觉编码器（通常是 ViT 或 SigLIP）变成一串视觉 Token，再由跨模态投影层（MLP 或 Q-Former）映射到语言模型的嵌入空间，与文本指令拼接后一起送入大语言模型生成回答。

![多模态大模型推理流水线与两大优化切入点（本文原创示意）](/images/posts/xinya-mllm-personalization-inference-acceleration/mllm-inference-pipeline.svg)

这条流水线上有两个事实，决定了本专题的技术版图：

- **视觉 Token 的数量决定注意力开销**。一张图常被切成数百甚至上千个 Token，视频更是成倍增长，而注意力计算随序列长度近似平方增长——这是推理加速的主要战场。
- **主干权重既是能力的来源，也是适配和压缩的对象**。个性化方法在视觉编码器、投影层、语言模型上插入或附加少量可训练参数；量化方法则直接压缩权重本身的比特宽度。

这条流水线不是凭空出现的，它建立在几个里程碑工作之上。CLIP（[ICML 2021](https://arxiv.org/abs/2103.00020)）用 4 亿图文对的对比学习证明：自然语言监督可以训练出具备零样本迁移能力的视觉表征，是多模态对齐的基石。

![CLIP 的对比预训练与零样本分类流程（来源：openai/CLIP GitHub 仓库，MIT License）](/images/posts/xinya-mllm-personalization-inference-acceleration/clip-architecture.png)

> 图片来源：[openai/CLIP](https://github.com/openai/CLIP) 仓库根目录 `CLIP.png`，MIT License，用于说明对比图文预训练的基本流程。

LLaVA（[NeurIPS 2023](https://arxiv.org/abs/2304.08485)）则把这条路推进到指令跟随：用 GPT-4 生成的视觉指令数据微调，让模型获得对话式的视觉理解能力，成为后来大多数开源多模态大模型的范式。

![LLaVA 与同期模型在多模态指令跟随上的行为对比（来源：haotian-liu/LLaVA GitHub 仓库，Apache-2.0）](/images/posts/xinya-mllm-personalization-inference-acceleration/llava-example-comparison.png)

> 图片来源：[haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) 仓库 `images/llava_example_cmp.png`，Apache-2.0，用于展示视觉指令微调模型的多模态对话能力。

## 主线一：个性化（训练侧）

个性化要回答的问题是：**在不动或少动预训练主干的前提下，如何让模型适配具体任务、具体领域、具体用户？**文献里有三条相互交织的技术线。

![个性化微调概念插画：为休眠的巨型机械体嵌入一枚小巧的可替换模块（本文原创，AI 辅助生成）](/images/posts/xinya-mllm-personalization-inference-acceleration/aigc-personalization-tuning.jpg)

### 参数高效微调：只训练 1% 的参数

全量微调一个几十亿参数的多模态模型，显存和数据成本都很高。参数高效微调（PEFT）的思路是冻结预训练权重，把任务差异压缩进极少量可训练参数。方法大致分三派：

![参数高效微调方法谱系（本文原创）](/images/posts/xinya-mllm-personalization-inference-acceleration/peft-method-map.svg)

- **加性方法**：往网络里插入小模块。Adapter 在层间插入瓶颈 MLP；Prompt Tuning / Visual Prompt Tuning（[ECCV 2022](https://arxiv.org/abs/2203.12119)）只学习输入侧的软提示向量；Yo'LLaVA 一类的个性化工作则用少量可学习 Token 记住用户指定的视觉主体。
- **重参数化方法**：以 LoRA（[ICLR 2022](https://arxiv.org/abs/2106.09685)）为代表，用低秩乘积改写权重增量。后续工作沿两个方向改进：DoRA（[ICML 2024](https://arxiv.org/abs/2402.09353)）把权重分解为方向与幅度分别微调；PiSSA、SVFT（均为 NeurIPS 2024）用奇异值分解来初始化适配参数，加速收敛。
- **量化协同方法**：QLoRA（[NeurIPS 2023](https://arxiv.org/abs/2305.14314)）先把主干量化到 4-bit NF4，再在上面挂 LoRA 微调，把 65B 模型的微调压进单张 48G 显卡；LoftQ、LQ-LoRA（均为 ICLR 2024）进一步在量化时同步初始化低秩分支。这一派正是个性化与推理加速的交汇点。

![QLoRA 的方法总览：4-bit NF4 量化主干 + 双重量化 + 分页优化器，支撑单卡微调 65B 模型](/images/posts/xinya-mllm-personalization-inference-acceleration/qlora-nf4-finetuning-overview.png)

*图源：Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)，NeurIPS 2023，方法总览图；取自作者 CC BY 4.0 arXiv 源码，原图用于论文解读。*

LoRA 值得单独看一张图，因为它是整条线的基石：

![LoRA 低秩适配机制（本文原创重绘，机制出自 arXiv:2106.09685）](/images/posts/xinya-mllm-personalization-inference-acceleration/lora-mechanism.svg)

直觉很朴素：下游任务需要的权重变化往往是低秩的，所以与其微调整个矩阵，不如只学两个小矩阵的乘积。推理时 BA 还能合并回原权重，不增加任何延迟——这也是 LoRA 在工业界流行的原因之一。

### 知识迁移：把大模型能力搬到分割、检测与推理感知

第二条线利用大模型已有的知识来提升具体视觉任务。Segment Anything（[ICCV 2023](https://arxiv.org/abs/2304.02643)）用 11 亿掩码训练出可提示分割的通用模型，其“图像编码器 + 提示编码器 + 掩码解码器”的三段式设计，成为后来大量工作的基础设施。

![SAM 的模型结构：图像编码器、提示编码器与掩码解码器（来源：facebookresearch/segment-anything GitHub 仓库，Apache-2.0）](/images/posts/xinya-mllm-personalization-inference-acceleration/sam-model-diagram.png)

> 图片来源：[facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) 仓库 `assets/model_diagram.png`，Apache-2.0，用于说明可提示分割模型的结构。

在此之上，LISA（[CVPR 2024](https://arxiv.org/abs/2308.00692)）让大语言模型输出特殊的 `[SEG]` Token 来驱动 SAM 解码器，实现需要常识推理的分割（“分割出最可能遮雨的物体”）；VideoLISA（[NeurIPS 2024](https://arxiv.org/abs/2409.19603)）和 GLUS（[CVPR 2025](https://arxiv.org/abs/2504.07962)）把它扩展到视频；DenseCLIP、RegionCLIP、GLIP、Grounding DINO 等工作则把 CLIP 式对齐迁移到稠密预测与开放词汇检测。

![LISA 的流程框架：多模态大模型生成 [SEG] Token，其末层嵌入经解码器变成分割掩码，训练时使用 LoRA 高效微调](/images/posts/xinya-mllm-personalization-inference-acceleration/lisa-reasoning-segmentation-framework.png)

*图源：Lai et al., [LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/abs/2308.00692)，CVPR 2024，流程框架图；取自作者 CC BY-NC-SA 4.0 arXiv 源码，原图用于论文解读。*

### 零样本与少样本学习

第三条线关注标注稀缺的场景：CLIP 的零样本分类、Flamingo（[NeurIPS 2022](https://arxiv.org/abs/2204.14198)）的图文交错少样本上下文学习，以及 ICLR/CVPR 2025 的两篇多模态少样本 3D 点云分割工作，都在探索“缺乏大规模标注时依旧保持竞争力”的边界。

## 主线二：推理加速（推理侧）

推理加速要回答的问题是：**模型已经训好了，如何让它在有限算力下跑得更快、更省？**三条经典路线之外，多模态场景还长出了第四条特有路线。

![推理加速概念插画：光之猎鹰挣脱沉重方块，化作光箭穿越数据隧道（本文原创，AI 辅助生成）](/images/posts/xinya-mllm-personalization-inference-acceleration/aigc-inference-acceleration-falcon.jpg)

### 量化：用更少比特存同一个模型

量化把 FP16/FP32 权重压缩为 INT8、INT4 甚至 1-bit 表示，显存占用和能耗同步下降。从 BinaryConnect（[NeurIPS 2015](https://arxiv.org/abs/1511.00363)）、XNOR-Net（[ECCV 2016](https://arxiv.org/abs/1603.05279)）的 1-bit 探索，到 AdaRound（[ICML 2020](https://arxiv.org/abs/2004.10568)）逐层学习取整方向，再到 GPTQ（[ICLR 2023](https://arxiv.org/abs/2210.17323)）用近似二阶信息逐列补偿误差，训练后量化已经能把大模型压到 4-bit 而精度损失很小。

![量化的基本概念与方法演进（本文原创示意）](/images/posts/xinya-mllm-personalization-inference-acceleration/quantization-concept.svg)

### 剪枝与视觉 Token 压缩：多模态特有的杠杆

剪枝去除冗余结构提高稀疏性。在多模态模型里，最肥沃的剪枝对象是**视觉 Token 序列**：图像 Patch 里大量是背景和冗余区域，并不需要全部进入语言模型。

![视觉 Token 剪枝的基本流程与代表工作（本文原创示意）](/images/posts/xinya-mllm-personalization-inference-acceleration/token-pruning-concept.svg)

这条线从纯视觉模型起步——DynamicViT（[NeurIPS 2021](https://arxiv.org/abs/2106.02034)）用预测模块逐层丢弃 Token，SPViT（[ECCV 2022](https://arxiv.org/abs/2112.13890)）做延迟感知的软剪枝，TokenLearner（[ICLR 2022](https://arxiv.org/abs/2202.07800)）学习 Token 重组——随后进入多模态大模型：FastV（[ECCV 2024](https://arxiv.org/abs/2403.06764)）发现第二层之后可以即插即用地丢掉一半视觉 Token，DivPrune（[CVPR 2025](https://arxiv.org/abs/2503.02175)）按多样性选 Token，LLaVA-PruMerge（[ICCV 2025](https://arxiv.org/abs/2403.15388)）把被剪 Token 合并进保留 Token，DyCoke（[CVPR 2025](https://arxiv.org/abs/2411.15024)）进一步处理视频 Token 的时序冗余。

![DyCoke 方法总览：预填充阶段做视频 Token 时序合并（左），解码阶段对 KV Cache 做动态剪枝（右），全程免训练](/images/posts/xinya-mllm-personalization-inference-acceleration/dycoke-video-token-compression-method.png)

*图源：Tao et al., [DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](https://arxiv.org/abs/2411.15024)，CVPR 2025，方法总览图；取自作者 CC BY 4.0 arXiv 源码，原图用于论文解读。*

### 蒸馏：大模型指导小模型

蒸馏用冻结的教师大模型产生的软标签、中间特征或输出分布来训练轻量学生模型，让学生以小得多的体量逼近教师能力。它与剪枝、量化互补且可叠加——经典的 Deep Compression（[ICLR 2016](https://arxiv.org/abs/1510.00149)）就是“剪枝 + 量化 + 哈夫曼编码”的组合拳。

![知识蒸馏的基本框架（本文原创示意）](/images/posts/xinya-mllm-personalization-inference-acceleration/distillation-concept.svg)

## 当前的主要挑战

这个领域远未收敛，至少有三个开放性挑战：

1. **任务与场景多样性**：不同领域对多模态交互的需求差异很大——医学影像要细粒度，视频理解要长时序，机器人要低延迟。如何设计通用且高效的个性化方案，仍是难题。
2. **大模型计算开销巨大**：数百亿参数的多模态模型在部署中常受限于算力和能耗，端侧和边缘场景尤其如此。
3. **个性化与高效性的平衡**：压缩可能损伤模型的细粒度感知能力，个性化模块也可能引入额外延迟。如何在保持性能的同时兼顾灵活性与推理效率，需要不断权衡和创新。

## 文献地图

以下 66 篇文献按“入门 → 进阶 → 领域相关”组织，链接均指向可公开访问的 arXiv 页面并逐一核验过。一句话定位只是阅读入口，不能代替读原文。

### 入门文献：骨干网络、多模态底座与基础方法

| 论文 | 发表 | 一句话定位 |
| --- | --- | --- |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | CVPR 2016 | 残差连接让百层网络可训练，现代骨干网络的起点 |
| [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | CVPR 2017 | 特征复用的极致：每层都与所有前层相连 |
| [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) | ECCV 2018 | 通道 + 空间双注意力，轻量即插即用 |
| [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | ICLR 2021 | 把图像切成 Patch 直接送进 Transformer |
| [P2T: Pyramid Pooling Transformer for Scene Understanding](https://arxiv.org/abs/2106.12011) | TPAMI 2022 | 金字塔池化注意力服务场景理解 |
| [Vision Transformers with Hierarchical Attention](https://arxiv.org/abs/2106.03180) | MIR 2024 | 层次化注意力改进 ViT 的多尺度能力 |
| [Exploiting Temporal State Space Sharing for Video Semantic Segmentation](https://arxiv.org/abs/2503.20824) | CVPR 2025 | 时序状态空间共享加速视频分割 |
| [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) | ICML 2021 | 图文对比学习奠定多模态对齐基石 |
| [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086) | ICML 2022 | 理解与生成统一的视觉语言预训练 |
| [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) | NeurIPS 2023 | 视觉指令微调范式的开山之作 |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | NeurIPS 2020 | 扩散模型奠基，生成侧的重要补充 |
| [Low-Rank Adaptation of Large Language Models (LoRA)](https://arxiv.org/abs/2106.09685) | ICLR 2022 | 低秩适配，个性化的基石方法 |
| [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119) | ECCV 2022 | 只学提示向量的视觉模型微调 |
| [DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification](https://arxiv.org/abs/2106.02034) | NeurIPS 2021 | 动态 Token 稀疏化的早期代表 |
| [Quantized Neural Networks](https://arxiv.org/abs/1609.07061) | JMLR 2018 | 训练感知量化的系统阐述 |

### 进阶文献：SAM 系、LoRA 变体、推理分割与 Token 压缩

| 论文 | 发表 | 一句话定位 |
| --- | --- | --- |
| [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) | ICCV 2023 | 可提示分割的通用基础模型 |
| [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) | ICLR 2025 | 把 SAM 扩展到视频统一分割 |
| [RemoteSAM: Towards Segment Anything for Earth Observation](https://arxiv.org/abs/2505.18022) | ACM MM 2025 | SAM 能力向遥感影像的个性化迁移 |
| [PiSSA: Principal Singular Values and Singular Vectors Adaptation](https://arxiv.org/abs/2404.02948) | NeurIPS 2024 | 用主奇异成分初始化 LoRA，收敛更快 |
| [SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors](https://arxiv.org/abs/2405.19597) | NeurIPS 2024 | 奇异向量方向的参数高效更新 |
| [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) | ICML 2024 | 方向与幅度分解，缩小与全量微调的差距 |
| [LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/abs/2308.00692) | CVPR 2024 | 用语言模型的常识推理驱动分割 |
| [One Token to Seg Them All (VideoLISA)](https://arxiv.org/abs/2409.19603) | NeurIPS 2024 | 一个 TRK Token 完成视频推理分割 |
| [GLUS: Global-Local Reasoning Unified into A Single LLM](https://arxiv.org/abs/2504.07962) | CVPR 2025 | 全局-局部推理统一的视频分割 |
| [Multimodality Helps Few-Shot 3D Point Cloud Semantic Segmentation](https://arxiv.org/abs/2410.22489) | ICLR 2025 | 多模态信息助力少样本 3D 分割 |
| [Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model](https://arxiv.org/abs/2503.16282) | CVPR 2025 | 视觉语言模型用于广义少样本 3D 分割 |
| [DivPrune: Diversity-based Visual Token Pruning](https://arxiv.org/abs/2503.02175) | CVPR 2025 | 按多样性保留视觉 Token |
| [LLaVA-PruMerge: Adaptive Token Reduction](https://arxiv.org/abs/2403.15388) | ICCV 2025 | 剪除与合并结合的自适应 Token 削减 |
| [Up or Down? Adaptive Rounding for Post-Training Quantization (AdaRound)](https://arxiv.org/abs/2004.10568) | ICML 2020 | 逐层学习取整方向的训练后量化 |
| [GPTQ/OPTQ: Accurate Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) | ICLR 2023 | 近似二阶信息补偿的 4-bit 量化 |
| [DyCoke: Dynamic Compression of Tokens for Fast Video LLMs](https://arxiv.org/abs/2411.15024) | CVPR 2025 | 视频大模型 Token 的动态压缩 |

### 领域相关文献：多模态底座、开放词汇视觉与压缩加速脉络

| 论文 | 发表 | 一句话定位 |
| --- | --- | --- |
| [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | NeurIPS 2022 | 图文交错输入的少样本上下文学习 |
| [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs](https://arxiv.org/abs/2301.12597) | ICML 2023 | 冻结两端，只学 Q-Former 跨模态接口 |
| [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500) | NeurIPS 2023 | 指令感知的视觉特征提取 |
| [MiniGPT-4: Enhancing Vision-Language Understanding with Advanced LLMs](https://arxiv.org/abs/2304.10592) | ICLR 2024 | 单投影层接通视觉编码器与 Vicuna |
| [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512) | ICLR 2023 | 按重要性自适应分配秩预算 |
| [Bayesian Low-rank Adaptation for Large Language Models](https://arxiv.org/abs/2308.13111) | ICLR 2024 | 给 LoRA 加不确定性建模 |
| [SVFit: Parameter-Efficient Fine-Tuning Using Singular Values](https://arxiv.org/abs/2409.05926) | arXiv 2024 | 只训奇异值的极简适配 |
| [RaSA: Rank-Sharing Low-Rank Adaptation](https://arxiv.org/abs/2503.12576) | ICLR 2025 | 跨层共享秩进一步省参数 |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | NeurIPS 2023 | 4-bit 主干 + LoRA，单卡微调 65B |
| [LQ-LoRA: Low-rank plus Quantized Matrix Decomposition](https://arxiv.org/abs/2311.12023) | ICLR 2024 | 低秩 + 量化联合分解初始化 |
| [LoftQ: LoRA-Fine-Tuning-aware Quantization](https://arxiv.org/abs/2310.08659) | ICLR 2024 | 为后续 LoRA 微调优化的量化初始化 |
| [Dynamic Low-Rank Sparse Adaptation](https://arxiv.org/abs/2502.14816) | ICLR 2025 | 低秩与稀疏动态结合的适配 |
| [DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting](https://arxiv.org/abs/2112.01518) | CVPR 2022 | CLIP 知识迁移到稠密预测 |
| [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://arxiv.org/abs/2202.11094) | CVPR 2022 | 纯文本监督下涌现分割能力 |
| [RegionCLIP: Region-based Language-Image Pretraining](https://arxiv.org/abs/2112.09106) | CVPR 2022 | 区域级图文对齐支撑开放词汇检测 |
| [Grounded Language-Image Pre-training (GLIP)](https://arxiv.org/abs/2112.03857) | CVPR 2022 | 定位与理解统一预训练 |
| [VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks](https://arxiv.org/abs/2305.11175) | NeurIPS 2023 | 用语言指令定制检测与分割输出 |
| [InternVL: Scaling up Vision Foundation Models](https://arxiv.org/abs/2312.14238) | CVPR 2024 | 视觉基础模型规模化并对齐 LLM |
| [Segment Everything Everywhere All at Once (SEEM)](https://arxiv.org/abs/2304.06718) | NeurIPS 2023 | 统一多种提示形式的全能分割 |
| [Grounding DINO: Marrying DINO with Grounded Pre-Training](https://arxiv.org/abs/2303.05499) | ECCV 2024 | 开放集检测的强基线 |
| [Learning To Prompt for Open-Vocabulary Object Detection](https://arxiv.org/abs/2203.14940) | CVPR 2022 | 用提示学习做开放词汇检测 |
| [Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models](https://arxiv.org/abs/2311.17095) | CVPR 2024 | 免训练的开放词汇分割 |
| [BinaryConnect: Training Deep Neural Networks with Binary Weights](https://arxiv.org/abs/1511.00363) | NeurIPS 2015 | 1-bit 权重训练的开端 |
| [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279) | ECCV 2016 | 二值卷积网络的 ImageNet 实践 |
| [Deep Compression: Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) | ICLR 2016 | 剪枝 + 量化 + 编码的组合压缩 |
| [Post Training 4-bit Quantization of Convolutional Networks](https://arxiv.org/abs/1810.05723) | NeurIPS 2019 | 4-bit 训练后量化的早期系统方案 |
| [An Image is Worth 1/2 Tokens After Layer 2 (FastV)](https://arxiv.org/abs/2403.06764) | ECCV 2024 | 即插即用的视觉 Token 减半 |
| [VoCo-LLaMA: Towards Vision Compression with Large Language Models](https://arxiv.org/abs/2406.12275) | CVPR 2025 | 用 LLM 自身注意力压缩视觉信息 |
| [FlashSloth: Lightning Multimodal Large Language Models via Embedded Visual Compression](https://arxiv.org/abs/2412.04317) | CVPR 2025 | 嵌入式视觉压缩的轻量多模态模型 |
| [Hybrid-Level Instruction Injection for Video Token Compression](https://arxiv.org/abs/2503.16036) | CVPR 2025 | 指令注入指导视频 Token 压缩 |
| [Boosting Multimodal Large Language Models with Visual Tokens Withdrawal](https://arxiv.org/abs/2405.05803) | AAAI 2025 | 生成阶段视觉 Token 撤离加速 |
| [SPViT: Enabling Faster Vision Transformers via Soft Token Pruning](https://arxiv.org/abs/2112.13890) | ECCV 2022 | 延迟感知的软 Token 剪枝 |
| [Not All Patches are What You Need (TokenLearner)](https://arxiv.org/abs/2202.07800) | ICLR 2022 | 学习式 Token 重组 |
| [AdaViT: Adaptive Vision Transformers for Efficient Image Recognition](https://arxiv.org/abs/2111.15668) | CVPR 2022 | 按输入难度自适应调整计算 |
| [Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation](https://arxiv.org/abs/2308.01045) | ICCV 2023 | 稠密预测任务的动态 Token 剪枝 |

## 建议的学习路径

多模态大模型的个性化与推理加速，是推动 AI 从实验室走向产业落地的关键一环。对入门同学，建议按四个阶段推进，每个阶段都“读论文 + 跑代码”并行：

![分阶段学习路径（本文原创）](/images/posts/xinya-mllm-personalization-inference-acceleration/learning-roadmap.svg)

基础学习资料（均可免费使用）：

- [《动手学深度学习》](https://d2l.ai/)：适合中文初学者的深度学习教材，理论、代码、实践一体；
- [《Deep Learning》](https://www.deeplearningbook.org/)：深度学习入门经典教材；
- [PyTorch 官方教程](https://pytorch.org/tutorials/)：掌握主流框架的最佳入口；
- [Google Colab](https://colab.research.google.com/)：免费云平台，不用安装软件就能跑 PyTorch 代码；
- [Kaggle](https://www.kaggle.com/)：海量数据集、竞赛与免费计算资源。

落到具体任务上，建议每位同学选一条线深入：个性化线可以从复现 LoRA 微调一个开源多模态模型开始，再读 DoRA、PiSSA 等变体，最后进入 LISA、VideoLISA 这类任务适配工作；加速线可以从 DynamicViT 的 Token 剪枝思想入手，再读 GPTQ 的量化补偿，最后跟进 DivPrune、DyCoke 等面向多模态大模型的最新压缩方法。期待大家在专题汇报中提出自己的见解，探索更多可能性。

## 说明

- 本文中标注“本文原创”的示意图均为笔者根据公开论文机制重绘，仅用于学习交流；三张概念插画（封面、个性化、推理加速）为本文原创的 AI 辅助生成图，仅用于栏目视觉呈现。
- CLIP、LLaVA、SAM 三张配图取自官方开源仓库，许可分别为 MIT 与 Apache-2.0；QLoRA、LISA、DyCoke 三张论文原图取自作者 arXiv 源码，许可分别为 CC BY 4.0 与 CC BY-NC-SA 4.0，出处均已在图注标明。
- 文献的一句话定位为笔者概述，具体方法与结论请以原文为准。
