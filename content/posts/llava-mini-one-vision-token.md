---
title: "LLaVA-Mini 精读：一个视觉 Token 如何加速图像与视频推理"
date: 2026-07-29 20:22:00
description: "拆解 ICLR 2025 的 LLaVA-Mini 如何用模态预融合与查询压缩，把送入主语言模型的视觉 Token 从 576 个降到 1 个，属于推理加速（推理侧）方向。"
series: 三大会论文精读
seriesOrder: 14
categories:
  - AI
tags:
  - 多模态大模型
  - 推理加速（推理侧）
  - 视觉 Token 压缩
  - 多模态预融合
  - 长视频理解
  - LLaVA-Mini
  - ICLR
hidden: true
haloPublished: true
draft: false
---

多模态大模型通常把一张图切成数百个 patch，再把对应的视觉 Token 和文本一起送进大语言模型。以 LLaVA-v1.5 的 CLIP ViT-L/336px 为例，一张图会产生 $24\times24=576$ 个视觉 Token；到了高分辨率图像和长视频，主语言模型的上下文、注意力计算和 KV Cache 都会迅速膨胀。

《LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token》提出了一个很激进的答案：**先在主语言模型外完成一次模态预融合，再把送进主语言模型的视觉 Token 压缩到 1 个**。论文报告，在标准分辨率图像上，其 FLOPs 从 LLaVA-v1.5 的 8.55T 降到 1.96T；A100 上延迟从 113.04 ms 降到 38.64 ms，同时 11 个图像基准的平均分由 56.3 提升到 57.9。

这篇论文属于本专题的 **推理加速（推理侧）**。它不是泛泛缩小模型，而是直接围绕多模态推理中的视觉 Token 数量，给出架构、消融、跨硬件延迟和长视频显存实验。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token** |
| 作者 | Shaolei Zhang、Qingkai Fang、Zhe Yang、Yang Feng |
| 会议 | ICLR 2025 |
| 方法 | LLaVA-Mini |
| 专题方向 | 推理加速（推理侧）：视觉 Token 压缩、低延迟图像推理、长视频显存优化 |
| 正式全文 | [arXiv:2501.03895 v2](https://arxiv.org/abs/2501.03895)，页面标注 Accepted to ICLR 2025，PDF 首页标注 Published as a conference paper at ICLR 2025 |
| 官方代码 | [ictnlp/LLaVA-Mini](https://github.com/ictnlp/LLaVA-Mini) |
| 模型权重 | [ICTNLP/llava-mini-llama-3.1-8b](https://huggingface.co/ICTNLP/llava-mini-llama-3.1-8b) |
| 代码与项目素材许可 | [Apache-2.0](https://github.com/ictnlp/LLaVA-Mini/blob/main/LICENSE) |

**为什么选它**：最新一篇专题文章 MM-FSS 属于训练侧个性化，本次轮换到推理侧。LLaVA-Mini 直接量化了视觉 Token 压缩对 FLOPs、延迟和显存的影响，正文、附录、代码、模型与官方原图都公开可核验；仓库既有索引也没有出现过该标题、论文 URL 或文章主题。

## 问题背景：多模态推理贵在哪里

普通 LLaVA 把文本指令嵌入 $H^q$ 与视觉编码器输出 $H^v$ 拼接后送进语言模型：

$$
\left\langle H^q_1,\ldots,H^q_k,H^v_1,\ldots,H^v_{l_v},H^q_{k+1},\ldots,H^q_{l_q}\right\rangle.
$$

当输入长度为 $L=l_q+l_v$ 时，自注意力的主要计算随 $L$ 增长，预填充阶段还要为这些 Token 生成并保存 KV。标准图像已经有 576 个视觉 Token；视频若按 1 fps 采样 $M$ 帧，就会带来 $576M$ 个视觉 Token。

直接合并、裁剪或平均池化虽然能缩短上下文，却容易删掉文字、小物体和空间关系。作者先问了一个更基础的问题：**视觉 Token 在主语言模型的每一层都同样重要吗？**

论文在 LLaVA-v1.5-7B/13B、LLaVA-v1.6-Mistral-7B 和 LLaVA-NeXT-7B 上分析注意力，观察到视觉 Token 在早期层获得较高关注，随后迅速下降；把视觉 Token 从第 1-4 层移除几乎会摧毁 GQA 和 MMBench，而从第 29-32 层移除影响很小。作者据此提出：早期层的关键作用是把视觉信息写入文本表示，既然如此，可以把这次融合前移到主语言模型之外。

这里要区分“观察”和“证明”。注意力分布与分层删除实验支持这个设计动机，但并不能证明所有架构都以同样方式处理视觉信息；论文的分析范围主要是 LLaVA 系列。

## 核心贡献

论文的贡献可以概括为三点：

1. **定位视觉 Token 的分层作用**：通过注意力权重、注意力熵和分层删除实验，发现完整视觉 Token 主要在早期层参与跨模态融合。
2. **提出预融合加极限压缩**：模态预融合先让文本吸收完整视觉信息，查询压缩再把送入主 LLM 的视觉 Token 缩到 $C^2$ 个；标准设置取 $C=1$。
3. **统一图像、高分辨率图像和视频**：标准图像每张 1 个视觉 Token，高分辨率版本使用 64 个，视频则把每帧压到 1 个并顺序拼接。

![LLaVA-Mini 方法架构](/images/posts/llava-mini-one-vision-token/llava-mini-iclr2025-figure6-architecture.png)

*图源：Zhang et al., [LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token](https://arxiv.org/abs/2501.03895), Figure 6, ICLR 2025；从作者 arXiv v2 源码中的原始矢量图直接栅格化，结构、标签和数值均未修改。arXiv 页面采用 non-exclusive distribution license；原图用于论文解读。*

## 方法总览：先融合，再压缩

LLaVA-Mini 仍保留视觉编码器、投影层和自回归语言模型，只在主 LLM 前新增两个模块：查询压缩和模态预融合。

### 查询压缩

视觉编码器和投影层先得到：

$$
H^v\in\mathbb{R}^{N^2\times d_h}.
$$

模型引入 $C\times C$ 个可学习查询 $Q^v$。加入二维正弦位置编码后，查询通过交叉注意力从全部视觉 Token 中提取信息：

$$
A=\operatorname{Softmax}\left((Q^v+PE(Q^v))(H^v+PE(H^v))^\top\right),
$$

$$
\hat H^v=A H^v,\qquad \hat H^v\in\mathbb{R}^{C^2\times d_h}.
$$

标准图像设置 $C=1$，因此 $\hat H^v$ 只有 1 个 Token。二维位置编码很重要，因为压缩查询不仅要判断“什么内容重要”，还要保留内容出现在哪里。

### 模态预融合

单纯把 576 个 Token 压成 1 个会明显损失信息。LLaVA-Mini 因而使用 $N_{fusion}$ 个与主 LLM 结构和超参数相同的 Transformer block，在主 LLM 之前处理完整视觉 Token 与文本 Token：

$$
\hat H^q=f(\operatorname{Concat}(H^v,H^q))[-l_q:].
$$

这里只保留输出中的文本位置，得到已经吸收视觉信息的 $\hat H^q$。最终送进主语言模型的是：

$$
\operatorname{Concat}(\hat H^v,\hat H^q),
$$

总长度由 $N^2+l_q$ 变为 $C^2+l_q$。

**最容易被标题掩盖的事实**是：一个视觉 Token 指的是**送入主 LLM 的压缩表示**。视觉编码器、查询压缩和预融合仍会看完整的 576 个视觉 Token；论文的价值在于把最昂贵的主 LLM 上下文缩短，而不是让整个系统从头到尾只处理一个视觉向量。

## 图像、高清视频与长视频流程

### 标准图像

1. CLIP ViT-L/336px 提取 576 个视觉 Token。
2. 四层预融合模块让文本 Token 读取完整视觉信息。
3. 查询压缩产生 1 个视觉 Token。
4. 主 LLM 接收预融合后的文本与 1 个视觉 Token，并生成回答。

### 高分辨率图像

对于 $672\times672$ 图像，作者把图像横纵各切一次得到四个局部块，同时保留原始全局图。预融合读取四组局部视觉 Token、全局视觉 Token和文本；压缩模块把四组局部 Token 压成 $C^2$ 个。LLaVA-Mini-HD 取 $C=8$，即 64 个视觉 Token。

### 视频

对 $M$ 帧视频，每帧分别生成 $C^2$ 个压缩视觉 Token和一组文本融合表示；所有帧的压缩 Token 顺序拼接，文本融合表示则跨帧池化。主 LLM 的输入视觉长度从 $MN^2$ 下降为 $MC^2$。论文默认 1 fps 且 $C=1$，因此一小时视频约对应 3600 个视觉 Token，而不是约 207 万个。

## 训练流程

LLaVA-Mini 延续 LLaVA 的两阶段训练：

| 阶段 | 数据 | 冻结模块 | 训练模块 |
| --- | --- | --- | --- |
| 视觉语言预训练 | 558K 图文描述 | 视觉编码器、LLM | 投影层 |
| 指令微调 | 665K LLaVA 指令数据 | 视觉编码器 | 投影、压缩、预融合、LLM |

两阶段 batch size 都是 256，训练分别为 1 和 2 个 epoch，使用 AdamW、cosine decay 与 0.03 warmup ratio。作者报告使用 8 张 NVIDIA A800 训练。标准公平比较采用 Vicuna-v1.5-7B；带星号的 LLaMA-3.1-8B 版本额外使用约 300 万图像/视频训练样本，不能与标准版本混为一谈。

## 实验设置与主要结果

论文覆盖 11 个图像基准和 7 个视频评测。图像侧包括 VQAv2、GQA、VisWiz、ScienceQA-IMG、TextVQA、POPE、MME、MMBench、SEED-Bench、LLaVA-Bench-in-the-Wild 和 MM-Vet；视频侧包括三组开放问答、Video-ChatGPT 五维生成评测、MVBench、MLVU 和 EgoSchema。

### 图像质量与效率

| 模型 | 主 LLM 视觉 Token | FLOPs | A100 延迟 | VQAv2 | GQA | MMBench | 11 项平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA-v1.5 Vicuna-7B | 576 | 8.55T | 113.04 ms | 78.5 | 62.0 | 64.3 | 56.3 |
| LLaVA-Mini Vicuna-7B | 1 | 1.96T | 38.64 ms | 77.6 | 60.9 | 65.6 | 57.9 |
| LLaVA-Mini-HD Vicuna-7B | 64 | 7.19T | 论文主表未单列 | 78.9 | 61.8 | 67.5 | 58.6 |

标准版相对 LLaVA-v1.5 的 FLOPs 降低 77.08%，A100 测得约 2.92 倍加速。精度不是所有任务都提升：VQAv2 和 GQA 分别下降 0.9 和 1.1，MMBench 提升 1.3，最终 11 项平均提升 1.6。这个分布比单说“性能不降”更准确。

论文对 HD 版本存在一处内部不一致：5.2 节正文写 8.13 TFLOPs，但 Figure 8 与附录 Table 14 均写 7.19T。上表采用能够互相印证的图表值 7.19T，并保留这条差异供复现时核对。

![LLaVA-Mini 官方效率总览](/images/posts/llava-mini-one-vision-token/llava-mini-iclr2025-official-efficiency-overview.png)

*图源：Zhang et al., [LLaVA-Mini 作者官方仓库](https://github.com/ictnlp/LLaVA-Mini/blob/main/assets/performance.png)，汇总论文 Figure 7 与 Figure 9 的标准分辨率 FLOPs、A100 延迟和长视频显存结果，ICLR 2025；图片按作者仓库原样保存，坐标轴、图例和数值未修改。仓库采用 [Apache-2.0](https://github.com/ictnlp/LLaVA-Mini/blob/main/LICENSE)；原图用于论文解读。*

补充材料还给出跨硬件结果：1-token LLaVA-Mini 在 RTX 3090、A100、A800 上分别为 64.52、38.64、27.43 ms，对应的 LLaVA-v1.5 分别为 198.75、113.04、87.43 ms。加速并非只出现在单一卡型上。

### 视频与长视频

| 模型 | 每帧视觉 Token | MSVD-QA Acc. | MSRVTT-QA Acc. | ActivityNet-QA Acc. | MVBench | MLVU | EgoSchema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Video-LLaVA | 256 | 70.7 | 59.2 | 45.3 | 43.1 | 未报告 | 38.4 |
| LLaVA-Mini | 1 | 70.9 | 59.5 | 53.5 | 44.5 | 42.8 | 51.2 |

LLaVA-Mini 每秒取一帧，而固定取 8 或 16 帧的视频模型更容易漏掉长视频中的关键事件。论文指出，它只在短于 1 分钟的视频上训练，却能在 MLVU 中处理超过 2 小时的视频，体现的是架构的长度外推能力。

显存部分需要谨慎表述。论文按每帧约 0.6 MB 的增量，推算 RTX 3090 24GB **理论上**可以容纳超过 10,000 帧，也就是 1 fps 下约 3 小时；这不是对 10,000 帧端到端吞吐、首 Token 延迟和生成质量的完整压力测试。

## 消融分析

### 预融合才是极限压缩的关键

| 设置 | 预融合层数 | 视觉 Token | FLOPs | VQAv2 | GQA | MMBench |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无预融合 | 0 | 1 | 0.96T | 72.4 | 54.2 | 57.7 |
| 有预融合 | 1 | 1 | 1.21T | 74.8 | 55.5 | 60.4 |
| 有预融合 | 2 | 1 | 1.46T | 76.0 | 57.6 | 63.1 |
| 有预融合 | 3 | 1 | 1.81T | 76.9 | 59.1 | 64.9 |
| 有预融合 | 4 | 1 | 1.96T | 77.6 | 60.9 | 65.6 |

没有预融合时，即使保留 144 个视觉 Token，VQAv2/GQA/MMBench 也只有 76.9/58.9/64.9。四层预融合只保留 1 个 Token，却达到 77.6/60.9/65.6。论文的真正创新不是查询池化本身，而是**把原本发生在主 LLM 早期层的视觉注入显式前移**。

### 查询压缩优于平均池化

在同为 1 个视觉 Token、总 FLOPs 约 1.96T 时，平均池化的 VQAv2/GQA/MMBench 为 76.1/59.8/64.0，查询压缩为 77.6/60.9/65.6，额外计算仅约 2.42G。查询会根据图像内容调整关注区域，不只是对所有 patch 一视同仁。

![LLaVA-Mini 查询压缩注意力可视化](/images/posts/llava-mini-one-vision-token/llava-mini-iclr2025-figure12-compression-attention.png)

*图源：Zhang et al., [LLaVA-Mini 作者官方仓库](https://github.com/ictnlp/LLaVA-Mini/blob/main/assets/compression.png)，对应论文 Figure 12, ICLR 2025；左侧为 LLaVA-Bench-in-the-Wild 原图，右侧为查询压缩交叉注意力热力图，所有子图、标签和热力分布均未修改。仓库采用 [Apache-2.0](https://github.com/ictnlp/LLaVA-Mini/blob/main/LICENSE)；原图用于论文解读。*

热力图显示，查询会聚焦人物、文字、接口和主体边缘；主体不明确时，注意力分布更分散。它提供了直观解释，但不能单独证明热区就是因果上决定答案的区域。

### Token 数量仍然是可调旋钮

标准分辨率下，从 1 个增加到 64 个视觉 Token，VQAv2/GQA/MMBench 从 77.6/60.9/65.6 升到 78.5/61.6/67.5。高分辨率下使用 576 个 Token 可达到 80.0/62.9/68.1。LLaVA-Mini 不是宣称“永远只需要一个”，而是把 Token 数量变成可按延迟预算调节的工程旋钮。

## 失败案例与局限

论文没有单列 limitation 章节，也没有系统展示 LLaVA-Mini 自身的失败案例。下面前两点是论文事实，后四点是基于实验设计的分析。

1. **论文事实：压缩仍有任务级损失**。1-token 标准版在 VQAv2、GQA、POPE 和 MME 上低于 LLaVA-v1.5，说明极限压缩并非无损。
2. **论文事实：MVBench 细项并不全面领先**。例如 Object Shuffle 为 29.5，低于 Video-LLaVA 的 40.5；Fine-grained Action 为 37.0，低于 Video-LLaVA 的 42.0。
3. **我的分析：预融合不是免费计算**。四个 Transformer block 仍需读取完整视觉 Token，只是其计算量远小于让整套 7B LLM 在长视觉上下文上运行。
4. **我的分析：注意力规律的外推范围有限**。主要诊断对象是 LLaVA 系列；交叉注意力、混合专家或原生多模态架构未必具有相同分层行为。
5. **我的分析：长视频容量不等于长视频服务质量**。超过 10,000 帧是显存估算，论文没有同时报告该长度下的端到端延迟、吞吐、时间定位精度和信息遗忘曲线。
6. **我的分析：部分评测依赖 GPT-3.5-turbo 判分**。开放式视频问答和生成质量的自动评估会引入裁判模型偏差；多选基准更可复核，但覆盖不了所有开放式行为。

## 可复现资源

- [论文 PDF 与 arXiv 源码](https://arxiv.org/abs/2501.03895)：正文、附录、矢量图和实验表格。
- [官方代码](https://github.com/ictnlp/LLaVA-Mini)：训练、图像/视频推理、Web Demo 和评测说明。
- [官方模型](https://huggingface.co/ICTNLP/llava-mini-llama-3.1-8b)：LLaMA-3.1-8B-Instruct 版本权重。
- [Evaluation.md](https://github.com/ictnlp/LLaVA-Mini/blob/main/docs/Evaluation.md)：图像和视频基准复现入口。

复现时应把模型版本写清楚：论文中与 LLaVA-v1.5 做公平对照的是 Vicuna-v1.5-7B、558K 预训练数据加 665K 指令数据的版本；公开权重则是使用更多数据的 LLaMA-3.1-8B 版本，两者的分数不能直接替换。

## 个人判断

LLaVA-Mini 最值得借鉴的不是“一张图只留一个 Token”这个宣传数字，而是它对计算位置的重新分配：在便宜的前置模块中完成高带宽跨模态融合，在昂贵的主 LLM 中只保留低带宽视觉表示。这个思路与单纯按相似度删 Token 不同，它承认视觉细节在融合前不可随意丢弃。

对工程团队而言，三点尤其有价值：

1. **先确认瓶颈所在层，再做压缩**。层级诊断比从视觉编码器出口直接剪 Token 更可靠。
2. **把 Token 预算做成部署参数**。标准图像可取 1，OCR 或高分辨率场景可取 64，延迟和质量之间不必只有一个固定点。
3. **评估必须覆盖全链路**。FLOPs、预填充延迟、KV Cache、视频长度和任务精度要一起报告，不能只用压缩率代表系统收益。

论文证明了“完整视觉信息不必贯穿整个主 LLM”是一个有效方向，但还没有证明 1-token 配置能覆盖所有细粒度、多图和长视频任务。更稳妥的落地方式，是把预融合深度与视觉 Token 数量共同纳入动态路由，根据输入分辨率、问题类型和延迟预算自适应选择。

## 参考资料

1. Zhang et al. [LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token](https://arxiv.org/abs/2501.03895). ICLR 2025.
2. Liu et al. [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485). NeurIPS 2023.
3. Li et al. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597). ICML 2023.
4. Shang et al. [LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](https://arxiv.org/abs/2403.15388).
5. Ye et al. [VoCo-LLaMA: Towards Vision Compression with Large Language Models](https://arxiv.org/abs/2406.12275).
6. [LLaVA-Mini 官方代码、模型与项目素材](https://github.com/ictnlp/LLaVA-Mini).
