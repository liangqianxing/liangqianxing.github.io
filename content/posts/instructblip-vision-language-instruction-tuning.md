---
title: "InstructBLIP 精读：让视觉特征听懂任务指令"
date: 2026-07-23 12:50:00
description: 从指令感知 Q-Former、26 个数据集的指令微调与零样本评测出发，拆解 NeurIPS 2023 的 InstructBLIP 如何适配多种视觉语言任务。
series: 三大会论文精读
seriesOrder: 9
categories:
  - AI
tags:
  - 多模态大模型
  - 个性化
  - 训练侧适配
  - 指令微调
  - Q-Former
  - Zero-shot
  - NeurIPS
hidden: true
haloPublished: true
---

同一张图片，用户可能要求描述场景、读取文字、回答常识问题，也可能要求判断两个物体的空间关系。如果视觉模块不看指令，总是向大语言模型输出同一组图像特征，那么真正与任务有关的信息只能等到语言模型阶段再筛选。

InstructBLIP 把指令提前送进 Q-Former：可学习查询在提取视觉特征时，就能根据当前任务选择信息。作者再将 26 个公开数据集统一成指令格式，以 13 个数据集训练、13 个数据集做 held-out 零样本评测，只更新 Q-Former，冻结图像编码器和大语言模型。

这篇论文直接属于本专题的**个性化（训练侧）**方向，更准确地说，是“面向场景与下游任务的多模态指令微调”。它不是用户身份或偏好建模，而是让同一个视觉语言模型通过参数高效适配不同任务，并提升对未见数据集和未见任务的 zero-shot 迁移。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning** |
| 作者 | Wenliang Dai、Junnan Li、Dongxu Li、Anthony Meng Huat Tiong、Junqi Zhao、Weisheng Wang、Boyang Li、Pascale N. Fung、Steven C. Hoi |
| 会议 | NeurIPS 2023，Main Conference Track |
| 专题子方向 | 个性化（训练侧）：多模态指令微调、任务适配、zero-shot 迁移 |
| 正式论文 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9a6a435e75419a836fe47ab6793623e6-Abstract-Conference.html) |
| 作者版本与许可 | [arXiv:2305.06500](https://arxiv.org/abs/2305.06500)，CC BY 4.0 |
| 官方补充材料 | [NeurIPS Supplemental](https://proceedings.neurips.cc/paper_files/paper/2023/file/9a6a435e75419a836fe47ab6793623e6-Supplemental-Conference.pdf) |
| 代码与模型 | [Salesforce LAVIS / InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) |

**选择理由**：上一篇专题文章是推理侧的 DeeR-VLA，今天切回训练侧。InstructBLIP 位于附件给出的优先文献池，直接研究多模态模型如何通过指令微调适配多种任务；正式论文、补充材料、作者源码图和官方实现均可访问，主结果、消融和许可也能交叉核验。

## 问题背景：指令不应只在最后出现

BLIP-2 已经用 Q-Former 把冻结视觉编码器接到冻结 LLM 上。它用一组可学习 query 从图像嵌入中提取固定长度的视觉表示，再把这些表示当作软提示交给 LLM。问题在于：BLIP-2 提取视觉表示时并不知道用户要完成什么任务。

例如，对一张烤箱图片，“哪张图里的披萨在烤箱内”和“描述厨房布局”需要关注的区域不同。任务指令如果只进入 LLM，Q-Former 仍会对两条指令输出相同视觉表示。InstructBLIP 的核心假设是：**任务适配不只是改变文本输出，还应该改变视觉信息的读取方式。**

![InstructBLIP 的训练与零样本评测数据划分](/images/posts/instructblip-vision-language-instruction-tuning/instructblip-fig2-datasets.png)

*图源：Dai et al., [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500), Figure 2, NeurIPS 2023；从作者 CC BY 4.0 arXiv 源码中的原始矢量图直接栅格化，任务、数据集、颜色与标签均未修改。原图用于论文解读。*

Figure 2 覆盖 11 类任务和 26 个数据集。黄色表示 held-in 数据，白色表示 held-out 数据。后者不只是“同任务换数据集”，还包括训练阶段完全未见的视觉推理、视频问答、视觉对话和图像分类任务。

## 核心贡献

论文的贡献可以归纳为四点：

1. **系统化视觉语言指令微调**：将 26 个公开数据集转换为统一指令格式，每类任务设计 10-15 个自然语言模板。
2. **指令感知 Q-Former**：指令同时进入 Q-Former 和 LLM，让视觉查询在读取图片时就受任务约束。
3. **兼顾规模的数据采样**：用数据集规模的平方根分配采样概率，避免小数据集过拟合、大数据集欠拟合。
4. **严格区分 held-in 与 held-out**：在 13 个未参与指令微调的数据集上做 zero-shot 评测，并把四类任务完全留作未见任务。

**论文结论**：InstructBLIP 在作者选择的 13 个 held-out 数据集上都取得当时新的 zero-shot 最优结果；最小的 FlanT5-XL 版本在与 Flamingo-80B 共有的 6 个评测上全部更高，平均相对提升 24.8%。

**我的判断**：最可信的证据不是“全部 SOTA”这个时间敏感结论，而是同骨干、同训练设置下的对照与消融。它们说明自然语言指令格式和指令感知视觉提取确实改善了跨数据集泛化。不过这是一轮 16 卡集中式多任务微调，不是面向单个用户的低成本即时个性化。

## 方法总览：任务先影响视觉查询

![InstructBLIP 的指令感知 Q-Former 架构](/images/posts/instructblip-vision-language-instruction-tuning/instructblip-fig3-architecture.png)

*图源：Dai et al., [InstructBLIP](https://arxiv.org/abs/2305.06500), Figure 3, NeurIPS 2023；从作者 CC BY 4.0 arXiv 源码中的原始矢量图直接栅格化，完整保留输入、冻结模块、注意力结构和输出标签。原图用于论文解读。*

设冻结图像编码器输出为 $E_v(I)$，$K$ 个可学习 query 为 $Q$，指令 token 为 $T$。用统一记号概括，指令感知视觉表示可以写成：

$$
Z=\operatorname{QFormer}(Q,T;E_v(I)),\qquad
V=W_p Z.
$$

$Q$ 与 $T$ 在 Q-Former 的 self-attention 中交互，query 再通过 cross-attention 读取图像嵌入；$W_p$ 把输出投影到 LLM 的输入空间。最终冻结 LLM 同时接收视觉软提示 $V$ 和指令 $T$，自回归生成回答。

这里的“参数高效”边界很明确：指令微调阶段只更新 Q-Former，ViT-g/14 图像编码器和 FlanT5/Vicuna LLM 都冻结。论文实验包含 FlanT5-XL（3B）、FlanT5-XXL（11B）、Vicuna-7B 和 Vicuna-13B 四种语言骨干。

### 与 BLIP-2 的关键差别

BLIP-2 的 query 只看图像，因此同一图片在不同任务下得到静态视觉表示。InstructBLIP 让指令 token 与 query 共享 self-attention：

- 指令要求空间关系时，query 可以更关注相关物体和位置；
- 指令要求 OCR 时，query 可以偏向文字区域；
- 指令要求简短回答或详细描述时，视觉表示也能围绕输出目标变化。

论文并没有解冻视觉编码器来获得这种适配能力。任务条件通过 Q-Former 注入，既保留 BLIP-2 的模块化结构，也限制了训练参数量。

## 数据构造与采样

作者把图像描述、VQA、知识问答、视觉推理、视频问答、视觉对话、分类等任务转成“图像 + 指令 -> 回答”。对通常只需短答案的数据集，模板会显式加入 `short`、`briefly` 等词，避免模型把所有输出都学成固定长度；带场景文字的数据还会把 OCR token 放进指令。

由于各数据集规模差异很大，作者没有按数据集均匀采样，也没有直接按样本总量采样。对规模为 $S_d$ 的第 $d$ 个数据集，其基础采样概率为：

$$
p_d=\frac{\sqrt{S_d}}{\sum_{i=1}^{D}\sqrt{S_i}}.
$$

开平方会压平头部数据集的优势，同时避免极小数据集被过度重复。论文还人工降低多选 A-OKVQA 的权重、提高开放式 OKVQA 的权重。这一点很实用，也意味着训练配方并非完全由公式自动决定。

模型仍使用标准语言建模目标，只在回答 token 上计算自回归损失。用 $y_{<t}$ 表示已生成回答，可写为：

$$
\mathcal L_{\mathrm{LM}}
=-\sum_t \log p(y_t\mid I,T,y_{<t}).
$$

## 训练与推理流程

### 训练

1. 从 13 个 held-in 数据集按平滑后的权重采样样本，并为所属任务均匀抽取一条指令模板。
2. 图像由冻结 ViT-g/14 编码；指令同时送入 Q-Former 和冻结 LLM。
3. 只更新 Q-Former，使 query 学会按指令提取任务相关视觉特征。
4. 最多训练 60K step，每 3K step 验证一次；所有数据集共用一个最优 checkpoint，不为每项评测单独选模型。
5. 使用 AdamW，$\beta_1=0.9$、$\beta_2=0.999$、weight decay 0.05；前 1000 step 将学习率从 $10^{-8}$ 线性升到 $10^{-5}$，之后余弦衰减到 0。

不同骨干的 batch size 分别为 192（3B）、128（7B）、64（11B/13B）。论文使用 16 张 NVIDIA A100 40GB，单个模型在 1.5 天内完成训练。这里的参数更新量较小，但总体数据吞吐与硬件成本仍然不低。

### 推理

- 图像描述与开放式 VQA：直接生成文本，再按对应数据集指标评估。
- 分类与多选问答：限制候选答案集合，计算候选序列的 log-likelihood，取分数最高者。
- 二分类：把正负标签扩展为 `yes/true` 与 `no/false` 等 verbalizer，减少单个词频带来的偏差。
- 视频问答：每个视频均匀抽取 4 帧，分别通过图像编码器和 Q-Former，再拼接视觉特征送入 LLM。

因此，论文中的“统一模型”不等于所有任务共享完全相同的解码方式；评测协议仍包含候选排序、OCR 输入和视频抽帧等任务特定处理。

## 实验设置与主要结果

13 个 held-out 数据集覆盖 NoCaps、Flickr30K、GQA、VSR、IconQA、TextVQA、Visual Dialog、HatefulMemes、VizWiz、ScienceQA 图像子集，以及 MSVD-QA、MSRVTT-QA、iVQA。指标并不统一：描述任务使用 CIDEr，HatefulMemes 使用 AUC，Visual Dialog 使用 MRR，其余多为 top-1 accuracy。

| 模型对照 | NoCaps CIDEr | GQA | TextVQA | ScienceQA 图像子集 | MSRVTT-QA | iVQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BLIP-2 FlanT5-XL | 104.5 | 44.0 | 43.1 | 54.9 | 16.2 | 40.4 |
| **InstructBLIP FlanT5-XL** | **119.9** | **48.4** | **46.6** | **70.4** | **25.0** | **53.1** |
| BLIP-2 FlanT5-XXL | 98.4 | 44.6 | 44.1 | 64.5 | 17.4 | 45.8 |
| **InstructBLIP FlanT5-XXL** | **120.0** | **47.9** | **46.6** | **70.6** | **25.6** | **53.8** |

同一 FlanT5-XL 骨干上，作者报告 13 个 held-out 数据集的平均相对提升为 15.0%。MSRVTT-QA 从 BLIP-2 的 16.2 提到 25.0，相对增加约 54%；论文所写“比此前 SOTA 提升 47.1%”对应的是与当时外部基线比较，不能与前一个比例混用。

模型规模也不是唯一解释。论文指出，约 4B 参数的 InstructBLIP FlanT5-XL 在与 Flamingo-80B 共有的 6 项 zero-shot 评测上全部更高。但这些模型的训练数据、架构和评测实现不同，因此这更适合说明指令微调的有效性，不应解读为“4B 模型一般都胜过 80B 模型”。

### 下游微调

作者还从 InstructBLIP checkpoint 出发继续适配单一数据集。它保持 $224\times224$ 分辨率并冻结视觉编码器，训练参数从对照方案的 1.2B 降到 188M。ScienceQA 图像子集上，BLIP-2 FlanT5-XXL 为 89.5，InstructBLIP 为 90.7；OCR-VQA 从 72.7 提到 73.3。

这说明多任务指令微调可以成为更好的任务专用初始化。但 OKVQA 上的 InstructBLIP Vicuna-7B 为 62.1，仍低于论文列出的 562B PaLM-E 的 66.1，不能把“四项都提升”写成“四项都是全局最优”。

## 消融：提升来自指令，还是只来自多任务数据？

![指令微调与普通多任务训练对比](/images/posts/instructblip-vision-language-instruction-tuning/instructblip-fig4-multitask.png)

*图源：Dai et al., [InstructBLIP](https://arxiv.org/abs/2305.06500), Figure 4, NeurIPS 2023；从作者 CC BY 4.0 arXiv 源码中的原始矢量图直接栅格化，完整保留 held-in/held-out 坐标轴、方法标签与数值。原图用于论文解读。*

Figure 4 是最关键的因果对照之一。所有方法都使用 BLIP-2 FlanT5-XL 和相同训练配置：

| 训练方式 | Held-out 平均 | Held-in 平均 |
| --- | ---: | ---: |
| BLIP-2 zero-shot | 46.1 | 67.8 |
| 普通输入训练，评测给自然语言指令 | 46.3 | 92.5 |
| 训练给任务/数据集标识，评测给自然语言指令 | 45.5 | 89.0 |
| 训练和评测都给任务/数据集标识 | 46.8 | 93.7 |
| **InstructBLIP 自然语言指令微调** | **52.9** | **93.8** |

普通多任务训练几乎可以追平 held-in 表现，却没有改善 held-out 泛化。自然语言指令的价值主要出现在未见数据上，而不是让模型更容易记住训练集。

### 指令感知视觉特征与数据平衡

| FlanT5-XL 变体 | Held-in 平均 | GQA | ScienceQA | IconQA | VizWiz | iVQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 完整 InstructBLIP | **94.1** | **48.4** | **70.4** | **50.0** | **32.7** | **53.1** |
| 移除指令感知视觉特征 | 89.8 | 45.9 | 63.4 | 45.8 | 25.1 | 47.5 |
| 移除平衡采样 | 92.6 | 46.8 | 66.0 | 49.9 | 31.8 | 51.1 |

移除指令感知视觉特征后，ScienceQA 降 7.0、VizWiz 降 7.6、iVQA 降 5.6。Vicuna-7B 版本的 iVQA 更从 52.2 降到 36.8。空间与时间推理依赖从图像中选择相关证据，这与方法动机一致。

数据平衡的影响较小但较稳定。论文还观察到，取消平衡后，不同数据集在相差很大的训练 step 才达到峰值，难以用单一 checkpoint 同时服务所有任务。

## 失败风险与局限

论文附录的 Broader Impact 明确指出，InstructBLIP 使用现成的冻结 LLM，因此会继承无依据生成和偏差；作者不建议在没有针对具体应用评估安全与公平性的情况下直接部署。

结合实验设计，还需要注意以下边界：

1. **Held-out 不等于从未见过**：作者避免了指令微调数据与评测集之间的显式重叠，但冻结骨干的大规模预训练数据很难完全审计。
2. **评测依赖任务特定接口**：多选排序、OCR token、二分类 verbalizer 和视频 4 帧抽样都会影响结果，不能把全部提升归因于一个统一生成策略。
3. **模型仍可能看错或编造**：指令感知查询提高了相关视觉证据的提取概率，却没有事实校验或拒答机制。
4. **训练侧高效不等于推理侧加速**：冻结大模块降低的是反向传播和可训练参数，推理时 ViT-g/14 与 3B-13B LLM 仍需完整运行。
5. **数据与许可复杂**：26 个数据集有各自条款；官方 README 还说明模型受 LLaVA 数据、LLaMA 与 Vicuna 许可约束，不能用论文的 CC BY 4.0 覆盖模型与训练数据。
6. **人工配方影响可迁移性**：OKVQA 与 A-OKVQA 的权重经过手工调整，新领域通常仍需重新搜索数据比例。

## 可复现资源

- [NeurIPS 正式页面与 18 页论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9a6a435e75419a836fe47ab6793623e6-Abstract-Conference.html)
- [NeurIPS 官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/file/9a6a435e75419a836fe47ab6793623e6-Supplemental-Conference.pdf)
- [arXiv 全文、源码与 CC BY 4.0 许可](https://arxiv.org/abs/2305.06500)
- [LAVIS 官方实现、模型配置与推理示例](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)
- [Hugging Face：Salesforce/instructblip-flan-t5-xl](https://huggingface.co/Salesforce/instructblip-flan-t5-xl)

LAVIS 提供 FlanT5-XL/XXL、Vicuna-7B/13B 的模型入口，以及生成、训练数据格式和 Gradio demo 示例。官方 README 将模型定位为 research use，并特别提示 Vicuna 版本必须遵守 LLaMA 与 Vicuna 的许可，训练中使用的 LLaVA 数据为 CC BY-NC 4.0。复现时应分别检查论文、代码、checkpoint、数据集和基础模型许可。

## 个人判断

InstructBLIP 的设计很克制：它没有增加更大的视觉编码器，也没有解冻整个 LLM，而是改变“指令在什么时候进入视觉链路”。这让论文提出了一个比泛泛的“多任务训练有效”更具体的机制：自然语言任务描述既是输出约束，也是视觉信息选择器。

对工程团队而言，最值得复用的是两层结构。第一层是参数接口：冻结昂贵骨干，把场景适配集中在桥接模块。第二层是评测接口：必须保留未见数据集和未见任务，才能区分“把训练任务拟合得更好”和“真正学会按指令迁移”。

它的局限同样清楚。188M 可训练参数仍不是 LoRA 级别的小参数包，16 张 A100 的多任务训练也不是个人设备上的即时定制；模型推理成本没有因此下降。因而我会把 InstructBLIP 定位为：一篇证明**指令感知视觉提取能提升多模态任务适配和 zero-shot 迁移**的训练侧基线，而不是一个已经解决用户级个性化、低成本训练或高效推理的完整方案。

## 参考资料

1. Dai et al., [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9a6a435e75419a836fe47ab6793623e6-Abstract-Conference.html), NeurIPS 2023.
2. Dai et al., [arXiv:2305.06500](https://arxiv.org/abs/2305.06500), CC BY 4.0.
3. Salesforce, [LAVIS InstructBLIP Implementation](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip).
4. Li et al., [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q.html), ICML 2023.
5. Alayrac et al., [Flamingo: a Visual Language Model for Few-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html), NeurIPS 2022.
