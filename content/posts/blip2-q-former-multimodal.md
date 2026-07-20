---
title: "BLIP-2 精读：用 Q-Former 接通冻结视觉编码器与大语言模型"
date: 2026-07-20 09:15:00
description: 从 Q-Former、三种表征学习目标与两阶段预训练出发，拆解 BLIP-2 如何用冻结视觉编码器和冻结 LLM 构建多模态模型。
series: 三大会论文精读
seriesOrder: 3
categories:
  - AI
tags:
  - 多模态
  - 视觉语言模型
  - BLIP-2
  - Q-Former
  - 参数高效训练
  - ICML
---

BLIP-2 追问的是一个很有工程意味的问题：**已经有强大的视觉编码器和大语言模型，能否不重新训练这两个大模块，只学习一座足够小的跨模态桥梁？**

论文的答案是 Q-Former（Querying Transformer）和两阶段预训练。第一阶段让一组可学习查询从冻结图像特征中提取“与文本有关”的信息；第二阶段把这些查询输出投影成冻结 LLM 能理解的软视觉提示。视觉编码器与 LLM 在预训练中都保持冻结。

这条路线显著减少了需要更新的参数，但“冻结”不等于训练免费，也不等于模型不会幻觉。BLIP-2 仍在 1.29 亿张图像上训练，最大实验用了 16 张 A100；它还会继承 LLM 的错误知识、偏见和隐私风险。本文会把论文报告的结果与我的工程判断分开。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** |
| 作者 | Junnan Li、Dongxu Li、Silvio Savarese、Steven Hoi |
| 会议 | ICML 2023，PMLR 202:19730-19742 |
| 主题 | 多模态预训练、视觉语言模型、冻结基础模型、参数高效对齐 |
| 官方论文 | [PMLR / ICML Proceedings](https://proceedings.mlr.press/v202/li23q.html) |
| 作者版本与许可 | [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)，CC BY 4.0 |
| 评审记录 | [OpenReview](https://openreview.net/forum?id=KU9UojoX7U) |
| 官方代码 | [Salesforce LAVIS - BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) |

**选题理由**：前两篇精读分别讨论 QLoRA 的高效训练与 DPO 的偏好对齐，这一篇切换到多模态。BLIP-2 的价值不只在某个榜单分数，而在它给出了一种可复用的模块化思路：冻结两端的大模型，把跨模态学习集中到一个信息瓶颈中。方法图、注意力掩码、训练目标与失败案例都足够完整，适合工程读者逐层核对。

## 问题背景：两端都很强，中间却没有接口

视觉语言预训练通常有两条昂贵路径：

1. 从头或端到端训练一个统一多模态模型，视觉和语言参数一起更新；
2. 冻结部分预训练模块，但让另一部分承担全部跨模态对齐压力。

冻结视觉编码器可以保留成熟的图像表征，冻结 LLM 可以保留语言生成和零样本迁移能力，也能缓解灾难性遗忘。但困难随之而来：LLM 在单模态预训练中从未见过图像，图像 patch 特征也不是它词嵌入空间中的合法输入。只用“看图生成文字”的语言建模损失，未必足以让一个小桥接模块学会细粒度图文对齐。

BLIP-2 的核心判断是：先别急着让 LLM 生成。先训练一个查询式瓶颈，把视觉特征压缩成与语言相关的表示；完成这一步后，再把这些表示接到 LLM。

![BLIP-2 两阶段框架总览](/images/posts/blip2-q-former-multimodal/blip2-overview.png)

*图源：Li et al., [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q.html), Figure 1, ICML 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2301.12597)。原图用于论文解读。*

## 核心贡献

论文的贡献可以拆成四层：

1. **Q-Former 桥接架构**：用固定数量的可学习查询从冻结视觉编码器中抽取信息，并与文本共享自注意力参数。
2. **两阶段预训练**：先做视觉-语言表征学习，再做视觉到语言的生成学习，降低冻结 LLM 学习跨模态对齐的负担。
3. **多目标注意力掩码**：用 ITC、ITG、ITM 三个目标和三种 query-text 可见性，在同一组参数上训练对齐、生成与细粒度匹配能力。
4. **模块化实证**：在 VQA、图像描述和图文检索上，以明显少于部分大模型基线的预训练可训练参数取得当时有竞争力的结果，并展示自然语言指令控制的零样本图生文能力。

**论文结论**：更强的冻结图像编码器或更强的冻结 LLM 都能带来更好的下游表现，说明 BLIP-2 可以承接单模态模型的进步。

**我的判断**：最可迁移的不是“32 个 query”这个具体数值，而是先训练信息接口、再接生成模型的顺序。它把“视觉表征是否有用”和“LLM 是否会生成”两个问题分开，消融也能更明确地定位跨模态桥接失败在哪里。

## Q-Former：一个有意做窄的视觉接口

Q-Former 由两个 Transformer 子模块组成：

- image transformer 接收可学习 query，并通过交叉注意力读取冻结图像特征；
- text transformer 既可作为文本编码器，也可作为文本解码器；
- 两者共享 self-attention 层，但 query 和图像特征之间的 cross-attention 每隔一个 Transformer block 插入一次。

模型从 BERT-base 初始化；新增的 cross-attention 随机初始化。论文使用 32 个 query，每个维度为 768。若视觉编码器 ViT-L/14 输出 $257\times1024$ 的 patch 表示，Q-Former 输出只有：

$$
Z\in\mathbb{R}^{32\times768}
$$

这个固定大小与输入分辨率无关。它不是为了无损保留所有视觉信息，而是配合训练目标，把与文本最相关的信息挤进有限的 query token。论文给出的 Q-Former 总参数量为 188M，query 本身也计入模型参数。

从接口角度看，可以把图像侧写成：

$$
H_v=E_v(I),\qquad
Z=Q_\phi(Q,H_v,T;M)
$$

其中 $E_v$ 是冻结视觉编码器，$Q$ 是可学习查询，$T$ 是可选文本输入，$M$ 是随任务变化的注意力掩码，只有 Q-Former 参数 $\phi$ 在预训练中更新。

## 第一阶段：先学会什么视觉信息值得交给语言

![Q-Former 第一阶段目标与注意力掩码](/images/posts/blip2-q-former-multimodal/q-former-stage1.png)

*图源：Li et al., [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q.html), Figure 2, ICML 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2301.12597)。原图用于论文解读。*

三个目标共用同一套 Q-Former 参数，但用不同掩码控制 query 与文本如何交流。

### 1. ITC：图文对比学习

Image-Text Contrastive Learning 让匹配图文靠近、不匹配图文远离。为避免文本信息泄漏到图像表示中，query 与 text 在这个任务下互相不可见。

一张图会产生 32 个 query 输出。论文不是先把它们平均，而是分别与文本的 `[CLS]` 表示计算相似度，再取最高值作为图文相似度。可概括为：

$$
s(I,T)=\max_i\operatorname{sim}(z_i,t_{\mathrm{CLS}})
$$

因为视觉编码器冻结，单卡能容纳更多样本，作者使用 batch 内负例，而没有沿用 BLIP 的 momentum queue。

### 2. ITG：基于图像的文本生成

Image-grounded Text Generation 要求给定图像生成对应文本。掩码采用多模态 causal 结构：query 可以互相注意，但看不到文本；每个文本 token 可以看到全部 query 和此前的文本 token。

对应的自回归损失可写为：

$$
\mathcal{L}_{\mathrm{ITG}}
=-\sum_j\log p_\phi(t_j\mid t_{<j},Z)
$$

因为文本 token 不能直接读取冻结图像编码器，生成所需的信息必须先经过 query。这迫使 Q-Former 不只学习一个粗粒度匹配分数，还要保留足以描述图像的内容。

### 3. ITM：图文匹配

Image-Text Matching 是二分类任务，判断图文是否匹配。此时 query 与文本使用双向注意力，能够充分交互。每个 query 输出经过二分类头，32 个 logit 取平均得到最终匹配分数；负例采用 hard negative mining。

ITC 更像全局检索对齐，ITM 负责融合后的细粒度判别，ITG 则要求表示真正承载可生成的视觉语义。三者的互补性比单独增加一个 loss 权重更重要。

## 第二阶段：把 query 变成 LLM 的软视觉提示

![BLIP-2 第二阶段生成学习](/images/posts/blip2-q-former-multimodal/vision-language-stage2.png)

*图源：Li et al., [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q.html), Figure 3, ICML 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2301.12597)。原图用于论文解读。*

第二阶段把 Q-Former 的输出 $Z$ 经过全连接层投影到 LLM 的词嵌入维度：

$$
V=ZW_p
$$

投影后的 $V$ 被放到文本 embedding 前面，充当 soft visual prompt。视觉编码器和 LLM 继续冻结，训练信号只更新 Q-Former 与投影层。

论文测试了两类 LLM：

- **decoder-only OPT**：用普通 language modeling loss，让 LLM 在视觉提示条件下生成整段文本；
- **encoder-decoder FlanT5**：随机把文本拆成 prefix 与 suffix，把视觉提示和 prefix 送入 encoder，由 decoder 生成 suffix。

对 decoder-only 模型，目标可概括为：

$$
\mathcal{L}_{\mathrm{LM}}
=-\sum_j\log p(t_j\mid V,t_{<j})
$$

关键不是把 Q-Former 输出伪装成可读单词，而是让投影后的向量落到 LLM 能利用的连续提示空间。第一阶段若已经抽取了语言相关视觉信息，第二阶段就不必同时从头解决“看懂图像”和“驱动语言模型”两个问题。

## 训练与推理流程

### 预训练数据

论文沿用 BLIP 的 1.29 亿张图像，包括 COCO、Visual Genome、CC3M、CC12M、SBU，以及 LAION-400M 中的 1.15 亿张图像。作者使用 CapFilt 扩充 web 图像描述：BLIP-large 为每张图生成 10 个候选 caption，再与原始 caption 一起由 CLIP ViT-L/14 排序，保留前两个；每个训练 step 随机取一个。

这意味着 BLIP-2 的参数训练是轻量化的，但数据管线仍然是大规模的。它不是只靠少量高质量样本完成跨模态对齐。

### 模型与超参数

| 项目 | 论文设置 |
| --- | --- |
| 图像编码器 | 冻结 CLIP ViT-L/14 或 EVA-CLIP ViT-g/14，使用倒数第二层特征 |
| LLM | 冻结 OPT 2.7B/6.7B，或 FlanT5 XL/XXL |
| 第一阶段 | 250k steps；ViT-L/ViT-g batch size 2320/1680 |
| 第二阶段 | 80k steps；OPT/FlanT5 batch size 1920/1520 |
| 图像 | $224\times224$，random resized crop 与 horizontal flip |
| 优化 | AdamW、weight decay 0.05、峰值学习率 $10^{-4}$、2k warmup、cosine decay |
| 精度 | 冻结 ViT 和 OPT 使用 FP16；FlanT5 使用 BF16 |

最大配置 ViT-g + FlanT5-XXL 在一台 16×A100 40GB 机器上，论文报告第一阶段少于 6 天、第二阶段少于 3 天。这个数字说明它比端到端更新所有大模型参数更可控，但仍不是普通单机实验。

正式 PDF 的优化器描述把两个系数都写成了 $\beta_1$（0.9 与 0.98），显然存在排版歧义。实际复现应以代码配置与优化器默认值再次核对，不能机械照抄这行文字。

### 推理

零样本图生文时，输入图像经过视觉编码器和 Q-Former，得到固定数量软视觉提示，再拼接自然语言 prompt 交给 LLM 生成。VQA 实验中：

- OPT 使用 `Question: {} Answer:`；
- FlanT5 使用 `Question: {} Short answer:`；
- beam width 为 5，length penalty 为 -1，偏向短答案。

下游微调并不总是继续冻结所有模块。图像描述与 VQA 微调会更新 Q-Former 和图像编码器，同时冻结 LLM；图文检索则直接使用第一阶段模型，不接 LLM，并联合微调图像编码器与 Q-Former。

## 实验设置与主要结果

论文覆盖零样本 VQA、零样本 NoCaps 图像描述、Flickr30K 零样本检索，以及 COCO 上的图像描述、VQA 和图文检索微调。

### 零样本总览

| 任务 | BLIP-2 | 论文中的关键基线 | 应如何解读 |
| --- | ---: | ---: | --- |
| VQAv2 test-dev accuracy | **65.0** | Flamingo80B 56.3 | 高 8.7 个百分点；论文称预训练可训练参数少 54× |
| NoCaps val CIDEr | **121.6** | BLIP 113.2；SimVLM 112.2 | 测试域外图像描述能力更强 |
| NoCaps val SPICE | **15.8** | BLIP 14.8 | 语义内容指标同步提升 |
| Flickr30K text retrieval R@1 | **97.6** | BLIP 96.7；BEIT-3 94.9 | 图找文的零样本召回 |
| Flickr30K image retrieval R@1 | **89.7** | BLIP 86.7；BEIT-3 81.5 | 文找图的零样本召回 |

论文 Table 1 用 188M 作为 BLIP-2 预训练可训练参数量；更细的零样本 VQA Table 2 按具体生成配置列出 103M-108M 可训练参数。两个表的统计口径不同，不应把“108M”当成整个 Q-Former 的总参数量。

### 不是所有任务都赢

ViT-g + FlanT5-XXL 在 VQAv2 test-dev 达到 65.0，在 GQA test-dev 达到 44.7；但 OK-VQA 为 45.9，低于 Flamingo80B 的 50.6。作者认为 OK-VQA 更依赖开放世界知识，而 Flamingo 使用的 70B Chinchilla 比 11B FlanT5-XXL 储存了更多知识。

这个对照很重要：Q-Former 可以改善视觉接口，但不能凭空补足冻结 LLM 中没有的知识。

微调结果也应分任务看：

- COCO 图像描述中，ViT-g + OPT-2.7B 的 CIDEr 为 145.8；
- VQAv2 开放式生成中，ViT-g + OPT-6.7B 的 test-std 为 82.30，高于论文表内其他开放式生成模型，但低于闭集分类模型 BEIT-3 的 84.03；
- COCO 检索微调中，ViT-g 的图找文 R@1 为 85.4、文找图 R@1 为 68.3；
- Flickr30K 零样本检索中，对应 R@1 为 97.6 与 89.7。

这些结果来自不同模型配置、训练目标和评估协议，不能被压缩成一个“BLIP-2 全面优于所有方法”的结论。

## 消融分析：第一阶段与 ITG 都不是装饰

### 去掉第一阶段表征学习

Figure 5 对比了完整两阶段训练与直接做第二阶段生成学习。没有第一阶段时，OPT 和 FlanT5 的零样本 VQA 都明显更差；OPT 的曲线还会随第二阶段训练推进而显著下降，作者将其解释为灾难性遗忘或跨模态桥接失败。

从图上近似读取，在 80k steps 附近：

- OPT-6.7B 完整方法约 54，去掉第一阶段约 15；
- FlanT5-XL 完整方法约 63，去掉第一阶段约 44。

这些是读图近似值，不是论文表格中的精确报告值。可靠结论是差距方向与量级，而不是小数点后的具体数字。

### ITG 对检索也有帮助

COCO 检索消融给出了精确数值：

| 微调目标 | 图找文 R@1 | 图找文 R@5 | 文找图 R@1 | 文找图 R@5 |
| --- | ---: | ---: | ---: | ---: |
| ITC + ITM | 84.5 | 96.2 | 67.2 | 87.1 |
| ITC + ITM + ITG | **85.4** | **97.0** | **68.3** | **87.7** |

ITG 没有直接优化检索排名，却让 query 必须保留足以生成文本的视觉信息，因此也改善图文对齐。这是三目标设计真正互补的实验证据。

### 更强的两端都能增益

Table 2 支持三个一致趋势：ViT-g 通常优于 ViT-L；同一家族中更大的 LLM 通常更好；指令微调过的 FlanT5 在 VQA 上明显优于只做无监督语言建模的 OPT。

但这不等于模块可以任意替换后零成本升级。更换图像编码器或 LLM 仍要重新训练桥接层，输入维度、数值精度、prompt 格式和显存需求也会变化。

## 失败案例与局限

![BLIP-2 错误输出案例](/images/posts/blip2-q-former-multimodal/failure-cases.png)

*图源：Li et al., [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q.html), Appendix Figure 6, ICML 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2301.12597)。原图用于论文解读。*

附录给出三类具体失败：把爱因斯坦名言归给错误人物；冬季去加拿大却忽略天气，激活错误推理路径；把图中的 iPhone 14 认成 iPhone 11，使用过时或错误信息。

论文明确列出的局限包括：

1. 给 LLM 提供 few-shot VQA 示例没有带来更好表现。作者认为预训练样本只有单个图文对，没有让模型学习同一序列内多组图文对之间的关系。
2. 生成错误可能来自 LLM 的不准确知识、错误推理路径，或对新图像内容缺少最新信息。
3. 冻结模型会继承已有 LLM 的攻击性语言、社会偏见与隐私泄漏风险。
4. 指令约束或过滤有害数据可以缓解风险，但论文没有给出系统的安全评测。

还需要补充几个工程边界：

- 固定 32 个 query 是强信息瓶颈，对 OCR、密集目标、细粒度空间关系等高带宽任务可能丢信息；论文没有系统扫描 query 数量。
- 论文强调可训练参数量，却没有把数据清洗、合成 caption、总 FLOPs、能源与端到端 wall-clock 成本做完整同口径比较。
- 零样本 VQA 使用固定 prompt、beam search 和短答案偏置，结果会受解码协议影响。
- “冻结 LLM”避免更新其权重，不代表推理便宜；12.1B 总参数配置仍需加载视觉编码器、Q-Former 与 FlanT5-XXL。

## 可复现资源与实现检查表

### 官方资源

- [ICML / PMLR 正式论文与 BibTeX](https://proceedings.mlr.press/v202/li23q.html)
- [作者 arXiv v3 与 CC BY 4.0 许可](https://arxiv.org/abs/2301.12597)
- [Salesforce LAVIS 官方实现](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
- [官方 instructed generation notebook](https://github.com/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb)
- [官方训练脚本](https://github.com/salesforce/LAVIS/tree/main/run_scripts/blip2/train)
- [官方评估脚本](https://github.com/salesforce/LAVIS/tree/main/run_scripts/blip2/eval)
- [Hugging Face Transformers 的 BLIP-2 集成](https://huggingface.co/docs/transformers/model_doc/blip-2)

LAVIS 代码采用 BSD 3-Clause License。论文图来自作者 CC BY 4.0 arXiv 版本，本文保留原始标签和图例，仅从作者源码 PDF 栅格化为 PNG 并注明论文、Figure 编号与来源。

官方仓库给出了 stage 1、stage 2 和下游任务配置，但公开 YAML 的数据集、batch size、warmup 与论文完整 1.29 亿图像实验并不完全相同。它们是可运行起点，不是论文全部结果的一键复现清单。

### 实现检查表

1. 确认视觉编码器与 LLM 真正冻结，optimizer 只包含 Q-Former 与投影层；下游微调时再按任务决定是否解冻 ViT。
2. ITC、ITG、ITM 的 attention mask 必须分别验证，尤其防止 ITC 中 query 偷看到文本。
3. 检查 32 个 query 输出与 LLM embedding 的维度投影，不能把离散 token id 和连续 soft prompt 混为一谈。
4. 分别记录 stage 1 与 stage 2 checkpoint，先验证检索/匹配能力，再验证生成，便于定位桥接失败。
5. 复现实验时固定图像预处理、prompt、beam width、length penalty 与答案规范化，否则 VQA 数字不可直接比较。
6. 对生成结果增加对象识别、OCR、事实性、时间敏感知识和安全性抽检，不能只看 CIDEr 或 VQA accuracy。
7. 记录总参数、可训练参数、峰值显存、吞吐、GPU 时长与数据量，避免只用“可训练参数少”代表全部成本。

## 个人判断

我认为 BLIP-2 最重要的贡献是把多模态模型拆成三个职责清晰的部件：视觉编码器负责感知，Q-Former 负责选择和翻译，LLM 负责语言生成。这个分工让团队可以复用成熟单模态模型，也让训练失败更容易归因。

它的代价同样来自这个分工。Q-Former 只有有限视觉带宽，LLM 又无法通过权重更新来适应视觉输入；一旦 query 没抽到关键信息，后面的 LLM 只能基于缺失或错误证据生成一个语言上很流畅的答案。模块化降低了训练耦合，却没有消除错误传播。

对今天的工程实践，BLIP-2 更适合作为“适配器设计”的基线，而不是默认最终架构。若任务主要是通用图像问答、描述或检索，冻结强模型加轻量桥接很有吸引力；若任务需要高分辨率 OCR、视频时序、复杂定位或领域知识更新，就应认真评估更高带宽视觉 token、交错图文训练、指令微调以及部分解冻是否值得额外成本。

## 一句话总结

> BLIP-2 用 Q-Former 把冻结视觉编码器的高维 patch 特征压缩成少量语言相关查询，再通过两阶段预训练将其变成冻结 LLM 可消费的软视觉提示；它降低了跨模态训练的可更新参数量，但数据成本、视觉瓶颈、知识错误与安全风险仍然存在。

## 参考资料

1. Li et al. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q.html). ICML 2023.
2. Li et al. [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://proceedings.mlr.press/v162/li22n.html). ICML 2022.
3. Radford et al. [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html). ICML 2021.
4. Alayrac et al. [Flamingo: a Visual Language Model for Few-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html). NeurIPS 2022.
5. Dai et al. [EVA-CLIP: Improved Training Techniques for CLIP at Scale](https://arxiv.org/abs/2303.15389). 2023.
