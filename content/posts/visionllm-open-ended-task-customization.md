---
title: "VisionLLM 精读：训练侧如何用语言指令定制检测与分割"
date: 2026-08-06 20:22:43
description: 拆解 NeurIPS 2023 VisionLLM 如何用 LoRA、语言引导图像 Tokenizer 与输出格式查询，把多模态大模型训练侧适配到检测、分割、视觉定位等可定制任务。
series: 三大会论文精读
seriesOrder: 19
categories:
  - AI
tags:
  - 多模态大模型
  - 个性化
  - 训练侧适配
  - LoRA
  - 视觉指令微调
  - 目标检测
  - 实例分割
  - NeurIPS
hidden: true
haloPublished: true
---

让多模态大模型“适配视觉任务”，通常意味着再训练一个检测头、分割头，或者把框坐标逐 Token 自回归生成。VisionLLM 选择了更激进的路线：把任务描述、目标类别、坐标范围和输出结构全部写进语言指令，再让一个经过 LoRA 适配的 Alpaca-7B 同时承担任务解析与视觉输出解码。

它因此直接属于本专题的 **个性化（训练侧）**：这里的“个性化”不是记住某个用户的脸或偏好，而是让同一套多模态模型按场景要求定制任务、目标对象和输出格式，并把 LLM 的知识迁移到目标检测、实例分割、视觉定位、图像描述与视觉问答。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks** |
| 作者 | Wenhai Wang、Zhe Chen、Xiaokang Chen、Jiannan Wu、Xizhou Zhu、Gang Zeng、Ping Luo、Tong Lu、Jie Zhou、Yu Qiao、Jifeng Dai |
| 会议 | NeurIPS 2023，37th Conference on Neural Information Processing Systems |
| 专题子方向 | 个性化（训练侧）：LoRA 适配、视觉任务知识迁移、任务/类别/输出格式定制 |
| 正式论文 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c1f7b1ed763e9c75e4db74b49b76db5f-Abstract-Conference.html) |
| 官方补充材料 | [NeurIPS Supplemental](https://proceedings.neurips.cc/paper_files/paper/2023/file/c1f7b1ed763e9c75e4db74b49b76db5f-Supplemental-Conference.pdf) |
| 作者版本与许可 | [arXiv:2305.11175 v2](https://arxiv.org/abs/2305.11175)，CC BY 4.0 |
| 官方仓库 | [OpenGVLab/VisionLLM](https://github.com/OpenGVLab/VisionLLM)，仓库为 Apache-2.0；截至本文核验时，v1 代码与模型仍标记为未发布 |

**选择理由**：上一篇专题文章 MMInference 属于推理加速，本轮切回训练侧。现有精读已经覆盖用户专属概念学习、可验证奖励个性化、视觉指令微调和轻量 Adapter；VisionLLM 则把训练侧适配推进到检测与分割等需要结构化空间输出的视觉任务，且正式论文、11 页补充材料、CC BY 4.0 作者源码和高质量原图均可核验。

## 问题背景：统一任务，不等于能够定制任务

通用视觉模型常把检测、分割、描述和问答预先编码成固定任务 ID 或固定输出头。它们可以共享参数，但用户仍不能自然地说：

- 只检测指定的 10 类对象；
- 把类别顺序改成当前业务词表；
- 用 8、16 或 24 个边界点输出实例掩码；
- 同一张图既给短描述，也给长描述或回答推理问题。

另一条路线是视觉 Prompt，例如在输入图上画掩码或示例。它适合视觉任务，却与 LLM 的自然语言接口不一致。VisionLLM 的核心问题是：**能否把纯视觉任务也改写成语言可描述、LLM 可解析、Token 可监督的任务？**

![VisionLLM 的任务、类别集合与输出格式定制示例](/images/posts/visionllm-open-ended-task-customization/visionllm-open-ended-task-customization-fig2-results.png)

*图源：Wang et al., [VisionLLM](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c1f7b1ed763e9c75e4db74b49b76db5f-Abstract-Conference.html), Figure 2(a-d), NeurIPS 2023；从 NeurIPS 正式 PDF 原页裁取，完整保留四个子图、子图标识、任务指令、模型输出与原始图注。作者 arXiv v2 以 CC BY 4.0 发布。原图用于论文解读。*

Figure 2 展示了三种不同粒度的定制：

1. **任务级**：检测、分割、描述和 VQA 由语言指令切换；
2. **对象级**：类别集合可以是标准类别、重新排序的类别，甚至问题或描述短语；
3. **输出级**：边界框、不同数量的多边形点和文本长度都可以写进指令。

需要先划清边界：这些展示来自训练分布内构造出的指令变化，并不自动等于面对任意新任务都能 zero-shot 泛化。论文对检测、分割、定位和描述给了量化结果，但 VQA 只提供定性示例。

## 核心贡献

论文的贡献可以拆成四点：

1. **统一语言指令**：把视觉语言任务与纯视觉任务都表达成“图像 + 任务描述 + 输出格式”。
2. **语言引导图像 Tokenizer**：用文本特征调制多尺度视觉特征，再通过 Deformable DETR 查询抽取固定数量、带语义和位置的图像 Token。
3. **LLM 开放任务解码器**：扩展 Alpaca-7B 的词表，使类别和坐标都能作为离散 Token 预测。
4. **输出格式即查询**：先把指令解析成结构模板，再用这些结构 Token 并行查询检测或分割结果，避免纯自回归逐框生成。

**论文结论**：共享参数的 VisionLLM 能仅通过更换语言指令切换多个任务；InternImage-H 版本在 COCO 检测上达到 60.2 AP，ResNet-50 版本在随机任务描述与类别集合下达到 44.6 AP。

**我的判断**：最有价值的不是“60+ AP”本身，因为它高度依赖强大的 InternImage-H 视觉骨干；更关键的是 LoRA、文本条件视觉 Tokenizer 和结构化并行解码共同让随机类别与任务描述训练得以收敛。它证明了 LLM 可以成为视觉任务的可编程接口，但还没有证明任意自然语言都能可靠定义新视觉任务。

## 方法总览：把图像和视觉输出都翻译成“外语”

![VisionLLM 总体架构](/images/posts/visionllm-open-ended-task-customization/visionllm-open-ended-task-customization-fig3-architecture.png)

*图源：Wang et al., [VisionLLM](https://arxiv.org/abs/2305.11175), Figure 3, NeurIPS 2023；从作者 CC BY 4.0 arXiv v2 源码中的原始矢量 PDF 直接栅格化，输入、三大模块、图像 Token、语言指令和输出标签均未修改。原图用于论文解读。*

设图像为 $X$，语言指令为 $s$。视觉骨干和文本编码器分别产生多尺度视觉特征与语言特征：

$$
F_v=B(X),\qquad F_l=E_{\text{text}}(s).
$$

语言特征通过 cross-attention 注入每个尺度的视觉特征。再用 $M$ 个可学习查询，经 Deformable DETR 风格的编码器/解码器得到图像 Token：

$$
\widetilde F_v=\operatorname{CrossAttn}(F_v,F_l),
$$

$$
T=\left\{(e_i,l_i)\right\}_{i=1}^{M}
=\operatorname{ImageTokenizer}(Q,\widetilde F_v).
$$

$e_i$ 表示语义嵌入，$l_i$ 表示该 Token 的绝对中心位置。论文默认 $M=100$，因此它不是把整张高分辨率图像切成密集 patch 后全部送给 LLM，而是抽取一组与当前指令相关的对象级 Token。

### 统一视觉指令

对描述和 VQA，指令格式与常见多模态模型相似：`<image>` 后跟任务文本。检测和分割则必须同时说明三件事：

- `<class>`：类别名称到类别 Token 的映射；
- `<range>`：坐标离散化范围；
- 输出元组：例如检测使用 $(c,x_1,y_1,x_2,y_2)$，分割使用 $(c,x_1,y_1,\ldots,x_N,y_N)$。

作者用 seed instruction 和 Self-Instruct 风格的扩写生成多种任务描述，并在训练时随机采样。这样模型不能只记住一个固定模板。

### 类别 Token 与位置 Token

Alpaca 的原始词表并不适合精确输出视觉坐标。VisionLLM 增加两类 Token：

$$
\mathcal V_p=\{\texttt{<p-512>},\ldots,\texttt{<p512>}\},
$$

$$
\mathcal V_c=\{\texttt{<c0>},\texttt{<c1>},\ldots,\texttt{<c511>}\}.
$$

位置 Token `p_k` 表示相对于图像 Token 中心的离散偏移，归一化值为 $k/512$。类别名称不直接作为输出，而是在指令中动态映射到 `c_i`。因此，同一个 `c0` 在一条指令里可以代表 `person`，在另一条指令里可以代表 `frisbee`。

这带来灵活性，也带来上限：类别语义必须通过当前指令正确绑定；LVIS 的 1203 类无法一次放入有限词表，作者训练时每轮随机抽 80 类，推理时把 1203 类拆成 16 组滑窗预测。

## 关键模块：输出格式即查询

传统 Pix2Seq 会把多个目标串成一个长序列，自回归生成类别和坐标。目标越多，序列越长；对象集合又没有天然顺序，训练很容易受排列影响。

VisionLLM 先让微调后的 Alpaca 把用户指令解析成标准结构。例如，检测被解析为多组：

```text
<cls> <x1> <y1> <x2> <y2>
```

这些结构 Token 不是最终答案，而是并行查询。检测训练和推理时，作者向解码器输入 100 组结构查询，一次生成 100 个候选对象，再按置信度筛选。

可以把它抽象为：

$$
q_{\tau}=\operatorname{Parse}(s),\qquad
\hat y=\operatorname{Decoder}_{\text{LLM}}(T,s,q_{\tau}).
$$

$q_{\tau}$ 由任务类型和输出格式决定。对检测、分割这类结构化任务，它允许并行预测；对描述任务，查询只是 `<bos>`，后续仍然自回归生成文本。

## 损失函数

总损失由图像 Tokenizer 和开放任务解码器两部分组成：

$$
\mathcal L=\mathcal L_{\text{tok}}+\mathcal L_{\text{dec}}.
$$

图像 Tokenizer 使用类别无关的 focal loss 和中心点 $L_1$ 回归：

$$
\mathcal L_{\text{tok}}
=\mathcal L_{\text{focal}}+\lambda\mathcal L_1.
$$

文本输出使用标准 next-token 监督。检测框等无序集合先通过匈牙利匹配把 100 个预测与真值对齐，再统一用交叉熵训练离散类别与坐标 Token。论文没有给出上述 $\lambda$ 的具体数值，因此复现时不能从公式直接恢复完整配方。

## 训练流程：先学检测，再做多任务适配

![VisionLLM 两阶段训练日程](/images/posts/visionllm-open-ended-task-customization/visionllm-open-ended-task-customization-figa-training-schedule.png)

*图源：Wang et al., [VisionLLM 官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/file/c1f7b1ed763e9c75e4db74b49b76db5f-Supplemental-Conference.pdf), Figure A, NeurIPS 2023；从作者 CC BY 4.0 arXiv v2 源码中的原始矢量 PDF 直接栅格化，坐标轴、两条训练曲线、阶段边界和图例均未修改。原图用于论文解读。*

### 阶段一：建立视觉与语言 Token 的桥梁

1. 用预训练 Deformable DETR、BERT 和 Alpaca-7B 初始化模型；
2. 训练视觉骨干与语言引导图像 Tokenizer；
3. 冻结 Alpaca 大部分参数，只更新注意力层 Q/K/V/O 上 rank 64 的 LoRA；
4. 只训练目标检测，同时随机改变任务描述和类别集合。

### 阶段二：共享参数的多任务训练

1. 冻结视觉骨干；
2. 加入检测、实例分割、视觉定位、图像描述和 VQA；
3. 用同一个模型，通过指令切换任务；
4. 继续训练 Tokenizer、桥接组件与 LoRA 参数。

Figure A 显示，两阶段方案在第一个阶段迅速达到约 45 AP，进入多任务阶段后短暂下降，再稳定在约 44 AP；单阶段训练到 50 epoch 仍只有约 28 AP。它说明课程式训练主要解决优化问题，不能解读为多任务模型最终超过单任务模型。

官方补充材料给出的默认训练为 50 epoch、$4\times8$ 张 NVIDIA A100、每卡 1 个样本、AdamW、初始学习率 $2\times10^{-4}$ 和余弦退火。这里存在一处官方文档不一致：NeurIPS 正文写 BERT-Large，补充材料写 BERT-Base。没有公开的 v1 代码和配置可用于消歧，复现者应把它视为实质性风险。

## 实验设置与主要结果

训练数据覆盖五类任务：

- COCO 2017：目标检测与实例分割；
- RefCOCO、RefCOCO+、RefCOCOg：合并超过 12 万个指代表达用于视觉定位；
- COCO Caption：图像描述；
- LLaVA-Instruct-150K：VQA 训练。

VQA 没有标准量化结果。作者明确说明，LLaVA-Instruct-150K 与标准 VQA 基准不兼容，因此只展示定性案例。

| 模型 | 检测 AP | 分割 AP | RefCOCO P@0.5 | BLEU-4 | CIDEr |
| --- | ---: | ---: | ---: | ---: | ---: |
| VisionLLM-R50，分任务训练 | 44.8 | 25.2 | 84.4 | 30.8 | 112.4 |
| **VisionLLM-R50，共享多任务** | **44.6** | **25.1** | **80.6** | **31.0** | **112.5** |
| **VisionLLM-H，共享多任务** | **60.2** | **30.6** | **86.7** | **32.1** | **114.2** |

共享多任务模型在 R50 上相对分任务模型略降：检测 $-0.2$ AP、分割 $-0.1$ AP、定位 $-3.8$ P@0.5；描述则基本持平并略有提升。论文把它归因于多任务冲突。

60.2 AP 来自 InternImage-H，而不是 Alpaca 或 LoRA 单独带来的提升。更公平的同骨干比较是 R50：VisionLLM 的 44.6 AP 比 Pix2Seq-R50 的 43.2 高 1.4 AP，但仍低于 Deformable DETR-R50 的 45.7 AP。

### 对象与输出格式定制

![VisionLLM 按类别集合执行检测](/images/posts/visionllm-open-ended-task-customization/visionllm-open-ended-task-customization-fige-category-customization.png)

*图源：Wang et al., [VisionLLM 官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/file/c1f7b1ed763e9c75e4db74b49b76db5f-Supplemental-Conference.pdf), Figure E, NeurIPS 2023；从作者 CC BY 4.0 arXiv v2 源码中的原始矢量 PDF 直接栅格化，四组指令、黄色类别集合、检测框和标签均未修改。原图用于论文解读。*

Figure E 的重点不是开放词表识别本身，而是**类别集合成为模型执行范围**：当指令只包含 `frisbee` 时，模型不再输出人物；当类别映射重新排序或加入“穿蓝色 T 恤的人”这类描述时，输出标签跟随当前指令变化。

COCO minival 的定量结果也显示，类别数量从 10、20、40 增加到 80 时，对应所选类别的平均 AP 分别为 48.9、52.7、49.3、44.6。不同子集难度不同，因此不能据此得出“类别越少越好”的单调结论。

分割输出格式从 8 个边界点增加到 24 个边界点时，mask AP 从 18.5 提升到 25.1，AP75 从 11.6 提升到 22.4。更多点能描述更细的轮廓，但也增加查询长度、显存与计算开销。

### 有限的推理速度证据

补充材料在单张 A100、batch size 1、$1024\times1024$ 输入上报告：

| 方法 | FPS | 单图时间 |
| --- | ---: | ---: |
| VisionLLM-R50 | 5.1 | 197.4 ms |
| Pix2Seq-R50 | 4.4 | 227.3 ms |
| VisionLLM-ViT-B | 4.0 | 251.7 ms |
| Pix2Seq v2-ViT-B | 3.4 | 294.1 ms |

这组结果支持“结构查询比逐 Token 输出检测框更快”，但范围很窄：没有说明精度对齐、预处理边界、功耗、显存、并发、尾延迟，也没有覆盖描述和 VQA 的长文本解码。因此它不应被归入本专题的推理加速论文。

## 消融分析：哪些设计真正不可少

### LoRA 不是装饰，而是收敛条件

| LoRA | 随机任务描述/类别 | COCO AP |
| --- | --- | ---: |
| 否 | 否 | 45.2 |
| 否 | 是 | 1.2 |
| **是** | **是** | **44.8** |

没有随机性时，冻结 LLM、不给 LoRA 仍能训练普通检测；一旦类别映射和任务描述随机变化，AP 直接跌到 1.2。加入 LoRA 后恢复到 44.8。论文据此认为 LoRA 是语言 Token 与视觉 Token 对齐的桥梁。

**我的判断**：这个消融是全文最直接的训练侧适配证据，但它并不能证明 LoRA 比其他 Adapter 更好，因为论文没有做相同参数预算下的替代模块比较。

### 文本条件主要帮助跨模态定位

| 设置 | COCO AP | RefCOCO P@0.5 |
| --- | ---: | ---: |
| 不使用 BERT | 44.7 | 48.1 |
| **使用并训练 BERT** | **44.8** | **84.1** |
| 使用但冻结 BERT | 1.3 | 34.3 |

检测只需在固定类别上找对象，去掉文本编码器几乎不影响 COCO；视觉定位必须理解指代表达，RefCOCO 从 84.1 跌到 48.1。更反常的是冻结 BERT 会让两项都崩溃，说明语言引导模块必须共同适配，而不是把预训练文本编码器当作静态特征源。

### 查询式 Tokenizer 与坐标分辨率

- 用平均池化 patch 替代查询式图像 Tokenizer，AP 从 44.8 降到 23.1；
- 位置 bin 从 257 增至 513、1025、2049 时，AP 为 34.9、40.8、44.8、44.8；
- 图像 Token 从 50 增至 100、200、300 时，AP 为 44.5、44.8、45.1、45.2。

1025 个位置 Token 已达到饱和。100 个图像 Token 相比 300 个只损失 0.4 AP，作者因此选择 100 作为计算与精度折中。

### 随机化带来可定制性，也付出精度

从固定检测开始逐步加入随机任务描述、随机类别和随机输出格式，AP 从 45.2 依次变为 45.1、44.8、44.6。标准检测精度小幅下降 0.6 AP，换来任务接口的灵活性。八种不同 Prompt 的 AP 位于 44.7-44.8，说明在作者设计的模板分布内，措辞与类别顺序变化较稳定。

## 失败案例与局限

1. **不是用户级个性化**：模型没有用户身份、长期记忆或个人偏好建模；它实现的是任务与输出接口的定制。
2. **不是纯 PEFT 训练**：Alpaca 主要通过 LoRA 更新，但阶段一还训练视觉骨干和图像 Tokenizer，整体成本不能只用 LoRA 参数量概括。
3. **分割边界较粗**：InternImage-H 的 mask AP50 为 61.2，但 AP75 只有 27.6。作者归因于坐标离散化、边界点数量受限和多边形表达弱于直接掩码预测。
4. **VQA 证据不足**：只提供定性示例，没有标准 VQA 分数。Figure 2(d) 的回答也加入了图像未必能支持的建议性内容，说明 LLM 幻觉仍存在。
5. **多任务冲突真实存在**：共享模型在检测、分割和视觉定位上均低于分任务训练，尤其 RefCOCO 下降 3.8 P@0.5。
6. **大词表需要分组推理**：LVIS 1203 类必须拆成 16 组，虽然得到 18.9 AP，但延迟会随分组增加。
7. **复现材料不完整**：官方 VisionLLM v1 README 至今仍把“Release code and models”列为未完成；正文与补充材料对 BERT-Large/BERT-Base 的描述也不一致。
8. **许可不能混为一谈**：论文作者版本为 CC BY 4.0，GitHub 仓库为 Apache-2.0；Alpaca/LLaMA 基座、训练数据和下游数据集仍有各自条款。
9. **“开放任务”有训练边界**：实验验证的是随机模板、类别集合与输出格式，不是从未训练过的新视觉算子或任意复杂指令。

## 可复现资源

- [NeurIPS 正式论文与 BibTeX](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c1f7b1ed763e9c75e4db74b49b76db5f-Abstract-Conference.html)
- [NeurIPS 官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/file/c1f7b1ed763e9c75e4db74b49b76db5f-Supplemental-Conference.pdf)
- [arXiv v2 全文、源码与 CC BY 4.0 许可](https://arxiv.org/abs/2305.11175)
- [OpenGVLab/VisionLLM 官方仓库](https://github.com/OpenGVLab/VisionLLM)
- [VisionLLM v1 README](https://github.com/OpenGVLab/VisionLLM/tree/main/VisionLLM)

截至 2026-08-06 核验，仓库顶层已转为 VisionLLM 系列入口，主要可运行代码集中在 VisionLLM v2；v1 子目录仍只有论文介绍、架构图和未完成的代码/模型发布清单。因此，读者可以复核论文机制、公式、图表和官方元数据，但不能按 v1 官方代码完整复现实验。

## 个人判断

VisionLLM 最值得学习的地方，是把“任务头”重新定义成一个可由语言编程的接口。类别集合、空间量化和输出结构都进入 Prompt，LLM 不只生成答案，还参与决定视觉特征该关注什么、结果该按什么结构返回。

工程上可以复用三条经验：

1. 对检测、分割等无序集合，不要机械照搬长序列自回归生成；结构查询加匈牙利匹配更符合问题性质。
2. PEFT 模块的价值需要放在困难训练分布里检验。固定检测上 LoRA 看似可有可无，加入随机任务和类别后才显出决定性作用。
3. “统一模型”必须同时报告共享模型和分任务模型，否则很容易把骨干规模带来的高分误写成任务统一本身的收益。

我会把 VisionLLM 定位为一篇**训练侧视觉任务适配的早期系统论文**：它成功把 LLM 接口推进到检测与分割，但离稳定的用户级个性化、任意任务 zero-shot、完整开源复现和生产级可靠性仍有明显距离。

## 参考资料

1. Wang et al., [VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c1f7b1ed763e9c75e4db74b49b76db5f-Abstract-Conference.html), NeurIPS 2023.
2. Wang et al., [VisionLLM Supplementary Materials](https://proceedings.neurips.cc/paper_files/paper/2023/file/c1f7b1ed763e9c75e4db74b49b76db5f-Supplemental-Conference.pdf), NeurIPS 2023.
3. Wang et al., [arXiv:2305.11175 v2](https://arxiv.org/abs/2305.11175), CC BY 4.0.
4. OpenGVLab, [VisionLLM Official Repository](https://github.com/OpenGVLab/VisionLLM), Apache-2.0.
5. Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022.
6. Zhu et al., [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159), ICLR 2021.
