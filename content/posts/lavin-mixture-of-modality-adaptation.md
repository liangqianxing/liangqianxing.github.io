---
title: "LaVIN 精读：用 3.8M 参数适配视觉语言指令"
date: 2026-07-31 20:30:00
description: "拆解 NeurIPS 2023 的 LaVIN 如何用 Mixture-of-Modality Adapter 和混合模态训练，以 3.8M 可训练参数适配文本与图文指令，属于个性化（训练侧）方向。"
series: 三大会论文精读
seriesOrder: 15
categories:
  - AI
tags:
  - 多模态大模型
  - 个性化（训练侧）
  - 参数高效微调
  - Adapter
  - 动态路由
  - 视觉指令微调
  - LaVIN
  - NeurIPS
hidden: true
haloPublished: true
draft: false
---

把一个冻结的大语言模型改造成视觉语言助手，通常不只是加一层投影那么简单：图像编码器与语言模型需要对齐，文本指令和图文指令又可能争用同一组参数。若直接全量微调 7B 或 13B 模型，训练、显存和 checkpoint 成本都会迅速上升；若只训练一个统一 Adapter，不同模态之间又可能互相干扰。

NeurIPS 2023 论文《Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models》提出 Mixture-of-Modality Adaptation（MMA），并将它应用到 CLIP ViT 与 LLaMA，得到 LaVIN。作者冻结视觉编码器和 LLM 主干，只训练 3.8M 或 5.4M 个适配参数；在 ScienceQA 上，LaVIN-7B/13B 分别达到 89.41/90.83，并把 8 张 A100 上的训练墙钟时间降到 1.4/2 小时。

这篇论文属于本专题的 **个性化（训练侧）**。它直接研究多模态大模型的参数高效适配：如何用 Adapter 把视觉知识接入 LLM，如何在文本与图文指令之间路由，以及如何用混合模态数据联合训练。论文也报告训练时间、显存和 checkpoint 存储开销，但这些是训练效率证据，不应误写成推理加速。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models** |
| 作者 | Gen Luo、Yiyi Zhou、Tianhe Ren、Shengxin Chen、Xiaoshuai Sun、Rongrong Ji |
| 会议 | NeurIPS 2023，Main Conference Track |
| 方法 | Mixture-of-Modality Adaptation（MMA）与 LaVIN |
| 专题方向 | 个性化（训练侧）：多模态 Adapter、视觉指令微调、模态路由 |
| 正式论文 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5e84e4413268b713f0d4a1b23a9dae57-Abstract-Conference.html)，DOI 10.52202/075280-1288 |
| 作者全文与许可 | [arXiv:2305.15023 v3](https://arxiv.org/abs/2305.15023)，CC BY-NC-SA 4.0 |
| 官方补充材料 | [NeurIPS Supplemental](https://proceedings.neurips.cc/paper_files/paper/2023/file/5e84e4413268b713f0d4a1b23a9dae57-Supplemental-Conference.pdf) |
| 官方代码 | [luogen1996/LaVIN](https://github.com/luogen1996/LaVIN) |

**为什么选它**：上一篇专题精读是推理侧的 LLaVA-Mini，本次按轮换回到训练侧。LaVIN 不是泛多模态模型，而是把参数高效微调作为核心问题，完整给出 Adapter 结构、模态路由、端到端训练、可训练参数量、墙钟时间、消融和失败案例；仓库中也没有出现过该标题、论文 URL 或方法主题。

## 问题背景：多模态适配为什么会变贵

论文把当时的多模态 LLM 方案概括为三类：

1. **专家系统**：LLM 解析指令，再调用 OCR、VQA、图像生成等视觉模型。它不必重训 LLM，但多个模型同时驻留，计算和存储都重。
2. **模块化训练**：在视觉编码器与 LLM 之间加入 Q-Former、投影层或其他 neck，先做大规模图文对齐，再做视觉指令微调。训练通常分阶段，数据和中间 checkpoint 成本较高。
3. **MMA**：冻结大主干，在视觉编码器和 LLM 中插入轻量 Adapter，用一批混合的文本/图文指令直接做端到端适配。

作者真正要解决的并不只是“少训练一些参数”。同一模型同时接收文本与图文指令时，两类输入需要的参数更新方向未必一致。一个统一 Adapter 可能为视觉任务让路，损伤原有语言能力；完全分开两套 Adapter 又需要确定当前输入走哪一条路径。MMA 因而同时设计了 **MM-Adapter** 和 **Mixture-of-Modality Training（MMT）**。

## 核心贡献

论文的贡献可以拆成三点：

1. **参数高效的多模态连接方式**：在冻结的 CLIP ViT 和 LLaMA 中插入轻量适配器，不再全量更新 7B/13B 主干。
2. **按输入模态路由 Adapter**：通过显式 modality token，在文本与图文指令之间选择不同的适配路径，减轻模态冲突。
3. **单阶段混合模态训练**：一个 mini-batch 同时采样 text-only 和 text-image 指令，只更新 Adapter，使视觉编码器与 LLM 的适配参数能联合优化。

论文结论是：LaVIN 以 3.8M/5.4M 可训练参数获得与早期全量微调多模态模型接近的 ScienceQA 表现，同时显著减少训练时间和 checkpoint 存储。我的判断是：这项工作的核心不是“低成本版 LLaVA”，而是用一个显式模态变量把共享主干与模态专用更新分开。

## 方法总览：冻结主干，只训练适配路径

![LaVIN 的 MMA 总体架构](/images/posts/lavin-mixture-of-modality-adaptation/lavin-neurips2023-figure2-mma-architecture.png)

*图源：Luo et al., [Cheap and Quick](https://arxiv.org/abs/2305.15023), Figure 2, NeurIPS 2023；从作者 CC BY-NC-SA 4.0 arXiv v3 源码中的原始矢量图直接栅格化，结构、标签和示例均未修改。原图用于论文解读。*

LaVIN 使用预训练 CLIP ViT-L/14 作为图像编码器，LLaMA-7B 或 LLaMA-13B 作为语言模型。ViT 每隔四层取一个 `[cls]` token，共得到 6 个视觉 token；轻量 visual adapter 将它们投影到 LLM 的隐藏维度。文本经 tokenizer 与 embedding 得到文本 token，二者与 modality token 拼接后进入 LLM。

图中的雪花表示冻结参数，火焰表示训练参数。视觉编码器和 LLM 的大矩阵保持不变，更新集中在：

- ViT 中插入的 Adapter；
- 视觉维度投影层；
- LLM 中的 MM-Adapter；
- modality embedding 和路由参数。

这仍然是端到端训练：损失会穿过 LLM、视觉 token 和图像编码器的计算图，只是梯度最终只更新这些小模块。

## MM-Adapter：两条支路如何路由

### 模态标记

作者用 one-hot 向量 $m\in\mathbb R^2$ 表示输入是 text-only 还是 text-image，并通过可学习表 $E_m$ 得到 modality token：

$$
t_m=mE_m,\qquad E_m\in\mathbb R^{2\times c}.
$$

对某层输入 $Z\in\mathbb R^{n\times c}$，MM-Adapter 的残差更新为：

$$
Z'=Z+s\cdot\operatorname{router}\left(f_{a_1}(Z),f_{a_2}(Z);f_w(t_m)\right),
$$

其中 $f_{a_1}$ 与 $f_{a_2}$ 是两条 RepAdapter 支路，二者共享降维投影以进一步省参数，$s$ 是缩放因子。

### 软路由

路由器先把 modality token 映射成两条支路的权重：

$$
\hat w=\operatorname{softmax}\left(\frac{t_mW_m+b_m}{\tau}\right),
$$

再做加权求和：

$$
\operatorname{router}\left(f_{a_1}(Z),f_{a_2}(Z)\right)
=\hat w_0f_{a_1}(Z)+\hat w_1f_{a_2}(Z).
$$

温度 $\tau$ 控制分布尖锐程度。论文为 7B/13B 模型分别设置 10/5，MM-Adapter bottleneck 维度为 8。

这里有一个容易被“动态路由”四个字掩盖的边界：$\hat w$ 只依赖 text-only/text-image 的 one-hot 模态标记，不读取当前图像或问题内容。因此它是**按模态动态切换**，不是按样本难度、语义或视觉内容做细粒度路由；同一层内，同一种模态的样本会得到相同类型的路由决策。

![LaVIN 在两种模态输入上的路由权重](/images/posts/lavin-mixture-of-modality-adaptation/lavin-neurips2023-supp-figure1-routing-paths.png)

*图源：Luo et al., [Cheap and Quick 官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/file/5e84e4413268b713f0d4a1b23a9dae57-Supplemental-Conference.pdf), Supplemental Figure 1, NeurIPS 2023；从作者 CC BY-NC-SA 4.0 arXiv v3 源码中的原始矢量图直接栅格化，10 层路由权重、箭头和示例均未修改。原图用于论文解读。*

补充材料展示了最后 10 层的权重。图文指令与纯文本指令大多选择相反支路，而且许多权重接近 0 或 1。它证明两类输入确实学出了不同路径；但图中只有各一个样本，也没有报告跨数据集的路由稳定性，因此不能据此断言支路已经形成可解释的“视觉专家”和“语言专家”。

## 视觉 token 与 LLM 输入

设 ViT 提取的多层 `[cls]` 特征为 $X\in\mathbb R^{n\times d}$，visual adapter 用一个窄瓶颈完成维度转换：

$$
X'=\sigma(XW_d+b_d)W_u+b_u,
$$

其中 $\sigma$ 为 SwiGLU，瓶颈维度 $d_h$ 远小于输入与 LLM 隐藏维度。最终 LLM 输入为：

$$
Z=
\begin{cases}
[t_m,X',Y], & \text{text-image},\\
[t_m,Y], & \text{text-only}.
\end{cases}
$$

与把全部 patch token 送入 LLM 的方案相比，LaVIN 只取 6 个分层 `[cls]` token，视觉 neck 默认维度也只有 128。这个选择降低了训练成本，但会牺牲局部文字、小物体和空间细节，后面的失败案例正好暴露了这一点。

## MMT：文本与图文指令一起训练

Mixture-of-Modality Training 冻结 $\phi$ 表示的视觉/语言主干，只优化适配参数 $\theta_a$。训练 batch 随机混合 text-only 与 text-image 样本，对回答序列做自回归最大似然；写成最小化负对数似然为：

$$
\mathcal L=-\sum_{i=1}^{B}\sum_{s=1}^{S_i}
\log p\left(R_s^i\mid Z^i,R_{<s}^i;\theta_a\right).
$$

它与普通视觉指令微调最大的区别是：两种模态从同一个训练入口进入，但通过 modality token 将更新分配到不同 Adapter 组合。共享主干保留通用能力，Adapter 吸收下游任务和模态差异。

## 训练与推理流程

### 训练

1. 图像经 CLIP ViT-L/14，抽取 6 个分层 `[cls]` token；文本经 LLaMA tokenizer 与 embedding。
2. 根据样本是否带图像选择 modality token，并构造 $[t_m,X',Y]$ 或 $[t_m,Y]$。
3. ViT Adapter、visual adapter 与 LLM MM-Adapter 参与前向，主干保持冻结。
4. 使用回答序列的自回归损失，只更新约 3.8M（7B）或 5.4M（13B）参数。

ScienceQA 训练使用 AdamW、20 个 epoch、batch size 32、学习率 $9\times10^{-3}$、weight decay 0.02 和 cosine decay。作者另报告 40 epoch 的 LaVIN-13B 最佳结果。解码采用 temperature 0.1、top-p 0.75。

多模态对话实验使用 Alpaca-52k 文本指令和 LLaVA 图文指令。论文内部这里存在一个小的不一致：引言写 152k text-image pairs，数据集小节与代码 README 写 158k；复现时应以代码实际数据清单为准，而不是混用两个数字。

### 推理

推理不再更新参数。纯文本请求跳过图像编码器并走文本路由；图文请求先编码图像，再把视觉 token 与文本 token 送入另一条路由组合。Adapter 增加少量前向计算，所以 LaVIN 的贡献是低成本训练与多模态适配，不是减少推理延迟。

## 实验设置与主要结果

ScienceQA 包含自然科学、社会科学和语言科学问题，训练/验证/测试集分别为 12,726/4,241/4,241 条，既有 text-only，也有 text-image 样本。论文主要指标是测试集平均准确率。

![LaVIN 的 ScienceQA 主结果与消融](/images/posts/lavin-mixture-of-modality-adaptation/lavin-neurips2023-tables1-2-scienceqa-ablation.png)

*图源：Luo et al., [Cheap and Quick 正式论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/5e84e4413268b713f0d4a1b23a9dae57-Paper-Conference.pdf), Tables 1-2, NeurIPS 2023；从正式 PDF 高分辨率裁取，两张表的列名、分组、数值、粗体和脚注均完整保留，未修改实验数据。作者 arXiv v3 采用 CC BY-NC-SA 4.0；原图用于论文解读。*

### ScienceQA

| 模型 | 可训练参数 | 平均准确率 |
| --- | ---: | ---: |
| LLaMA-Adapter | 1.8M | 85.19 |
| LaVIN-7B | 3.8M | 89.41 |
| LaVIN-13B（20 epochs） | 5.4M | 90.50 |
| LaVIN-13B（40 epochs） | 5.4M | **90.83** |
| LLaVA-13B | 13B | 90.92 |
| MM-CoT Large | 738M | **91.68** |

LaVIN-13B 与全量微调的 LLaVA-13B 基本持平，但没有超过当时的 MM-CoT Large。LaVIN-7B 相比 LLaMA-Adapter 提升 4.22 个点；40-epoch 13B 相比 LLaMA-Adapter 提升 5.64 个点。

论文 Table 3 的红色增量标记并不统一：LaVIN-7B 的 +4.22 以 LLaMA-Adapter 为基线，而 LaVIN-13B 的 +5.02 实际对应 $90.83-85.81$，即以 LLaVA 为基线；正文又写与 LLaMA-Adapter 相比 +5.64。本文因此只使用原始分数和明确写出的比较基线。

### COCO captioning 与零样本评估

在 COCO Karpathy test split 上，LaVIN-13B 不做图文预训练时为 36.4 BLEU-4/126.9 CIDEr；加入 0.6M 预训练图文对后为 37.8/131.7。它高于 LLaMA-Adapter V2 的 36.2/122.2，但仍低于使用 129M 图文对的 BLIP-2（43.7/145.3）。这更像是成本-质量折中，而不是全面取代大规模预训练。

零样本 TruthfulQA 上，LaVIN 为 47.9，LLaMA base 为 38.7，LLaVA 为 16.4；MME Cognition/Perception 上，LaVIN 为 963.6/249.6，BLIP-2 为 1293.8/290.0。结果支持“语言能力保留得更好”，但也说明视觉泛化仍落后于大规模预训练模型。

## 消融：收益来自哪里

Table 2 给出了逐步加模块的 ScienceQA 结果：

| 设置 | 可训练参数 | 平均准确率 | 相对 text-only 累计提升 |
| --- | ---: | ---: | ---: |
| Text only | 1.8M | 82.65 | 0.00 |
| + Vision modality（MMT） | 2.4M | 86.32 | +3.67 |
| + Joint optimization（MMT） | 2.5M | 87.34 | +4.69 |
| + 更强图像编码器 | 2.9M | 88.33 | +5.68 |
| + MM-Adapter | 3.8M | 89.41 | +6.76 |
| + 13B LLM | 5.4M | 90.50 | +7.85 |

最大的单步收益来自把视觉模态纳入 MMT（+3.67）；在冻结主干的前提下联合优化视觉侧与语言侧 Adapter 再增加 1.02。MM-Adapter 本身以额外 0.9M 参数带来 1.08 个点，说明路由有效，但它不是全部收益来源。

### 训练成本

论文 Table 6 在统一的 8×A100 设置下报告：

| 模型 | 可训练参数 | 峰值显存 | 墙钟时间 | checkpoint 存储 |
| --- | ---: | ---: | ---: | ---: |
| BLIP-2 | 188M | 未报告 | >200 小时 | 未报告 |
| LLaVA | 13B | OOM | N/A | N/A |
| LLaVA（显存优化） | 13B | 36.8G | 7 小时 | 26GB |
| LaVIN-7B | 3.8M | 33.9G | 1.4 小时 | 15M |
| LaVIN-13B | 5.4M | 55.9G | 2 小时 | 20M |

这里的 1.4/2 小时是 8 张 A100 上的墙钟时间，不是单卡 GPU-hours。原表把存储写成 15M/20M，却没有在列名中明确单位；结合参数规模可推测是 MB 级，但本文保留原表记法，不擅自补单位。论文声称相对 LLaVA 节省超过 99.9% 磁盘存储，这主要来自不再保存完整 LLM 更新参数。

## 失败案例与局限

![LaVIN 的官方失败案例](/images/posts/lavin-mixture-of-modality-adaptation/lavin-neurips2023-supp-figure5-failure-cases.png)

*图源：Luo et al., [Cheap and Quick 官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/file/5e84e4413268b713f0d4a1b23a9dae57-Supplemental-Conference.pdf), Supplemental Figure 5, NeurIPS 2023；从官方补充 PDF 高分辨率裁取，四组提示、回答、原图和 Figure 编号均保留。作者 arXiv v3 采用 CC BY-NC-SA 4.0；原图用于论文解读。*

论文正文与补充材料给出的边界很具体：

1. **复杂语言逻辑失败**：把 “think again” 直译为“再觉得一下”，古诗翻译也丢失原意，说明保住一般 NLP 分数不等于稳健理解复杂表达。
2. **事实幻觉**：对补办出生证明的问题，模型编造了从 National Archives 获取个人出生记录的说法。
3. **细粒度视觉不足**：作者明确指出 LaVIN 难以识别文字字符等细节；只取 6 个 `[cls]` token 的视觉瓶颈是可能原因之一。
4. **仍会生成错误或虚构回答**：这是作者在 limitation 章节直接承认的结论，并非本文推测。
5. **路由粒度有限**：路由只看模态标签，不能根据 OCR 难度、图像复杂度或问题类型自适应选择计算路径。
6. **对话证据偏定性**：多轮对话图由 GPT-4 打分，但论文没有给出大规模盲测、人工一致性或显著性分析，不能把案例图当作通用聊天能力证明。
7. **数据口径有小差异**：152k/158k 图文指令和 Table 3 增量基线不一致，会增加精确复现成本。

## 可复现资源

- [NeurIPS 正式论文与 BibTeX](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5e84e4413268b713f0d4a1b23a9dae57-Abstract-Conference.html)
- [NeurIPS 官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/file/5e84e4413268b713f0d4a1b23a9dae57-Supplemental-Conference.pdf)
- [arXiv v3 全文、源码与 CC BY-NC-SA 4.0 许可](https://arxiv.org/abs/2305.15023)
- [LaVIN 官方代码、训练脚本与 checkpoint 说明](https://github.com/luogen1996/LaVIN)
- [ScienceQA 数据集](https://github.com/lupantech/ScienceQA)

官方仓库提供 ScienceQA、多模态对话、4-bit 单卡训练和 MME 评测入口。论文 Table 6 的成本比较统一使用 8×A100；当前 README 的 7B 复现命令改为 2×A100，13B 命令仍使用 8×A100。因此，实际复现时间应按所用脚本与卡数记录，不能直接套用成本表的 1.4/2 小时。LaVIN-lite 可在单卡运行，但 7B/13B 分别约需 29/42 小时。

截至本文核验时，官方仓库根目录没有检测到 LICENSE 文件，GitHub API 也没有识别出代码许可证。论文图片可按 arXiv 的 CC BY-NC-SA 4.0 条款用于非商业解读，但代码的复制、修改和再发布需要单独向作者确认授权边界。

## 个人判断

LaVIN 最值得保留的工程思想，是把“共享能力”和“模态差异”放在不同参数层级：冻结的 ViT/LLaMA 负责通用表征，两条低秩瓶颈支路吸收文本与图文任务差异，一个极小的路由器决定组合方式。它比为每个任务保存一份 7B/13B 权重更适合多租户或多场景适配。

但这套设计也带有明显的 2023 年阶段性限制。视觉输入被压成 6 个 `[cls]` token，路由只依赖二值模态标签，评测集中在 ScienceQA、COCO、MME 与少量 GPT-4 打分案例。今天把它扩展到 OCR、文档、多图或视频任务时，更合理的方向是：保留参数高效 Adapter，但让路由同时读取问题、视觉复杂度和任务标签，并增加 token 级视觉表征，而不是沿用固定的 text/text-image 二分。

因此，我会把 LaVIN 定位为一篇把**多模态参数高效适配**讲得很完整的早期工作：它证明了只更新百万级参数也能把冻结 LLM 迁移到视觉指令任务，并用消融说明混合模态训练和路由各自有效；但它没有证明同样的压缩与路由粒度能覆盖细粒度视觉、多图推理或现代多模态基准。

## 参考资料

1. Luo et al. [Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5e84e4413268b713f0d4a1b23a9dae57-Abstract-Conference.html). NeurIPS 2023.
2. Luo et al. [arXiv:2305.15023 v3](https://arxiv.org/abs/2305.15023). CC BY-NC-SA 4.0.
3. Zhang et al. [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](https://arxiv.org/abs/2303.16199). 2023.
4. Liu et al. [Visual Instruction Tuning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html). NeurIPS 2023.
5. Li et al. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q.html). ICML 2023.
6. Lu et al. [Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering](https://proceedings.neurips.cc/paper/2022/hash/11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.html). NeurIPS 2022.

本文中的 4 张论文原图均用于论文解读，未修改实验数值、图例或标签；版权与再使用范围以作者 arXiv v3 页面标注的 CC BY-NC-SA 4.0 为准。
