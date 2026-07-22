---
title: "QLoRA 精读：4-bit 量化如何把 65B 微调压进单卡"
date: 2026-07-18 09:15:00
description: 从 NF4、双重量化、Paged Optimizer 与全层 LoRA 出发，拆解 QLoRA 的显存账本、训练流程、实验边界和复现要点。
series: 三大会论文精读
seriesOrder: 1
categories:
  - AI
tags:
  - LLM
  - QLoRA
  - LoRA
  - 模型量化
  - 参数高效微调
  - NeurIPS
hidden: true
haloPublished: true
---

QLoRA 解决的问题很具体：**在不更新大模型主体权重的前提下，能否让梯度穿过一个 4-bit 量化模型，只训练少量 LoRA 参数，并尽量保持 16-bit 微调的效果？**

论文给出的答案是肯定的。它把 65B LLaMA 的微调显存从论文估算的 780 GB 以上降到 48 GB 以内，并把量化、参数高效微调和统一内存分页组合成一套可以实际运行的训练方案。

但这篇论文也很容易被过度概括。它没有证明“任何 4-bit 微调都等价于全量微调”，也没有在 33B 和 65B 上直接完成 16-bit 全量微调对照。本文会把论文验证过的结论、作者的推断和我的工程判断分开。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **QLoRA: Efficient Finetuning of Quantized LLMs** |
| 作者 | Tim Dettmers、Artidoro Pagnoni、Ari Holtzman、Luke Zettlemoyer |
| 会议 | NeurIPS 2023 |
| 主题 | 低比特量化、LoRA、指令微调、显存优化 |
| 官方论文 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html) |
| 扩展版与许可 | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)，CC BY 4.0 |
| 官方代码 | [artidoro/qlora](https://github.com/artidoro/qlora) |

**选题理由**：QLoRA 不只是一个节省显存的技巧。它把数据类型设计、量化尺度、低秩适配器、显存峰值管理和训练评估放进同一条工程链路，至今仍是理解低成本 LLM 微调的重要入口。

## 问题背景：LoRA 仍然背着一个大模型

全量微调需要为每个参数保存至少三类大对象：模型权重、梯度和优化器状态。以 Adam 为例，优化器还要维护一阶、二阶矩。即使参数采用 16-bit，优化器状态常常仍以更高精度保存，因此总显存远大于“参数量乘以 2 字节”。论文估算 65B LLaMA 的常规 16-bit 微调需要超过 780 GB GPU 显存。

LoRA 冻结预训练权重 $W$，只训练一个低秩增量：

$$
Y = XW + sXL_1L_2
$$

其中 $L_1 \in \mathbb{R}^{h\times r}$、$L_2 \in \mathbb{R}^{r\times o}$，且 $r$ 远小于原矩阵维度。LoRA 大幅减少了可训练参数、参数梯度和优化器状态，但有两个成本仍然存在：

1. 冻结的基础模型仍要常驻显存；
2. 反向传播仍需经过基础模型，激活和输入梯度不会凭空消失。

论文给出的一个 7B 示例很说明问题：常见的 LoRA 权重约占原模型参数的 0.2%，其参数只占 26 MB，但 LoRA 输入梯度可占 567 MB；启用 gradient checkpointing 后，平均每条序列仍约有 18 MB 输入梯度，而 4-bit 基础模型本身约占 5,048 MB。继续削减 LoRA 参数的收益已经很小，真正的大头是基础模型和激活。

![全量微调、LoRA 与 QLoRA 的显存和梯度路径对比](/images/posts/qlora-efficient-finetuning/qlora-method.png)

*图源：Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html), Figure 1, NeurIPS 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2305.14314)。原图用于论文解读。*

## 核心贡献

论文的核心不是“4-bit + LoRA”这六个字，而是五个互相配合的设计：

1. **4-bit NormalFloat（NF4）**：针对近似零均值正态分布的预训练权重设计非均匀量化码本。
2. **Double Quantization（双重量化）**：再次量化第一层量化的尺度常数，平均再节省 0.373 bit/parameter。
3. **Paged Optimizers**：借助 NVIDIA Unified Memory，在显存峰值出现时把优化器状态分页到 CPU 内存。
4. **BF16 计算路径**：权重以 NF4 存储，参与矩阵乘时解量化到 BF16；LoRA 参数也以 BF16 训练。
5. **全线性层 LoRA**：适配器覆盖所有 Transformer 线性层，而不只放在 attention 的 query/value 投影上。

前三项主要解决显存，后两项主要保证训练质量。把它们拆开看，QLoRA 的关键工程思想是：**低精度负责存储，高精度负责计算，只有小规模增量参数负责学习。**

## 方法总览

### 1. 分块量化：先控制离群值影响

若直接对整个张量使用同一个绝对最大值做缩放，一个离群值就可能拉大量化范围，使其余数值只能挤在少数几个量化桶中。分块量化把张量展平后切成大小为 $B$ 的连续块，每块独立计算量化常数 $c_i$。

以 FP32 到 Int8 为例：

$$
X^{\mathrm{Int8}}
= \operatorname{round}\left(
\frac{127}{\operatorname{absmax}(X^{\mathrm{FP32}})}X^{\mathrm{FP32}}
\right)
$$

块越小，局部尺度通常越准确，但量化常数数量也越多。QLoRA 对基础权重使用 block size 64，这正是后续双重量化要处理的开销来源。

### 2. NF4：让 16 个码点适配正态权重

均匀 4-bit 量化把区间切成等宽桶，但预训练权重并不均匀分布。NF4 从标准正态分布 $N(0,1)$ 的分位点构造 16 个码值，使各量化桶在理论分布下拥有近似相同的概率质量。

论文把第 $i$ 个代表值写成相邻分位点的中点：

$$
q_i = \frac{1}{2}\left[
Q_X\left(\frac{i}{2^k+1}\right)
+ Q_X\left(\frac{i+1}{2^k+1}\right)
\right]
$$

其中 $Q_X$ 是标准正态分布的分位函数。实际 NF4 还做了两项处理：

- 将码本归一化到 $[-1,1]$；
- 分别构造正负区间并合并，保留一个精确的零点。

这里的“信息论最优”有前提：权重需要近似零均值正态分布，并以论文定义的分位量化目标衡量。附录对 LLaMA-7B 做 Shapiro-Wilk 检验，约 7.5% 的神经元权重被判为非正态，而 5% 显著性水平本就预期约 5% 假阳性。这个结果支持近似假设，但不是所有层、所有模型都严格服从正态分布的证明。

### 3. 双重量化：量化“量化常数”

第一次量化后，每 64 个权重需要一个 32-bit 常数。摊到单个参数上，这部分成本是：

$$
\frac{32}{64}=0.5\ \text{bit/parameter}
$$

QLoRA 再用 8-bit 浮点量化这些常数，并以 256 为第二层 block size：

$$
\frac{8}{64}+\frac{32}{64\times256}
=0.127\ \text{bit/parameter}
$$

因此平均节省：

$$
0.5-0.127=0.373\ \text{bit/parameter}
$$

对 65B 参数模型，这约等于 3 GB。实现时，第一层量化常数均为正数，论文先减去其均值，使第二层量化的数据以零为中心。

### 4. Paged Optimizer：处理峰值，不是压缩所有状态

Gradient checkpointing 通过重算激活降低常态显存，但长序列或特殊 batch 仍可能产生短时峰值。Paged Optimizer 把优化器状态分配为 NVIDIA Unified Memory；当 GPU 显存紧张时，系统把页面迁移到 CPU RAM，需要更新参数时再迁回。

要注意，Paged Optimizer 主要解决**偶发峰值导致的 OOM**，不是把所有训练计算免费搬到 CPU。论文没有给出系统性的分页吞吐曲线，只报告 65B、48 GB GPU、batch size 16 的测试中，paged 和普通 optimizer 训练速度相同，并明确把分页在何种条件下变慢留给后续工作。

### 5. 全层 LoRA：覆盖位置比 rank 更关键

原始 LoRA 常只作用于注意力的 query/value 投影。QLoRA 的消融显示，在 LLaMA-7B Alpaca 设置中，只放 attention 或 FFN 的 LoRA 都无法追平强 16-bit 基线；把 LoRA 放到所有线性层后，4-bit 结果可以匹配该基线。相反，只要覆盖全层，$r$ 在论文搜索的 $8$ 到 $256$ 范围内与最终性能没有明显关系。

![LoRA 覆盖层位置的消融结果](/images/posts/qlora-efficient-finetuning/lora-layers-ablation.png)

*图源：Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html), Figure 2, NeurIPS 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2305.14314)。原图用于论文解读。*

**论文结论**：全层覆盖是恢复 16-bit 基线性能的关键超参数，LoRA rank 在覆盖充分后不敏感。

**我的判断**：这说明“可训练参数占比越低越好”不是合理目标。QLoRA 已经把主要显存花在基础权重和激活上，多放一些 LoRA 模块的边际显存很小，却能显著增加更新路径的表达能力。

## 关键公式：一次 QLoRA 线性层如何计算

论文把带双重量化和 LoRA 的线性层写成：

$$
Y^{\mathrm{BF16}}
=X^{\mathrm{BF16}}
\operatorname{doubleDequant}
(c_1^{\mathrm{FP32}},c_2^{\mathrm{kbit}},W^{\mathrm{NF4}})
+X^{\mathrm{BF16}}L_1^{\mathrm{BF16}}L_2^{\mathrm{BF16}}
$$

双重解量化为：

$$
\operatorname{doubleDequant}(c_1,c_2,W)
=\operatorname{dequant}
(\operatorname{dequant}(c_1,c_2),W)
$$

数据类型分工如下：

| 对象 | 数据类型 | 是否更新 |
| --- | --- | --- |
| 基础权重 $W$ | NF4 存储，计算时解量化到 BF16 | 否 |
| 第一层量化常数 | FP8 存储 | 否 |
| 第二层量化常数 | FP32 | 否 |
| LoRA 参数 $L_1,L_2$ | BF16 | 是 |
| 前向与反向矩阵乘 | BF16 | 参与梯度计算 |

基础权重虽然冻结，梯度仍要穿过其 BF16 解量化结果，才能得到输入梯度并继续传给更早的 LoRA 模块；只是不计算和保存 $W$ 的参数梯度。

## 训练与推理流程

### 训练

1. 以 block size 64 把预训练权重量化为 NF4。
2. 对第一层量化常数做均值中心化，再以 FP8、block size 256 做第二次量化。
3. 在基础模型的所有线性层插入 BF16 LoRA 模块。
4. 前向时按需把 NF4 权重双重解量化到 BF16，完成基础分支和 LoRA 分支计算。
5. 反向传播经过基础权重，但只为 LoRA 参数累积梯度。
6. 优化器只更新 LoRA 参数；出现显存峰值时，Paged Optimizer 可将优化器页面迁移到 CPU。

论文复现 Guanaco 时的共同设置包括 NF4、double quantization、BF16 计算、$r=64$、$\alpha=16$，并在所有线性层加入 LoRA。7B/13B 使用 0.1 LoRA dropout，33B/65B 使用 0.05；优化器设置包含 Adam $\beta_2=0.999$、max grad norm 0.3 和常数学习率。

### 推理

推理不需要参数梯度、优化器状态或分页。运行时保留 4-bit 基础权重和训练后的 LoRA 参数，前向时仍按计算内核要求完成解量化与矩阵乘。论文重点评估的是训练可行性，没有系统比较 adapter 合并、推理 kernel 或端到端服务延迟，因此不能从本文结果直接推出 QLoRA 一定提升推理速度。

## 实验设置与主要结果

论文一共训练了 1,000 多个模型，覆盖 RoBERTa、T5、LLaMA，规模从 80M 到 65B。主要实验可以分成三层。

### 1. 量化数据类型本身

作者先做训练前的量化评估，覆盖 OPT、BLOOM、LLaMA、Pythia 的 125M 到 65B 模型。Pile Common Crawl 的聚合困惑度为：

| 数据类型 | Mean PPL，越低越好 |
| --- | ---: |
| Int4 | 34.34 |
| FP4 E2M1 | 31.07 |
| FP4 E3M0 | 29.48 |
| NF4 + DQ | **27.41** |

![NF4、FP4 与双重量化的零样本结果](/images/posts/qlora-efficient-finetuning/nf4-results.png)

*图源：Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html), Figure 3, NeurIPS 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2305.14314)。原图用于论文解读。*

Figure 3 的横轴是模型总 bit 数，不是单纯参数量；它展示的是量化后 LLaMA 的零样本准确率。NF4 和 NF4 + DQ 都明显高于同规模 FP4，双重量化没有表现出明显精度损失。

### 2. QLoRA 能否匹配 16-bit 微调

论文在两个层级做比较：

- RoBERTa-large 与 T5 80M 到 11B：比较全量 BF16、BF16 LoRA、Int8/FP4/NF4 QLoRA，在 GLUE 和 Super-NaturalInstructions 上结果接近。
- LLaMA 7B、13B、33B、65B：比较 BF16 LoRA、FP4 LoRA、NF4 + DQ LoRA，在 Alpaca/FLAN v2 微调后做 5-shot MMLU。

LLaMA 八个“模型规模 × 数据集”设置的平均 MMLU 为：BF16 LoRA 53.0、FP4 52.2、NF4 + DQ 53.1。这个结果支持“NF4 QLoRA 匹配 BF16 LoRA”，也显示 FP4 平均落后约 0.8 个百分点。

需要严格区分：

- **论文直接建立的证据**：较小模型上，QLoRA 接近 16-bit 全量微调；7B 到 65B 上，QLoRA 接近 16-bit LoRA。
- **论文没有直接建立的证据**：33B/65B 上 QLoRA 与 16-bit 全量微调等价。作者在 Limitations 中明确承认没有做这组昂贵对照。

### 3. 显存与大模型可训练性

附录按 batch size 1、sequence length 512、gradient checkpointing 估算显存构成。7B、13B、33B、65B 的总占用分别约为 6.9、11.3、24.7、45.0 GB；33B 略超 24 GB，正是 Paged Optimizer 用来跨过峰值的场景。

![不同 LLaMA 规模的 QLoRA 显存构成](/images/posts/qlora-efficient-finetuning/memory-breakdown.png)

*图源：Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Supplemental-Conference.pdf), Supplemental Figure 6, NeurIPS 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2305.14314)。原图用于论文解读。*

图中蓝色基础模型权重始终是最大项：7B 为 5,046 MB，65B 为 37,074 MB。LoRA adapter 很薄，但 optimizer 和输入梯度随模型增大仍不可忽略。这也解释了为什么 QLoRA 不能把训练显存简单算成“参数量 × 0.5 字节”。

### 4. Guanaco 与聊天机器人评估

作者在 8 个指令数据集上训练 Guanaco，并在 80 条 Vicuna prompts 与 953 条 OpenAssistant prompts 上用 GPT-4 和人类评分。最常被引用的“99.3% of ChatGPT”来自 GPT-4 对 80 条 Vicuna prompts 的相对打分：Guanaco-65B 相对 ChatGPT 的均值为 99.3%，95% 置信区间为 4.4%。

这不是“综合能力达到 ChatGPT 的 99.3%”。论文自己发现：

- GPT-4 对先出现的回答有明显顺序偏置；
- GPT-4 与人类在样本级的一致性只有 Fleiss $\kappa=0.25$；
- 不同基准会改变 ChatGPT 与 Guanaco 的相对排序；
- MMLU 强不代表聊天能力强，反之亦然。

因此，这部分更重要的结论不是一个百分比，而是**评估协议本身不稳定，数据与目标任务的匹配度会强烈左右结论**。

## 消融分析

论文中最有工程价值的消融可以压缩成四条：

| 消融 | 论文观察 | 工程含义 |
| --- | --- | --- |
| LoRA 覆盖位置 | all linear layers 才能匹配强 16-bit 基线 | 优先保证覆盖，不要只盯参数比例 |
| LoRA rank | 全层覆盖后，$r=8\sim256$ 无明显单调收益 | rank 不是首要旋钮 |
| NF4 vs FP4 | NF4 在困惑度、零样本与微调结果上更稳 | 4-bit 不是一种单一精度，码本设计很重要 |
| Double Quantization | 额外省 0.373 bit/parameter，未观察到性能下降 | 对 33B/65B 的“最后几 GB”很关键 |

指令微调还有两个有意思的结果：

1. 7B 模型只在回答部分计算损失时，四个数据集的平均 MMLU 为 38.6；对 instruction 和 response 都计算损失时为 37.5。
2. 在 Chip2、Unnatural Instructions、FLAN v2 上，数据量和训练 epoch 带来的 MMLU 变化多为 0.0 到 0.5，而数据集之间差异达到 1.5 到 8.0。作者据此强调数据适配性与质量比单纯规模更重要。

## 失败案例与局限

### 论文展示的失败案例

Guanaco-65B 的 “lemon-picked” 样例包括：

- 对冷门事实给出自信但错误的实体和年份；
- 算术过程前后矛盾，甚至在同一回答中给出两个错误结论；
- 被简单提示注入诱导后泄露“secret word”；
- 对无害的字符串反转请求随机拒绝；
- 在 Theory of Mind 场景中假设并不存在的信息传递。

这些案例说明，低成本微调可以改善指令风格，但不会自动修复基础模型的事实性、数学能力、安全边界或因果推理。

### 作者明确列出的限制

1. 没有在 33B/65B 上完成 QLoRA 与 16-bit 全量微调的直接对照。
2. 指令模型只评估了 MMLU、Vicuna 和 OA，没有覆盖 BigBench、RAFT、HELM 等更广测试。
3. 责任 AI 评估只做了有限的 CrowS 偏见测试，不能代表完整安全性。
4. 没有系统探索 3-bit 等更低精度，也没有比较其他 PEFT 方法。
5. Paged Optimizer 缺少完整的吞吐、迁移频率和硬件条件分析。

### 我的补充判断

论文的最大长期价值是训练机制，不是 Guanaco 的排行榜名次。聊天模型与评估器都来自 2023 年，绝对排名早已过时；NF4、双重量化、低精度存储与高精度计算分离、全层 adapter 覆盖这些设计仍然直接影响今天的微调栈。

## 可复现资源

- [NeurIPS 官方论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)
- [NeurIPS 官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Supplemental-Conference.pdf)
- [QLoRA 官方代码](https://github.com/artidoro/qlora)，MIT License
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)，提供 k-bit 量化与相关算子
- [PEFT](https://github.com/huggingface/peft)，LoRA 等参数高效微调实现
- [Guanaco 训练数据](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

复现时优先核对这些参数：`load_in_4bit`、NF4 quant type、double quant、BF16 compute dtype、LoRA target modules、gradient checkpointing、序列长度和真实 global batch size。论文仓库提醒其训练参数名与实际全局 batch 语义存在历史差异，不能只照抄字段名。

## 个人判断

我认为 QLoRA 最值得记住的不是“单卡微调 65B”，而是三条更通用的系统原则：

1. **优化存储不等于降低计算精度**：权重可以 4-bit 存储，关键矩阵乘仍在 BF16 中完成。
2. **节省大项后，要继续处理元数据和峰值**：量化常数的 0.5 bit/parameter、偶发激活峰值，在 65B 规模都变成 GB 级问题。
3. **参数高效不应等同于参数越少越好**：当 adapter 已不是显存主项时，扩大覆盖范围比继续压 rank 更有价值。

QLoRA 也给工程读者一个很好的警示：论文中的“匹配 16-bit”必须追问匹配的是全量微调还是 LoRA、在哪些模型规模、哪些数据集和哪些指标上成立。只有把这四个限定补齐，结论才可迁移到自己的训练任务。

## 参考资料

1. Dettmers et al. [QLoRA: Efficient Finetuning of Quantized LLMs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html). NeurIPS 2023.
2. Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
3. Dettmers et al. [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html). NeurIPS 2022.
4. Touvron et al. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971). 2023.
