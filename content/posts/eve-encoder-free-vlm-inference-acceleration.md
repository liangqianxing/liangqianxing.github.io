---
title: "EVE 精读：移除视觉编码器能加速多模态推理吗"
date: 2026-08-03 07:00:00
description: "拆解 NeurIPS 2024 Spotlight 论文 EVE：用无视觉编码器的 decoder-only 架构降低多模态视觉前处理开销，并核对其速度、精度、训练稳定性与端到端延迟边界，属于推理加速（推理侧）方向。"
series: 三大会论文精读
seriesOrder: 16
categories:
  - AI
tags:
  - 多模态大模型
  - 推理加速（推理侧）
  - 视觉编码器
  - 视觉语言模型
  - Decoder-only
  - 推理延迟
  - EVE
  - NeurIPS
hidden: true
haloPublished: true
draft: false
---

高分辨率图像进入多模态大模型时，视觉编码器往往需要先把图片 resize、切块，再与语言模型串联执行。这个视觉前处理在自回归解码很长时未必是总时延主项，但在首 token 响应、短答案或高分辨率部署里会变得显著。NeurIPS 2024 Spotlight 论文《Unveiling Encoder-Free Vision-Language Models》提出 EVE：不把 CLIP 等深视觉编码器放进线上推理路径，而是直接以轻量 Patch Embedding Layer（PEL）将图片 patch 接到一个 Vicuna-7B decoder-only 主干。

这篇论文属于本专题的 **推理加速（推理侧）**：它直接报告了多模态模型的视觉部分 FLOPs 与时延，并通过删除深度视觉编码器降低部署计算。标准 EVE-7B 相对 LLaVA-1.5，把表中视觉部分从 372 GFLOPs / 0.033 s 降到 42 GFLOPs / 0.003 s；不过两者表中的语言部分都仍为 15.2 TFLOPs / 0.48 s。也就是说，它有很强的**图像编码阶段**加速证据，却不能被概括成“每个请求端到端快 10 倍”。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **Unveiling Encoder-Free Vision-Language Models** |
| 作者 | Haiwen Diao、Yufeng Cui、Xiaotong Li、Yueze Wang、Huchuan Lu、Xinlong Wang |
| 会议 | NeurIPS 2024，Spotlight |
| 方法 | EVE（Encoder-free Vision-language modEl） |
| 专题方向 | **推理加速（推理侧）**：移除深视觉编码器，缩短视觉前处理与降低部署开销 |
| 官方论文 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5e2217482fa75556f1970be809acd3f8-Abstract-Conference.html) |
| 作者全文 | [arXiv:2406.11832 v2](https://arxiv.org/abs/2406.11832) |
| 官方代码与模型 | [baaivision/EVE](https://github.com/baaivision/EVE)，MIT License |

**为什么选它**：上一篇专题文章是 LaVIN，属于训练侧的多模态 Adapter 适配；本次按轮换回到推理侧。EVE 不只是泛称“高效”的 VLM，而是将视觉编码器的参数、FLOPs 与 latency 作为明确的部署问题，并在 Table 6 中给出可复核的推理侧数字。仓库现有文章、README 索引、本任务记录和附件候选文献中均没有出现 EVE 或该论文 URL。

## 问题背景：视觉编码器为什么会拖慢首段推理

典型 LLaVA 类模型的路径是：图片先经 CLIP/EVA 等视觉编码器，得到视觉 token；投影器把它们映射到 LLM 隐空间；最后 LLM 做自回归生成。高分辨率图片还经常被缩放、填充或切为多块，视觉编码器需重复处理。作者将问题归为三类：

1. 固定预训练尺寸/长宽比与真实图像不匹配，带来 resize、padding 或切图成本；
2. 视觉模型与语言模型串行部署，视觉编码器规模可从 0.4B 到 22B；
3. 两个独立大主干的容量如何匹配没有统一准则。

![编码器式与无编码器式 VLM 的对比](/images/posts/eve-encoder-free-vlm-inference-acceleration/eve-neurips2024-figure1-encoder-free-overview.png)

*图源：Diao et al., [Unveiling Encoder-Free Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5e2217482fa75556f1970be809acd3f8-Paper-Conference.pdf), Figure 1, NeurIPS 2024；从正式论文 PDF 等比例裁取，模块、标注和数据均未修改。作者官方 [代码仓库](https://github.com/baaivision/EVE) 声明项目内容为 MIT License；原图用于论文解读。*

EVE 的选择不是在现有 LLM 内部剪 token，而是改变输入侧架构：保留一个语言 decoder，使用较浅的 PEL 直接把图片 patch 表示放到语言 token 之前。这会避免一个深视觉主干在线运行，并天然允许任意长宽比；代价是模型必须在统一 decoder 内自己学到视觉表征，训练难度明显上升。

## 核心贡献

1. **无视觉编码器的 VLM 推理路径**：以 PEL 取代深预训练 vision encoder，图片与文本 token 一起进入同一个因果 decoder。
2. **LLM-guided pre-aligning**：先冻结 Vicuna-7B，只训练视觉输入/对齐小模块，让随机初始化的视觉侧先接上既有语言空间，避免随后大规模训练崩溃。
3. **Patch Aligning Layer（PAL）监督**：训练时用 CLIP-ViT-L/14 的 patch 特征做逐位置 MSE 蒸馏，并用多源 VLM 生成的文字标签做 next-token CE；PAL 在推理时删除。
4. **可核验的部署实验**：Table 6 分别报告视觉与 LLM 部分的 FLOPs、时延，给出了加速发生在哪个阶段的证据。

## 方法总览：浅 PEL 上线，深 PAL 只在训练时存在

![EVE 的整体架构、PEL 与 PAL](/images/posts/eve-encoder-free-vlm-inference-acceleration/eve-neurips2024-figures2-3-architecture.png)

*图源：Diao et al., [Unveiling Encoder-Free Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5e2217482fa75556f1970be809acd3f8-Paper-Conference.pdf), Figures 2-3, NeurIPS 2024；从正式论文 PDF 等比例裁取，图号、模块、箭头、符号和说明完整保留。作者官方 [代码仓库](https://github.com/baaivision/EVE) 声明项目内容为 MIT License；原图用于论文解读。*

给定任意尺寸图片 $I\in\mathbb{R}^{H\times W\times3}$，PEL 先以卷积得到二维特征，再在不交叠 slice 内做平均池化与局部 cross-attention（CA1）。一个 `<CLS>` 通过 CA2 汇聚所有 patch，行末插入可学习的 `<SPL>`，保留二维换行结构；最后经过两层 FFN 的视觉 token 与文本 embedding 拼接，送进 Vicuna-7B 的 causal decoder。

这个输入长度仍决定 LLM 的注意力成本。若视觉 token 数为 $N_v$、文本/生成上下文为 $N_t$，单层全注意力的常见复杂度可记为：

$$
\mathcal{O}\left((N_v+N_t)^2d\right).
$$

这不是论文给出的专门公式，而是解释其系统取舍的工程记号：EVE 没有魔法般消除后续 decoder 的视觉 token 成本。它把主要收益放在**不再执行深视觉编码器**，并以 PEL 的 pooling 和局部 CA1 控制前端成本；高分辨率下仍需警惕视觉 token 与自回归 KV cache 的增长。

### PAL：用离线教师弥补上线时没有编码器

训练阶段，PAL 从 EVE 的若干中间层取 patch feature，删除 `<CLS>/<SPL>`，reshape 回二维、adaptive pool 到教师空间，再由 CA3 将多层特征汇合；它与冻结的 CLIP-ViT-L/14 对应 patch 特征做 $\ell_2$ 归一化后的 MSE。另一路以合成 caption/问答文本做 CE。

可以把训练目标概括为：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{CE}}(y,\hat y)+\lambda\mathcal{L}_{\mathrm{MSE}}(\operatorname{norm}(z_{\mathrm{EVE}}),\operatorname{norm}(z_{\mathrm{VE}})).
$$

论文明确采用这两个目标，但没有给出上式中的统一 $\lambda$ 写法；这里仅用它说明监督来源，不能把它当作作者报告的精确超参数。关键部署点是：**PAL 和 CLIP 教师仅用于训练，推理时都会删除**。

## 三阶段训练与推理流程

### 训练

1. **Stage 1，LLM-guided pre-aligning**：冻结 Vicuna-7B，只训练 PEL/PAL；从 33M 重述后的公开图文对中使用 16M，建立视觉到语言的初始连接。
2. **Stage 2，generative pre-training**：解冻 PEL、PAL 与完整 LLM，在 33M 图文对上继续用 CE + patch MSE。
3. **Stage 3，SFT**：标准 EVE-7B 用 LLaVA-mix-665K；EVE-HD 再引入合计 1.2M 的 OCR、文档、图表等指令数据。论文报告标准 EVE-7B 在两台 8xA100 40GB 节点上约训练 9 天。

训练中实际用到 CLIP 教师，不等于线上也要运行它。这个区分很重要：EVE 用较重的**离线训练**换取较轻的**在线视觉路径**，并不是训练与推理都无视觉编码器。

### 推理

1. 输入保持原始长宽比；PEL 用卷积、池化、局部 CA 形成 patch token、`<CLS>` 与每行 `<SPL>`。
2. 视觉 token 与 prompt token 拼接，进入单个 Vicuna-7B causal decoder。
3. 仅保留 PEL + decoder；PAL 和 CLIP-ViT-L/14 监督分支被移除。
4. decoder 按自回归方式生成答案，常规的 KV cache、batching、量化或 speculative decoding 仍是独立的系统优化空间，EVE 没有替代它们。

## 实验设置与主要结果

论文以 Vicuna-7B 为语言主干。标准版最长边为 672，HD 版为 1344；PEL 卷积 stride 14、average pooling stride 2，所有 cross-attention 头数为 8，PAL 的跨层间隔因子为 4。评测覆盖 VQAv2、GQA、VizWiz、ScienceQA-IMG、TextVQA、POPE、MME、MMBench、SEED-Bench 与 MM-Vet。

### 能力：HD 版接近部分 encoder-based 基线，但并非全面领先

Table 3 中，EVE-7B (HD) 在 VQAv2/GQA/VizWiz/ScienceQA-IMG/TextVQA 分别为 78.6/62.6/51.1/64.9/56.8；标准 EVE-7B 为 75.4/60.8/41.8/63.0/51.9。与论文列出的 LLaVA-1.5（78.5/62.0/50.0/66.8/58.2）相比，HD 版在前三项接近或略高，但 ScienceQA-IMG 与 TextVQA 仍低。

作者也明确写出标准 EVE 在 MME、MMBench、ScienceQA-IMG、MM-Vet 较弱，原因包括只用视觉语言数据会损伤语言能力、对选项字母/二值指令不够可靠。因而“能竞争”应理解为若干指标上的成本-性能折中，不是取代所有 encoder-based VLM 的结论。

### 效率：加速主要发生在视觉前端

![缩放趋势、视觉监督消融与部署效率表](/images/posts/eve-encoder-free-vlm-inference-acceleration/eve-neurips2024-figures7-8-table6-efficiency.png)

*图源：Diao et al., [Unveiling Encoder-Free Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5e2217482fa75556f1970be809acd3f8-Paper-Conference.pdf), Figures 7-8 and Table 6, NeurIPS 2024；从正式论文 PDF 等比例裁取，曲线、图例、坐标轴、表头和数值均未修改。作者官方 [代码仓库](https://github.com/baaivision/EVE) 声明项目内容为 MIT License；原图用于论文解读。*

论文 Table 6 的原始数值如下。FLOPs 与 time 分别按视觉部分、LLM 部分单列，不能混成单一的端到端加速率。

| 模型 | 视觉 FLOPs | 视觉 time | LLM FLOPs | LLM time |
| --- | ---: | ---: | ---: | ---: |
| LLaVA-1.5 | 372 G | 0.033 s | 15.2 T | 0.48 s |
| EVE-7B | 42 G | 0.003 s | 15.2 T | 0.48 s |
| LLaVA-1.6 (HD) | 1,860 G | 0.13 s | 76.1 T | 2.07 s |
| EVE-7B (HD) | 170 G | 0.013 s | 60.8 T | 1.52 s |

标准分辨率下，视觉部分 FLOPs 从 372 G 到 42 G，约减少 88.7%；视觉编码 latency 从 33 ms 到 3 ms，约减少 90.9%。但 LLM 部分仍是 0.48 s，所以把表中两个阶段简单相加，也只会从约 0.513 s 降到约 0.483 s，约 5.8%。HD 对比中，视觉部分约减少 90.9% FLOPs、90.0% time；表中的 LLM 部分也从 2.07 s 降到 1.52 s。论文没有交代 Table 6 的硬件、batch size、输出 token 数、warm-up 或统计口径，工程选型前必须在目标服务栈复测这些端到端数值。

## 消融：真正的稳定器是 Stage 1，不是把教师越训越久

![PEL/PAL、三阶段流程与跨层间隔的消融](/images/posts/eve-encoder-free-vlm-inference-acceleration/eve-neurips2024-figures5-6-ablation.png)

*图源：Diao et al., [Unveiling Encoder-Free Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5e2217482fa75556f1970be809acd3f8-Paper-Conference.pdf), Tables 4-5 and Figures 5-6, NeurIPS 2024；从正式论文 PDF 等比例裁取，表头、数值、曲线、坐标轴与图例完整保留，未修改实验数据。作者官方 [代码仓库](https://github.com/baaivision/EVE) 声明项目内容为 MIT License；原图用于论文解读。*

Table 5 与 Figure 5 是最有工程价值的一组消融：没有 Stage 1 时，继续 Stage 2 的 4M 数据能暂时上升，但扩到 8M 后 VQAv2/GQA/MMBench/SEED 从 64.6/54.1/40.6/45.4 掉到 50.2/42.5/26.8/36.2，并伴随 loss collapse。加入 Stage 1 后，同样的 4M、8M 则稳定升到 69.4/56.5/42.0/48.7 和 71.2/58.9/44.0/50.3。

PAL 也有帮助，但收益随数据变大而变弱：在 4M 预训练时，去掉 PAL 的 VQAv2/GQA 从 69.4/56.5 降到 66.4/55.3；8M 时从 71.2/58.9 降到 69.4/57.3。Figure 8 报告到 24M 时，有无视觉教师监督的差约 0.3-0.8 个点。作者的实验结论是，冻结 LLM 的 Stage 1 才是防崩溃和可扩展训练的关键，PAL 更像有限数据下的视觉表征助推器。

## 失败案例与局限

1. **端到端加速被 LLM 解码掩盖**：标准设置中视觉部分虽快一个数量级，LLM 0.48 s 不变。输出长、prompt 长或 batch 大时，EVE 不能替代 attention kernel、KV cache、量化与服务调度优化。
2. **语言遗忘与指令脆弱性**：作者报告只用视觉语言数据训练时，ScienceQA-IMG 从 65.3% 降到 63.0%，并指出 option letter、二元问答易出错。
3. **更高训练换更轻部署**：标准版需约 33M 公开图文样本、两台 8xA100 节点约 9 天；这是部署预算与预训练预算之间的转移，并非免费加速。
4. **对比不完全可归因**：HD 配置除架构外还改变了分辨率、SFT 数据量等，Table 6 又未给出完整计时协议；不能把差异全归因于“删除视觉编码器”。
5. **定性案例仍会失真**：附录 Figure 9 中，EVE 对镜中自拍的描述漏掉或误判多个细节；HD 版更详细，但第二个 OCR 样例仍反复生成文字。这与其保留几乎无损 patch、但尚未完全掌握细粒度视觉语言对齐的张力一致。

## 可复现资源

- [NeurIPS 正式论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5e2217482fa75556f1970be809acd3f8-Paper-Conference.pdf) 与 [arXiv 源码](https://arxiv.org/e-print/2406.11832)
- [作者 EVE 仓库](https://github.com/baaivision/EVE)，MIT License；提供训练、评测、demo 与 checkpoint 说明
- [EVE-7B](https://huggingface.co/BAAI/EVE-7B-v1.0) 和 [EVE-7B-HD](https://huggingface.co/BAAI/EVE-7B-HD-v1.0) 模型页；使用时还须遵守底座 Vicuna/Llama 2 的许可证

复现建议先固定论文的 PEL stride 14、pooling stride 2、CA head 8、PAL interval 4、最长边 672/1344 和三阶段 learning rate（$4\times10^{-4}$、$4\times10^{-5}$、$2\times10^{-5}$）。测速则应将 image encoding、prefill、decode 分开记录，同时报告硬件、精度、batch、输入尺寸、输出长度、warm-up 次数和 p50/p95；否则很容易把视觉阶段的局部收益误标为总吞吐收益。

## 我的判断

EVE 最有启发性的地方，不是“视觉编码器必然应该删除”，而是把多模态推理成本拆开：一个深 encoder 的首段计算，与一个 decoder 的 token 生成，是两种完全不同的瓶颈。它针对前者给出结构级答案，并用训练期教师让轻量上线路径继承视觉归纳偏置。

对短回答、高分辨率、首 token 敏感的产品，这种设计值得测；对长上下文、多轮生成或大量视频帧的服务，decoder attention 和 KV memory 可能仍是主导成本。更务实的路线是把 EVE 视为视觉前端的一个选项，再与视觉 token 压缩、量化、paged KV cache、continuous batching 等互补技术组合，而不是把它当成完整的多模态服务优化栈。

## 参考资料

1. Diao et al., [Unveiling Encoder-Free Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5e2217482fa75556f1970be809acd3f8-Abstract-Conference.html), NeurIPS 2024 Spotlight.
2. Diao et al., [arXiv:2406.11832 v2](https://arxiv.org/abs/2406.11832)，公开全文、源文件与版本记录。
3. [BAAI-Vision/EVE](https://github.com/baaivision/EVE)，官方实现、模型和 MIT License。
4. Liu et al., [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)，LLaVA-1.5 对照基线。
