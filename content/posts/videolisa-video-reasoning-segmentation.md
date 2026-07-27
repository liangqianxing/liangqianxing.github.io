---
title: "VideoLISA 精读：多模态模型的训练侧视频分割适配"
date: 2026-07-27 16:28:22
description: 从 LoRA 适配、稀疏-稠密视频采样与 One-Token-Seg-All 训练出发，拆解 NeurIPS 2024 的 VideoLISA 如何把多模态大模型迁移到视频推理分割，属于个性化（训练侧）方向。
series: 三大会论文精读
seriesOrder: 11
categories:
  - AI
tags:
  - 多模态大模型
  - 个性化（训练侧）
  - 参数高效微调
  - Reasoning Segmentation
  - 视频分割
  - VideoLISA
  - NeurIPS
hidden: true
haloPublished: true
---

用户说“找出跑得更快的物体”或“把游戏输掉的孩子分割出来”，模型不能只做逐帧目标检测：它既要理解整段视频的运动变化，也要用常识或规则推断语言真正指向谁，最后还要输出跨帧一致的像素掩码。

VideoLISA 把图像推理分割模型 LISA 扩展到视频。它用稀疏-稠密采样把 32 帧的时间上下文压进多模态大模型，再让同一个 `<TRK>` token 提示 SAM 解码器处理整段视频。训练时，这个 token 被同时监督多个帧，因此不只记住某一帧的位置，而要学习可跨帧匹配的目标语义。

这篇论文直接属于本专题的 **个性化（训练侧）** 方向：作者以 LLaVA-Phi-3-V 为多模态骨干，用 LoRA 和分割监督把通用图文模型适配到视频 reasoning segmentation、referring VOS 和运动理解任务。它也涉及视觉 token 压缩，但论文自己明确承认总体计算成本仍高，因此不应把它归为推理加速成果。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos** |
| 作者 | Zechen Bai、Tong He、Haiyang Mei、Pichao Wang、Ziteng Gao、Joya Chen、Lei Liu、Zheng Zhang、Mike Zheng Shou |
| 会议 | NeurIPS 2024 |
| 方法名 | VideoLISA |
| 专题子方向 | 个性化（训练侧）：多模态大模型任务适配、LoRA、视频推理分割与知识迁移 |
| 正式论文 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0cf3e7eefb9d643e93e16ff1d94090a7-Abstract-Conference.html)，DOI 10.52202/079017-0219 |
| 作者全文与许可 | [arXiv:2409.19603](https://arxiv.org/abs/2409.19603)，CC BY-NC-SA 4.0 |
| 代码与训练说明 | [showlab/VideoLISA](https://github.com/showlab/VideoLISA)，Apache 2.0 |
| 模型权重 | [VideoLISA-3.8B](https://huggingface.co/ZechenBai/VideoLISA-3.8B) |
| 新基准 | [ReasonVOS](https://github.com/showlab/VideoLISA/blob/main/BENCHMARK.md) |

**选择理由**：最近的专题文章已覆盖视觉 token 推理加速，今天切回训练侧。VideoLISA 位于附件的优先文献池，直接研究多模态大模型向视频 reasoning segmentation 的适配；20 页全文、训练代码、3.8B 权重、结果表、消融和失败案例均公开，方法与配图都能从作者 arXiv 原文核验。

## 问题背景：视频比图像多出两个难点

LISA 已经展示了如何让大语言模型输出一个特殊 `[SEG]` token，再把该 token 的隐藏状态变成 SAM 的提示，从而回答“分割出最可能遮雨的物体”这类需要常识的图像问题。直接逐帧运行 LISA 并不能可靠解决视频任务，原因有两个。

第一，输入端需要理解时间。对于“移动更快的物体”，单帧没有速度信息；对于“输掉游戏的人”，关键事件可能发生在视频后半段。只让 LLM 看第一帧，会让语言推理建立在不完整的视觉证据上。

第二，输出端需要保持身份一致。同一目标会移动、遮挡、改变尺度，画面还可能出现外观相似的干扰对象。每帧独立生成提示容易发生 mask drift：前几帧分对，后几帧却跟到另一个对象。

作者的目标因此不是简单增加视频帧，而是在固定计算预算下同时保留时间跨度、空间细节和跨帧对象身份。

## 核心贡献

论文的贡献可以归纳为四点：

1. **Sparse Dense Sampling**：保留少量帧的完整视觉 token，同时把更多时间位置的每帧特征池化为一个 token，在长时序与像素细节之间折中。
2. **One-Token-Seg-All**：新增一个 `<TRK>` token，用同一隐藏表示提示全部视频帧的 mask decoder，避免每帧维护一套独立提示。
3. **跨帧监督**：训练 `<TRK>` 时并行计算多个稠密帧的分割损失，迫使它学习可跨帧泛化的语义表示，而不是单帧位置捷径。
4. **ReasonVOS 基准**：基于 MOSE、MeViS、VIPSeg 和 BURST 构造需要复杂推理、世界知识或时间理解的视频分割评测。

**论文结论**：VideoLISA-3.8B 在 MeViS、ReasonVOS 和 Ref-DAVIS-17 上显著优于论文列出的图像 LLM 或传统视频分割对照；同一个模型也保持了较强的图像 reasoning segmentation 能力。

**我的判断**：最关键的证据不是“3.8B 胜过 13B”，因为模型、数据和训练方式并不相同；更可信的是同一 VideoLISA 框架内的消融：跨帧输入、稀疏-稠密采样和多帧监督分别带来增益。论文证明的是一个有效的任务适配配方，而不是单纯依靠模型规模获胜。

## 方法总览：两个视觉通道，一个 TRK Token

![VideoLISA 方法总览](/images/posts/videolisa-video-reasoning-segmentation/videolisa-figure1-framework.png)

*图源：Bai et al., [One Token to Seg Them All](https://arxiv.org/abs/2409.19603), Figure 1, NeurIPS 2024；从作者 CC BY-NC-SA 4.0 arXiv PDF 原图裁切，结构、标签、箭头和示例均未修改。原图用于论文解读。*

VideoLISA 有两条视觉路径：

- **语义推理路径**：LLaVA visual tokenizer 把采样视频帧变成视觉 token，与文字指令一起输入 LLM。LLM 经过 LoRA 适配后生成文本，并输出特殊 `<TRK>` token。
- **像素解码路径**：SAM vision encoder 为待分割帧提取高分辨率特征；`<TRK>` 最后一层隐藏状态经 MLP 投影为 prompt embedding，再由 SAM mask decoder 输出掩码。

这套拆分保留了两个基础模型各自擅长的能力：LLM 负责语言、常识与时序推理，SAM 负责像素级边界。连接点不是框或点坐标，而是一个从整段视频与语言共同生成的连续向量。

## 关键模块一：稀疏-稠密采样

设视频均匀采样 $T_{sparse}$ 帧，每帧原本有 $L$ 个视觉 token。把所有帧完整送入 LLM，需要处理：

$$
N_{full}=T_{sparse}L.
$$

VideoLISA 再从中选择 $T_{dense}$ 个稠密帧，保留它们的全部 $L$ 个 token；与此同时，把 $T_{sparse}$ 个时间位置都做全局平均池化，每帧只保留一个稀疏 token。最终长度为：

$$
N_{SD}=T_{sparse}+T_{dense}L.
$$

论文默认 $T_{sparse}=32$、$T_{dense}=4$。若按仓库所用 CLIP ViT-L/14@336 的 $24\times24=576$ 个 patch token 估算，完整 32 帧为 18,432 个视觉 token，稀疏-稠密方案为 2,336 个，输入 LLM 的视觉 token 约减少 87.3%。这是根据论文公式与公开配置做的计算，不是论文报告的端到端延迟。

设计直觉是：稀疏 token 负责回答“前后发生了什么”，稠密 token 负责回答“具体边界在哪里”。只用稠密的少量帧会丢时间事件，只做池化又会丢像素级细节。

## 关键模块二：为什么一个 Token 能分割整段视频

![单一提示跨帧分割的对比](/images/posts/videolisa-video-reasoning-segmentation/videolisa-figure2-one-token.png)

*图源：Bai et al., [One Token to Seg Them All](https://arxiv.org/abs/2409.19603), Figure 2, NeurIPS 2024；从作者 CC BY-NC-SA 4.0 arXiv PDF 原图裁切，三组提示对比和帧序列均完整保留。原图用于论文解读。*

Figure 2 给出了方法动机。固定 box prompt 强依赖首帧位置，目标移动后很快失效；图像版 LISA 的提示更有语义，但它没看完整视频，也只在单帧上受监督，遇到大幅运动或相似干扰对象仍会漂移。

VideoLISA 从两个方向修正：

1. `<TRK>` 生成前，LLM 已通过稀疏-稠密输入看到整段视频的时间上下文。
2. 训练时，同一个 `<TRK>` 同时负责多个稠密帧，任何只编码单帧坐标的捷径都无法稳定降低全部分割损失。

设第 $t$ 个稠密帧的 SAM 特征为 $V_t$，`<TRK>` 的隐藏状态为 $h_{trk}$，投影层为 $g$，则掩码可概括为：

$$
M_t=\operatorname{MaskDecoder}(V_t,g(h_{trk})),
\qquad t\in\mathcal D.
$$

关键是所有 $t$ 共用同一个 $h_{trk}$。论文把它解释为“semantic kernel”：帧特征是变化的上下文，统一提示负责指出应该在这些上下文中寻找哪个实体。

## 训练目标与参数适配

训练数据分为图像和视频两部分。图像部分沿用 LISA 的语义分割、指代表达分割和 239 条 ReasonSeg 数据；视频部分使用 YouTube-VOS、Refer-YouTube-VOS 和 MeViS。图像会被复制成伪视频，与视频样本一起训练。

文本答案使用标准自回归生成损失，分割部分结合逐像素 BCE 与 Dice loss。论文中的总目标可以写成：

$$
\mathcal L
=\lambda_{txt}\mathcal L_{txt}
+\lambda_{seg}
\left(\lambda_{bce}\mathcal L_{BCE}
+\lambda_{dice}\mathcal L_{Dice}\right).
$$

作者设置 $\lambda_{txt}=\lambda_{seg}=1$、$\lambda_{bce}=2$、$\lambda_{dice}=0.5$。视频样本在 $T_{dense}$ 个稠密帧上并行计算分割损失后取平均。

Figure 1 明确标出 LLM 使用 LoRA，官方训练仓库也提供 LoRA 合并脚本。这里的个性化不是记住某个用户身份，而是用参数高效更新把通用多模态骨干迁移到特定下游能力：视频时序理解、目标跟踪和像素级 reasoning segmentation。

## 训练与推理流程

### 训练

1. 从视频均匀取 32 个稀疏帧，再随机取 4 个稠密帧。
2. 用 visual tokenizer 构建稀疏-稠密视觉 token，与指令一起输入 LLM。
3. 让 LLM 生成文本答案与 `<TRK>`，并计算回答 token 上的语言损失。
4. 把 `<TRK>` 隐藏状态投影为 SAM prompt，在 4 个稠密帧上并行解码掩码。
5. 联合优化语言损失、BCE 和 Dice loss，使文本推理与像素输出对齐。

最终模型基于 LLaVA-Phi-3-V，LLM 为 Phi-3 3.8B。作者使用 64 张 NVIDIA A10 24GB、DeepSpeed、每卡 batch size 2、AdamW，学习率 $3\times10^{-4}$、weight decay 0、warmup 100 step。消融训练 3,000 次迭代约 10 小时，最终对比模型训练 6,000 次迭代约 20 小时。

### 推理

1. 同样采样 32 个稀疏帧和 4 个均匀分布的稠密帧。
2. LLM 只生成一次 `<TRK>`。
3. 所有视频帧逐帧进入 SAM vision encoder 和 mask decoder，始终复用同一个 `<TRK>` prompt。
4. 可选地把 4 个稠密帧的高置信掩码交给 XMem++，传播和修正其余帧。

“一个 token”不代表只运行一次分割解码。LLM 的语义提示只生成一次，但 SAM 仍需逐帧编码和解码；可选 XMem++ 还会增加额外跟踪成本。因此它主要简化跨帧提示与身份关联，不等同于端到端一次前向完成整段视频。

## 实验设置与主要结果

视频评测采用区域相似度 $\mathcal J$、轮廓准确度 $\mathcal F$ 及二者平均 $\mathcal J\&\mathcal F$。ReasonVOS 是作者新建的 zero-shot 评测：91 段视频、458 条视频-指令-掩码样本，其中 205 条短查询、253 条长查询。种子数据由人工标注，再用 Claude 3 改写扩增并进行人工复核。

![VideoLISA 在三个视频分割基准上的主要结果](/images/posts/videolisa-video-reasoning-segmentation/videolisa-tables1-3-results.png)

*图源：Bai et al., [One Token to Seg Them All](https://arxiv.org/abs/2409.19603), Tables 1-3, NeurIPS 2024；从作者 CC BY-NC-SA 4.0 arXiv PDF 原表裁切，方法名、指标、粗体、下划线和数值均未修改。原图用于论文解读。*

| 模型 | Refer-YouTube-VOS J&F | Ref-DAVIS-17 J&F | MeViS J&F | ReasonVOS J&F |
| --- | ---: | ---: | ---: | ---: |
| LISA-13B | 52.6 | 60.7 | - | - |
| TrackGPT-13B | 59.5 | 66.5 | - | - |
| VideoLISA-3.8B，One-Token-Seg-All | 61.7 | 67.7 | 42.3 | 45.1 |
| VideoLISA-3.8B，含后处理 | 63.7 | 68.8 | 44.4 | 47.5 |

在 Ref-DAVIS-17 上，纯 One-Token-Seg-All 已达到 67.7，加入 XMem++ 后为 68.8。MeViS 上，论文列出的此前最好 LMPM 为 37.2，VideoLISA 不带后处理为 42.3；ReasonVOS 上，OnlineRefer 为 38.7，VideoLISA 为 45.1。

Refer-YouTube-VOS 的边界更值得注意：传统专用方法 SgMg 为 65.7，仍高于 VideoLISA 的 61.7 和后处理后的 63.7。论文的强项是兼顾复杂语言推理与视频分割，不是每个传统 RVOS 指标都全局最优。

图像 ReasonSeg 上，VideoLISA-3.8B 的验证集 gIoU/cIoU 为 61.4/67.1，高于论文列出的 LISA-13B 微调版 56.2/62.9。作者将其归因于图像与视频联合训练，以及多帧监督带来的推理能力；但由于骨干与数据配方同时变化，这个跨模型比较不能单独证明某一个模块的因果作用。

## 消融：收益究竟来自哪里

### 时间建模

在统一 3,000 step 设置下，直接把 LISA-7B 用视频数据微调，MeViS J&F 只从 43.2 提到 44.8，ReasonSeg gIoU 还从 51.7 降到 48.6。仅仅“加入视频数据”并不足以获得稳定迁移。

同一 VideoLISA-3.8B 框架中：

| 时间建模方式 | ReasonSeg gIoU | MeViS J&F | Ref-DAVIS-17 J&F |
| --- | ---: | ---: | ---: |
| n-frame 直接拼接 | 55.6 | 49.9 | 65.5 |
| 空间-时间池化 | 56.0 | 50.8 | 62.2 |
| Slow-Fast Pooling | 54.0 | 50.2 | 65.7 |
| **Sparse Dense Sampling** | **58.9** | **51.7** | **67.8** |

全部池化会破坏分割需要的细节，只拼接少量完整帧又限制时间跨度。稀疏-稠密采样在三项指标上取得最稳定的折中。

### 跨帧监督

固定时间建模后，只让 `<TRK>` 监督一个帧的 One-Token-Seg-One 在 MeViS 上为 46.1；改为同一个 token 监督多个帧后提升到 51.7，Ref-DAVIS-17 从 60.2 提到 67.8。这个对照最直接地支持“训练目标决定 token 是否学会跨帧关联”。

加入 XMem++ 后，两项分别升到 54.5 和 68.7。需要把这个结果视为组合系统：提升并非全部来自 VideoLISA 自身，后处理依赖额外跟踪模型和多参考掩码。

### 数据配方

只用图像分割数据时，ReasonSeg gIoU 为 57.2，但 MeViS 只有 46.0；只用视频分割时，MeViS 升到 49.3，ReasonSeg 却降到 41.4。联合图像与视频分割后，两者分别达到 58.9 和 51.7。

加入 Image-QA 数据反而令三项主指标下降；再加入 Video-QA 后 ReasonSeg 有所恢复，但 Ref-DAVIS-17 仍低于纯分割数据版本。作者据此把多任务兼容性留作未来工作。更多数据类型不一定自动带来更好的密集预测。

## 失败案例与局限

![VideoLISA 失败案例](/images/posts/videolisa-video-reasoning-segmentation/videolisa-figure4-failures.png)

*图源：Bai et al., [One Token to Seg Them All](https://arxiv.org/abs/2409.19603), Figure 4, NeurIPS 2024；从作者 CC BY-NC-SA 4.0 arXiv PDF 原图裁切，问题文本、视频帧和掩码均未修改。原图用于论文解读。*

Figure 4 展示了两类典型错误。第一段视频中，一辆车撞进商店，但模型没有把“打破平静的不寻常物体”定位到车，作者认为基础 MLLM 对画面内容产生了错误识别。第二段要求判断谁输掉游戏，模型缺少游戏规则知识；在提示里补充“输家会被玩具打脸”后，它才找到正确目标。

这说明 reasoning segmentation 的失败不只有 mask 边界问题。语言模型对事件看错、缺知识或幻觉时，像素解码器即使边界精确，也只会精确地分割错误对象。

论文主动指出两个主要限制：

1. **计算成本仍高**：3.8B LLM 已小于此前 7B/13B 方法，但相对传统 VOS 仍昂贵。论文没有报告端到端 FPS、峰值显存或各模块延迟。
2. **缺少视频预训练视觉骨干**：SAM 图像编码器不专门建模运动；引入视频骨干可能改善跟踪，但要同时兼容 LLM 与 SAM decoder 并不简单。

工程上还应补充四个边界：

1. **ReasonVOS 规模较小**：458 个样本适合诊断复杂查询，但不足以覆盖开放世界视频事件。
2. **后处理混合了能力来源**：XMem++ 提升了指标，也增加系统组件、显存和延迟，部署评测应分别报告前后处理版本。
3. **单 token 可能成为瓶颈**：一个向量要表达目标语义、外观变化与多实例关系；复杂遮挡或多个同类目标时容量是否足够仍未系统分析。
4. **隐私与误用风险**：论文 broader impact 提到监控与医疗应用，但没有对人群偏差、隐私保护、误跟踪风险或拒答策略做实验。

## 可复现资源

- [NeurIPS 2024 正式页面与正式 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0cf3e7eefb9d643e93e16ff1d94090a7-Abstract-Conference.html)
- [作者 arXiv 全文、源码与 CC BY-NC-SA 4.0 许可](https://arxiv.org/abs/2409.19603)
- [官方代码仓库](https://github.com/showlab/VideoLISA)
- [VideoLISA-3.8B 权重](https://huggingface.co/ZechenBai/VideoLISA-3.8B)
- [ReasonVOS 数据与评测说明](https://github.com/showlab/VideoLISA/blob/main/BENCHMARK.md)
- [MeViS 数据集](https://github.com/henghuiding/MeViS)
- [XMem++ 后处理实现](https://github.com/mbzuai-metaverse/XMem2)

官方仓库提供环境安装、图像与视频数据组织、DeepSpeed 训练、LoRA 权重合并，以及 MeViS、ReasonVOS、Ref-YouTube-VOS、Ref-DAVIS-17 和图像分割评测脚本。README 同时说明作者使用 8 个节点、共 64 张 A10 24GB；小规模硬件可以尝试降低 batch size 与学习率，但论文没有验证这些替代配方能复现主结果。

论文和本文使用的原图采用 CC BY-NC-SA 4.0。代码、模型权重、训练数据、LLaVA、Phi-3、SAM 与 XMem++ 仍有各自许可，不能用论文许可统一覆盖。

## 个人判断

VideoLISA 最值得复用的不是“把 `[SEG]` 改成 `<TRK>`”，而是输入表示和训练监督形成了闭环：稀疏-稠密采样让 token 看见时间，One-Token-Seg-All 再要求同一个 token 对多个时间位置负责。只改输入或只改输出都不够。

它也展示了一种实用的多模态任务适配范式：保留成熟的 LLM 与 SAM 分工，用 LoRA 和一个紧凑接口把世界知识迁移到像素任务。对于遥感时序、工业视频检查或机器人操作，类似思路可以把专门的密集预测解码器接到通用多模态骨干上。

但当前证据还不能支持“统一视频分割基础模型”的强结论。训练使用 64 张 A10，推理仍逐帧运行 SAM，可选后处理又引入 XMem++；ReasonVOS 只有 458 个样本，系统指标也不完整。下一步更有价值的工作应同时报告任务质量、TTFT、逐帧吞吐、峰值显存和长视频稳定性，并在更大规模开放事件上测试幻觉与身份漂移。

因此，我会把 VideoLISA 定位为：一篇把 **多模态大模型通过 LoRA 与跨帧监督适配到视频 reasoning segmentation** 讲清楚的训练侧论文。它的主要贡献是任务迁移与时序表示，不是已经完成的低成本视频推理系统。

## 参考资料

1. Bai et al., [One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0cf3e7eefb9d643e93e16ff1d94090a7-Abstract-Conference.html), NeurIPS 2024.
2. Show Lab, [VideoLISA Code, Model and ReasonVOS](https://github.com/showlab/VideoLISA).
3. Lai et al., [LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/abs/2308.00692), CVPR 2024.
4. Kirillov et al., [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023.
5. Cheng and Schwing, [XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model](https://arxiv.org/abs/2207.07115), ECCV 2022.
