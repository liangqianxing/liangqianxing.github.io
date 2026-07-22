---
title: "M3 精读：可伸缩视觉 Token 如何加速多模态推理"
date: 2026-07-20 15:15:49
description: 从嵌套视觉 Token、多尺度训练目标与预填充成本出发，拆解 ICLR 2025 的 M3 如何在同一多模态大模型中按需切换推理粒度。
series: 三大会论文精读
seriesOrder: 4
categories:
  - AI
tags:
  - 多模态大模型
  - 推理加速
  - 视觉 Token 压缩
  - M3
  - LLaVA
  - ICLR
hidden: true
haloPublished: true
---

多模态大模型常把一张图像展开成数百乃至数千个视觉 token，再把它们作为前缀送进语言模型。问题是：**不是每个问题都需要看清每个局部，但固定长度表示会为简单样本支付同样的预填充成本。**

M3（Matryoshka Multimodal Models）的答案是把同一张图表示成一组由粗到细的视觉 token 尺度，并在训练时让同一套权重适应全部尺度。部署时可以按算力预算选择 1、9、36、144 或 576 个视觉 token，而不必为每个长度维护一个独立模型。

这篇论文直接属于本专题的**多模态推理加速**方向。需要先划清结论边界：M3 证明了“一个模型支持多档视觉粒度”可行，也报告了明显的语言模型预填充成本下降；它没有给出可部署的自动尺度选择器，附录中的延迟还是 roofline 模型估算，不是端到端实测吞吐。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **Matryoshka Multimodal Models** |
| 作者 | Mu Cai、Jianwei Yang、Jianfeng Gao、Yong Jae Lee |
| 会议 | ICLR 2025 |
| 专题子方向 | 多模态大模型的推理加速：视觉 token 压缩与按预算推理 |
| 正式评审 | [OpenReview: Uhj5OxAz7I](https://openreview.net/forum?id=Uhj5OxAz7I) |
| 作者版本与许可 | [arXiv:2405.17430](https://arxiv.org/abs/2405.17430)，CC BY 4.0 |
| 项目主页 | [Matryoshka Multimodal Models](https://matryoshka-mm.github.io/) |
| 代码与模型 | [mu-cai/matryoshka-mm](https://github.com/mu-cai/matryoshka-mm)，代码 Apache 2.0 |

**选择理由**：M3 讨论的是多模态模型推理链路中的真实瓶颈，而不是泛化的视觉语言建模。它把 token 数量变成可控部署参数，覆盖图像、OCR 和视频场景；论文同时给出方法图、性能曲线、样本级失败案例、消融与系统成本分析，事实和配图都能从作者全文及 CC BY 4.0 源码核验。

## 问题背景：固定 576 个 Token 并不总是合理

以 LLaVA-1.5 使用的 CLIP ViT-L/14@336 为例，视觉编码器把图像变成 $24\times24=576$ 个 patch token。LLaVA-NeXT 为高分辨率图像切分多个网格后，每个网格仍可贡献 576 个 token；视频再叠加多帧，视觉前缀会迅速变长。

对语言模型而言，这些 token 会进入 prefill：每一层都要处理更长的上下文，KV cache 也要保存相应状态。固定长度策略有两个问题：

1. 自然场景中的粗粒度问题可能只需少量视觉信息，却仍支付完整 prefill 成本；
2. 文档、图表和 OCR 又确实依赖细节，粗暴裁剪会直接丢失答案所需文字。

已有 token pruning、merging 方法通常为一个输入产生一个确定长度。M3 追求的不是寻找唯一压缩率，而是训练一个**可在多个压缩率之间切换**的模型。

![M3 从粗到细的嵌套视觉表示](/images/posts/matryoshka-multimodal-models-inference-acceleration/m3-concept.png)

*图源：Cai et al., [Matryoshka Multimodal Models](https://arxiv.org/abs/2405.17430), Figure 1, ICLR 2025；取自作者 CC BY 4.0 arXiv 源码。原图用于论文解读。*

图 1 表达了论文的直觉：少量 token 先保留“餐厅里有个女孩”这样的全局语义，增加 token 后才逐步出现衣服、纸袋和饮料杯等细节。同一个问题可以选择不同粒度，输出信息密度也随之变化。

## 核心贡献

论文的贡献可以归纳为四点：

1. **沿 token 长度构造 Matryoshka 表示**：不是裁短特征维度，而是让 1、9、36、144、576 个视觉 token 形成由粗到细的多尺度表示。
2. **一套权重覆盖多个部署预算**：训练时同时优化所有尺度，推理时由用户控制视觉粒度，无需分别训练五个模型。
3. **把尺度变成数据分析工具**：不同数据集对视觉细节的需求差异很大，自然场景基准在低 token 下较稳，OCR 与文档理解下降明显。
4. **量化潜在效率收益**：减少视觉前缀可降低语言模型 prefill 的 FLOPs、激活内存和理论延迟，并可与量化、低秩压缩等方法叠加。

**论文结论**：M3 在完整 token 下基本保持基线能力，在少量 token 下明显优于只在推理时临时池化或采样的方案。

**我的判断**：真正有工程价值的是“一个 checkpoint，多档服务等级”。例如同一服务可以为离线文档分析保留 576 token，为实时自然图像问答使用 9 或 36 token。但论文还没有完成自动路由这一步，所以它更像可控执行层，而不是闭环调度器。

## 方法总览：空间池化得到五档粒度

![M3 模型架构与粒度控制器](/images/posts/matryoshka-multimodal-models-inference-acceleration/m3-architecture.png)

*图源：Cai et al., [Matryoshka Multimodal Models](https://arxiv.org/abs/2405.17430), Figure 3, ICLR 2025；取自作者 CC BY 4.0 arXiv 源码。原图用于论文解读。*

M3 沿用 LLaVA 的主干：CLIP 视觉编码器产生视觉特征，投影后作为前缀送入 Vicuna 7B。它没有新增可学习模块，而是在视觉 token 网格上顺序执行平均池化：

$$
24\times24 \rightarrow 12\times12 \rightarrow 6\times6 \rightarrow 3\times3 \rightarrow 1\times1
$$

对应尺度集合为：

$$
\mathcal{S}=\{576, 144, 36, 9, 1\}.
$$

设最细粒度视觉表示为 $X_{576}$，池化算子为 $g(\cdot)$，则可以写成：

$$
X_{144}=g(X_{576}),\quad
X_{36}=g(X_{144}),\quad
X_{9}=g(X_{36}),\quad
X_{1}=g_{3\times3}(X_{9}).
$$

论文用 $X_{S_{i-1}}\subset X_{S_i}$ 表达“嵌套”。严格来说，平均池化后的向量不是原 token 的字面子集，而是由细尺度确定性派生出的粗尺度表示。工程上更准确的理解是：**所有粗尺度都共享同一条细到粗的空间聚合链路。**

平均池化还有一个重要偏置：它保留二维邻域关系。论文比较了顺序采样和空间采样，顺序采样打乱了这种空间结构，结果最差。

## 多尺度训练目标

给定文本问题 $X_q$、某一视觉尺度 $X_{S_i}$ 和答案 token 序列 $X_a=(x_1,\ldots,x_L)$，每个尺度仍使用标准自回归生成概率：

$$
P_\theta(X_a\mid X_{S_i},X_q)
=\prod_{j=1}^{L}
P_\theta(x_j\mid X_{S_i},X_q,X_{a,<j}).
$$

关键变化只是把五个尺度的负对数似然取平均：

$$
\mathcal{L}_{M^3}(\theta)
=\frac{1}{M}\sum_{i=1}^{M}
-\log P_\theta(X_a\mid X_{S_i},X_q),
\qquad M=5.
$$

这里的 $\theta$ 包含视觉编码器和后续语言模型的可训练参数。论文从已经完成视觉指令调优的 LLaVA-1.5 或 LLaVA-NeXT 初始化，而不是从纯文本 Vicuna 直接开始；随后让同一个样本在五个尺度上都学习回答。

这与“训练时随机抽一个尺度”不同。所有尺度同时参与损失，会迫使模型对每个样本都适应嵌套表示。表 8 的消融显示，从 LLaVA 初始化并平均所有尺度损失的组合表现最好。

## 训练与推理流程

### 训练

1. 用 CLIP ViT-L/14@336 把图像编码为 $24\times24$ 网格；
2. 依次平均池化，得到五组视觉 token；
3. 将每组 token 分别与同一文本问题拼接，执行答案的 teacher forcing；
4. 平均五个尺度的语言建模损失，更新视觉编码器和语言模型；
5. 对 LLaVA-1.5-M3 与 LLaVA-NeXT-M3 各训练 1 个 epoch。

论文使用 8 张 NVIDIA H100。LLaVA-1.5-M3 的 LLM 学习率为 $2\times10^{-5}$，LLaVA-NeXT-M3 为 $1\times10^{-5}$；两者视觉编码器学习率均为 $2\times10^{-5}$。训练数据沿用相应 LLaVA 版本的视觉指令数据。

### 推理

推理时仍先运行视觉编码器，再从五档尺度中选一档送进语言模型。较少的视觉 token 主要减少 LLM prefill、激活和 KV cache 成本；解码阶段每步生成文本 token 的成本也会因上下文更短而下降。

要注意，论文的“Granularity Controller”在架构图中是一个外部控制入口，不是训练出来的预测网络。当前实现通过参数指定尺度，无法自动判断某张图或某个问题该用 9 还是 576 个 token。

## 实验设置

论文在两个基座上实验：

- **LLaVA-1.5-7B-M3**：验证固定分辨率图像下的多尺度能力；
- **LLaVA-NeXT-7B-M3**：覆盖多图块高分辨率输入，并零样本迁移到视频问答。

图像基准包括 MMBench、GQA、POPE、VizWiz、SEEDBench、ScienceQA、MMMU，以及 TextVQA、DocVQA、ChartQA、AI2D 等文档/OCR 任务。视频基准包括 MSVD-QA、MSRVTT-QA、ActivityNet-QA、NExT-QA、IntentQA 和 EgoSchema。

论文还训练了“每个尺度一个独立模型”的 Specific Scale（SS）基线，并比较三种无需训练的推理期压缩：平均池化、空间采样和顺序采样。

## 主要结果：低 Token 对自然场景友好，对 OCR 不友好

![M3 在 MMBench 上的视觉 token 数与性能曲线](/images/posts/matryoshka-multimodal-models-inference-acceleration/m3-mmbench-oracle.png)

*图源：Cai et al., [Matryoshka Multimodal Models](https://arxiv.org/abs/2405.17430), Figure 2, ICLR 2025；取自作者 CC BY 4.0 arXiv 源码。原图用于论文解读。*

在 LLaVA-1.5-M3 上，MMBench 从 576 token 的 65.9 降到 9 token 的 63.1、1 token 的 59.5；原始 LLaVA-1.5 使用 576 token 得到 64.8。也就是说，9 token 并没有完全无损，但仍保留了大部分能力。

LLaVA-NeXT-M3 的数据更能说明“任务依赖”：

| 每网格 Token | MMBench | TextVQA | DocVQA | AI2D |
| ---: | ---: | ---: | ---: | ---: |
| 576 | 67.96 | 63.13 | 72.61 | 66.71 |
| 144 | 69.50 | 62.61 | 66.48 | 68.07 |
| 36 | 68.56 | 58.71 | 55.94 | 67.36 |
| 9 | 67.35 | 51.97 | 43.52 | 66.77 |
| 1 | 62.97 | 38.92 | 31.63 | 64.57 |

**论文结论**：MMBench、AI2D、ScienceQA 等基准在低 token 下相对稳定，TextVQA 和 DocVQA 等密集感知任务需要更多视觉细节。

**我的分析**：不能把“COCO 风格数据集约 9 token 足够”推广成所有多模态请求的默认值。表中文字、票据号码、小物体属性和图表刻度都可能在平均池化中消失。部署策略至少需要结合任务类型、图像分辨率和问题中的 OCR 信号。

### 视频结果

LLaVA-NeXT-M3 把均匀采样的 6 帧拼成图像网格。完整配置使用 2880 个视觉 token；压到每网格 9 个、总计 45 个 token 时：

| 基准 | 2880 Token | 45 Token | 差值 |
| --- | ---: | ---: | ---: |
| ActivityNet-QA | 53.9 | 53.2 | -0.7 |
| IntentQA | 58.8 | 58.7 | -0.1 |
| EgoSchema | 36.8 | 38.8 | +2.0 |
| MSVD-QA | 78.2 | 75.8 | -2.4 |
| NExT-QA | 63.1 | 59.5 | -3.6 |

这说明部分视频基准确实只需要稀疏语义，冗长上下文甚至可能干扰模型；但不同基准并非一致无损。论文没有报告真实视频服务吞吐，也只采样 6 帧，因此不能据此推断长视频端到端系统已经解决。

## Oracle：有潜力，但不是可部署结果

论文定义的 oracle 会在得到每个测试样本的预测后，选择“能够答对且 token 最少”的尺度。以 MMBench 为例，oracle 平均只用 8.90 个 token，却得到 74.35；固定 576-token M3 为 67.96。

这个差距说明样本级动态选择很有价值，但 oracle 使用了答案正确性这一测试后信息。它是上界，不是可上线的路由器。论文在结论中也把“缺少有效视觉 token 预测器”列为主要限制。

## 推理成本：显著降低 Prefill，但数据是理论估算

附录用 LLM-Viewer 的 roofline 模型估算 Tesla V100 上的 LLaVA-1.5 prefill，假设图像原本产生 576 个视觉 token，文本提示为 30 token：

| 视觉 Token | FLOPs | Prefill 时间 | 总内存 | 激活内存 |
| ---: | ---: | ---: | ---: | ---: |
| 576 | 8.0 TB | 58.1 ms | 21.6 GB | 3.8 GB |
| 144 | 2.2 TB | 19.5 ms | 15.0 GB | 0.7 GB |
| 36 | 0.9 TB | 18.0 ms | 13.8 GB | 0.3 GB |
| 9 | 0.5 TB | 17.7 ms | 13.6 GB | 0.2 GB |
| 1 | 0.4 TB | 17.6 ms | 13.5 GB | 0.1 GB |

从 576 压到 9 token，理论 FLOPs 降到约 $1/16$，prefill 时间从 58.1 ms 降到 17.7 ms，约为 3.28 倍加速；总内存只从 21.6 GB 降到 13.6 GB，因为模型权重等固定成本仍在。

**论文结论**：视觉 token 减少能显著降低 LLM prefill 计算和激活内存。

**我的判断**：FLOPs 并不会等比例变成延迟收益，表中 144 到 1 token 的估算时间几乎饱和就是证据。更重要的是，这张表没有计入完整视觉编码、数据搬运、调度和文本解码，也不是实机端到端基准。工程评估还需要真实 GPU 上的 TTFT、TPOT、吞吐、峰值显存和不同 batch size 曲线。

## 消融分析

### 多尺度训练优于推理时临时压缩

在 LLaVA-NeXT 的 MMBench 上，M3 与三种推理期启发式方法的差距随 token 减少而扩大：

| Token | M3 | 推理期平均池化 | 空间采样 | 顺序采样 |
| ---: | ---: | ---: | ---: | ---: |
| 144 | 69.50 | 61.68 | 65.81 | 60.14 |
| 36 | 68.56 | 50.77 | 60.05 | 44.76 |
| 9 | 67.35 | 45.45 | 45.45 | 31.96 |
| 1 | 62.97 | 19.33 | 26.29 | 22.42 |

这说明仅在推理时把原模型 token 池化掉，语言模型没有学会解释压缩表示；M3 的收益主要来自多尺度联合训练，而不只是平均池化算子本身。

### 平均池化优于采样

表 6 比较 M3 内部的 token 构造方式。平均池化在 TextVQA、MMBench、AI2D 的各尺度上整体优于顺序采样和空间采样。作者推测原因是平均池化更好地保留了局部空间信息。

这也暴露了方法偏置：平均池化适合平滑聚合，但可能抹掉小字和稀有局部对象。OCR 曲线的快速下降与这一机制是一致的。

### 必须让 LLM 适应多尺度分布

只训练 CLIP、冻结 LLM 会明显下降。以 9 token 为例：

- TextVQA：完整微调 51.97，冻结 LLM 为 36.15；
- MMBench：67.35 对 61.08；
- DocVQA：43.52 对 28.36。

因此，M3 不是一个完全即插即用的视觉压缩器。要得到低 token 下的鲁棒性，需要更新整个多模态模型，训练成本和重新验证成本都不能忽略。

## 失败案例与局限

![TextVQA 在不同视觉 token 尺度下的成功与失败样本](/images/posts/matryoshka-multimodal-models-inference-acceleration/m3-textvqa-scales.png)

*图源：Cai et al., [Matryoshka Multimodal Models](https://arxiv.org/abs/2405.17430), Figure 4, ICLR 2025；取自作者 CC BY 4.0 arXiv 源码。原图用于论文解读。*

图 4 给出三类 TextVQA 样本：有些问题在所有尺度都能答对；有些需要至少 36 或 144 个 token；还有些即使用完整 token 也答错。这说明“更多 token”只是提高可见细节的必要条件之一，并不保证视觉识别或语言推理正确。

论文明确承认的主要限制是：

1. **缺少自动尺度预测器**：当前只能人工或外部策略指定 token 档位，无法逼近 oracle；
2. **继承 LLaVA 的偏差与风险**：训练数据包含 GPT-4/GPT-4V 生成内容，相应偏差不会因 token 压缩而消失。

结合实验，我认为还应补充五点：

1. 效率数据来自 roofline 理论模型，不是端到端实测；
2. 视觉编码器仍先产生完整网格，M3 主要节省语言模型侧成本；
3. 平均池化对 OCR、小目标和密集图表不友好；
4. 只有 Vicuna 7B、LLaVA 系列基座，跨架构泛化仍需验证；
5. 低 token 有时提高分数，可能包含基准偏差或减少干扰，不能简单解释为压缩后表征“更强”。

## 可复现资源

- [作者代码仓库](https://github.com/mu-cai/matryoshka-mm)：基于 LLaVA，提供训练、评测和命令行推理入口；
- [模型列表](https://github.com/mu-cai/matryoshka-mm/blob/main/docs/MODEL_ZOO.md)：发布 LLaVA-1.5-M3 与 LLaVA-NeXT-M3 权重；
- [项目主页](https://matryoshka-mm.github.io/)：集中展示图像、视频与多粒度样例；
- [arXiv 全文与源码](https://arxiv.org/abs/2405.17430)：论文采用 CC BY 4.0，本文配图均从该官方作者源码提取；
- [OpenReview 记录](https://openreview.net/forum?id=Uhj5OxAz7I)：用于核对 ICLR 2025 收录与评审信息。

代码仓库标注 Apache 2.0，但项目同时提醒：LLaVA 数据、Vicuna/Llama 基座和模型权重各有自己的许可与使用条款。复现实验不能只看仓库代码许可证，还要逐项检查数据和 checkpoint 授权。

## 个人判断：M3 解决了执行档位，还没解决调度

M3 最值得复用的思想，是把视觉 token 预算从训练时常量变成推理时接口。它适合与服务系统组合：请求分类器先判断 OCR、文档、自然场景或视频，再结合延迟 SLO 和 GPU 负载选择尺度；高风险请求还可以从 9 token 升档到 36 或 144 token 重试。

但论文当前最关键的缺口也恰好在这里：没有可靠的尺度预测器，就只能用任务级静态规则，无法获得 oracle 展示的样本级上界。一个更完整的系统需要同时优化三件事：

1. 预测最低可用视觉粒度，并估计不确定性；
2. 把视觉编码、prefill、decode 和 batch 调度纳入同一成本模型；
3. 在真实硬件上报告质量、TTFT、吞吐和显存的 Pareto 前沿。

因此，我会把 M3 定位为**多模态弹性推理的表征基础**：它证明了同一模型可以承受跨度很大的视觉 token 预算，也清楚展示了任务之间的粒度差异；距离自动、可靠、端到端最优的推理加速系统，还差路由与系统验证两层。

## 参考资料

1. Cai et al. [Matryoshka Multimodal Models](https://openreview.net/forum?id=Uhj5OxAz7I). ICLR 2025.
2. Cai et al. [Matryoshka Multimodal Models, arXiv:2405.17430](https://arxiv.org/abs/2405.17430). CC BY 4.0.
3. Liu et al. [Visual Instruction Tuning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html). NeurIPS 2023.
4. Liu et al. [LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/). 2024.
5. Bolya et al. [Token Merging: Your ViT But Faster](https://openreview.net/forum?id=JroZRaRw7Eu). ICLR 2023.
6. Yuan et al. [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363). 2024.
