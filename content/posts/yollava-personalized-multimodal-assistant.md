---
title: "Yo'LLaVA 精读：用 16 个软 Token 记住你的专属视觉概念"
date: 2026-07-21 15:07:30
description: 从可学习身份 Token、视觉属性软提示与难负样本出发，拆解 NeurIPS 2024 的 Yo'LLaVA 如何用 5 张图完成多模态大模型个性化。
series: 三大会论文精读
seriesOrder: 6
categories:
  - AI
tags:
  - 多模态大模型
  - 个性化
  - 视觉概念
  - 软提示
  - YoLLaVA
  - NeurIPS
---

通用多模态大模型知道什么是“狗”，却不知道哪一只狗是用户的宠物。把几张参考图每次都塞进上下文可以临时补充信息，但视觉 token 很长，也无法让模型在后续纯文本对话中直接使用这个名字。

Yo'LLaVA 研究的是另一种接口：给定一个主体的少量照片，为它增加一个可读写的身份 token，再用一小组连续软 token 保存视觉属性。论文默认只用 5 张照片和 16 个软 token，就让冻结的 LLaVA-1.5-13B 学会识别该主体，并围绕它进行图文或纯文本对话。

这篇论文直接属于本专题的**多模态大模型个性化**方向，更具体地说，是“用户专属视觉概念的轻量注入”。它不是用户偏好推荐系统，也没有形成长期记忆管理系统；它解决的是更基础的一步：让模型把“一个类别”细化成“用户指定的那一个实例”。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **Yo'LLaVA: Your Personalized Language and Vision Assistant** |
| 作者 | Thao Nguyen、Haotian Liu、Yuheng Li、Mu Cai、Utkarsh Ojha、Yong Jae Lee |
| 会议 | NeurIPS 2024，Main Conference Track |
| 专题子方向 | 多模态大模型个性化：用户专属人物、宠物、物体与地标 |
| 官方论文 | [NeurIPS Proceedings](https://papers.neurips.cc/paper_files/paper/2024/hash/48088756ec0ce6ba362bddc7ebeb3915-Abstract-Conference.html) |
| 作者版本与许可 | [arXiv:2406.09400 v2](https://arxiv.org/abs/2406.09400)，CC BY 4.0 |
| 项目主页 | [Yo'LLaVA](https://thaoshibe.github.io/YoLLaVA/) |
| 代码与数据 | [WisconsinAIVision/YoLLaVA](https://github.com/WisconsinAIVision/YoLLaVA)、[Hugging Face 数据集](https://huggingface.co/datasets/thaoshibe/YoLLaVA) |

**选择理由**：最近两篇专题文章都落在推理加速，今天切换到个性化。Yo'LLaVA 明确以用户指定主体为研究对象，不是泛多模态任务；方法只改少量 token，工程边界清楚。官方全文、源码、代码和数据均公开，论文同时给出方法图、完整主表、消融和失败案例，关键事实可以交叉核验。

## 问题背景：类别知识不等于个人知识

用户说“给 `<bo>` 买什么生日礼物”，需要模型同时知道两件事：`<bo>` 是谁，以及通用世界知识中什么礼物适合这类主体。普通 LMM 只有后者。即便图中出现 `<bo>`，模型通常也只能回答“这是一只柴犬”，无法知道它就是用户命名的那只狗。

常见的上下文方案有两个局限：

1. 文本描述很难覆盖区分相似实例所需的细节；“戴眼镜的短发男子”可能匹配很多人。
2. 每次附上多张参考图会消耗大量视觉 token，并且不能直接支持没有参考图的后续对话。

Yo'LLaVA 因而把个性化表示放进模型词表：身份 token 负责让人和模型都能引用主体，连续软 token 负责承载难以写成文字的视觉属性。

![Yo'LLaVA 个性化图文与纯文本对话总览](/images/posts/yollava-personalized-multimodal-assistant/figure1-overview.png)

*图源：Nguyen et al., [Yo'LLaVA: Your Personalized Language and Vision Assistant](https://arxiv.org/abs/2406.09400), Figure 1, NeurIPS 2024；从作者 CC BY 4.0 arXiv v2 源码直接栅格化。原图用于论文解读。*

图 1 展示了目标能力：模型既能在新图中认出 `<bo>` 或 `<A>`，也能在没有输入图像时围绕该主体对话。后者意味着个性化视觉信息必须进入可持久化参数，而不只是停留在一次会话上下文里。

## 核心贡献

论文的贡献可以归纳为四点：

1. **定义个性化 LMM 任务**：从少量无文本标注的主体图片出发，要求模型完成新图识别、个性化视觉问答和无图文本问答。
2. **用软提示保存专属视觉概念**：新增一个身份 token 和 $k$ 个潜在 token，冻结视觉编码器、投影器与语言模型主体。
3. **用难负样本学习实例边界**：从 LAION 检索视觉相似但身份不同的图像，防止模型把所有相似宠物或人物都认成目标主体。
4. **构建 40 个主体的评测集**：覆盖人物、宠物、地标、物体和虚构角色，并与 LLaVA、GPT-4V 提示方案及 MyVLM 比较。

**论文结论**：16 个可学习 token 在主体识别和无图属性问答上优于同等长度的文本提示，并接近使用 5 张参考图、约 5000 token 的 GPT-4V 图像提示。

**我的判断**：它最有价值的地方不是某个 0.924 分数，而是把“个人视觉概念”变成一个很小、可独立存储的参数包。不过论文没有解决这些参数包如何授权、撤销、加密和多用户隔离，因此它还是概念注入原型，不是完整的个人记忆基础设施。

## 方法总览：身份 Token 与视觉属性 Token 分工

对一个名为 `<sks>` 的主体，Yo'LLaVA 定义如下软提示：

$$
\texttt{<sks> is <token}_1\texttt{><token}_2\texttt{>}\cdots\texttt{<token}_k\texttt{>.}
$$

其中 `<sks>` 是新增词表项，承担两个方向的接口：用户可以在问题中输入它，模型也能在答案中生成它。$k$ 个潜在 token 不对应离散单词，而是直接优化的连续向量，用来保存外观、纹理和局部特征。

![Yo'LLaVA 的训练管线与可学习参数](/images/posts/yollava-personalized-multimodal-assistant/figure2-training-pipeline.png)

*图源：Nguyen et al., [Yo'LLaVA: Your Personalized Language and Vision Assistant](https://arxiv.org/abs/2406.09400), Figure 2, NeurIPS 2024；从作者 CC BY 4.0 arXiv v2 源码直接栅格化。原图用于论文解读。*

设语言模型隐藏维度为 $C$、原词表大小为 $N$。新增 `<sks>` 后，输出分类头从 $C\times N$ 扩为 $C\times(N+1)$。论文实际更新的参数集合为：

$$
\theta=\{\texttt{<sks>},\texttt{<token}_1\texttt{>},\ldots,
\texttt{<token}_k\texttt{>},W_{:,N+1}\}.
$$

也就是 $k+1$ 个新增输入向量，以及让模型能够输出 `<sks>` 的最后一列分类权重。视觉编码器、视觉投影器和 LLM 其余参数全部冻结。

给定问题与目标回答 $X_a=(x_1,\ldots,x_L)$，训练仍使用回答 token 上的自回归损失：

$$
\mathcal L(\theta)=-\sum_{j=1}^{L}
\log p_\theta(x_j\mid I,X_q,X_{a,<j}).
$$

方法没有新增识别头。个性化概念通过普通语言建模接口进入和离开模型，因此识别与对话共享同一套表示。

## 关键模块一：三类训练数据各司其职

### 正样本与简单负样本

只用主体照片训练“图中有 `<sks>` 吗”，模型很容易学会一律回答“是”。论文为每个主体随机加入约 100 张 LAION 图像作为简单负样本，让正负答案都出现在训练中。

### CLIP 检索的难负样本

简单负样本只能教会类别边界。例如目标是一只黄色狗形玩偶，随机负样本会让模型学会“玩偶”，却可能把所有黄色玩偶都当成 `<sks>`。

论文对每张训练图 $I^i$ 用 CLIP 图像嵌入从 LAION 检索最相似的 $m$ 张图片。最终每个主体使用 100 张简单负样本和 $n\times m$ 张难负样本；默认总负样本约 200 张。难负样本迫使模型寻找区分同类实例的细粒度特征。

### 无图属性对话

作者为人物和非人物主体分别编写 10 个基础属性问题，例如发色、配饰、材质、颜色和纹理，再让原始 LLaVA 根据每张训练图生成答案。

关键细节是：训练这些属性对话时，论文**移除原始图片**。否则模型可以直接看图回答“它是什么颜色”，没有动力把颜色写入软 token。移除图像后，答案只能依赖个性化提示，视觉属性才会被蒸馏到连续向量中。

## 训练与推理流程

### 训练

1. 用户提供某个主体的 5 张照片，并分配唯一身份 token。
2. 系统加入正样本、约 100 张随机负样本和约 100 张 CLIP 检索难负样本。
3. 用 30 组正/负识别模板生成图像问答，再用 10 个属性问题合成无图对话。
4. 在 LLaVA-1.5-13B 上冻结原参数，只优化身份 token、16 个潜在 token 和对应输出列。
5. 使用 AdamW、学习率 0.001，最多训练 15 个 epoch；按训练集识别准确率保存最佳 checkpoint。

论文全部实验在单张 NVIDIA RTX A6000 上完成，但没有报告每个主体的实际训练时长、峰值显存或软提示文件大小。因此“训练快、存储轻”在结构上成立，系统成本仍缺少完整基准。

### 推理

- **新图识别**：输入待测图和“图中是否有 `<sks>`”，模型直接生成 Yes/No。
- **个性化视觉问答**：输入新图，询问 `<sks>` 的动作、位置或与环境的关系。
- **无图文本问答**：只输入身份 token 和问题，属性信息从 16 个潜在 token 读取。

每个主体需要单独训练并加载自己的 token 参数。论文没有实现多主体冲突消解、动态选择参数包或跨设备同步。

## 实验设置

作者收集 40 个主体：10 个人物、5 只宠物、5 个地标、15 个物体和 5 个虚构角色。每个主体有 10-20 张图片，划分训练与测试；默认取 5 张训练图。

识别评测包含 333 个正样本和 13,320 个负样本。因为负样本远多于正样本，论文没有直接报告总体正确率，而是使用正负准确率的等权平均：

$$
\text{Weighted Accuracy}
=0.5\times\text{Acc}_{pos}+0.5\times\text{Acc}_{neg}.
$$

问答评测共 571 道二选一题，其中 171 道带图视觉问答、400 道无图文本问答。主要基线包括无个性化的 LLaVA、人工或自动生成文本描述的 LLaVA/GPT-4V，以及给 GPT-4V 直接提供 1 张或 5 张参考图的图像提示。

## 主要结果：16 个 Token 换来了什么

### 主体识别

| 方法 | 提示规模 | 正样本准确率 | 负样本准确率 | 加权准确率 |
| --- | ---: | ---: | ---: | ---: |
| Yo'LLaVA | 16 个可学习 token | 0.949 | 0.898 | **0.924** |
| LLaVA，无提示 | 0 | 0.000 | 1.000 | 0.500 |
| LLaVA，自动短描述 | 约 16 token | 0.734 | 0.903 | 0.819 |
| LLaVA，自动长描述 | 约 1300 token | 0.320 | 0.980 | 0.650 |
| LLaVA，人工描述 | 约 16 token | 0.740 | 0.903 | 0.822 |
| GPT-4V，1 张参考图 | 约 1000 token | 0.809 | 0.992 | 0.901 |
| GPT-4V，5 张参考图 | 约 5000 token | 0.851 | 0.998 | **0.925** |

Yo'LLaVA 与 5 图 GPT-4V 的加权准确率几乎相同，但错误结构不同：Yo'LLaVA 的正样本召回更高，负样本准确率更低。不能只看 0.924 与 0.925 接近，就认为两个系统完全等价。

长文本描述反而从 0.819 降到 0.650，主要损失来自正样本准确率 0.734 降到 0.320。论文推测，拼接所有图像描述引入了背景等无关信息。这支持“表示应压缩主体稳定属性”，但不证明任意软提示都优于任意文本摘要。

![提示 token 数量与主体识别准确率](/images/posts/yollava-personalized-multimodal-assistant/figure4-token-efficiency.png)

*图源：Nguyen et al., [Yo'LLaVA: Your Personalized Language and Vision Assistant](https://arxiv.org/abs/2406.09400), Figure 4, NeurIPS 2024；从作者 CC BY 4.0 arXiv v2 源码直接栅格化，坐标轴、图例与对数横轴均保留。原图用于论文解读。*

图 4 的绿色星标说明 16 个软 token 达到 0.924；红色圆点则显示 GPT-4V 需要从约 16 个文本 token 扩展到约 5000 个图像 token 才逐步接近这一结果。这里比较的是**上下文 token 数**，不是端到端延迟或 GPU 成本，不能直接把横轴换算为同倍数加速。

### 个性化问答

Yo'LLaVA 的视觉问答准确率为 0.929，无图文本问答为 0.883。LLaVA 文本提示在无图问答上的最好结果为 0.803，说明连续提示确实保存了更多可查询视觉属性。

需要指出正式论文中的一处表述不一致：正文称 Yo'LLaVA 的 0.929 是视觉问答“领先结果”，但 Table 5 中 GPT-4V 的两种文本提示分别为 0.932 和 0.936。严格按表格，0.929 并非所有列最高。更可靠的结论是：它优于论文列出的 LLaVA 变体，并在不携带参考图的前提下保持竞争力。

无图文本问答也有同样边界。Yo'LLaVA 的 0.883 高于文本提示，但 GPT-4V 携带 1 张或 5 张参考图时达到 0.982/0.987。论文证明的是更紧凑的持久化表示，不是绝对能力超过带完整视觉证据的闭源模型。

### 与 MyVLM 比较

按照 MyVLM 的 29 个对象和评测协议，Yo'LLaVA 的正/负识别准确率为 97.0%/95.7%，加权 96.4%，MyVLM 为 96.6%/90.9%、加权 93.8%。主体 token 在图像描述中出现的 recall 为 100%，MyVLM 为 96%。

Yo'LLaVA 不需要额外人脸或物体识别器，这是系统简化；但结果来自另一套较小数据集，不能与前面 40 主体实验的 0.924 直接横向比较。

## 消融分析

![软 token 数与训练图片数的消融](/images/posts/yollava-personalized-multimodal-assistant/figure5-ablation.png)

*图源：Nguyen et al., [Yo'LLaVA: Your Personalized Language and Vision Assistant](https://arxiv.org/abs/2406.09400), Figure 5, NeurIPS 2024；从作者 CC BY 4.0 arXiv v2 源码直接栅格化，六个子图、图例、坐标轴与置信区域均保留。原图用于论文解读。*

### Token 数量

固定 10 张训练图时，只训练身份 token（$k=0$）的识别准确率约为 24%。随着潜在 token 增加，正样本识别明显改善；论文选择 $k=16$，在该消融中约为 91%，兼顾准确率和提示长度。曲线并非单调，32 个 token 后也没有持续提升，因此“更多 token 必然更好”不成立。

### 图片数量

固定 $k=16$ 时，1 张图的加权准确率约 0.68，5 张图首次超过 0.90，10 张约 0.92。5 张是作者在该数据上的成本折中，不是跨主体的理论最小值；小脸、姿态变化大或外观高度相似的主体可能需要更多覆盖。

### 数据组成

Table 7 给出更直接的增量结果：

| 训练数据 | 加权识别准确率 | 能否回答未见过的属性问题 |
| --- | ---: | --- |
| 原始 LLaVA | 0.500 | 否 |
| 仅识别问答 | 0.707 | 否 |
| 加入属性对话 | 0.754 | 是 |
| 再加入检索难负样本 | **0.914** | 是 |

属性对话主要补上“能谈论这个主体”，难负样本则带来最大的识别增益。这说明软 token 只是容器，训练数据决定其中学到的是类别捷径还是实例特征。

### 遗忘评测

在 POPE 三个切分上，Yo'LLaVA 比原 LLaVA 各低约 0.01；MMBench 英文均为 0.68，LLaVA-Wild 均为 72.3。核心权重冻结确实把遗忘控制在很小范围，但这只覆盖三个通用基准，并不等于所有基础能力严格不变。

## 失败案例与局限

论文附录 Table 15 主动展示了四类错误：目标人脸在图中较小时出现假阴性；相似人物出现假阳性；无依据编造某人的生日是 12 月 25 日；描述中虚构并不存在的手表。作者将前两类归因于细粒度识别不足，将后两类归因于基础语言模型的幻觉。

除此之外，工程落地还需要关注：

1. **隐私与身份安全**：人物软 token 可能编码近似生物特征。论文没有成员推断、身份冒用、删除证明或访问控制实验。
2. **评测规模有限**：只有 40 个主体，问题主要是模板生成和人工二选一，不能覆盖真实多轮对话与开放世界误识别。
3. **每个主体都要训练**：论文未报告多用户参数管理、并发加载、版本迁移或训练失败恢复。
4. **偏差会被继承**：作者明确指出 CLIP 与 LLaMA/Vicuna 的偏差可能传递到个性化回答，人物属性推断尤其敏感。
5. **负样本授权复杂**：难负样本来自 LAION。官方仓库明确说明不拥有这些检索图像的版权，仅供研究。
6. **复现成熟度有限**：截至核验时，官方 README 仍称代码“under construction”，且未提供独立 LICENSE 文件；不能把论文开放与代码可直接商用混为一谈。

## 可复现资源

- [NeurIPS 正式页面与正式 PDF](https://papers.neurips.cc/paper_files/paper/2024/hash/48088756ec0ce6ba362bddc7ebeb3915-Abstract-Conference.html)
- [arXiv v2 全文、源码与 CC BY 4.0 许可](https://arxiv.org/abs/2406.09400)
- [官方代码仓库](https://github.com/WisconsinAIVision/YoLLaVA)
- [官方项目主页](https://thaoshibe.github.io/YoLLaVA/)
- [40 主体数据集](https://huggingface.co/datasets/thaoshibe/YoLLaVA)

项目基于 LLaVA，仓库给出的入口是 `train-multi-token.py`，测试脚本为 `test-sks.py` 和 `test-sks-qa.py`。项目页注明数据集为 CC BY-NC 4.0，代码、checkpoint 和数据仅面向研究，并受 CLIP、LLaMA、Vicuna 等上游许可约束。复现前应分别核查论文图片、数据、代码和基础模型的许可，而不是用一个许可证概括全部资产。

## 个人判断

Yo'LLaVA 把个性化拆成了两个很干净的接口：一个离散身份 token 负责命名，一组连续 token 负责视觉记忆。这种表示比每次携带多张参考图更适合作为服务端“用户概念槽位”，也可以与检索式用户档案、偏好记忆和动态路由组合。

但论文最值得后续补齐的并不是再刷一点识别分数，而是**生命周期**：谁能创建一个人物 token，如何证明获得授权，如何撤销，如何发现两个 token 指向同一人，以及如何阻止一个用户加载另一个用户的概念参数。个性化一旦从“我的玩偶”扩展到“我的朋友”，参数效率就不再是唯一指标，治理能力会成为系统设计的一部分。

因此，我会把 Yo'LLaVA 定位为：一个结构简洁、实验证据充分的个性化视觉概念注入基线。它证明了 5 张图和 16 个软 token 可以让冻结 LMM 获得实例级记忆，但还没有证明这种记忆已具备生产环境需要的可靠性、隐私和运维属性。

## 参考资料

1. Nguyen et al., [Yo'LLaVA: Your Personalized Language and Vision Assistant](https://papers.neurips.cc/paper_files/paper/2024/hash/48088756ec0ce6ba362bddc7ebeb3915-Abstract-Conference.html), NeurIPS 2024.
2. Nguyen et al., [arXiv:2406.09400 v2](https://arxiv.org/abs/2406.09400), 2024.
3. WisconsinAIVision, [YoLLaVA Code and Dataset](https://github.com/WisconsinAIVision/YoLLaVA).
4. Alaluf et al., [MyVLM: Personalizing VLMs for User-Specific Queries](https://arxiv.org/abs/2403.14599), 2024.
