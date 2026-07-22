---
title: Transformer、ViT 与 CLIP 基础：从三篇代表论文到高频面试题
date: 2026-07-13 10:00:00
description: 以三篇代表论文为主线，串起 Transformer、Vision Transformer 与 CLIP 的结构、目标和面试高频题。
series: LLM 核心原理
seriesOrder: 1
categories:
  - AI
tags:
  - Transformer
  - ViT
  - CLIP
  - 多模态
  - 深度学习
  - 面试
hidden: true
haloPublished: true
---

Transformer、Vision Transformer（ViT）与 CLIP 是理解现代大模型、多模态模型和视觉基础模型最重要的三个入口。它们分别回答了三个问题：

- **Transformer**：不使用 RNN 和 CNN，能否只靠注意力完成序列建模？
- **ViT**：图像能否像文本一样，被切成 token 后交给 Transformer？
- **CLIP**：图像和自然语言能否在同一个语义空间中对齐，从而实现零样本识别？

本文以三篇代表性论文为主线：

1. *Attention Is All You Need*（Vaswani et al., 2017）
2. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*（Dosovitskiy et al., 2020）
3. *Learning Transferable Visual Models From Natural Language Supervision*（Radford et al., 2021）

目标不是逐句翻译论文，而是建立一套适合面试复习的知识框架：**动机、结构、公式、训练目标、优缺点、模型关系与高频追问**。

## 1. 一张表看懂三者关系

| 模型 | 输入 | 核心结构 | 训练目标 | 解决的问题 |
| --- | --- | --- | --- | --- |
| Transformer | token 序列 | Multi-Head Attention + FFN | 自回归或序列到序列损失 | 长距离依赖与并行序列建模 |
| ViT | 图像 Patch 序列 | Transformer Encoder | 图像分类损失 | 将 Transformer 扩展到视觉 |
| CLIP | 图像与文本对 | 图像编码器 + 文本编码器 | 图文对比学习 | 学习可迁移的跨模态语义空间 |

三者并不是互相独立的模型：

> ViT 把图像转换成 Transformer 能处理的 token；CLIP 再使用 ViT 或 ResNet 编码图像，使用 Transformer 编码文本，并通过对比学习把二者对齐。

---

## 2. Transformer：Attention Is All You Need

### 2.1 为什么要摆脱 RNN

Transformer 出现前，机器翻译通常依赖 RNN、LSTM 或 GRU。循环网络有两个突出问题：

1. **难以并行**：第 $t$ 个时间步依赖第 $t-1$ 个时间步，训练必须顺序执行。
2. **长距离依赖路径长**：相距很远的两个 token，需要经过多次状态传递才能交互。

Self-Attention 让任意两个 token 可以在一层内直接建立联系，而且整个序列可以并行计算。

### 2.2 Scaled Dot-Product Attention

给定输入表示 $X$，通过三个线性映射得到：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

- Query：当前 token 想查找什么信息；
- Key：每个 token 可以用什么特征被匹配；
- Value：真正被加权汇总的内容。

注意力计算为：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其计算过程可以拆成四步：

1. 使用 $QK^\top$ 计算 token 两两相似度；
2. 除以 $\sqrt{d_k}$ 控制数值尺度；
3. 使用 Softmax 转换为权重；
4. 对 $V$ 做加权求和。

#### 为什么要除以 $\sqrt{d_k}$？

如果 $Q$ 和 $K$ 的各维独立且方差约为 1，那么点积的方差会随维度 $d_k$ 增大。过大的点积会让 Softmax 进入饱和区，梯度变小。缩放后可以让训练更稳定。

### 2.3 Multi-Head Attention

单头注意力只在一个表示子空间中计算关系。多头注意力将特征拆分为多个头：

$$
\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
\operatorname{MultiHead}(Q,K,V)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
$$

不同头可以关注不同类型的关系，例如局部搭配、句法结构、指代关系或全局语义。需要注意：**多头并不保证每个头都天然具有可解释语义**，它首先是一种增加表示能力的参数化方式。

![Transformer 原论文中的编码器与解码器架构](/images/posts/transformer-vit-clip/transformer-figure-1.png)

*图源：Vaswani et al., [Attention Is All You Need](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need), Figure 1, NeurIPS 2017。原图用于论文结构解读。*

### 2.4 Encoder 与 Decoder

原论文中的 Transformer 是 Encoder-Decoder 架构。

#### Encoder Layer

每层包含：

1. Multi-Head Self-Attention；
2. Position-wise Feed-Forward Network；
3. 每个子层外都有残差连接与 LayerNorm。

FFN 对每个 token 独立应用相同的两层 MLP：

$$
\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

注意力负责 **token 之间的信息混合**，FFN 负责 **每个 token 内部的通道变换**。

#### Decoder Layer

Decoder 比 Encoder 多一个 Cross-Attention，并使用 Masked Self-Attention：

- Masked Self-Attention：当前 token 不能看到未来 token；
- Cross-Attention：Query 来自 Decoder，Key 和 Value 来自 Encoder 输出。

### 2.5 为什么需要位置编码

Self-Attention 本身对输入顺序没有感知。如果交换 token 的位置，同时交换对应的表示，注意力结果也会相应交换。因此需要显式加入位置信息。

原论文使用正弦与余弦位置编码：

$$
PE_{(pos,2i)}=\sin\left(pos/10000^{2i/d_{model}}\right)
$$

$$
PE_{(pos,2i+1)}=\cos\left(pos/10000^{2i/d_{model}}\right)
$$

这种编码不需要训练，并允许模型通过线性关系推断相对位置。后续模型也常使用可学习位置编码、RoPE、ALiBi 或相对位置偏置。

### 2.6 原论文 Base 配置

经典 Transformer Base 的常见参数是：

- Encoder 6 层，Decoder 6 层；
- $d_{model}=512$；
- 8 个注意力头；
- FFN 隐藏维度 2048；
- Dropout 0.1。

这些数字不是 Transformer 的定义，只是原论文中的代表性配置。

### 2.7 Self-Attention 的复杂度

长度为 $n$、隐藏维度为 $d$ 时，标准注意力需要构造 $n\times n$ 的注意力矩阵：

- 时间复杂度通常写作 $O(n^2d)$；
- 注意力矩阵空间复杂度为 $O(n^2)$。

这也是长序列模型研究稀疏注意力、线性注意力、FlashAttention 和状态空间模型的重要原因。

### 2.8 Transformer 高频面试题

#### Q1：Self-Attention 与 Cross-Attention 有什么区别？

Self-Attention 的 $Q,K,V$ 来自同一序列；Cross-Attention 的 Query 来自一个序列，Key 和 Value 来自另一个序列。

#### Q2：为什么 Transformer 比 RNN 更容易并行？

Transformer 一层内所有 token 的 $Q,K,V$ 可以同时计算；RNN 的隐藏状态存在时间步依赖。

#### Q3：Residual 与 LayerNorm 分别有什么作用？

残差连接改善深层网络的梯度传播，并保留原始表示；LayerNorm 稳定特征尺度。现代大模型常使用 Pre-Norm，即先归一化再进入子层。

#### Q4：Encoder-only、Decoder-only、Encoder-Decoder 分别适合什么任务？

- Encoder-only：双向理解，如 BERT、分类、检索；
- Decoder-only：自回归生成，如 GPT；
- Encoder-Decoder：输入到输出的条件生成，如翻译、摘要、T5。

#### Q5：Attention Mask 有哪些？

- Padding Mask：屏蔽补齐位置；
- Causal Mask：屏蔽未来 token；
- 任务自定义 Mask：控制局部窗口、块稀疏或可见区域。

---

## 3. ViT：把图像变成 token 序列

### 3.1 ViT 的核心假设

ViT 的核心非常直接：

> 一张图像可以被切成固定大小的 Patch，每个 Patch 类似 NLP 中的一个 token。

假设输入图像尺寸为 $H\times W$，Patch 尺寸为 $P\times P$，那么 Patch 数量为：

$$
N=\frac{H\times W}{P^2}
$$

例如 $224\times224$ 图像使用 $16\times16$ Patch，会得到：

$$
N=\frac{224\times224}{16\times16}=196
$$

每个 Patch 展平后维度为 $P^2C$，再经过线性层映射到 $D$ 维 token embedding。

![ViT 原论文中的模型总览](/images/posts/transformer-vit-clip/vit-figure-1.png)

*图源：Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929), Figure 1, ICLR 2021。原图用于模型流程解读。*

### 3.2 ViT 的输入序列

ViT 的输入可以写为：

$$
z_0=[x_{class};x_p^1E;x_p^2E;\ldots;x_p^NE]+E_{pos}
$$

其中：

- $x_p^i$：第 $i$ 个图像 Patch；
- $E$：Patch Embedding 线性投影；
- $x_{class}$：可学习的 `[CLS]` token；
- $E_{pos}$：可学习的位置编码。

经过多层 Transformer Encoder 后，使用 `[CLS]` 对应的最终表示完成分类。

### 3.3 Patch Embedding 与卷积的关系

Patch Embedding 可以用 `Conv2d(kernel_size=P, stride=P)` 实现。因为卷积核大小和步长都等于 Patch 大小，所以每次处理一个不重叠 Patch。

区别在于：

- CNN 会逐层使用局部卷积建立层级特征；
- 原始 ViT 一开始就把 Patch 当作序列 token，后续主要依赖全局注意力交互。

### 3.4 ViT 为什么需要大数据预训练

CNN 天然具有两个重要归纳偏置：

1. **局部性**：相邻像素更容易相关；
2. **平移等变性**：同一个卷积核可以在不同位置复用。

ViT 的归纳偏置更弱。它能够学习更灵活的全局关系，但在数据量较小时不一定优于 CNN。原论文显示，随着预训练数据规模增加，ViT 的性能和迁移能力会明显增强。

这也是论文标题中 “at Scale” 的关键含义：ViT 的优势与训练规模密切相关。

### 3.5 Patch 越小越好吗

Patch 越小，保留的局部细节越多，但 token 数量会平方级增加。

以 $224\times224$ 图像为例：

| Patch 大小 | token 数量 | 特点 |
| --- | ---: | --- |
| $32\times32$ | 49 | 计算便宜，但细节较粗 |
| $16\times16$ | 196 | 常见性能与成本折中 |
| $8\times8$ | 784 | 细节更丰富，注意力成本显著增加 |

标准注意力复杂度与 token 数量平方相关，因此 Patch 大小是视觉 Transformer 的关键计算旋钮。

### 3.6 ViT-B/16 怎么读

- `B`：Base 模型规模；
- `/16`：Patch 大小为 $16\times16$。

类似地，ViT-L/16 表示 Large 规模、16 像素 Patch；ViT-H/14 表示 Huge 规模、14 像素 Patch。

### 3.7 ViT 高频面试题

#### Q1：ViT 中的位置编码为什么重要？

Patch token 本身不包含二维位置信息。没有位置编码，模型难以区分某个 Patch 位于左上角还是右下角。

#### Q2：ViT 的 `[CLS]` token 有什么作用？

它作为全局聚合 token，通过多层注意力从所有 Patch 收集信息，最终用于分类。后续工作也会使用所有 token 的平均池化代替 `[CLS]`。

#### Q3：ViT 与 CNN 的主要差别是什么？

CNN 具有强局部归纳偏置和层级结构；ViT 使用全局 Self-Attention，归纳偏置较弱，更依赖数据和预训练规模。

#### Q4：图像分辨率变化时位置编码怎么办？

常见做法是把二维位置编码恢复为网格后进行插值，再适配新的 Patch 网格尺寸。

#### Q5：ViT 的 Attention Map 能否直接解释模型？

它能提供一定的可视化线索，但注意力权重不等于严格因果解释。通常还需结合 Attention Rollout、梯度或遮挡实验。

---

## 4. CLIP：用自然语言监督视觉表示

### 4.1 CLIP 想解决什么问题

传统图像分类模型依赖固定类别和人工标注，例如 ImageNet 的 1000 个类别。模型训练完成后，分类头只能预测预先定义的标签。

CLIP 的思路是：互联网上天然存在大量“图像 + 文本”配对数据，能否直接利用自然语言作为开放词汇监督？

CLIP 使用约 4 亿图文对进行预训练，学习一个共享语义空间，使匹配的图像与文本相似，不匹配的图像与文本远离。

### 4.2 双塔编码器

CLIP 包含两个独立编码器：

- Image Encoder：可以是 ResNet，也可以是 ViT；
- Text Encoder：Transformer。

图像和文本分别编码为向量，再进行 L2 归一化：

$$
v_i=\frac{f_{image}(I_i)}{\|f_{image}(I_i)\|},\quad
t_i=\frac{f_{text}(T_i)}{\|f_{text}(T_i)\|}
$$

归一化后，点积等价于余弦相似度。

![CLIP 原论文中的对比预训练与零样本分类流程](/images/posts/transformer-vit-clip/clip-figure-1.png)

*图源：Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html), Figure 1, ICML 2021。原图用于训练目标与零样本流程解读。*

### 4.3 对比学习目标

一个 batch 中有 $N$ 对图像和文本。计算所有图像与所有文本的相似度，得到 $N\times N$ 矩阵：

$$
S_{ij}=\frac{v_i^\top t_j}{\tau}
$$

其中 $\tau$ 是温度参数。矩阵对角线 $(i=j)$ 是正样本，其余位置是负样本。

CLIP 使用对称的交叉熵损失：

$$
\mathcal{L}_{\mathrm{image}\rightarrow\mathrm{text}}
=-
\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(S_{ii})}{\sum_{j=1}^{N}\exp(S_{ij})}
$$

$$
\mathcal{L}_{\mathrm{text}\rightarrow\mathrm{image}}
=-
\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(S_{ii})}{\sum_{j=1}^{N}\exp(S_{ji})}
$$

最终的对称损失为：

$$
\mathcal{L}=\frac{1}{2}
\left(
\mathcal{L}_{\mathrm{image}\rightarrow\mathrm{text}}
+
\mathcal{L}_{\mathrm{text}\rightarrow\mathrm{image}}
\right)
$$

它同时训练：

- 给定图像，找出匹配文本；
- 给定文本，找出匹配图像。

### 4.4 为什么需要温度参数

温度控制 Softmax 分布的尖锐程度：

- 温度较小：模型更强调最相似样本，分布更尖锐；
- 温度较大：相似度分布更平滑。

CLIP 中通常学习的是 logit scale，它决定归一化相似度在进入 Softmax 前被放大的程度。

### 4.5 CLIP 如何做零样本分类

假设分类类别为 `cat`、`dog`、`car`：

1. 为每个类别构造文本模板，如 `a photo of a cat`；
2. 使用 Text Encoder 得到类别文本向量；
3. 使用 Image Encoder 得到待分类图像向量；
4. 计算图像与所有类别文本的相似度；
5. 选择相似度最高的类别。

这相当于把文本编码器动态生成的向量作为分类器权重，因此无需重新训练固定分类头。

### 4.6 Prompt Engineering 为什么有效

只输入 `cat` 和输入 `a photo of a cat` 的分布不同。CLIP 在自然语言描述上训练，使用更接近训练语料的模板通常效果更好。

论文还使用 Prompt Ensembling：为同一类别设计多个模板，对文本向量进行组合，以降低单个模板带来的偏差。

### 4.7 CLIP 的能力与局限

#### 优点

- 开放词汇与零样本迁移；
- 图文检索天然统一；
- 可作为生成模型、检测、分割和多模态大模型的视觉基础编码器；
- 类别可以通过自然语言动态扩展。

#### 局限

- 训练数据来自互联网，可能包含偏见、噪声和不当内容；
- 对细粒度计数、空间关系、文字识别等任务可能不稳定；
- 零样本性能依赖 Prompt；
- 对训练分布之外的概念仍可能失败；
- 双塔模型交互高效，但细粒度跨模态融合弱于 Cross-Encoder。

### 4.8 CLIP 高频面试题

#### Q1：CLIP 是生成模型吗？

原始 CLIP 不是生成模型，而是图文双塔对比学习模型。它输出跨模态表示和相似度。Stable Diffusion 等生成模型会使用 CLIP 文本编码器作为条件编码器。

#### Q2：为什么 CLIP 适合检索？

图像和文本可以离线编码到同一向量空间，在线只需做向量相似度搜索，不必对每一对图文运行复杂交互网络。

#### Q3：CLIP 与普通多分类有什么区别？

普通分类器学习固定类别权重；CLIP 使用文本编码器生成类别表示，类别集合可以在推理时改变。

#### Q4：CLIP 的负样本从哪里来？

同一个 batch 中，除匹配图文对之外的其他组合都作为 batch 内负样本。因此更大的 batch 往往能提供更多负样本。

#### Q5：双塔与 Cross-Encoder 如何取舍？

双塔可提前编码，适合大规模召回；Cross-Encoder 让图文 token 深度交互，精度通常更高但计算昂贵，适合重排序。

---

## 5. Transformer、ViT 与 CLIP 如何串起来

### 5.1 结构继承关系

```text
Transformer
  ├─ 用 Self-Attention 建模 token 序列
  ├─ 文本 token + 位置编码
  └─ Encoder / Decoder 基础结构

ViT
  ├─ 图像切成 Patch token
  ├─ 加入 [CLS] 与位置编码
  └─ 复用 Transformer Encoder

CLIP
  ├─ Image Encoder：ResNet 或 ViT
  ├─ Text Encoder：Transformer
  └─ 图文对比学习 + 零样本分类
```

### 5.2 三者关注的层级不同

- Transformer 是一种通用的 **序列建模架构**；
- ViT 是 Transformer 在视觉分类上的 **输入表示与架构应用**；
- CLIP 是围绕图文数据设计的 **预训练目标与跨模态系统**。

面试时不要把它们放在同一层级比较。例如，“ViT 和 CLIP 有什么区别”更准确的回答是：ViT 通常是视觉编码器架构，CLIP 是图文对比预训练框架，CLIP 的图像编码器可以使用 ViT。

---

## 6. 综合高频八股速答

### 6.1 为什么 Q 和 K 使用不同参数矩阵？

Query 表示“我要寻找什么”，Key 表示“我拥有什么可供匹配的特征”。使用不同投影可以学习非对称的匹配空间；即使在 Self-Attention 中输入相同，角色也不同。

### 6.2 为什么 V 不参与注意力权重计算？

Q 和 K 决定路由权重，V 携带被聚合的信息。分离匹配与内容表示能增强灵活性。

### 6.3 Attention 一定比卷积好吗？

不一定。卷积在小数据、局部结构和高分辨率场景具有很强归纳偏置与计算优势。Attention 更擅长全局交互，但成本和数据需求可能更高。

### 6.4 ViT 为什么常用 LayerNorm 而不是 BatchNorm？

Transformer 沿特征维度归一化每个 token，不依赖 batch 统计，因此对 batch 大小和序列长度更稳定，也与 NLP Transformer 结构保持一致。

### 6.5 CLIP 的 embedding 能直接做向量数据库检索吗？

可以。通常将 embedding 归一化后使用余弦相似度或内积建立 ANN 索引。需要保证图像和文本使用同一 CLIP 模型及一致的预处理。

### 6.6 对比学习中 batch size 为什么重要？

batch 内其他样本提供负例。batch 越大，负例通常越丰富，但也可能出现“假负样本”：语义相关却未被标记为配对。

### 6.7 CLIP 能否理解文本中的否定？

不一定可靠。CLIP 更擅长整体语义对齐，对组合关系、否定、数量和精细空间关系可能表现有限，需要具体评测验证。

### 6.8 ViT 如何用于检测和分割？

需要保留 Patch token 的空间网格，而不是只使用 `[CLS]`。常结合多尺度特征、特征金字塔、检测头或掩码解码器。

### 6.9 为什么现代视觉模型会使用分层或窗口注意力？

高分辨率图像 token 很多，全局注意力成本高。分层结构和局部窗口可以降低复杂度，同时建立类似 CNN 的多尺度表示。

### 6.10 FlashAttention 改变了 Attention 的数学结果吗？

标准情况下不改变目标计算结果。它通过分块、重计算和减少 HBM 读写，使精确注意力更符合 GPU IO 特性。

---

## 7. 面试回答模板

如果面试官让你快速介绍三者，可以按下面的顺序回答：

> Transformer 用缩放点积注意力替代循环结构，使 token 可以并行建模全局依赖；ViT 把图像切成 Patch 并映射成 token，加入位置编码和分类 token 后送入 Transformer Encoder；CLIP 再把 ViT 或 ResNet 作为图像编码器、Transformer 作为文本编码器，通过对称对比损失学习图文共享空间，从而实现图文检索和零样本分类。三者的主线是：通用序列建模、视觉 token 化、跨模态语义对齐。

回答后再根据追问展开公式、复杂度、归纳偏置或训练目标，不要一开始就堆细节。

---

## 8. 复习清单

### Transformer

- [ ] 能写出 Attention 公式并解释缩放项；
- [ ] 能区分 Self-Attention、Masked Attention 与 Cross-Attention；
- [ ] 能解释多头、位置编码、残差、LayerNorm 和 FFN；
- [ ] 能分析 $O(n^2)$ 复杂度；
- [ ] 能区分 Encoder-only、Decoder-only 与 Encoder-Decoder。

### ViT

- [ ] 能计算 Patch token 数量；
- [ ] 能解释 Patch Embedding、`[CLS]` 和位置编码；
- [ ] 能比较 ViT 与 CNN 的归纳偏置；
- [ ] 能说明 Patch 大小对精度与计算量的影响；
- [ ] 能解释为什么 ViT 依赖大规模预训练。

### CLIP

- [ ] 能画出图像塔、文本塔和相似度矩阵；
- [ ] 能解释对称对比损失和温度参数；
- [ ] 能描述零样本分类流程；
- [ ] 能比较双塔与 Cross-Encoder；
- [ ] 能说明 CLIP 的优势、偏见与能力边界。

---

## 9. 代表论文

1. Vaswani et al. **Attention Is All You Need**. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Dosovitskiy et al. **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**. ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
3. Radford et al. **Learning Transferable Visual Models From Natural Language Supervision**. ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

如果只记住一句话：

> Transformer 定义了 token 如何全局交互，ViT 定义了图像如何变成 token，CLIP 定义了图像 token 与语言语义如何对齐。
