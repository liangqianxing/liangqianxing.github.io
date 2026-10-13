# Attention Is All You Need

**题目：**只需注意力机制

**作者：**Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Łukasz Kaiser、Illia Polosukhin

**来源：**NeurIPS 2017，第 31 届神经信息处理系统大会，美国长滩。

> 说明：本文根据用户提供的论文 PDF 翻译。公式、变量、代码仓库地址和引用编号保留原样。

## 摘要

主流的序列转换模型建立在复杂的循环神经网络或卷积神经网络之上，通常包含编码器和解码器。性能最好的模型还会通过注意力机制连接编码器和解码器。

我们提出一种新的简单网络架构 Transformer，它完全建立在注意力机制之上，彻底舍弃循环和卷积。两个机器翻译任务上的实验表明，Transformer 在质量上优于已有模型，同时更易于并行化，训练所需时间也显著更少。

在 WMT 2014 英德翻译任务上，我们的模型取得 28.4 BLEU，比包括集成模型在内的已有最佳结果高出 2 个以上 BLEU。在 WMT 2014 英法翻译任务上，模型只使用 8 块 GPU 训练 3.5 天，就取得了 41.0 的单模型最新 BLEU 分数，而训练成本只是文献中最佳模型的一小部分。

## 1 引言

循环神经网络，尤其是长短期记忆网络 [12] 和门控循环神经网络 [7]，已经成为序列建模和序列转换问题（例如语言建模与机器翻译）的事实标准 [29, 2, 5]。此后，大量工作继续推动循环语言模型以及编码器-解码器架构的发展 [31, 21, 13]。

循环模型通常沿输入和输出序列的符号位置分解计算。把序列位置与计算时间步对齐后，它们根据前一时刻的隐藏状态 `h_{t-1}` 和位置 `t` 的输入，生成一系列隐藏状态 `h_t`。这种天然的顺序性阻止了训练样本内部的并行化；当序列变长时，由于内存限制，跨样本进行批处理的能力也会下降，因此这一问题更加关键。近期工作通过因式分解技巧 [18] 和条件计算 [26] 显著提高了计算效率，后者还改善了模型性能，但顺序计算这一根本限制仍然存在。

注意力机制已经成为许多优秀序列建模和序列转换模型的重要组成部分。它可以在不考虑输入或输出位置之间距离的情况下建模依赖关系 [2, 16]。然而，除少数例外 [22] 外，这些注意力机制通常仍与循环网络结合使用。

本文提出 Transformer：一种舍弃循环、完全依赖注意力机制来建立输入与输出之间全局依赖关系的模型架构。Transformer 可以实现大幅并行化，只需在 8 块 P100 GPU 上训练 12 小时，就能在翻译质量上达到新的最好结果。

## 2 背景

减少顺序计算的目标，也是 Extended Neural GPU [20]、ByteNet [15] 和 ConvS2S [8] 的基础。这些模型都把卷积神经网络作为基本构件，为输入和输出的所有位置并行计算隐藏表示。

在这些模型中，关联任意两个输入或输出位置所需的操作数，会随着位置间距离增长：ConvS2S 呈线性增长，ByteNet 呈对数增长。这使得模型更难学习远距离依赖 [11]。在 Transformer 中，这个操作数被降低为常数，但代价是有效分辨率下降，因为注意力会对不同位置进行加权平均。我们通过第 3.2 节介绍的多头注意力来抵消这一影响。

自注意力（self-attention），有时也称为内部注意力（intra-attention），是一种关联同一序列中不同位置的注意力机制，用于计算整个序列的表示。自注意力已成功用于阅读理解、抽象式摘要、文本蕴含和学习与任务无关的句子表示等多种任务 [4, 22, 23, 19]。

端到端记忆网络使用循环注意力机制，而不是与序列位置对齐的循环结构；它们在简单语言问答和语言建模任务上表现良好 [28]。

据我们所知，Transformer 是第一个完全依靠自注意力计算输入和输出表示、而不使用与序列位置对齐的 RNN 或卷积的序列转换模型。下面我们将描述 Transformer，说明采用自注意力的动机，并讨论它相对于 [14, 15] 和 [8] 等模型的优势。

## 3 模型架构

大多数有竞争力的神经序列转换模型都采用编码器-解码器结构 [5, 2, 29]。编码器把输入符号表示序列 `(x_1, ..., x_n)` 映射为连续表示序列 `z = (z_1, ..., z_n)`。给定 `z` 后，解码器逐个生成输出符号序列 `(y_1, ..., y_m)`。

在每一步，模型都是自回归的 [9]：生成下一个符号时，把此前已经生成的符号作为额外输入。Transformer 保留了整体的编码器-解码器结构，但编码器和解码器都使用堆叠的自注意力层与逐位置全连接层，如图 1 左右两部分所示。

### 3.1 编码器和解码器堆栈

**编码器：**编码器由 `N = 6` 个相同层堆叠而成。每层包含两个子层：第一个是多头自注意力机制，第二个是简单的逐位置全连接前馈网络。我们在两个子层外都使用残差连接 [10]，然后进行层归一化 [1]。也就是说，每个子层的输出为：

`LayerNorm(x + Sublayer(x))`

其中 `Sublayer(x)` 是该子层本身实现的函数。为了能够使用残差连接，模型中的所有子层以及嵌入层都产生维度为 `d_model = 512` 的输出。

**解码器：**解码器同样由 `N = 6` 个相同层堆叠而成。除了编码器每层中的两个子层外，解码器还插入第三个子层，对编码器堆栈的输出执行多头注意力。

与编码器类似，我们在每个子层外使用残差连接，再进行层归一化。我们还修改了解码器堆栈中的自注意力子层，使某个位置不能关注后续位置。这种掩码与输出嵌入向右偏移一个位置相结合，可以保证位置 `i` 的预测只能依赖位置 `i` 之前已经知道的输出。

### 3.2 注意力

注意力函数可以描述为：把一个查询（query）和一组键-值对（key-value pairs）映射为一个输出；查询、键、值和输出都是向量。输出是值的加权和，每个值的权重由查询与对应键之间的兼容性函数计算得到。

#### 3.2.1 缩放点积注意力

我们使用的具体注意力函数称为“缩放点积注意力”（Scaled Dot-Product Attention），如图 2 所示。输入包含维度为 `d_k` 的查询和键，以及维度为 `d_v` 的值。我们计算查询与所有键的点积，把每个点积除以 `√d_k`，再应用 softmax 函数得到各个值的权重。

实践中，我们会同时对一组查询计算注意力，并把它们打包成矩阵 `Q`。键和值也分别打包成矩阵 `K` 和 `V`。输出矩阵计算如下：

`Attention(Q, K, V) = softmax(QK^T / √d_k)V`  （1）

最常用的两种注意力函数是加性注意力 [2] 和点积（乘性）注意力。除缩放因子外，点积注意力与我们的算法相同。加性注意力使用一个带单隐藏层的前馈网络来计算兼容性函数。两者在理论复杂度上相似，但实践中点积注意力更快、更节省空间，因为它可以使用高度优化的矩阵乘法代码实现。

当 `d_k` 较小时，两种机制表现相近；但当 `d_k` 较大时，不带缩放的加性注意力优于点积注意力 [3]。我们推测，当 `d_k` 较大时，点积的数值幅度会变大，使 softmax 进入梯度极小的区域。为了抵消这种影响，我们用 `1/√d_k` 对点积进行缩放。

为了说明点积为什么会变大，假设 `q` 和 `k` 的各个分量相互独立，均值为 0、方差为 1。那么点积 `q · k = Σ q_i k_i` 的均值为 0，方差为 `d_k`。

#### 3.2.2 多头注意力

我们发现，与其对维度为 `d_model` 的键、值和查询执行一次注意力函数，不如先使用不同的、可学习的线性投影，把查询、键和值分别投影 `h` 次，映射到维度 `d_k`、`d_k` 和 `d_v`，然后在每组投影后的查询、键和值上并行执行注意力函数，得到维度为 `d_v` 的输出。最后把这些输出拼接起来，再进行一次线性投影，得到最终结果，如图 2 所示。

多头注意力使模型可以在不同位置、联合关注不同表示子空间中的信息。只有一个注意力头时，平均操作会抑制这种能力。

`MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O`

其中：

`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`

投影矩阵为：

`W_i^Q ∈ R^(d_model × d_k)`，`W_i^K ∈ R^(d_model × d_k)`，`W_i^V ∈ R^(d_model × d_v)`，`W^O ∈ R^(h d_v × d_model)`。

本文使用 `h = 8` 个并行注意力层，即 8 个头。每个头使用 `d_k = d_v = d_model / h = 64`。由于每个头的维度降低，总计算成本与完整维度的单头注意力相近。

#### 3.2.3 注意力在本文模型中的应用

Transformer 以三种不同方式使用多头注意力：

- **编码器-解码器注意力：**查询来自前一个解码器层，记忆键和值来自编码器输出。因此，解码器的每个位置都可以关注输入序列的所有位置。这类似于 [31, 2, 8] 等序列到序列模型中的典型编码器-解码器注意力机制。
- **编码器自注意力：**编码器中的键、值和查询都来自同一个位置，即编码器前一层的输出。因此，编码器中的每个位置都可以关注编码器前一层的所有位置。
- **解码器自注意力：**解码器中的每个位置都可以关注解码器中截至该位置的所有位置。为了保持自回归性质，必须阻止信息向左之外的未来位置流动。我们在缩放点积注意力中通过掩码实现这一点：把 softmax 输入中对应非法连接的值设置为 `−∞`。

### 3.3 逐位置前馈网络

除了注意力子层外，编码器和解码器的每一层还包含一个逐位置、参数共享的全连接前馈网络。它由两个线性变换组成，中间使用 ReLU 激活：

`FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`  （2）

不同位置使用相同的线性变换，但不同层使用不同参数。换一种描述方式，它相当于两个 kernel size 为 1 的卷积。

输入和输出维度为 `d_model = 512`，内层维度为 `d_ff = 2048`。

### 3.4 嵌入与 Softmax

与其他序列转换模型类似，我们使用可学习的嵌入，把输入 token 和输出 token 转换为维度为 `d_model` 的向量。我们还使用通常的可学习线性变换和 softmax 函数，把解码器输出转换为下一个 token 的预测概率。

与 [24] 类似，我们在两个嵌入层和 softmax 前的线性变换之间共享同一个权重矩阵。在嵌入层中，我们把这些权重乘以 `√d_model`。

### 3.5 位置编码

由于我们的模型不包含循环和卷积，为了让模型利用序列顺序，必须把关于 token 相对位置或绝对位置的信息注入模型。为此，我们在编码器和解码器堆栈底部的输入嵌入上添加“位置编码”。位置编码与嵌入具有相同维度 `d_model`，因此二者可以直接相加。位置编码有许多可选方案，包括可学习和固定形式 [8]。

本文使用不同频率的正弦和余弦函数：

`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`

`PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

其中 `pos` 表示位置，`i` 表示维度。也就是说，位置编码的每个维度都对应一个正弦曲线。波长构成从 `2π` 到 `10000 · 2π` 的几何级数。

我们选择这个函数，是因为假设它可以让模型更容易学习按相对位置进行注意：对于任意固定偏移量 `k`，`PE_{pos+k}` 都可以表示为 `PE_pos` 的线性函数。

我们还尝试了可学习的位置嵌入 [8]，发现两种版本产生的结果几乎相同（表 3 的 E 行）。我们选择正弦版本，是因为它可能允许模型外推到训练时没有遇到过的更长序列。

### 表 1：不同层类型的最大路径长度、逐层复杂度和最少顺序操作数

`n` 是序列长度，`d` 是表示维度，`k` 是卷积 kernel size，`r` 是受限自注意力中的邻域大小。

| 层类型 | 每层复杂度 | 顺序操作数 | 最大路径长度 |
|---|---|---|---|
| 自注意力 | `O(n² · d)` | `O(1)` | `O(1)` |
| 循环层 | `O(n · d²)` | `O(n)` | `O(n)` |
| 卷积层 | `O(k · n · d²)` | `O(1)` | `O(log_k(n))` |
| 受限自注意力 | `O(r · n · d)` | `O(1)` | `O(n/r)` |

## 4 为什么使用自注意力

本节比较自注意力层与通常用于把一个变长符号表示序列 `(x_1, ..., x_n)` 映射到另一个等长序列 `(z_1, ..., z_n)` 的循环层和卷积层。这里 `x_i, z_i ∈ R^d`；这种映射可以看作典型序列转换编码器或解码器中的一个隐藏层。我们从三个要求出发来说明采用自注意力的动机。

第一个要求是每层的总计算复杂度。第二个要求是可并行化的计算量，用执行所需的最少顺序操作数衡量。第三个要求是网络中长距离依赖之间的路径长度。

学习长距离依赖是许多序列转换任务中的关键挑战。影响模型学习这些依赖能力的一个关键因素，是前向和反向信号必须在网络中经过的路径长度。输入和输出位置任意组合之间的路径越短，模型就越容易学习长距离依赖 [11]。因此，我们也比较了由不同层类型组成的网络中，任意输入位置与输出位置之间的最大路径长度。

如表 1 所示，自注意力层用常数个顺序执行的操作连接所有位置，而循环层需要 `O(n)` 个顺序操作。在计算复杂度方面，当序列长度 `n` 小于表示维度 `d` 时，自注意力层比循环层更快；在机器翻译最新模型使用的词片段 [31] 和字节对 [25] 表示中，通常就是这种情况。

为了提高超长序列任务的计算性能，可以把自注意力限制在输入序列中、以相应输出位置为中心的大小为 `r` 的邻域内。这样会把最大路径长度增加到 `O(n/r)`。我们计划在未来工作中进一步研究这种方法。

当 kernel 宽度 `k < n` 时，单个卷积层并不能连接所有输入位置和输出位置。对于连续 kernel，需要堆叠 `O(n/k)` 个卷积层；对于扩张卷积 [15]，则需要 `O(log_k(n))` 个卷积层。这会增加网络中任意两个位置之间最长路径的长度。

卷积层通常比循环层更昂贵，复杂度多出一个 `k` 的因子。不过，可分离卷积 [6] 可以显著降低复杂度，使其达到 `O(k · n · d + n · d²)`。即使 `k = n`，可分离卷积的复杂度也等于一个自注意力层加一个逐位置前馈层，这正是我们模型采用的方法。

自注意力还有一个附带好处：可能产生更容易解释的模型。我们检查了模型中的注意力分布，并在附录中展示和讨论了一些例子。不同注意力头不仅明显学会了不同任务，许多头还表现出与句子的句法和语义结构相关的行为。

## 5 训练

本节介绍模型的训练方案。

### 5.1 训练数据和批处理

我们在标准 WMT 2014 英德数据集上训练，该数据集包含约 450 万个句子对。句子使用字节对编码 [3]，源语言和目标语言共享约 37,000 个 token 的词表。

对于英法任务，我们使用规模明显更大的 WMT 2014 英法数据集，包含 3,600 万个句子，并按照 [31] 把 token 划分为 32,000 个词片段词表。

我们按照近似序列长度把句子对组成 batch。每个训练 batch 包含一组句子对，约有 25,000 个源 token 和 25,000 个目标 token。

### 5.2 硬件和训练计划

我们在一台配备 8 块 NVIDIA P100 GPU 的机器上训练模型。对于全文中描述超参数的 base 模型，每个训练步骤约耗时 0.4 秒。base 模型总共训练 100,000 步，即 12 小时。

对于 big 模型（表 3 最后一行），每步耗时 1.0 秒。big 模型训练 300,000 步，即 3.5 天。

### 5.3 优化器

我们使用 Adam 优化器 [17]，参数为 `β_1 = 0.9`、`β_2 = 0.98`、`ε = 10^−9`。训练期间，学习率按照下式变化：

`lrate = d_model^−0.5 · min(step_num^−0.5, step_num · warmup_steps^−1.5)`  （3）

这意味着在最初的 `warmup_steps` 个训练步骤中，学习率线性增加；之后随着步数增加，学习率按步数平方根的倒数下降。我们使用 `warmup_steps = 4000`。

### 5.4 正则化

训练中使用三种正则化方法。

**残差 Dropout：**我们在每个子层的输出上应用 dropout [27]，位置是在把它加到子层输入并进行归一化之前。此外，我们还对编码器和解码器堆栈中嵌入与位置编码的和应用 dropout。对于 base 模型，我们使用 `P_drop = 0.1`。

**标签平滑：**训练时使用取值 `ε_ls = 0.1` 的标签平滑 [30]。它会损害 perplexity，因为模型会学得更加不确定，但能够提高准确率和 BLEU 分数。

## 6 结果

### 6.1 机器翻译

在 WMT 2014 英德翻译任务上，big Transformer 模型（表 2 中的 Transformer (big)）比此前报告的最佳模型（包括集成模型）高出 2.0 个以上 BLEU，取得 28.4 的新最佳 BLEU。该模型配置列在表 3 的最后一行。训练使用 8 块 P100 GPU，耗时 3.5 天。

即使是 base 模型也超过了此前发表的所有模型和集成模型，而训练成本只是竞争模型的一小部分。

在 WMT 2014 英法翻译任务上，我们的 big 模型取得 41.0 BLEU，超过此前发表的所有单模型结果，而训练成本不到此前最佳模型的四分之一。英法 Transformer (big) 模型使用的 dropout 率为 `P_drop = 0.1`，而不是 0.3。

对于 base 模型，我们对最后 5 个 checkpoint 求平均；这些 checkpoint 每隔 10 分钟写入一次。对于 big 模型，我们对最后 20 个 checkpoint 求平均。

我们使用 beam size 为 4、长度惩罚 `α = 0.6` 的 beam search [31]。这些超参数是在开发集上实验后选择的。推理时，我们把最大输出长度设为输入长度加 50，但在可能时提前终止 [31]。

表 2 汇总了结果，并将我们的翻译质量和训练成本与文献中的其他模型架构进行比较。我们通过训练时间、使用的 GPU 数量，以及估计的每块 GPU 持续单精度浮点计算能力的乘积，估计训练模型所使用的浮点运算次数。

### 表 2：Transformer 在英德和英法 newstest2014 测试集上以很低的训练成本取得更高 BLEU

| 模型 | 英德 BLEU | 英法 BLEU | 英德训练成本（FLOPs） | 英法训练成本（FLOPs） |
|---|---:|---:|---:|---:|
| ByteNet [15] | 23.75 | — | — | — |
| Deep-Att + PosUnk [32] | — | 39.2 | `1.0 · 10^20` | — |
| GNMT + RL [31] | 24.6 | 39.92 | `2.3 · 10^19` | `1.4 · 10^20` |
| ConvS2S [8] | 25.16 | 40.46 | `9.6 · 10^18` | `1.5 · 10^20` |
| MoE [26] | 26.03 | 40.56 | `2.0 · 10^19` | `1.2 · 10^20` |
| Deep-Att + PosUnk Ensemble [32] | — | 40.4 | — | `8.0 · 10^20` |
| GNMT + RL Ensemble [31] | 26.30 | 41.16 | `1.8 · 10^20` | `1.1 · 10^21` |
| ConvS2S Ensemble [8] | 26.36 | 41.29 | `7.7 · 10^19` | `1.2 · 10^21` |
| Transformer（base） | 27.3 | 38.1 | `3.3 · 10^18` | — |
| Transformer（big） | 28.4 | 41.0 | `2.3 · 10^19` | — |

### 6.2 模型变体

为了评估 Transformer 不同组件的重要性，我们用不同方式改变 base 模型，并测量英德翻译开发集 `newstest2013` 上性能的变化。我们使用上一节所述的 beam search，但不进行 checkpoint 平均。结果见表 3。

在表 3 的 A 行中，我们改变注意力头数量、注意力键和值的维度，同时保持计算量不变，具体方法见第 3.2.2 节。单头注意力比最佳设置低 0.9 BLEU；但头数过多时，质量也会下降。

在 B 行中，降低注意力键的维度 `d_k` 会损害模型质量。这说明确定兼容性并不容易，比点积更复杂的兼容性函数可能会有帮助。

在 C 行和 D 行中可以看到，正如预期，更大的模型效果更好，而 dropout 对避免过拟合非常有帮助。在 E 行中，我们用可学习的位置嵌入 [8] 替换正弦位置编码，结果与 base 模型几乎相同。

### 表 3：Transformer 架构变体

未列出的参数与 base 模型相同。所有指标都在英德翻译开发集 `newstest2013` 上测量。列出的 perplexity 是按照字节对编码计算的每个词片段 perplexity，不应与按词计算的 perplexity 直接比较。

| 配置 | `N` | `d_model` | `d_ff` | `h` | `d_k` | `d_v` | `P_drop` | `ε_ls` | 训练步数 | PPL | BLEU | 参数量（百万） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.92 | 25.8 | 65 |
| A（1 头） | 6 | 512 | 2048 | 1 | 512 | 512 | 0.1 | 0.1 | 100K | 5.29 | 24.9 | — |
| A（4 头） | 6 | 512 | 2048 | 4 | 128 | 128 | 0.1 | 0.1 | 100K | 5.00 | 25.5 | — |
| A（16 头） | 6 | 512 | 2048 | 16 | 32 | 32 | 0.1 | 0.1 | 100K | 4.91 | 25.8 | — |
| A（32 头） | 6 | 512 | 2048 | 32 | 16 | 16 | 0.1 | 0.1 | 100K | 5.01 | 25.4 | — |
| B（`d_k=16`） | 6 | 512 | 2048 | 8 | 16 | 64 | 0.1 | 0.1 | 100K | 5.16 | 25.1 | 58 |
| B（`d_k=32`） | 6 | 512 | 2048 | 8 | 32 | 64 | 0.1 | 0.1 | 100K | 5.01 | 25.4 | 60 |
| C（`N=2`） | 2 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 6.11 | 23.7 | 36 |
| C（`N=4`） | 4 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 5.19 | 25.3 | 50 |
| C（`N=8`） | 8 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.88 | 25.5 | 80 |
| C（`d_model=256`） | 6 | 256 | 2048 | 8 | 32 | 32 | 0.1 | 0.1 | 100K | 5.75 | 24.5 | 28 |
| C（`d_model=1024`） | 6 | 1024 | 2048 | 8 | 128 | 128 | 0.1 | 0.1 | 100K | 4.66 | 26.0 | 168 |
| C（`d_ff=1024`） | 6 | 512 | 1024 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 5.12 | 25.4 | 53 |
| C（`d_ff=4096`） | 6 | 512 | 4096 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.75 | 26.2 | 90 |
| D（dropout=0.0） | 6 | 512 | 2048 | 8 | 64 | 64 | 0.0 | 0.1 | 100K | 5.77 | 24.6 | — |
| D（dropout=0.2） | 6 | 512 | 2048 | 8 | 64 | 64 | 0.2 | 0.1 | 100K | 4.95 | 25.5 | — |
| D（标签平滑=0.0） | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.0 | 100K | 4.67 | 25.3 | — |
| D（标签平滑=0.2） | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.2 | 100K | 5.47 | 25.7 | — |
| E（可学习位置嵌入） | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.92 | 25.7 | — |
| big | 6 | 1024 | 4096 | 16 | 64 | 64 | 0.3 | 0.1 | 300K | 4.33 | 26.4 | 213 |

## 7 结论

本文提出 Transformer，这是第一个完全建立在注意力机制上的序列转换模型，用多头自注意力替代了编码器-解码器架构中最常使用的循环层。

在翻译任务上，Transformer 的训练速度显著快于基于循环层或卷积层的架构。在 WMT 2014 英德和英法翻译任务上，我们都取得了新的最佳结果。在英德任务上，我们的最佳模型甚至超过了此前报告的所有集成模型。

我们对基于注意力的模型前景感到兴奋，并计划把它们应用到其他任务。我们计划将 Transformer 扩展到文本之外的输入和输出模态，并研究局部、受限的注意力机制，以高效处理图像、音频和视频等大规模输入输出。让生成过程变得更少依赖顺序，也是我们的另一个研究目标。

训练和评估模型所用的代码地址为：<https://github.com/tensorflow/tensor2tensor>。

## 致谢

我们感谢 Nal Kalchbrenner 和 Stephan Gouws 提供富有帮助的评论、修正和启发。

## 参考文献

以下条目保留原文书目信息，便于检索：

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.

[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.

[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.

[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.

[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.

[7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

[8] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.

[9] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[11] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[12] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735–1780, 1997.

[13] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

[14] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.

[15] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.

[16] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.

[17] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

[18] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.

[19] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.

[20] Samy Bengio and Łukasz Kaiser. Can active memory replace attention? In Advances in Neural Information Processing Systems (NIPS), 2016.

[21] Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.

[22] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.

[23] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.

[24] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.

[25] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

[26] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

[27] Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929–1958, 2014.

[28] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In Advances in Neural Information Processing Systems 28, pages 2440–2448, 2015.

[29] Ilya Sutskever, Oriol Vinyals, and Quoc V. V. Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.

[30] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.

[31] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

[32] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.
