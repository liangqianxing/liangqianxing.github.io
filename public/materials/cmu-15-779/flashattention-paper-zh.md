# FlashAttention：具有 IO 感知的快速、内存高效精确注意力

**作者**：Tri Dao†、Daniel Y. Fu†、Stefano Ermon†、Atri Rudra‡、Christopher Ré†  
†斯坦福大学计算机科学系；‡纽约州立大学布法罗分校计算机科学与工程系  
**日期**：2022 年 6 月 24 日（arXiv:2205.14135v2）

## 摘要

Transformer 在长序列上运行缓慢且占用大量内存，因为自注意力的时间和内存复杂度都关于序列长度呈二次增长。近似注意力方法试图通过牺牲模型质量来降低计算复杂度，但常常不能带来实际的壁钟时间加速。我们认为，缺失的原则是让注意力算法具备 **IO 感知**：计算 GPU 各级内存之间的读写。

我们提出 FlashAttention，一种具备 IO 感知的精确注意力算法。它利用分块（tiling）减少 GPU 高带宽内存（HBM）与 GPU 芯片上 SRAM 之间的内存读写次数。我们分析 FlashAttention 的 IO 复杂度，证明它比标准注意力需要更少的 HBM 访问，并且在一系列 SRAM 大小下达到最优。我们还将 FlashAttention 扩展到块稀疏注意力，得到一种比现有近似注意力方法都更快的近似算法。

FlashAttention 能加速 Transformer 训练：在序列长度 512 的 BERT-large 上，相比 MLPerf 1.1 训练速度纪录，端到端壁钟时间加速 15%；在序列长度 1K 的 GPT-2 上加速 3 倍；在序列长度 1K–4K 的 Long Range Arena 上加速 2.4 倍。FlashAttention 和块稀疏 FlashAttention 还支持更长上下文，从而提高模型质量（GPT-2 困惑度提升 0.7，长文档分类提升 6.4 个百分点），并带来新能力：Transformer 首次在 Path-X（序列长度 16K，准确率 61.4%）和 Path-256（序列长度 64K，准确率 63.1%）上取得高于随机猜测的性能。

## 1 引言

Transformer 模型已成为自然语言处理和图像分类等应用中最广泛使用的架构。模型规模不断变大、层数不断加深，但让它们处理更长上下文仍然困难，因为核心的自注意力模块在时间和内存上的复杂度都随序列长度二次增长。一个重要问题是：让注意力更快、更节省内存，能否帮助 Transformer 解决长序列的运行时间和内存挑战？

许多近似注意力方法试图降低注意力的计算和内存需求，包括稀疏近似、低秩近似及其组合。虽然它们把计算量降到线性或近线性，但很多方法相对于标准注意力并没有壁钟时间加速，因此没有得到广泛采用。主要原因之一是它们只关注 FLOP 数量（而 FLOP 未必与壁钟时间相关），忽略了内存访问（IO）开销。

本文提出让注意力算法具备 IO 感知：仔细计算不同层次的快速、慢速内存之间的读写，例如 GPU 芯片上快速的 SRAM 与相对较慢的 GPU 高带宽内存 HBM 之间的读写（图 1 左）。在现代 GPU 上，计算速度已超过内存速度，Transformer 中的大多数操作都受内存访问瓶颈限制。对数据库连接、图像处理、数值线性代数等内存受限操作，IO 感知算法一直很关键；然而 PyTorch、TensorFlow 等深度学习常用 Python 接口无法对内存访问进行细粒度控制。

![GPU 内存层次、FlashAttention 分块及 GPT-2 加速（原文图 1）](/images/materials/cmu-15-779/flashattention/figure-01.webp)

我们提出 FlashAttention，一种通过大幅减少内存访问来计算精确注意力的新算法。主要目标是避免把注意力矩阵在 HBM 中读写。为此需要：(i) 在不访问完整输入的情况下完成 softmax 归约；(ii) 在反向传播时不保存巨大的中间注意力矩阵。我们采用两项成熟技术解决这些问题：把输入分成块，多次遍历输入块，以增量方式执行 softmax 归约（也称分块）；在前向传播中保存 softmax 归一化因子，在反向传播时快速地在片上重新计算注意力，这比从 HBM 读取中间注意力矩阵更快。

我们在 CUDA 中实现 FlashAttention，以便精细控制内存访问，并把全部注意力操作融合到一个 GPU kernel。即使因重计算导致 FLOP 增加，该算法仍运行更快（GPT-2 上最高 7.6 倍），且内存使用量仅随序列长度线性增长。

我们分析 FlashAttention 的 IO 复杂度。当头维度为 $d$、SRAM 大小为 $M$ 时，FlashAttention 需要 $O(N^2d^2M^{-1})$ 次 HBM 访问，而标准注意力需要 $\Omega(Nd+N^2)$ 次。对典型的 $d$ 和 $M$，FlashAttention 的 HBM 访问少很多（图 2 中最高少 9 倍）。我们还给出下界，证明对所有 SRAM 大小，没有精确注意力算法能在渐近意义上进一步减少 HBM 访问。

FlashAttention 也可作为近似注意力的基础。我们实现了块稀疏 FlashAttention，它比 FlashAttention 还快 2–4 倍，可扩展到序列长度 64K；其 IO 复杂度相对 FlashAttention 按稀疏率成比例降低。第 5 节讨论多 GPU 注意力、核回归和块稀疏矩阵乘等扩展。我们开源 FlashAttention 实现，便于基于这一原语开展工作。

实验验证了 FlashAttention 能加速训练并通过更长上下文提升模型质量：

- **更快训练。** BERT-large（长度 512）比 MLPerf 1.1 训练纪录快 15%；GPT-2（长度 1K）比 HuggingFace 和 Megatron-LM 基线最多快 3 倍；Long Range Arena（长度 1K–4K）快 2.4 倍。
- **更高质量。** 更长序列使 GPT-2 困惑度改善 0.7，长文档分类提升 6.4 个百分点；FlashAttention 让 Transformer 首次在 Path-X 上超过随机性能（长度 16K），块稀疏版本在 Path-256（长度 64K）上超过随机性能。
- **注意力基准。** 在常见的 128–2K 序列长度上，FlashAttention 比标准实现最多快 3 倍，并可扩展到 64K。长度不超过 512 时，它同时比所有已有方法更快、更省内存；超过 1K 后部分近似方法（如 Linformer）开始更快，而块稀疏 FlashAttention 比作者所知的所有近似方法都快。

## 2 背景

### 2.1 硬件性能

这里聚焦 GPU，其他硬件加速器具有相似特征。GPU 内存层次包含多种大小和速度不同的内存，越小通常越快。以 A100 为例：40–80 GB HBM，带宽 1.5–2.0 TB/s；108 个流式多处理器各有 192 KB 芯片上 SRAM，带宽约 19 TB/s。SRAM 比 HBM 快一个数量级，但容量小许多数量级。随着计算相对内存越来越快，操作越来越受 HBM 访问限制，因此利用快速 SRAM 更重要。

**执行模型。** GPU 使用大量线程执行称为 kernel 的操作。每个 kernel 从 HBM 把输入加载到寄存器和 SRAM，执行计算，再把输出写回 HBM。

**性能特征。** 根据计算与内存访问的平衡，操作可分为计算受限和内存受限，常用算术强度（每字节内存访问对应的算术操作数）衡量：

1. **计算受限：**运行时间由算术操作数量决定，HBM 访问时间较小。大内维矩阵乘、大通道数卷积是典型例子。
2. **内存受限：**运行时间由内存访问次数决定，计算时间较小。逐元素操作（激活、dropout）和归约操作（求和、softmax、批归一化、层归一化）都属于此类。

**Kernel 融合。** 加速内存受限操作的常用方式是 kernel 融合：如果多个操作作用于同一输入，只需从 HBM 加载一次，而不是每个操作加载一次。编译器能自动融合许多逐元素操作。然而在模型训练中，中间值仍需写入 HBM 供反向传播使用，因而朴素 kernel 融合的效果受限。

### 2.2 标准注意力实现

给定输入序列 $Q,K,V\in\mathbb{R}^{N\times d}$，其中 $N$ 是序列长度、$d$ 是头维度，注意力输出 $O\in\mathbb{R}^{N\times d}$ 为

$$
S=QK^T\in\mathbb{R}^{N\times N},\qquad P=\operatorname{softmax}(S)\in\mathbb{R}^{N\times N},\qquad O=PV\in\mathbb{R}^{N\times d},
$$


其中 softmax 按行计算。

标准实现会把 $S$ 和 $P$ 实体化并写入 HBM，需 $O(N^2)$ 内存。通常 $N\gg d$（如 GPT-2 中 $N=1024,d=64$）。由于 softmax 等操作是内存受限的，大量内存访问导致壁钟时间缓慢。对注意力矩阵的 mask、对 $P$ 的 dropout 还会进一步放大问题，因此已有工作尝试融合 mask 与 softmax 等逐元素操作。

**算法 0：标准注意力实现**

输入：位于 HBM 的 $Q,K,V\in\mathbb{R}^{N\times d}$。

1. 按块从 HBM 加载 $Q,K$，计算 $S=QK^T$，把 $S$ 写回 HBM。
2. 从 HBM 读取 $S$，计算 $P=\operatorname{softmax}(S)$，把 $P$ 写回 HBM。
3. 按块加载 $P,V$，计算 $O=PV$，把 $O$ 写回 HBM。
4. 返回 $O$。

## 3 FlashAttention：算法、分析与扩展

我们展示如何在不为反向传播保存巨大中间矩阵的情况下，以更少的 HBM 读写计算精确注意力。这既节省内存，也缩短壁钟时间。我们分析 IO 复杂度，并将算法扩展到块稀疏注意力。

### 3.1 使用分块和重计算的高效注意力算法

输入 $Q,K,V\in\mathbb{R}^{N\times d}$ 位于 HBM，希望计算并写回 $O\in\mathbb{R}^{N\times d}$，目标是让 HBM 访问次数关于 $N$ 次二次以下。

**分块。** Softmax 把 $K$ 的各列耦合在一起。对向量 $x\in\mathbb{R}^B$，定义

$$
m(x)=\max_i x_i,\qquad f(x)_i=e^{x_i-m(x)},\qquad \ell(x)=\sum_i f(x)_i,\qquad \operatorname{softmax}(x)=f(x)/\ell(x).
$$


将 $x=[x^{(1)},x^{(2)}]$ 拼接，可用额外统计量递归计算：

$$
m(x)=\max(m(x^{(1)}),m(x^{(2)})),
$$

$$
f(x)=\left[e^{m(x^{(1)})-m(x)}f(x^{(1)}),\ e^{m(x^{(2)})-m(x)}f(x^{(2)})\right],
$$

$$
\ell(x)=e^{m(x^{(1)})-m(x)}\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\ell(x^{(2)}).
$$


因此只要维护 $(m,\ell)$，便能一次处理一个块。我们把 $Q,K,V$ 分块，将每块的 softmax 值与统计量算出，再合并结果。

**重计算。** 反向传播通常需要 $S,P\in\mathbb{R}^{N\times N}$。我们保存前向输出 $O$ 及 softmax 统计量 $(m,\ell)$，反向传播时从片上 SRAM 中的 $Q,K,V$ 块重新计算 $S,P$，无需保存二次规模的中间值。这是一种选择性梯度检查点。虽然 FLOP 增加，但减少 HBM 访问后反向传播反而更快。

**Kernel 融合。** 分块使算法可由一个 CUDA kernel 完成：从 HBM 加载输入，在片上执行矩阵乘、softmax、可选 mask/dropout 和第二次矩阵乘，再把结果写回 HBM，避免反复读写输入和输出。

**算法 1：FlashAttention**

输入：位于 HBM 的 $Q,K,V\in\mathbb{R}^{N\times d}$，片上 SRAM 大小 $M$。设置块大小

$$
B_c=\left\lfloor\frac{M}{4d}\right\rfloor,\qquad B_r=\min\left(\left\lfloor\frac{M}{4d}\right\rfloor,d\right).
$$

初始化 $O=0_{N\times d}$、$\ell=0_N$、$m=-\infty_N$（均在 HBM）。把 $Q$ 分成 $T_r=\lceil N/B_r\rceil$ 个块，把 $K,V$ 分成 $T_c=\lceil N/B_c\rceil$ 个块。对每个列块 $j$：

1. 将 $K_j,V_j$ 从 HBM 加载至片上 SRAM。
2. 对每个行块 $i$，加载 $Q_i,O_i,\ell_i,m_i$。
3. 片上计算 $S_{ij}=Q_iK_j^T$。
4. 计算该块行最大值 $\tilde m_{ij}$、$\tilde P_{ij}=\exp(S_{ij}-\tilde m_{ij})$ 和行和 $\tilde\ell_{ij}$。
5. 更新

   $$
   m_i^{\rm new}=\max(m_i,\tilde m_{ij}),\quad \ell_i^{\rm new}=e^{m_i-m_i^{\rm new}}\ell_i+e^{\tilde m_{ij}-m_i^{\rm new}}\tilde\ell_{ij}.
   $$

6. 更新并写回

   $$
   O_i\leftarrow \operatorname{diag}(\ell_i^{\rm new})^{-1}\left(\operatorname{diag}(\ell_i)e^{m_i-m_i^{\rm new}}O_i+e^{\tilde m_{ij}-m_i^{\rm new}}\tilde P_{ij}V_j\right).
   $$

7. 写回 $\ell_i\leftarrow\ell_i^{\rm new},m_i\leftarrow m_i^{\rm new}$。

最后返回 $O$。算法进行 $O(N^2d)$ FLOP，除输入和输出外仅需 $O(N)$ 额外内存。

### 3.2 FlashAttention 的 IO 复杂度

设序列长度为 $N$、头维度为 $d$、SRAM 大小为 $M$，且 $d\le M\le Nd$。标准注意力需要

$$
\Theta(Nd+N^2)
$$

次 HBM 访问，而 FlashAttention 需要

$$
\Theta(N^2d^2M^{-1})
$$

次 HBM 访问。证明思路是：片上可容纳大小为 $\Theta(M)$ 的 $K,V$ 块；每个块需遍历所有 $Q$ 块，因此对 $Q$ 有 $\Theta(Nd/M)$ 次遍历，每次加载 $\Theta(Nd)$ 个元素。

**命题 3（下界）。** 不存在一个精确注意力算法，对所有 $M\in[d,Nd]$ 都使用 $o(N^2d^2M^{-1})$ 次 HBM 访问。令 $M=\Theta(Nd)$，输入和输出本身就有 $Nd$ 个元素，算法至少要进行 $\Omega(Nd)$ 次访问，与假设矛盾。

图 2（左）显示，虽然 FlashAttention 因反向重计算 FLOP 更多，但 HBM 访问显著更少，因此 GPT-2 上运行更快。图 2（中）改变块大小 $B_c$：块越大，遍历次数越少、HBM 访问越少、前向运行越快；超过 256 后，运行时间转而受算术操作等因素限制，块太大也无法装入 SRAM。

![FlashAttention 运行时间、块大小与稀疏率（原文图 2）](/images/materials/cmu-15-779/flashattention/figure-02.webp)

### 3.3 扩展：块稀疏 FlashAttention

给定 $Q,K,V$ 和块掩码 $\widetilde M\in\{0,1\}^{N\times N}$，计算

$$
S=QK^T,\quad P=\operatorname{softmax}(S\odot\mathbf 1_{\widetilde M}),\quad O=PV,
$$

其中掩码为 0 的位置置为 $-\infty$。要求掩码按块定义：块掩码矩阵 $M$ 的每个元素决定对应 $B_r\times B_c$ 子块。

算法与算法 1 相同，只跳过零块。若非零块比例为 $s$，其 HBM 访问复杂度为

$$
\Theta\left(Nd+N^2d^2M^{-1}s\right).
$$

因此稀疏化直接按 $s$ 改善较大的 IO 项。长序列中常取 $s=N^{-1/2}$ 或 $s=N^{-1}\log N$，对应 $\Theta(N\sqrt N)$ 或 $\Theta(N\log N)$ 的 IO 复杂度。本文实验使用固定 butterfly 稀疏模式。图 2（右）验证运行时间随稀疏率近似成比例改善；在 LRA 上块稀疏 FlashAttention 加速 2.8 倍，同时精度与标准注意力相当。

## 4 实验

我们评估 FlashAttention 对 Transformer 训练时间、模型质量及注意力运行时间和内存的影响。

- **训练速度：** BERT 训练比 MLPerf 1.1 纪录快 15%；GPT-2 比 HuggingFace 最多快 3 倍、比 Megatron 快 1.8 倍；LRA 快 2.4 倍。
- **质量：** 4K 上下文的 GPT-2 比 1K 上下文困惑度好 0.7；长文档分类提升 6.4 个百分点；在 Path-X（16K）和 Path-256（64K）首次取得高于随机的 Transformer 性能。
- **基准：** FlashAttention 内存随序列长度线性增长，在长度不超过 2K 时比标准注意力最多快 3 倍；块稀疏版本也随长度线性增长，并快于所有测试的近似基线。

### 4.1 使用 FlashAttention 的更快模型

**BERT。** 在 Wikipedia 上训练 BERT-large，使用 MLPerf 1.1 相同初始化和设置。达到掩码语言模型准确率 72.0% 的时间（8×A100，10 次平均）：Nvidia MLPerf 20.0±1.5 分钟，FlashAttention 17.4±1.4 分钟，快约 15%。

**GPT-2。** 在 OpenWebText 上，FlashAttention 相比 HuggingFace 和 Megatron-LM 的端到端训练最多快 3 倍和 1.7 倍，困惑度相同。GPT-2 small：HuggingFace 9.5 天、Megatron 4.7 天、FlashAttention 2.7 天；GPT-2 medium：21.0 天、11.5 天、6.9 天。

**Long Range Arena。** 在五个任务（ListOps、Text、Retrieval、Image、Pathfinder）上，FlashAttention 平均准确率 59.8、加速 2.4 倍；块稀疏版本平均 59.6、加速 2.8 倍。表中 Linformer、Linear Attention、Performer、Local Attention、Reformer、Smyrf 的平均准确率分别为 54.9、59.6、58.9、56.0、57.6、57.9。

### 4.2 使用更长序列的更好模型

**长上下文语言建模。** FlashAttention 让 GPT-2 上下文长度扩大 4 倍，同时仍比 Megatron 的 1K 上下文实现快 30%，困惑度更好 0.7。GPT-2 small：Megatron 1K 困惑度 18.2、4.7 天；FlashAttention 1K 为 18.2、2.7 天，2K 为 17.6、3.0 天，4K 为 17.5、3.6 天。

**长文档分类。** MIMIC-III 是带多标签的重症监护出院摘要数据，ECtHR 是欧洲人权法院案件。MIMIC 平均文档 2,395 token、最长 14,562；ECtHR 平均 2,197、最长 49,392。用预训练 RoBERTa 重复位置嵌入后，MIMIC 长度 16K 相比 512 提升 4.3 个百分点，ECtHR 长度 8K 相比 512 提升 8.5 个百分点。

**Path-X 和 Path-256。** 任务要求判断黑白 128×128 或 256×256 图像中两点间是否存在路径，像素逐个输入 Transformer。先在 Path-64 预训练，再空间插值位置嵌入。FlashAttention 在 Path-X（16K）达到 61.4% 准确率；块稀疏版本支持 Path-256（64K），达到 63.1%。

### 4.3 注意力基准

在一块 40 GB HBM 的 A100 上，加入 dropout 和 padding mask，改变序列长度比较多种精确、近似和稀疏注意力。

**运行时间。** 图 3 左显示前向+反向时间。精确注意力的时间随长度二次增长，FlashAttention 比 PyTorch 最多快 3 倍。虽然许多近似/稀疏方法随长度线性增长，但短序列中 FlashAttention 因 HBM 访问更少仍更快；长度在 512–1024 之间时近似方法才与 FlashAttention 交叉。块稀疏 FlashAttention 在全部长度上都快于作者所知的精确、稀疏和近似实现。

**内存占用。** 图 3 右显示 FlashAttention 和块稀疏版本均线性增长，比精确基线最多节省 20 倍，也比近似基线更省。除 Linformer 外其他算法在 64K 前均在 A100 上耗尽内存；即使与 Linformer 相比，FlashAttention 仍节省约 2 倍内存。

![注意力运行时间与内存占用（原文图 3）](/images/materials/cmu-15-779/flashattention/figure-03.webp)

## 5 局限与未来方向

**编译到 CUDA。** 当前 IO 感知实现需要为每种注意力实现编写新的 CUDA kernel，语言层次低于 PyTorch、工程工作量大，而且实现可能不能跨 GPU 架构移植。未来需要支持以 PyTorch 等高级语言编写注意力，再编译为 IO 感知 CUDA 实现，类似图像处理中的 Halide。

**IO 感知深度学习。** 注意力是 Transformer 中最耗内存的计算，但深度网络的每一层都会访问 GPU HBM。希望 IO 感知方法能扩展到更多模块（附录 D）。

**多 GPU IO 感知方法。** 单 GPU 上 FlashAttention 在常数意义下最优，但注意力可跨多 GPU 并行。多 GPU 引入新的 IO 层级，需要分析 GPU 间数据传输；这是未来方向。

## 致谢

实现以 Apex 的 FMHA 代码为起点，感谢 Young-Jun Ko 对实现和 CUDA 的解释。感谢 Sabri Eyuboglu、Megan Leszczynski、Laurel Orr、Yuhuai Wu、Beidi Chen、Xun Huang 等对早期稿件的反馈，也感谢 Markus Rabe 和 Charles Staats 关于注意力算法的讨论。研究获得 NIH、NSF、ARL、ONR、NXP、Xilinx、Intel、IBM、Microsoft、Google Cloud、Stanford Data Science Initiative、美国国防部 NDSEG 以及 Stanford DAWN 项目成员等支持。

## 附录 A 相关工作

### IO 感知运行时优化

针对快速/慢速内存读写进行优化在计算机科学中历史悠久。本文与 IO 复杂度分析最直接相关，同时也与工作集模型、数据局部性、Roofline 算术强度模型、可扩展性分析和计算机体系结构教材中的内存层次概念相连。我们希望深度学习社区将这些思想用于更多系统组件。

### 结构化矩阵与高效机器学习模型

矩阵乘是机器学习模型的核心瓶颈。结构化矩阵（参数量和运行时间均为次二次）包括稀疏矩阵、低秩矩阵和信号处理中常见的快速变换；更一般的类别包括 Toeplitz-like、低位移秩和准可分矩阵。Butterfly 模式之所以用于块稀疏注意力，是因为 butterfly 矩阵及其乘积能以近乎最优的运行时间和参数量表达结构化矩阵。结构化矩阵虽理论高效，但难转化为壁钟加速（硬件彩票现象），后续工作让 butterfly 更适合硬件。

### 稀疏训练

块稀疏 FlashAttention 可视为让稀疏模型训练更高效的一步。剪枝、稀疏权重矩阵已用于推理压缩；彩票假设认为大稠密网络包含性能相当的小子网络。本文固定 butterfly 模式训练注意力，观察其在 LRA 上几乎达到稠密 FlashAttention 的性能。

### 高效 Transformer

Transformer 在 NLP 和计算机视觉中广泛使用，但时间和内存关于序列长度二次增长。已有路线包括基于哈希的稀疏近似（Reformer、Smyrf）、低秩近似（Performer），以及稀疏和低秩组合（Longformer、BigBird、Scatterbrain、Long-short Transformer、Combiner）。还包括沿序列压缩、使用前序列状态延长上下文（Transformer-XL、Compressive Transformer）。HiPPO/S4 等状态空间模型、LambdaNetworks、AFT 和 FLASH 则尝试用其他模块替代注意力。

## 附录 B 算法细节

### B.1 内存高效的前向传播

注意力内存效率的主要挑战是 softmax 将 $K$ 的各列耦合。将 softmax 归一化常数单独计算即可解耦。省略 max-shifting 时，定义

$$
L_i=\sum_j e^{q_i^Tk_j},\qquad o_i=\sum_j\frac{e^{q_i^Tk_j}}{L_i}v_j.
$$

先计算所有 $L_i$（额外 $O(N)$ 内存），再重复累加得到 $o_i$（额外 $O(d)$ 内存），前向传播只需线性额外内存。

### B.2 内存高效的反向传播

设标量损失为 $\phi$，输出梯度为 $dO$。有 $dV=P^TdO$，即

$$
dv_j=\sum_i\frac{e^{q_i^Tk_j}}{L_i}do_i.
$$

由 $dP=dOV^T$ 和 softmax 雅可比矩阵，令

$$
D_i=P_{i:}dP_{i:}^T=do_i^To_i,
$$

则

$$
dS_{ij}=P_{ij}(dP_{ij}-D_i),
$$

$$
dq_i=\sum_j\frac{e^{q_i^Tk_j}}{L_i}(do_i^Tv_j-D_i)k_j,
$$

$$
dk_j=\sum_i\frac{e^{q_i^Tk_j}}{L_i}(do_i^Tv_j-D_i)q_i.
$$

因此反向传播也只需 $O(N)$ 额外内存。

### B.3 FlashAttention 前向传播

完整计算为

$$
S=\tau QK^T,\quad S_{masked}=\operatorname{mask}(S),\quad P=\operatorname{softmax}(S_{masked}),
$$

$$
P_{dropped}=\operatorname{dropout}(P,p_{drop}),\quad O=P_{dropped}V,
$$

其中 $\tau$ 通常为 $1/\sqrt d$。算法 2 保存输出 $O$、softmax 统计量 $\ell,m$ 以及伪随机数发生器状态 $R$；逐块计算 mask、softmax、dropout 并更新输出。返回 $O,\ell,m,R$。

### B.4 FlashAttention 反向传播

算法 3 的标准反向传播需要把 $P,dP,dS$ 等二次规模矩阵写入 HBM。FlashAttention 反向传播只保存前向伪随机数状态，在反向时重新生成 dropout mask；并用 $D_i=do_i^To_i$ 计算 softmax 梯度，无需在 SRAM 中归约长度为 $N$ 的向量。算法 4 逐块重算 $S$、mask、$P$，累加 $dQ,dK,dV$，只需线性额外内存，FLOP 为 $O(N^2d)$，IO 复杂度为 $\Theta(N^2d^2M^{-1})$。

### B.5 与 Rabe 和 Staats 方法的比较

两者都对注意力矩阵分块，前向不保存大矩阵、反向重算。Rabe 和 Staats 主要优化总显存，FlashAttention 主要优化内存读写次数；后者通常比标准实现快 2–4 倍。前者为每个块保存临时输出，最后再合并；FlashAttention 逐块增量更新，只保存一份输出。反向时前者使用梯度检查点重算注意力和临时输出，FlashAttention 只重算注意力并用解析梯度公式，因而内存更小、速度更快。

## 附录 C 证明

### 定理 1

内层矩阵乘 $Q_iK_j^T$ 和 $\widetilde P_{ij}V_j$ 各需 $O(B_rB_cd)$ FLOP，共 $T_cT_r=N^2/(B_cB_r)$ 个块，因此总 FLOP 为 $O(N^2d)$。统计量 $(\ell,m)$ 仅需 $O(N)$ 额外内存。对列块索引 $j$ 归纳：处理前 $j$ 个块后，$m$ 等于已处理分数的行最大值，$\ell$ 等于移位指数的行和，$O$ 等于已处理 softmax 与 $V$ 的乘积。递推更新正是拼接 softmax 的代数聚合公式；处理完全部块后得到 $O=\operatorname{softmax}(QK^T)V$。

### 定理 2

标准实现的三个阶段分别读写 $Q,K,S,P,V,O$，总 HBM 访问为 $\Theta(Nd+N^2)$。FlashAttention 每个 $K,V$ 元素只加载一次，对 $Q,O$ 做 $T_c$ 次遍历；块大小满足 $B_cd=O(M)$、$B_rd=O(M)$、$B_rB_c=O(M)$，取 $B_c=\Theta(M/d)$、$B_r=\Theta(\min(M/d,d))$，所以 $T_c=\Theta(Nd/M)$，总访问为 $\Theta(N^2d^2/M)$。

### 命题 3

若对所有 $M$ 都有 $o(N^2d^2/M)$ 次访问，则在 $M=\Theta(Nd)$ 时为 $o(Nd)$。但输入 $Q,K,V$ 和输出 $O$ 共含 $Nd$ 个元素，至少需要 $\Omega(Nd)$ 次访问，矛盾。

### 定理 5

标准反向传播每步都读写 $Nd$ 或 $N^2$ 大小的输入输出，访问为 $\Theta(Nd+N^2)$。FlashAttention 反向中 $K,V,dK,dV$ 各只读/写一次，对 $Q,O,dO,dQ$ 做 $T_c$ 次遍历，因此访问为 $\Theta(N^2d^2/M)$。

## 附录 D 扩展细节

### D.1 块稀疏 FlashAttention

算法 5 与算法 2 相同，只在块掩码 $M_{ij}\ne0$ 时执行。非零块比例为 $s$ 时，需要写出 $O$，因此访问数为

$$
\Theta\left(Nd+\frac{N^2d^2}{M}s\right).
$$


### D.2 潜在扩展

**多 GPU 注意力。** 大语言模型通常在同一节点的 4–8 张 GPU 上划分注意力。除 GPU SRAM 和 HBM 外，还存在其他 GPU 的 HBM；超长序列可利用不同层次内存的不对称性协同计算。

**稀疏 MLP 层。** 稠密 MLP 通常是计算受限的，但稀疏权重的 MLP 可能受内存限制，实际加速往往不与稀疏率成比例。IO 感知实现可能释放稀疏性的收益。

**核机器学习。** FlashAttention 利用 $QK^T$ 低秩（秩 $d\ll N$），反复加载 $Q,K$ 并重算所需注意力块，从而减少 HBM 访问。核机器学习中 $N\times N$ 核矩阵的每个元素也是两个 $d$ 维向量的函数；KeOps 是通过减少内存读写加速核运算的成功例子。

## 附录 E 完整实验结果

### E.1 BERT

按照 MLPerf 1.1 参考实现训练 BERT-large：LAMB 优化器、学习率 $3.75\times10^{-3}$、批大小 448，最多 7100 步，达到掩码语言模型验证准确率 72.0% 即停止。使用 Apex AMP O2 和 FP16，在 8×A100-80GB 上运行 10 次，每次 16–19 分钟并取平均。训练/验证划分与 Nvidia 基线相同。

### E.2 GPT-2

使用 HuggingFace Transformers 和 Megatron-LM 的 GPT-2 实现，遵循 Megatron 训练配方。有效批大小 512，AdamW，GPT-2 small 学习率 $6\times10^{-4}$，medium 为 $1.5\times10^{-4}$，权重衰减 0.1，训练 400K 步，使用 PyTorch AMP。OpenWebText 随机抽取 0.5% 为验证集，所有实现使用相同验证集。在 8×A100-40GB 上测量壁钟时间。FlashAttention 与 HuggingFace 的验证困惑度曲线几乎重合。

![GPT-2 small 和 medium 的验证困惑度（原文图 4）](/images/materials/cmu-15-779/flashattention/figure-04.webp)

### E.3 LRA 细节

遵循 Long Range Arena 论文、代码库和 Nyströmformer 复现实验的超参数。若无法复现某基线，则报告原论文或复现工作中较好的结果。除 Performer（混合精度不稳定）和 Local Attention（仅支持 FP32）外均采用混合精度。总体加速取五个任务壁钟加速的几何平均。

Path-X/Path-256 先在 Path-64 预训练 200 个 epoch，再网格插值位置嵌入，在下游任务微调 200 个 epoch，线性 warmup 一个 epoch、余弦学习率衰减。Path-X 还对验证集最佳检查点再微调 200 个 epoch，可额外提升约 4 个百分点，但之后会过拟合。

### E.4 与 Apex FMHA 比较

Apex FMHA 面向最多 512 长度、头维度 64、A100，融合 $\operatorname{dropout}(\operatorname{softmax}(\operatorname{mask}(QK^T)))V$，前向把注意力矩阵写入 HBM。FlashAttention 以其代码为起点，加入分块和重计算，因此可支持最长 64K、头维度 16/32/64/128 及 Turing/Ampere GPU。短序列上，FlashAttention 前向略快、反向略慢；综合时间在长度 128 慢约 4%，长度 256 快 8%，长度 512 快 5%。

### E.5 不同硬件与配置

在 A100（批大小 8、头维度 64、12 头）上，FlashAttention 通常加速 2–4 倍，启用 dropout 和 mask 时因 kernel 融合加速更明显。头维度 128 会因块更大而降低加速，但因果 mask 可跳过一半块，仍最高约 3 倍。RTX 3090 因 HBM 带宽较低（约 900 GB/s 对 A100 的 1.5 TB/s）加速约 2.5–4.5 倍。T4 的 SRAM 更小，块尺寸必须更小，因此加速低于 A100；论文同时报告仅前向的推理加速。

![A100 上不同序列长度的加速（原文图 5）](/images/materials/cmu-15-779/flashattention/figure-05.webp)

![A100 上头维度 128 时的加速（原文图 6）](/images/materials/cmu-15-779/flashattention/figure-06.webp)

![RTX 3090 上不同序列长度的加速（原文图 7）](/images/materials/cmu-15-779/flashattention/figure-07.webp)

![T4 上不同序列长度的加速（原文图 8）](/images/materials/cmu-15-779/flashattention/figure-08.webp)

### E.6 完整基准

基线包括 PyTorch/HuggingFace、Megatron 的精确注意力，Reformer、Local Attention、Linformer、Smyrf、LongShortFormer 等近似注意力，以及 OpenAI Block-Sparse、Longformer、BigBird 等稀疏注意力。实验在单张 40 GB A100 上进行，8 个头、头维度 64、批大小 16，长度从 128 到 65,536；Q、K、V 为随机向量，dropout 为 0.1，padding mask 长度在总长度及其减 20 之间均匀采样。每个运行时间为 100 次平均，内存测量一次。

下表给出无 dropout、无 mask 的前向+反向内存（MB）：

|方法|128|256|512|1024|2048|4096|8192|16384|32768|65536|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PyTorch|36|104|336|1184|4416|17024|-|-|-|-|
|Megatron|36|104|336|1184|4416|-|-|-|-|-|
|Reformer|377|754|1508|3016|6033|12067|24134|-|-|-|
|Local Attention|53|110|232|592|1696|3392|6784|13568|27136|-|
|Linformer|25|52|114|287|832|1652|3292|6572|13132|26252|
|Smyrf|217|434|868|1737|3474|6947|13894|27788|-|-|
|LSformer|72|152|333|796|2540|5068|10125|20240|-|-|
|Block Sparse|33|82|228|408|910|2401|-|-|-|-|
|Longformer|30|61|124|277|681|1370|2748|-|-|-|
|BigBird|33|66|131|294|708|1431|2872|-|-|-|
|FlashAttention|22|44|104|209|418|836|1672|3344|6688|13376|
|块稀疏 FlashAttention|22|44|104|209|418|836|1672|3344|6690|13384|

论文表 9–20 还报告了有/无 dropout 和 mask 时，各方法在前向、反向及组合运行时间（毫秒）及 128–65,536 的完整数值；总体趋势是 FlashAttention 在短序列上优于精确基线，块稀疏 FlashAttention 在全部长度上最快，并保持线性内存增长。


## 图表与算法补充译文

### 图 1：内存层次与 FlashAttention 分块

左图：FlashAttention 使用分块，避免在相对较慢的 GPU HBM 中实体化巨大的 $N\times N$ 注意力矩阵（虚线框）。外层循环遍历 $K,V$ 块并将其加载到快速片上 SRAM；每个块内层循环遍历 $Q$ 块，在 SRAM 中计算，并将注意力输出写回 HBM。右图：GPT-2 上相对 PyTorch 注意力的加速。FlashAttention 不把 $N\times N$ 矩阵读写到 HBM，因此注意力计算最高加速 7.6 倍。

### 图 2：运行时间、块大小和稀疏率

左：GPT-2 medium（长度 1024、头维度 64、16 头、批大小 64）在 A100 上的前向+反向时间。标准注意力 66.6 GFLOP、40.3 GB HBM 读写、41.7 ms；FlashAttention 75.2 GFLOP、4.4 GB HBM 读写、7.3 ms。中：块大小从 64、128、256 到 512 时，HBM 访问先减少、前向时间先下降后受计算限制。右：长度 4K 时，块稀疏 FlashAttention 随非零块比例降低而按比例加速。

### 图 3：注意力运行时间与内存

左：前向+反向运行时间随序列长度增长；FlashAttention 和块稀疏版本与 PyTorch、Megatron、Linformer、OpenAI Sparse 等基线比较。右：内存占用随序列长度增长；FlashAttention 和块稀疏版本呈线性增长。

### 表 1：BERT-large 训练时间

|实现|训练时间（分钟）|
|---|---:|
|Nvidia MLPerf 1.1|20.0 ± 1.5|
|FlashAttention（本文）|17.4 ± 1.4|

### 表 2：GPT-2 训练

|实现|OpenWebText 困惑度|训练时间（加速）|
|---|---:|---:|
|GPT-2 small – HuggingFace|18.2|9.5 天（1.0×）|
|GPT-2 small – Megatron-LM|18.2|4.7 天（2.0×）|
|GPT-2 small – FlashAttention|18.2|2.7 天（3.5×）|
|GPT-2 medium – HuggingFace|14.2|21.0 天（1.0×）|
|GPT-2 medium – Megatron-LM|14.3|11.5 天（1.8×）|
|GPT-2 medium – FlashAttention|14.3|6.9 天（3.0×）|

### 表 3：Long Range Arena

|模型|ListOps|Text|Retrieval|Image|Pathfinder|平均|加速|
|---|---:|---:|---:|---:|---:|---:|---:|
|Transformer|36.0|63.6|81.6|42.3|72.7|59.3|–|
|FlashAttention|37.6|63.9|81.4|43.5|72.7|59.8|2.4×|
|块稀疏 FlashAttention|37.0|63.0|81.3|43.6|73.3|59.6|2.8×|
|Linformer|35.6|55.9|77.7|37.8|67.6|54.9|2.5×|
|Linear Attention|38.8|63.2|80.7|42.6|72.5|59.6|2.3×|
|Performer|36.8|63.6|82.2|42.1|69.9|58.9|1.8×|
|Local Attention|36.1|60.2|76.7|40.6|66.6|56.0|1.7×|
|Reformer|36.5|63.8|78.5|39.6|69.4|57.6|1.3×|
|Smyrf|36.1|64.1|79.0|39.6|70.5|57.9|1.7×|

### 表 4：GPT-2 长上下文

|实现|上下文长度|困惑度|训练时间（加速）|
|---|---:|---:|---:|
|Megatron-LM|1K|18.2|4.7 天（1.0×）|
|FlashAttention|1K|18.2|2.7 天（1.7×）|
|FlashAttention|2K|17.6|3.0 天（1.6×）|
|FlashAttention|4K|17.5|3.6 天（1.3×）|

### 表 5：长文档分类（micro-$F_1$）

|数据集|512|1024|2048|4096|8192|16384|
|---|---:|---:|---:|---:|---:|---:|
|MIMIC-III|52.8|50.7|51.7|54.6|56.4|57.1|
|ECtHR|72.2|74.3|77.1|78.6|80.7|79.2|

### 表 6：Path-X 与 Path-256

|模型|Path-X|Path-256|
|---|---:|---:|
|Transformer|随机|随机|
|Linformer|随机|随机|
|Linear Attention|随机|随机|
|Performer|随机|随机|
|Local Attention|随机|随机|
|Reformer|随机|随机|
|SMYRF|随机|随机|
|FlashAttention|61.4|随机|
|块稀疏 FlashAttention|56.0|63.1|

### 表 7：与 Apex FMHA 的运行时间（毫秒）

|注意力方法|128|256|512|
|---|---:|---:|---:|
|Apex FMHA 前向|0.10|0.29|1.14|
|FlashAttention 前向|0.08|0.22|0.81|
|Apex FMHA 反向|0.17|0.52|1.81|
|FlashAttention 反向|0.20|0.53|2.00|
|Apex FMHA 前向+反向|0.27|0.81|2.95|
|FlashAttention 前向+反向|0.28|0.75|2.81|

### 算法 2：FlashAttention 前向传播（完整伪代码）

输入 $Q,K,V$、SRAM 大小 $M$、缩放常数 $\tau$、mask 函数和 dropout 概率 $p_{drop}$。

1. 初始化伪随机数发生器状态 $R$ 并保存到 HBM。
2. 设置 $B_c=\lfloor M/(4d)\rfloor$，$B_r=\min(\lfloor M/(4d)\rfloor,d)$。
3. 在 HBM 初始化 $O=0,\ell=0,m=-\infty$，按块划分 $Q,K,V,O,\ell,m$。
4. 对 $j=1,\ldots,T_c$：加载 $K_j,V_j$ 到 SRAM。
5. 对 $i=1,\ldots,T_r$：加载 $Q_i,O_i,\ell_i,m_i$；计算 $S_{ij}=\tau Q_iK_j^T$；应用 mask；计算块行最大值 $\tilde m_{ij}$、$\tilde P_{ij}$、$\tilde\ell_{ij}$；更新 $m_i,\ell_i$；对 $\tilde P_{ij}$ 应用 dropout；按算法 1 的归一化公式更新 $O_i$ 并写回。
6. 返回 $O,\ell,m,R$。

### 算法 3：标准注意力反向传播

输入 $Q,K,V,dO\in\mathbb{R}^{N\times d}$ 和 HBM 中的 $P\in\mathbb{R}^{N\times N}$。

1. 分块加载 $P,dO$，计算 $dV=P^TdO$ 并写回。
2. 分块加载 $dO,V$，计算 $dP=dOV^T$ 并写回。
3. 读取 $P,dP$，计算 $dS_{ij}=P_{ij}(dP_{ij}-\sum_lP_{il}dP_{il})$ 并写回。
4. 分块加载 $dS,K$，计算 $dQ=dSK$。
5. 分块加载 $dS,Q$，计算 $dK=dS^TQ$。
6. 返回 $dQ,dK,dV$。

### 算法 4：FlashAttention 反向传播（完整伪代码）

输入前向的 $Q,K,V,O$、输出梯度 $dO$、统计量 $\ell,m$、缩放常数、mask、dropout 概率和随机状态 $R$。

1. 将随机状态设回 $R$，设置块大小并划分所有输入和梯度。
2. 初始化 $dQ,dK,dV=0$。
3. 对每个列块 $j$，加载 $K_j,V_j$，在 SRAM 初始化 $\widetilde{dK}_j,\widetilde{dV}_j$。
4. 对每个行块 $i$：加载 $Q_i,O_i,dO_i,dQ_i,\ell_i,m_i$；重算 $S_{ij}$、mask 和 $P_{ij}$；重新生成 dropout mask $Z_{ij}$；计算 $P^{dropped}_{ij}$、$dV_j$、$dP_{ij}$；计算 $D_i=\operatorname{rowsum}(dO_i\circ O_i)$；计算 $dS_{ij}=P_{ij}\circ(dP_{ij}-D_i)$；更新 $dQ_i$、$\widetilde{dK}_j$。
5. 将累积的 $dK_j,dV_j$ 写回 HBM，返回 $dQ,dK,dV$。

### 算法 5：块稀疏 FlashAttention 前向传播

输入算法 2 的全部参数以及块稀疏掩码 $M\in\{0,1\}^{N/B_r\times N/B_c}$。外层遍历列块 $j$，内层遍历行块 $i$；仅当 $M_{ij}\ne0$ 时加载 $Q_i,O_i,\ell_i,m_i$，计算分数、mask、softmax、dropout 并更新输出；零块直接跳过。最后返回 $O,\ell,m,R$。

### 表 9–21：完整运行时间与内存结果

表 9–20 分别给出带 dropout/掩码、仅掩码、仅 dropout、无 dropout 和掩码时，各方法在前向、反向及组合阶段的运行时间（毫秒）；表 21 给出组合阶段内存（MB）。列为序列长度 128、256、512、1024、2048、4096、8192、16384、32768、65536。方法名保留原文，数值完整转录如下：

```text
Table 9: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with dropout and masking. Best in bold, second best underlined.

               Attention Method    128    256    512     1024    2048    4096    8192    16384    32768     65536
               PyTorch Attention   0.36   0.34   0.78     2.54    9.33   36.33      -       -        -        -
                       Megatron    0.40   0.40   1.10     3.65   16.19      -       -       -        -        -
                       Reformer    2.03   3.15   5.67    11.02   22.59   46.14   97.38   212.13      -        -
                 Local Attention   0.83   0.86   1.01     2.20    7.13   14.32   28.60    57.79   117.67      -
                       Linformer   0.67   0.52   0.69     0.71    1.65    3.18    6.15    12.16    24.17    52.39
                           Smyrf   2.27   2.34   3.91     7.44   14.71   29.22   58.27   116.41      -        -
                       LSformer    1.18   1.27   1.34     3.38   11.40   22.55   44.95    89.76   179.66      -
                    Block Sparse   1.12   1.11   2.13     2.77    6.95   20.91      -       -        -        -
                     Longformer    1.22   1.14   1.08     1.95    5.72   12.98      -       -        -        -
                         BigBird   1.13   1.12   1.12     1.77    6.03   13.68      -       -        -        -
                FlashAttention     0.04   0.06   0.21     0.82    2.85   10.41   41.74   167.19   670.76   2682.35
   Block-Sparse FlashAttention     0.06   0.06   0.06    0.12    0.44    0.86    1.70     3.29     6.55     13.34



    We report timing results on the forward pass, backward pass, and combined forward + backward pass.
We measure each method with and without dropout, masking, or both—except for Block Sparse, Longformer,
and BigBird. These methods did not successfully run the backward pass with masking due to a bug in
external libraries, so we measured them without masking to be generous. We use FP16 for all measurements,
except for Local Attention, whose implementation only supports FP32.
    For each baseline, we increase sequence length until it runs out of memory on the GPU, except for the
following exceptions: The Megatron implementation does not support sequence lengths longer than 2048.
Block-Sparse (OpenAI) does not support sequence lengths longer than 4096. Longformer and BigBird do not
support sequence lengths longer than 8092.
    We measure memory usage on the combined forward + backward pass, without dropout or masking.

Results Table 8 summarizes all the experimental configurations and contains pointers to the results tables.




                                                        31

Table 10: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with dropout and masking. Best in bold, second best underlined.

              Attention Method     128     256     512     1024    2048    4096      8192    16384     32768     65536
              PyTorch Attention    0.37    0.49    1.66     5.81   22.32   87.67       -        -        -         -
                      Megatron     0.35    0.32    0.77     2.42    8.43      -        -        -        -         -
                      Reformer     2.37    4.59    8.91    17.68   35.13   70.05    140.01      -        -         -
                Local Attention    0.55    0.62    1.49     4.03   13.78   27.61    55.20    110.27    221.40      -
                      Linformer    0.89    0.80    0.81     0.93    2.48    4.75     9.29     18.27    36.53       -
                          Smyrf    1.41    2.83    5.43    10.72   21.25   42.31    84.48    168.95      -         -
                      LSformer     1.75    1.76    3.01     7.50   20.07   39.08    76.39    150.82      -         -
                   Block Sparse    1.29    1.28    2.18     3.04    7.27   21.16       -        -        -         -
                    Longformer     1.27    1.31    1.29     2.04    5.24   10.74     25.95      -        -         -
                        BigBird    1.33    1.28    1.32     1.81    5.55   11.44     27.45      -        -         -
               FlashAttention      0.30    0.26    0.68     2.02    6.84   26.89    105.70   418.96   1666.89   6660.44
  Block-Sparse FlashAttention      0.30    0.27    0.29    0.59    1.50    2.94      5.82    11.85     23.98     47.61



Table 11: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by
sequence length, with dropout and masking. Best in bold, second best underlined.

              Attention Method     128     256      512    1024    2048     4096     8192    16384     32768      65536
              PyTorch Attention    0.84    0.86     2.35   8.29    31.75   124.19      -        -        -          -
                      Megatron     0.87    0.89     1.33   4.21    16.50       -       -        -        -          -
                      Reformer     4.30    7.76    14.60   28.74   57.79   116.34   237.57      -        -          -
                Local Attention    1.40    1.60     2.06   6.06    20.94    42.01    84.08   168.48   339.45        -
                      Linformer    1.57    1.49     1.55   1.60     4.19     8.04    15.71    30.92    61.47        -
                          Smyrf    3.41    5.08     9.35   18.18   36.03    71.68   143.04   285.87      -          -
                      LSformer     3.08    3.10     4.26   10.90   31.59    61.72   121.51   241.18      -          -
                   Block Sparse    2.54    2.52     3.71   5.44    13.29    39.19      -        -        -          -
                    Longformer     2.47    2.49     2.51   3.10    10.39    22.49    60.44      -        -          -
                        BigBird    2.51    2.49     2.52   3.40    10.97    23.89    63.28      -        -          -
               FlashAttention      0.43    0.41     0.95   2.55     9.56   37.49    147.75   586.61   2339.11    9341.30
  Block-Sparse FlashAttention      0.44    0.44    0.45    0.89    1.95     4.12     7.64    16.60     32.73      64.11



Table 12: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with masking. Best in bold, second best underlined.

               Attention Method     128     256     512    1024    2048    4096      8192    16384    32768      65536
               PyTorch Attention    0.30    0.30    0.63    1.93    7.08   27.45    112.90       -       -         -
                       Megatron     0.45    0.41    0.43    1.52    5.80      -        -         -       -         -
                       Reformer     1.87    3.00    5.37   10.43   21.40   43.83    92.80    203.24      -         -
                 Local Attention    0.70    0.81    1.02    2.09    6.64   13.34     26.77   54.02    110.11       -
                       Linformer    0.63    0.50    0.67    0.65    1.36    2.60     5.04      9.92    19.69     43.47
                           Smyrf    2.38    2.32    3.76    7.16   14.14   28.09    55.98    111.73      -         -
                       LSformer     1.22    1.29    1.44    3.28   10.99   21.72    43.29     86.32   172.76       -
                    Block Sparse    0.96    1.04    1.66    2.16    5.41   16.15       -         -       -         -
                     Longformer     0.99    0.98    0.99    1.56    4.79   11.07     32.98       -       -         -
                         BigBird    0.96    1.02    1.02    1.48    5.05   11.59     34.16       -       -         -
                FlashAttention      0.03    0.04    0.17    0.68    2.28    8.40     33.55   134.14   537.50    2150.88
   Block-Sparse FlashAttention      0.05    0.04    0.05   0.11    0.35    0.68      1.33     2.54     5.34      10.73



Table 13: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with masking. Best in bold, second best underlined.

              Attention Method     128     256     512     1024    2048    4096      8192    16384     32768     65536
              PyTorch Attention    0.44    0.46    1.53     5.33   20.34   79.87       -        -        -         -
                      Megatron     0.29    0.31    0.65     1.95    6.49      -        -        -        -         -
                      Reformer     2.31    4.47    8.68    17.20   34.14   68.09    136.02      -        -         -
                Local Attention    0.51    0.62    1.30     3.81   13.33   26.72    53.41    106.82    214.15      -
                      Linformer    0.76    0.81    0.94     0.87    2.24    4.25     8.35     16.38    32.67     72.11
                          Smyrf    1.34    2.77    5.30    10.46   20.73   41.27    82.41    164.86      -         -
                      LSformer     1.66    1.61    3.09     7.42   19.68   38.35    74.92    147.86      -         -
                   Block Sparse    1.24    1.25    2.04     2.91    6.78   19.67       -        -        -         -
                    Longformer     1.27    1.23    1.24     1.85    4.99   10.21    24.89       -        -         -
                        BigBird    1.43    1.50    1.44     1.69    5.25   10.86    26.26       -        -         -
               FlashAttention      0.21    0.22    0.62     1.84    5.77   22.25    86.21    338.91   1343.91   5361.09
  Block-Sparse FlashAttention      0.22    0.22    0.26    0.57    1.55    3.13      5.98    12.21     23.49     47.85




                                                           32

Table 14: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by
sequence length, with masking. Best in bold, second best underlined.

              Attention Method     128     256      512    1024     2048      4096     8192    16384     32768      65536
              PyTorch Attention    0.80    0.81     2.08   7.23     27.51    107.58      -        -        -          -
                      Megatron     0.81    0.83     1.09   3.36     12.39        -       -        -        -          -
                      Reformer     4.16    7.46    14.06   27.68    55.66    112.15   229.37      -        -          -
                Local Attention    1.39    1.68     2.08   5.83     20.04     40.16    80.44   161.35   325.11        -
                      Linformer    1.51    1.42     1.56   1.67      3.67      6.99    13.63    26.77    53.36     117.56
                          Smyrf    3.38    4.93     9.07   17.66    34.94     69.55   138.72   277.41      -          -
                      LSformer     3.08    3.10     4.26   10.90    31.59     61.72   121.51   241.18      -          -
                   Block Sparse    2.39    2.40     3.31   5.02     12.25     35.94      -        -        -          -
                    Longformer     2.36    2.34     2.38   2.94      9.83     21.35    58.12      -        -          -
                        BigBird    2.35    2.35     2.37   3.25     10.36     22.57    60.63      -        -          -
               FlashAttention      0.32    0.30     0.83   2.37      7.95    30.77    119.98   473.65   1883.43    7513.01
  Block-Sparse FlashAttention      0.34    0.34    0.36    0.69     1.85      3.89     7.16    14.85     30.46      60.03



Table 15: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with dropout. Best in bold, second best underlined.

               Attention Method     128     256     512     1024     2048     4096     8192    16384    32768      65536
               PyTorch Attention    0.26    0.24    0.57     1.80     6.56    25.34       -       -        -         -
                       Megatron     0.27    0.27    0.56     1.88     6.56       -        -       -        -         -
                       Reformer     1.83    2.96    5.31    10.33    21.19    43.42    91.96   201.34      -         -
                 Local Attention    0.51    0.60    0.78     2.01     6.23    12.52    25.07    50.50   102.18       -
                       Linformer    0.47    0.37    0.49    0.52      1.37     2.65     5.12    10.13    20.25     44.16
                           Smyrf    2.12    2.01    3.15     5.97    11.83    23.36    46.48    92.72      -         -
                       LSformer     1.28    1.33    1.51     3.39    11.40    22.54    44.96    89.85   179.73       -
                    Block Sparse    1.03    1.00    1.72     2.39     5.96    17.88       -       -        -         -
                     Longformer     1.02    1.03    1.03     1.73     5.10    11.63    34.22      -        -         -
                         BigBird    0.99    1.03    1.01     1.58     5.36    12.27    35.56      -        -         -
                FlashAttention      0.10    0.10    0.22     0.83     2.81    10.38    41.63   167.01   668.74    2678.11
   Block-Sparse FlashAttention      0.54    0.51    0.68     0.61    0.67     1.10     1.89     3.71     7.18      14.41



Table 16: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with dropout. Best in bold, second best underlined.

              Attention Method     128     256     512     1024     2048     4096      8192    16384     32768      65536
              PyTorch Attention    0.44    0.35    0.90     2.94    10.77    41.67       -        -        -          -
                      Megatron     0.28    0.33    0.92     2.94    10.80       -        -        -        -          -
                      Reformer     2.24    4.34    8.39    16.62    33.02    65.77    131.52      -        -          -
                Local Attention    0.51    0.58    1.41     3.71    12.96    25.98    51.94    103.72    207.78       -
                      Linformer    0.84    0.74    0.79     0.85     2.28     4.37     8.66     17.02    33.78        -
                          Smyrf    1.27    2.56    4.90     9.66    19.16    38.13    76.17    152.39      -          -
                      LSformer     1.67    1.77    3.03     7.52    20.10    39.13    76.35    150.83      -          -
                   Block Sparse    1.27    1.36    2.15     3.04     7.27    21.18       -        -        -          -
                    Longformer     1.28    1.34    1.38     1.98     5.24    10.74     25.95      -        -          -
                        BigBird    1.48    1.47    1.50     1.81     5.57    11.38     27.43      -        -          -
               FlashAttention      0.15    0.18    0.58     1.86     6.50    26.21    104.27   416.10   1661.92    6643.01
  Block-Sparse FlashAttention      0.17    0.17    0.17    0.40     1.10     2.04      4.43     9.33     18.28      37.31



Table 17: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by
sequence length, with dropout. Best in bold, second best underlined.

              Attention Method     128     256      512    1024     2048      4096     8192    16384     32768      65536
              PyTorch Attention    0.66    0.67     1.43    4.82    17.47     67.29      -        -        -          -
                      Megatron     0.88    0.90     1.49   4.73     17.41        -       -        -        -          -
                      Reformer     4.06    7.28    13.68   26.98    54.27    109.39   223.80      -        -          -
                Local Attention    1.09    1.40     1.99   5.61     19.23     38.62    77.30   154.63   311.12        -
                      Linformer    1.31    1.21     1.30   1.39      3.73      7.15    14.05    27.69    55.00        -
                          Smyrf    3.00    4.37     8.05   15.66    31.04     61.64   123.04   245.65      -          -
                      LSformer     3.07    3.17     4.31   10.89    31.54     61.78   121.56   240.94      -          -
                   Block Sparse    2.54    2.52     3.71   5.44     13.29     39.19      -        -        -          -
                    Longformer     2.47    2.49     2.51   3.10     10.39     22.49    60.44      -        -          -
                        BigBird    2.51    2.49     2.52   3.40     10.97     23.89    63.28      -        -          -
               FlashAttention      0.35    0.36    0.80    2.52      9.16    36.70    146.13   583.45   2332.01    9323.63
  Block-Sparse FlashAttention      0.91    0.83     0.94   0.92     1.83      3.50     7.02    13.56     26.71      53.92




                                                           33

Table 18: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length.
Best in bold, second best underlined.

                Attention Method     128      256      512     1024    2048     4096    8192     16384     32768       65536
                PyTorch Attention    0.21     0.22     0.43    1.27     4.32    16.47   67.77       -         -          -
                        Megatron     0.24     0.26     0.42    1.33     4.28       -       -        -         -          -
                        Reformer     1.77     2.82     5.01    9.74    20.03    41.11   87.39    192.40       -          -
                  Local Attention    0.48     0.57     0.80    1.90    5.76     11.56   23.13     46.65     94.74        -
                        Linformer    0.46     0.36     0.45    0.50    1.09      2.09    4.01     7.90     15.70       35.40
                            Smyrf    1.94     1.96     3.01    5.69    11.26    22.23   44.21     88.22       -          -
                        LSformer     1.21     1.34     1.34    3.31    11.01    21.71   43.27     86.32    172.85        -
                     Block Sparse    0.96     1.04     1.66    2.16    5.41     16.15      -        -         -          -
                      Longformer     0.99     0.98     0.99    1.56    4.79     11.07   32.98       -         -          -
                          BigBird    0.96     1.02     1.02    1.48    5.05     11.59   34.16       -         -          -
                 FlashAttention      0.08     0.09     0.18    0.68     2.40     8.42   33.54    134.03    535.95     2147.05
    Block-Sparse FlashAttention      0.56     0.52     0.63    0.65    0.61     0.96    1.69      3.02      5.69       11.77



Table 19: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length.
Best in bold, second best underlined.

              Attention Method      128      256     512      1024    2048     4096      8192    16384      32768       65536
              PyTorch Attention     0.26     0.29    0.78      2.44    8.82    33.87       -        -         -           -
                      Megatron      0.29     0.30    0.80      2.59    8.86       -        -        -         -           -
                      Reformer      2.18     4.21    8.14     16.12   32.02    63.84    127.60      -         -           -
                Local Attention     0.51     0.64    1.28      3.60   12.52    25.08    50.22    100.23     200.66        -
                      Linformer     0.69     0.76    0.69      0.80    2.04     3.88     7.67     15.04     30.11       63.15
                          Smyrf     1.24     2.49    4.77      9.42   18.65    37.12    74.15    148.35       -           -
                      LSformer      1.68     1.61    3.02      7.40   19.72    38.27    74.89    147.99       -           -
                   Block Sparse     1.24     1.25    2.04      2.91    6.78    19.67       -        -         -           -
                    Longformer      1.27     1.23    1.24      1.85    4.99    10.21    24.89       -         -           -
                        BigBird     1.43     1.50    1.44      1.69    5.25    10.86    26.26       -         -           -
               FlashAttention       0.11     0.16    0.52      1.62    5.45    21.57    84.75    336.00    1338.56     5343.19
  Block-Sparse FlashAttention       0.11     0.12    0.16     0.38    1.20     2.34      4.69     9.10      18.74       37.04



Table 20: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by
sequence length. Best in bold, second best underlined.

              Attention Method      128      256      512     1024    2048      4096     8192     16384      32768       65536
              PyTorch Attention     0.67     0.70     1.18    3.67    13.22     50.44      -         -         -           -
                      Megatron      0.74     0.65     1.23     3.80   13.21        -       -         -         -           -
                      Reformer      3.93     7.01    13.15    25.89   52.09    105.00   215.13       -         -           -
                Local Attention     1.09     1.27     1.99    5.38    18.32     36.77    73.67    147.29    296.35         -
                      Linformer     1.31     1.25     1.30    1.29     3.20      6.10    11.93     23.39     46.72      100.52
                          Smyrf     2.98     4.23     7.78    15.12   29.96     59.45   118.60    237.02       -           -
                      LSformer      3.03     3.05     4.26    10.70   30.77     60.15   118.33    234.94       -           -
                   Block Sparse     2.39     2.40     3.31    5.02    12.25     35.94      -         -         -           -
                    Longformer      2.36     2.34     2.38    2.94     9.83     21.35    58.12       -         -           -
                        BigBird     2.35     2.35     2.37    3.25    10.36     22.57    60.63       -         -           -
               FlashAttention       0.31     0.31    0.73     2.29     7.64    30.09    118.50    470.51    1876.08     7492.85
  Block-Sparse FlashAttention       0.74     0.77     0.82    0.88    1.71      3.21     6.56     12.60      24.93       50.39



Table 21: Memory usage (MB) of various exact/approximate/sparse attention mechanisms by sequence length. Best
in bold, second best underlined.

                 Attention Method      128     256     512     1024   2048      4096    8192     16384     32768     65536
                 PyTorch Attention     36      104     336     1184   4416     17024       -        -         -        -
                         Megatron      36      104     336     1184   4416        -        -        -         -        -
                         Reformer      377     754     1508    3016   6033     12067    24134       -         -        -
                   Local Attention     53      110     232      592   1696      3392    6784     13568     27136       -
                         Linformer     25       52     114      287    832     1652      3292    6572      13132     26252
                             Smyrf     217     434     868     1737   3474      6947    13894    27788        -        -
                         LSformer      72      152     333      796   2540      5068    10125    20240        -        -
                      Block Sparse     33       82     228      408    910     2401        -        -         -        -
                       Longformer      30       61     124      277    681     1370      2748       -         -        -
                           BigBird     33       66     131      294    708     1431      2872       -         -        -
                  FlashAttention       22      44      104     209    418       836     1672     3344      6688      13376
     Block-Sparse FlashAttention       22       44     104      209    418      836      1672     3344      6690     13384




                                                              34


```

表格中的 “-” 表示该实现因显存或实现限制无法运行；加粗/下划线的最佳与次佳标记在纯文本转录中省略，但正文结论与数值保持不变。


## 参考文献

以下按原论文编号保留完整书目信息（论文标题和作者名保留原文，便于检索）。

[1] Alok Aggarwal and Jeffrey S. Vitter. *The input/output complexity of sorting and related problems*. Communications of the ACM, 31(9):1116–1127, 1988.  
[2] Irwan Bello. *LambdaNetworks: Modeling long-range interactions without attention*. arXiv:2102.08602, 2021.  
[3] Iz Beltagy, Matthew E. Peters, and Arman Cohan. *Longformer: The long-document transformer*. arXiv:2004.05150, 2020.  
[4] L. Susan Blackford et al. *An updated set of basic linear algebra subprograms (BLAS)*. ACM TOMS, 28(2):135–151, 2002.  
[5] Tom Brown et al. *Language models are few-shot learners*. NeurIPS 33:1877–1901, 2020.  
[6] Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. *Neural legal judgment prediction in English*. ACL, 2019.  
[7] Ilias Chalkidis et al. *Paragraph-level rationale extraction through regularization*. NAACL, 2021.  
[8] Benjamin Charlier et al. *Kernel operations on the GPU, with autodiff, without memory overflows*. JMLR 22(74):1–6, 2021.  
[9] Beidi Chen et al. *Scatterbrain: Unifying sparse and low-rank attention*. NeurIPS, 2021.  
[10] Tianqi Chen et al. *Training deep nets with sublinear memory cost*. arXiv:1604.06174, 2016.  
[11] Rewon Child et al. *Generating long sequences with sparse transformers*. arXiv:1904.10509, 2019.  
[12] Krzysztof M. Choromanski et al. *Rethinking attention with Performers*. ICLR, 2020.  
[13] Xiang Dai et al. *Revisiting transformer-based models for long document classification*. arXiv:2204.06683, 2022.  
[14] Zihang Dai et al. *Transformer-XL: Attentive language models beyond a fixed-length context*. ACL, 2019.  
[15] Tri Dao et al. *Learning fast algorithms for linear transforms using butterfly factorizations*. ICML, 2019.  
[16] Tri Dao et al. *Kaleidoscope: An efficient, learnable representation for all structured linear maps*. ICLR, 2020.  
[17] Tri Dao et al. *Pixelated butterfly: Simple and efficient sparse training for neural network models*. ICLR, 2022.  
[18] Tri Dao et al. *Monarch: Expressive structured matrices for efficient and accurate training*. ICML, 2022.  
[19] Giannis Daras et al. *SMYRF: Efficient attention using asymmetric clustering*. NeurIPS 33:6476–6489, 2020.  
[20] Christopher De Sa et al. *A two-pronged progress in structured dense matrix vector multiplication*. SODA, 2018.  
[21] Peter J. Denning. *The working set model for program behavior*. CACM, 11(5):323–333, 1968.  
[22] Jacob Devlin et al. *BERT: Pre-training of deep bidirectional transformers for language understanding*. 2019.  
[23] Xin Dong et al. *Learning to prune deep neural networks via layer-wise optimal brain surgeon*. arXiv:1705.07565, 2017.  
[24] Alexey Dosovitskiy et al. *An image is worth 16x16 words: Transformers for image recognition at scale*. ICLR, 2020.  
[25] Y. Eidelman and I. Gohberg. *On a new class of structured matrices*. Integral Equations and Operator Theory, 34(3):293–324, 1999.  
[26] Jean Feydy et al. *Fast geometric learning with symbolic matrices*. NeurIPS 33, 2020.  
[27] Jörg Flum and Martin Grohe. *Parameterized Complexity Theory*. Springer, 2006.  
[28] Jonathan Frankle and Michael Carbin. *The lottery ticket hypothesis: Finding sparse, trainable neural networks*. ICLR, 2018.  
[29] Jonathan Frankle et al. *Stabilizing the lottery ticket hypothesis*. arXiv:1903.01611, 2019.  
[30] Jonathan Frankle et al. *Linear mode connectivity and the lottery ticket hypothesis*. ICML, 2020.  
[31] Karan Goel et al. *It’s raw! Audio generation with state-space models*. ICML, 2022.  
[32] Aaron Gokaslan et al. *OpenWebText corpus*. 2019.  
[33] Jim Gray et al. *Data cube: A relational aggregation operator generalizing group-by, cross-tab, and sub-totals*. Data Mining and Knowledge Discovery, 1(1):29–53, 1997.  
[34] Andreas Griewank and Andrea Walther. *Evaluating derivatives: Principles and techniques of algorithmic differentiation*. SIAM, 2008.  
[35] Albert Gu et al. *HiPPO: Recurrent memory with optimal polynomial projections*. NeurIPS, 2020.  
[36] Albert Gu et al. *Combining recurrent, convolutional, and continuous-time models with linear state space layers*. NeurIPS 34, 2021.  
[37] Albert Gu, Karan Goel, and Christopher Ré. *Efficiently modeling long sequences with structured state spaces*. ICLR, 2022.  
[38] Song Han et al. *Learning both weights and connections for efficient neural networks*. arXiv:1506.02626, 2015.  
[39] Song Han et al. *Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding*. ICLR, 2016.  
[40] John Hennessy and David Patterson. *Memory hierarchy design*. In Computer Architecture: A Quantitative Approach, 2003.  
[41] Sara Hooker. *The hardware lottery*. arXiv:2009.06489, 2020.  
[42] Weizhe Hua et al. *Transformer quality in linear time*. arXiv:2202.10447, 2022.  
[43] Andrei Ivanov et al. *Data movement is all you need: A case study on optimizing transformers*. PMLS 3:711–732, 2021.  
[44] Zhe Jia and Peter Van Sandt. *Dissecting the Ampere GPU architecture via microbenchmarking*. GTC, 2021.  
[45] Zhe Jia et al. *Dissecting the NVIDIA Volta GPU architecture via microbenchmarking*. arXiv:1804.06826, 2018.  
[46] Zhe Jia et al. *Dissecting the Graphcore IPU architecture via microbenchmarking*. arXiv:1912.03413, 2019.  
[47] Alistair E. W. Johnson et al. *MIMIC-III, a freely accessible critical care database*. Scientific Data, 3(1):1–9, 2016.  
[48] Norman P. Jouppi et al. *In-datacenter performance analysis of a tensor processing unit*. ISCA, 2017.  
[49] Thomas Kailath et al. *Displacement ranks of matrices and linear equations*. JMAA, 68(2):395–407, 1979.  
[50] Angelos Katharopoulos et al. *Transformers are RNNs: Fast autoregressive transformers with linear attention*. ICML, 2020.  
[51] Nikita Kitaev et al. *Reformer: The efficient transformer*. ICML, 2020.  
[52] Zhenzhong Lan et al. *ALBERT: A lite BERT for self-supervised learning of language representations*. ICLR, 2020.  
[53] Mingzhen Li et al. *The deep learning compiler: A comprehensive survey*. IEEE TPDS, 32(3):708–727, 2020.  
[54] Valerii Likhosherstov et al. *Sub-linear memory: How to make Performers slim*. arXiv:2012.11346, 2020.  
[55] Ji Lin et al. *Runtime neural pruning*. NeurIPS 30, 2017.  
[56] Yinhan Liu et al. *RoBERTa: A robustly optimized BERT pretraining approach*. arXiv:1907.11692, 2019.  
[57] Xuezhe Ma et al. *Luna: Linear unified nested attention*. NeurIPS 34, 2021.  
[58] Peter Mattson et al. *MLPerf training benchmark*. PMLR 2:336–349, 2020.  
[59] Frank McSherry et al. *Scalability! But at what COST?* HotOS XV, 2015.  
[60] Maxim Milakov and Natalia Gimelshein. *Online normalizer calculation for softmax*. arXiv:1805.02867, 2018.  
[61] NVIDIA. *NVIDIA Tesla V100 GPU architecture*. 2017.  
[62] NVIDIA. *NVIDIA A100 Tensor Core GPU architecture*. 2020.  
[63] NVIDIA. *NVIDIA H100 Tensor Core GPU architecture*. 2022.  
[64] D. Stott Parker. *Random butterfly transformations with applications in computational linear algebra*. 1995.  
[65] Adam Paszke et al. *PyTorch: An imperative style, high-performance deep learning library*. NeurIPS 32, 2019.  
[66] Markus N. Rabe and Charles Staats. *Self-attention does not need O(n²) memory*. arXiv:2112.05682, 2021.  
[67] Alec Radford et al. *Language models are unsupervised multitask learners*. OpenAI blog, 2019.  
[68] Jack Rae and Ali Razavi. *Do transformers need deep long-range memory?* ACL, 2020.  
[69] Jack W. Rae et al. *Compressive transformers for long-range sequence modelling*. ICLR, 2020.  
[70] Jonathan Ragan-Kelley et al. *Halide: A language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines*. ACM SIGPLAN Notices, 48(6):519–530, 2013.  
[71] Raghu Ramakrishnan and Johannes Gehrke. *Database management systems*. McGraw-Hill, 2003.  
[72] Benjamin Recht and Christopher Ré. *Parallel stochastic gradient algorithms for large-scale matrix completion*. Mathematical Programming Computation, 5(2):201–226, 2013.  
[73] Hongyu Ren et al. *Combiner: Full attention transformer with sparse computation cost*. NeurIPS 34, 2021.  
[74] Aurko Roy et al. *Efficient content-based sparse attention with routing transformers*. TACL, 9:53–68, 2021.  
[75] Amit Sabne. *XLA: Compiling machine learning for peak performance*. 2020.  
[76] Victor Sanh et al. *Movement pruning: Adaptive sparsity by fine-tuning*. arXiv:2005.07683, 2020.  
[77] Mohammad Shoeybi et al. *Megatron-LM: Training multi-billion parameter language models using model parallelism*. arXiv:1909.08053, 2019.  
[78] Vikas Sindhwani et al. *Structured transforms for small-footprint deep learning*. NeurIPS, 2015.  
[79] Sainbayar Sukhbaatar et al. *Adaptive attention span in transformers*. ACL, 2019.  
[80] Yi Tay et al. *Long Range Arena: A benchmark for efficient transformers*. ICLR, 2020.  
[81] Yi Tay et al. *Efficient transformers: A survey*. arXiv:2009.06732, 2020.  
[82] Ashish Vaswani et al. *Attention is all you need*. NeurIPS 30, 2017.  
[83] Hongyu Wang et al. *DeepNet: Scaling transformers to 1,000 layers*. arXiv:2203.00555, 2022.  
[84] Sinong Wang et al. *Linformer: Self-attention with linear complexity*. arXiv:2006.04768, 2020.  
[85] Samuel Williams et al. *Roofline: An insightful visual performance model for multicore architectures*. CACM, 52(4):65–76, 2009.  
[86] Michael E. Wolf and Monica S. Lam. *A data locality optimizing algorithm*. PLDI, 1991.  
[87] Thomas Wolf et al. *Transformers: State-of-the-art natural language processing*. EMNLP Demonstrations, 2020.  
[88] David P. Woodruff. *Optimal space lower bounds for all frequency moments*. SODA, 2004.  
[89] Felix Wu et al. *Pay less attention with lightweight and dynamic convolutions*. ICLR, 2019.  
[90] Yunyang Xiong et al. *Nyströmformer: A Nyström-based algorithm for approximating self-attention*. AAAI 35:14138, 2021.  
[91] Li Yuan et al. *Tokens-to-token ViT: Training vision transformers from scratch on ImageNet*. ICCV, 2021.  
[92] Manzil Zaheer et al. *Big Bird: Transformers for longer sequences*. NeurIPS 33, 2020.  
[93] Shuangfei Zhai et al. *An attention free transformer*. arXiv:2105.14103, 2021.  
[94] Chen Zhu et al. *Long-short transformer: Efficient transformers for language and vision*. NeurIPS 34, 2021.

---

**译注：** 公式中的符号、算法编号和表格数值依照论文原文；图 1–8 的图注已在正文相应位置解释，实验表 9–21 的完整数值已在“图表与算法补充译文”中转录。 
