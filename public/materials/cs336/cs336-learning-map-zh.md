# Stanford CS336 Assignment 1：从零构建语言模型与工程核算导读

更新于 2026-10-27。

Stanford CS336（Language Modeling from Scratch）要求在不调用 `torch.nn` 预置层与 `torch.nn.functional` 的前提下，仅基于张量基础运算手写实现现代大语言模型的全部核心组件。本导读对照官方 Assignment 1 讲义（PDF 见右侧/下方原始文档），梳理 **Byte-Level BPE 分词器**、**Pre-Norm Transformer 架构（RMSNorm + RoPE + SwiGLU）**、**参数量 / FLOPs / 显存精确核算** 以及 **AdamW 训练与 Top-$p$ 解码闭环** 的完整实现脉络。

## 课程与代码入口

- [Stanford CS336 课程官网](https://cs336.stanford.edu/)
- [Assignment 1 官方仓库 (`stanford-cs336/assignment1-basics`)](https://github.com/stanford-cs336/assignment1-basics)
- [个人完整实现与测试仓库 (`liangqianxing/cs336-labs`)](https://github.com/liangqianxing/cs336-labs)
- [专栏主页与实验附件列表](https://liangqianxing.github.io/learning/cs336/)

## 8 篇学习记录与对应模块

| 顺序 | 学习记录 | 实现模块 (`cs336_basics/`) | 通过的官方测试 |
| --- | --- | --- | --- |
| 01 | [从 Linear、Embedding 搞清张量形状](/posts/cs336-linear-embedding-shapes/) | `linear.py`、`embedding.py` | `test_linear`、`test_embedding` |
| 02 | [RMSNorm 的广播与 SwiGLU 的门控](/posts/cs336-rmsnorm-swiglu/) | `rmsnorm.py`、`activations.py`、`swiglu.py` | `test_rmsnorm`、`test_silu_matches_pytorch`、`test_swiglu` |
| 03 | [Softmax 的数值稳定与 Attention Mask](/posts/cs336-softmax-attention-mask/) | `activations.py`、`attention.py` | `test_softmax_matches_pytorch`、`test_scaled_dot_product_attention`、`test_4d_scaled_dot_product_attention` |
| 04 | [多头注意力怎样拆头、计算和合头](/posts/cs336-multihead-attention-shapes/) | `attention.py` (`MultiHeadSelfAttention`) | `test_multihead_self_attention` |
| 05 | [RoPE 旋转位置编码与带位置的多头注意力](/posts/cs336-rope-rotary-position-embedding/) | `rope.py`、`attention.py` | `test_rope`、`test_multihead_self_attention_with_rope` |
| 06 | [Pre-Norm Transformer Block 与完整语言模型装配](/posts/cs336-pre-norm-transformer-block-lm/) | `transformer.py` (`TransformerBlock`, `TransformerLM`) | `test_transformer_block`、`test_transformer_lm`、`test_transformer_lm_truncated_input` |
| 07 | [Byte-Level BPE 分词器训练、倒排索引加速与流式编码](/posts/cs336-bpe-tokenizer-training-encoding/) | `bpe.py`、`tokenizer.py` | `test_train_bpe*` (3 项)、`test_tokenizer.py` (23 项) |
| 08 | [数值稳定交叉熵、AdamW、余弦调度与完整训练闭环](/posts/cs336-cross-entropy-adamw-training-loop/) | `nn_utils.py`、`optimizer.py`、`data.py`、`serialization.py`、`decoding.py`、`train.py` | `test_nn_utils.py` (3 项)、`test_optimizer.py` (2 项)、`test_get_batch`、`test_checkpointing` |

---

## 一、Byte-Level BPE 分词器核心要点（讲义第 2 节）

1. **为什么使用 UTF-8 字节级 BPE**：
   - 任何 Unicode 字符串编码为 UTF-8 后都是 $0 \sim 255$ 的字节序列，基础词表固定为 256，天然消除 `<unk>`（Out-of-Vocabulary）。
   - 解码时必须先拼接所有 token 的 `bytes` 再调用 `.decode("utf-8", errors="replace")`，不能逐字节解码，否则多字节汉字或 Emoji 会触发 `UnicodeDecodeError`。
2. **两级预切分（Special Tokens + GPT-2 正则）**：
   - 先用按长度降序排列的 `(|...|)` 正则切出 `<|endoftext|>` 等特殊 token，严禁跨特殊 token 边界合并。
   - 普通文本段使用 `regex.finditer(PAT, segment)` 拆成独立预分词单元（pre-tokens），统计唯一单词频次表。
3. **倒排索引增量合并（`bpe.py`）**：
   - 维护 `pair_counts` 与 `pair_to_words`（每个字节对出现在哪些唯一单词下标中）。
   - 每次选出 `(count, pair)` 字典序最大的字节对合并，只局部更新包含该字节对的单词，将 `test_train_bpe_speed` 耗时从 ~3.0s 降至 **0.18s**。

---

## 二、Pre-Norm Transformer 架构与资源核算（讲义第 3 节）

### 1. 子层结构与公式汇总

- **RMSNorm**（升 `float32` 计算均方根再转回原 `dtype`）：
  $$\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \varepsilon}} \odot g$$
- **RoPE 旋转位置编码**（对每头 $d_k$ 维度的奇偶相邻对 $(x_{2k-2}, x_{2k-1})$ 施加旋转角 $\theta_{i,k} = i / \Theta^{(2k-2)/d_k}$）：
  $$\begin{bmatrix} x'_{2k-2} \\ x'_{2k-1} \end{bmatrix} = \begin{bmatrix} \cos\theta_{i,k} & -\sin\theta_{i,k} \\ \sin\theta_{i,k} & \cos\theta_{i,k} \end{bmatrix} \begin{bmatrix} x_{2k-2} \\ x_{2k-1} \end{bmatrix}$$
- **SwiGLU 门控前馈层**（3 个无偏置线性矩阵 $W_1, W_3 \in \mathbb{R}^{d_{\text{ff}} \times d}$，$W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$）：
  $$\operatorname{SwiGLU}(x) = \big(\operatorname{SiLU}(x W_1^\top) \odot (x W_3^\top)\big) W_2^\top$$
- **Pre-Norm Transformer Block**：
  $$h = x + \operatorname{MHA}(\operatorname{RMSNorm}_1(x)), \qquad y = h + \operatorname{SwiGLU}(\operatorname{RMSNorm}_2(h))$$

### 2. 参数量与 FLOPs 闭合公式（Problem `transformer_accounting`）

设词表大小 $V$、上下文长度 $S$、层数 $L$、隐藏维 $d$、FFN 维 $d_{\text{ff}}$：

- **可训练参数总数**：
  $$N_{\text{params}} = 2 V d + L\big(4 d^2 + 3 d d_{\text{ff}} + 2 d\big) + d$$
- **单样本（$B=1$）前向传播矩阵乘法总 FLOPs**：
  $$\text{FLOPs}_{\text{fwd}} = L\big(8 S d^2 + 4 S^2 d + 6 S d d_{\text{ff}}\big) + 2 S d V$$

以讲义要求的 **GPT-2 XL**（$V=50{,}257, S=1{,}024, L=48, d=1{,}600, H=25, d_{\text{ff}}=4{,}288$）为例：
- 参数量 $N_{\text{params}} = 1{,}640{,}452{,}800 \approx 1.640\text{ B}$（FP32 权重占 $6.56\text{ GB}$）；
- 单样本前向 FLOPs 为 **$4.513 \times 10^{12}$（4.51 TFLOPs）**，其中 SwiGLU FFN 占 **67.25%**，QKVO 投影占 **22.30%**，注意力矩阵乘法 $QK^\top + PV$ 占 **7.14%**，`lm_head` 占 **3.31%**。

---

## 三、交叉熵、AdamW 显存核算与训练闭环（讲义第 4～6 节）

1. **LogSumExp 交叉熵**：
   $$\ell(o, y) = \log \sum_{a=1}^V \exp(o_a - m) - (o_y - m), \qquad m = \max_a o_a$$
2. **解耦权重衰减 AdamW**：
   $$\theta_t = \theta_{t-1} - \alpha \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \frac{m_t}{\sqrt{v_t} + \varepsilon} - \alpha \lambda \theta_{t-1}$$
3. **FP32 AdamW 训练显存公式（Problem `adamw_accounting`）**：
   - 静态显存（参数 + 梯度 + $m, v$ 状态）$= (4 + 4 + 8) N_{\text{params}} = 16 N_{\text{params}}$ 字节。
   - 对 GPT-2 XL，静态部分占 $26.25\text{ GB}$，每样本激活值约 $14.9\text{ GB}$，80GB H100 在纯 FP32 下最大支持 `batch_size = 3`。
4. **全量测试验证命令**：
   ```bash
   bash scripts/check_completed.sh
   # ======================== 46 passed, 2 skipped in 4.66s =========================
   ```
