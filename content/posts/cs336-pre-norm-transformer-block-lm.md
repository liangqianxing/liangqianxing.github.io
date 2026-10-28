---
title: 'Stanford CS336 学习记录 06：Pre-Norm Transformer Block 与完整语言模型装配'
date: '2026-10-23'
description: '将 RMSNorm、带 RoPE 的因果多头自注意力和 SwiGLU 组装成 Pre-Norm Transformer Block，搭建完整 TransformerLM，并核算 GPT-2 XL 的参数量与前向 FLOPs。'
series: Stanford CS336
seriesOrder: 6
cover: /images/posts/cs336-pre-norm-transformer-block-lm/cs336-pre-norm-transformer-block-lm-cover.webp
coverAlt: 干净残差流主干穿过多个 Pre-Norm 子层并汇聚到词表投影层的 AI 辅助概念插画
codeUrl: https://github.com/liangqianxing/cs336-labs/tree/b5c93ad0cc73178b2b2a62e518a0900da47a0fb7
attachmentUrl: /materials/cs336/experiments/cs336-06-transformer-block-lm-20261023.zip
categories:
  - 学习笔记
tags:
  - CS336
  - PyTorch
  - Transformer
  - Pre-Norm
  - FLOPs
draft: false
hidden: false
haloSync: false
haloPublished: false
---

前五篇把 `Linear`、`Embedding`、`RMSNorm`、`SwiGLU`、`RotaryPositionalEmbedding` 和因果 `MultiHeadSelfAttention` 全部单独打通了。这一篇进入 Assignment 1 第 3.5 节：把这些积木拼成 **Pre-Norm `TransformerBlock`** 和完整的 **`TransformerLM`**，并完成课程要求的资源核算（参数量与前向 FLOPs 拆解）。

[查看代码仓库](https://github.com/liangqianxing/cs336-labs/tree/b5c93ad0cc73178b2b2a62e518a0900da47a0fb7) · [下载本节代码与测试 ZIP](/materials/cs336/experiments/cs336-06-transformer-block-lm-20261023.zip)

## Pre-Norm 与 Post-Norm：为什么残差主干上不放归一化

原始 Vaswani et al. (2017) 采用的是 Post-Norm 结构：$x \leftarrow \operatorname{Norm}(x + \operatorname{SubLayer}(x))$，每次残差相加之后都要经过一次归一化。而在深层网络中，把归一化移到每个子层**入口**的 **Pre-Norm** 结构能保持一条从底层 Embedding 直通顶层输出的干净“残差流（Residual Stream）”：

$$
\begin{aligned}
h &= x + \operatorname{MultiHeadSelfAttention}\big(\operatorname{RMSNorm}_1(x)\big), \\
y &= h + \operatorname{SwiGLU}\big(\operatorname{RMSNorm}_2(h)\big).
\end{aligned}
$$

反向传播时，梯度可以沿着恒等映射项 $\frac{\partial y}{\partial x} = I + \dots$ 直接回传到浅层，显著缓解深层 Transformer 训练初期的梯度消失与不稳定问题。作为代价，经过 `num_layers` 层累加后的残差流激活幅度会随层数增长，因此在进入最后的词表线性头 `lm_head` 之前，必须额外加一层 **`ln_final`（最终 RMSNorm）**。

## 对齐官方 `state_dict` 键名与 `nn.ModuleList`

在编写 `cs336_basics/transformer.py` 时，除了数学公式正确，模块属性名还必须与官方参考权重字典 `ts_state_dict` 严格一致，这样 `load_state_dict` 才能直接加载而无需手写繁琐的键名重映射。

单个 `TransformerBlock` 的子模块命名对照如下：

| 属性名 | 模块类型 | 内部参数键名 | 参数形状 |
| --- | --- | --- | --- |
| `self.ln1` | `RMSNorm(d_model)` | `ln1.weight` | `(d_model,)` |
| `self.attn` | `MultiHeadSelfAttention` | `attn.{q,k,v,output}_proj.weight` | 各为 `(d_model, d_model)` |
| `self.ln2` | `RMSNorm(d_model)` | `ln2.weight` | `(d_model,)` |
| `self.ffn` | `SwiGLU(d_model, d_ff)` | `ffn.w1.weight`、`ffn.w2.weight`、`ffn.w3.weight` | `(d_ff, d_model)`、`(d_model, d_ff)`、`(d_ff, d_model)` |

而在拼装顶层 `TransformerLM` 时，有多层 `TransformerBlock`。这里必须使用 `nn.ModuleList` 而不是普通 Python `list`：

```python
self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
self.layers = nn.ModuleList(
    [
        TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            device=device,
            dtype=dtype,
        )
        for _ in range(num_layers)
    ]
)
self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
```

如果把 `self.layers` 写成普通 Python 列表 `[...]`，PyTorch 的 `nn.Module` 不会把列表里的子模块注册进参数树——`model.parameters()` 会漏掉所有 Transformer 层的权重，`state_dict()` 也不会生成 `layers.0.attn.q_proj.weight` 这样的层级键名。

## 为什么 `TransformerLM` 返回未归一化的 Logits

阅读讲义图 1 时，示意图最顶端画了一个 `Softmax -> Output Probabilities`。但对照 `adapters.py` 中 `run_transformer_lm` 的返回值契约：

> `Float[Tensor, "batch_size sequence_length vocab_size"]`: Tensor with the predicted unnormalized next-word distribution for each token.

`TransformerLM.forward` 的最后一步是 `return self.lm_head(x)`，直接返回未归一化的 logits，**不在模型内部调用 Softmax**：

```python
def forward(self, in_indices: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
    seq_len = in_indices.shape[-1]
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_indices.device, dtype=torch.long)
    x = self.token_embeddings(in_indices)
    for layer in self.layers:
        x = layer(x, token_positions=token_positions)
    x = self.ln_final(x)
    return self.lm_head(x)
```

这样设计有两个工程原因：
1. 训练阶段计算交叉熵损失时，需要将 `log` 与 `softmax` 合并为数值稳定的 `LogSumExp` 消去指数项；如果模型先输出了概率 `softmax(o)`，一旦某个目标概率下溢为 `0.0`，再取 `log` 就会直接产生 `-inf` / `NaN`。
2. 推理采样阶段往往要先对 logits 除以温度 $\tau$（`logits / temperature`）再做 Softmax，直接输出 logits 最灵活。

此外，当输入序列长度 `seq_len < context_length`（对应官方测试 `test_transformer_lm_truncated_input`）时，`token_positions` 按实际 `seq_len` 生成 `0 .. seq_len - 1`，RoPE 缓存和因果 mask 自动按当前序列长度切片，无需把输入强行 padding 到 `context_length`。

## 资源核算（Problem `transformer_accounting`）：以 GPT-2 XL 为例

课程讲义第 3.5 节要求对采用本作业架构（RMSNorm + RoPE + SwiGLU + 无偏置 Linear + 非绑定词表权重）的 **GPT-2 XL** 配置进行精确的参数量与前向 FLOPs 核算：
- `vocab_size` $V = 50{,}257$，`context_length` $S = 1{,}024$
- `num_layers` $L = 48$，`d_model` $d = 1{,}600$，`num_heads` $H = 25$，`d_ff` $d_{\text{ff}} = 4{,}288$

### 1. 可训练参数量与 FP32 权重显存

每层 `TransformerBlock` 的参数包含：
- Attention 四个投影矩阵 $W_Q, W_K, W_V, W_O$：$4 \times d^2 = 4 \times 1{,}600^2 = 10{,}240{,}000$
- SwiGLU 三个矩阵 $W_1, W_2, W_3$：$3 \times d \times d_{\text{ff}} = 3 \times 1{,}600 \times 4{,}288 = 20{,}582{,}400$
- 两个 RMSNorm 缩放向量：`ln1` 与 `ln2` 共 $2 \times d = 3{,}200$
- 单层合计：$30{,}825{,}600$ 个参数；48 层共 $48 \times 30{,}825{,}600 = 1{,}479{,}628{,}800$ 个参数。

加上词表层与最终归一化：
- `token_embeddings`（$V \times d$）与 `lm_head`（$V \times d$）：$2 \times 50{,}257 \times 1{,}600 = 160{,}822{,}400$
- `ln_final`：$1{,}600$

**总可训练参数量**：

$$
N_{\text{params}} = 1{,}479{,}628{,}800 + 160{,}822{,}400 + 1{,}600 = 1{,}640{,}452{,}800 \approx 1.640\text{ B}.
$$

按单精度浮点数（FP32，每参数 4 字节）存储，仅加载模型权重需要 $6{,}561{,}811{,}200\text{ bytes} \approx 6.56\text{ GB}$（或 $6.11\text{ GiB}$）。

### 2. 单样本前向传播矩阵乘法 FLOPs 拆解（$B = 1$）

利用矩阵乘法规则 $(m \times n) @ (n \times p)$ 需要 $2mnp$ FLOPs：

| 模块位置 | 矩阵乘法运算 | 单层 FLOPs 公式 | 48 层总 FLOPs（$S=1024$） | 占总 FLOPs 比例 |
| --- | --- | --- | ---: | ---: |
| Attention 投影 ($Q, K, V, O$) | $4 \times (S \times d) @ (d \times d)$ | $8 S d^2$ | $1.0066 \times 10^{12}$ | 22.30% |
| Attention 分数与加权 ($QK^\top, PV$) | $2 \times H \times (S \times d_k) @ (d_k \times S)$ | $4 S^2 d$ | $0.3221 \times 10^{12}$ | 7.14% |
| SwiGLU FFN ($W_1, W_2, W_3$) | $3 \times (S \times d) @ (d \times d_{\text{ff}})$ | $6 S d d_{\text{ff}}$ | $3.0351 \times 10^{12}$ | 67.25% |
| 输出层 `lm_head` | $(S \times d) @ (d \times V)$ | 整网一次 $2 S d V$ | $0.1492 \times 10^{12}$ | 3.31% |
| **合计** | — | — | **$4.5131 \times 10^{12}$ (4.51 TFLOPs)** | **100.00%** |

可以看到：
1. 在常规上下文长度 $S = 1{,}024$ 下，**SwiGLU FFN 占据了 67.25% 的主导计算量**，加上 QKVO 投影的 22.30%，线性投影层合计超过 89%。
2. 随着模型规模从 GPT-2 Small（$d=768$）增大到 GPT-2 XL（$d=1600$），与 $d^2$ 成正比的线性层（FFN + 投影）占比持续上升，而与 $V d$ 成正比的 `lm_head` 占比从约 19% 降至 3.31%。
3. 反过来，如果保持 GPT-2 XL 不变，把上下文长度 $S$ 扩大 16 倍到 $16{,}384$，线性层 FLOPs 呈线性放大 $16\times$，而与 $S^2$ 成正比的注意力矩阵乘法（$4 L S^2 d$）会暴增 $256\times$，达到 $82.46\text{ TFLOPs}$，占整网总计算量（$146.15\text{ TFLOPs}$）的 **56.42%**，成为新的性能瓶颈。这也正是长上下文场景必须引入 FlashAttention 的根本原因。

## 官方测试验证结果

运行 `test_model.py` 中的全部 13 项模型测试：

```text
tests/test_model.py::test_transformer_block PASSED
tests/test_model.py::test_transformer_lm PASSED
tests/test_model.py::test_transformer_lm_truncated_input PASSED
```

至此，`test_model.py` 全模块 13 个测试用例 100% 通过。下一篇回到文本入口，完成 [Byte-Level BPE 分词器训练与流式编码实现](/posts/cs336-bpe-tokenizer-training-encoding/)。

*封面使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 辅助生成，属于概念插画；准确的架构细节与 FLOPs 核算以正文为准。*

## 参考资料

- [Stanford CS336 Assignment 1 官方仓库](https://github.com/stanford-cs336/assignment1-basics)
- [Xiong et al., 2020: On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [Kaplan et al., 2020: Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
