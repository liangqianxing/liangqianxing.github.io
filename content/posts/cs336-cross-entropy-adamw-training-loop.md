---
title: 'Stanford CS336 学习记录 08：数值稳定交叉熵、AdamW、余弦调度与完整训练闭环'
date: '2026-10-27'
description: '用 LogSumExp 消去交叉熵中的指数溢出，手写解耦权重衰减的 AdamW 优化器、全局 L2 梯度裁剪、Warmup 余弦退火调度、memmap 数据采样、Checkpoint 序列化与 Top-p 解码，完成 CS336 Assignment 1 全部 48 项测试。'
series: Stanford CS336
seriesOrder: 8
cover: /images/posts/cs336-cross-entropy-adamw-training-loop/cs336-cross-entropy-adamw-training-loop-cover.webp
coverAlt: 滑动窗口采样、余弦退火曲线、双动量优化轨迹与检查点归档构成的训练闭环 AI 辅助概念插画
codeUrl: https://github.com/liangqianxing/cs336-labs/tree/b5c93ad0cc73178b2b2a62e518a0900da47a0fb7
attachmentUrl: /materials/cs336/experiments/cs336-08-adamw-training-loop-20261027.zip
categories:
  - 学习笔记
tags:
  - CS336
  - PyTorch
  - AdamW
  - CrossEntropy
  - 训练闭环
draft: false
hidden: false
haloSync: false
haloPublished: false
---

作为 Stanford CS336 Assignment 1 系列的收官篇，我们已经拥有了分词器（`bpe.py`、`tokenizer.py`）和完整的语言模型架构（`transformer.py`）。要让模型在语料上真正训练并生成文本，还需要完成最后四大支柱：**数值稳定的交叉熵损失**、**AdamW 优化器与学习率/梯度控制**、**流式数据采样与 Checkpoint 序列化**，以及 **Temperature + Top-$p$ 自回归解码**。

本篇完成后，`liangqianxing/cs336-labs` 仓库通过了官方测试套件的全部测试。

[查看代码仓库](https://github.com/liangqianxing/cs336-labs/tree/b5c93ad0cc73178b2b2a62e518a0900da47a0fb7) · [下载本节代码与测试 ZIP](/materials/cs336/experiments/cs336-08-adamw-training-loop-20261027.zip)

## 数值稳定 Cross-Entropy：把 $\log$ 和 $\exp$ 消掉

给定未归一化 logits $o \in \mathbb{R}^{B \times V}$ 和真实类别下标 $y \in \{0, \dots, V-1\}^B$，单样本交叉熵损失定义为负对数似然：

$$
\ell(o, y) = -\log \frac{\exp(o_y)}{\sum_{a=1}^{V} \exp(o_a)}.
$$

如果先调用 `softmax(o)` 再取 `-torch.log(prob)`，当 logits 被放大到 `1000.0 * inputs`（对应官方测试 `test_cross_entropy` 的溢出测试分支）时，非最大类的概率 $\exp(o_y - m)$ 会直接下溢成浮点数 `0.0`，随后 `log(0.0)` 变成 `-inf`。

正确做法是先把分式拆开，在减去行最大值 $m = \max_a o_a$ 后直接化简为 **LogSumExp** 形式，把分子上的 $\log$ 与 $\exp$ 解析抵消：

$$
\ell(o, y)
= -\log \frac{\exp(o_y - m)}{\sum_{a=1}^{V} \exp(o_a - m)}
= \log \sum_{a=1}^{V} \exp(o_a - m) - (o_y - m).
$$

在 `cs336_basics/nn_utils.py` 中，用 `torch.gather` 取出目标类别对应的平移后 logit，只需 6 行即可同时支持任意前导 batch 维度与极端大数值输入：

```python
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = torch.amax(inputs, dim=-1, keepdim=True)
    shifted = inputs - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = torch.gather(shifted, dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()
```

语言模型的困惑度（Perplexity）则是序列平均交叉熵的指数：$\operatorname{PPL} = \exp\left(\frac{1}{m}\sum_{i=1}^m \ell_i\right)$。

## 全局 $\ell_2$ 梯度裁剪（Gradient Clipping）

训练过程中遇到异常样本可能产生幅度极高的梯度，导致参数更新一步飞出稳定区域。全局梯度裁剪不是按单个参数矩阵分别裁剪，而是先计算**所有可训练参数梯度的联合 $\ell_2$ 范数**：

$$
\|g\|_2 = \sqrt{\sum_{p \in \Theta,\; p.\text{grad} \neq \text{None}} \|p.\text{grad}\|_2^2}.
$$

当 $\|g\|_2 > M$ 时，将每个参数的梯度原地乘以缩放系数 $\frac{M}{\|g\|_2 + \varepsilon}$（其中 $\varepsilon = 10^{-6}$）：

```python
grads = [p.grad for p in parameters if p.grad is not None]
total_norm = torch.sqrt(sum(torch.sum(g.detach().to(torch.float32) ** 2) for g in grads))
if total_norm > max_l2_norm:
    scale = max_l2_norm / (total_norm + eps)
    for g in grads:
        g.detach().mul_(scale.to(g.dtype))
```

注意官方测试 `test_gradient_clipping` 会特意把最后一个参数设为 `requires_grad_(False)`，它的 `p.grad` 为 `None`，必须先过滤掉 `p.grad is not None` 才能汇总范数。

## 解耦权重衰减的 AdamW 优化器与训练显存核算

按照 Loshchilov & Hutter (2019) 与讲义算法 1，在第 $t = 1, 2, \dots, T$ 步，AdamW 对每个参数 $\theta$ 维护一阶动量 $m$ 和二阶动量 $v$：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \\
\alpha_t &= \alpha \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}, \\
\theta_t &= \theta_{t-1} - \alpha_t \frac{m_t}{\sqrt{v_t} + \varepsilon} - \alpha \lambda \theta_{t-1}.
\end{aligned}
$$

这里最关键的区别在于：**权重衰减项 $-\alpha \lambda \theta$ 直接使用原始学习率 $\alpha$ 与解耦的权重衰减率 $\lambda$，不混入梯度 $g_t$ 和二阶动量分母 $\sqrt{v_t}$**。状态字典 `self.state[p]` 中保存步数 `step`、一阶矩 `exp_avg` 和二阶矩 `exp_avg_sq`，从而天然支持 `optimizer.state_dict()` 序列化。

### AdamW 训练峰值显存核算（Problem `adamw_accounting`）

假设全部使用 FP32（每元素 4 字节），训练時の显存由四部分组成：
1. **模型参数（Parameters）**：$4 N_{\text{params}}$ 字节；
2. **参数梯度（Gradients）**：$4 N_{\text{params}}$ 字节；
3. **AdamW 优化器状态（Optimizer State，$m$ 与 $v$）**：$2 \times 4 N_{\text{params}} = 8 N_{\text{params}}$ 字节。

仅这三项静态状态就高达 $16 N_{\text{params}}$ 字节。对于上一篇核算的 GPT-2 XL（$N_{\text{params}} \approx 1.640 \times 10^9$），静态显存基数 $b = 16 \times 1.64045 \times 10^9 \approx 26.25\text{ GB}$。再加上每层前向激活值（RMSNorm、QKV、注意力分数矩阵 $B H S^2$、注意力权重、SwiGLU 中间激活等）随 `batch_size` 线性增长的斜率项 $a \cdot \text{batch\_size}$（在 $S=1024$ 下约为 $14.9\text{ GB} / \text{sample}$），在单张 80GB H100 GPU 上不启用激活重计算（Activation Checkpointing）或混合精度时，最大只能塞下 `batch_size = 3`。

## Warmup + 余弦退火学习率调度

`get_lr_cosine_schedule` 将训练步 $t$ 划分为三个阶段：

$$
\alpha_t =
\begin{cases}
\dfrac{t}{T_w} \alpha_{\max}, & t < T_w \quad\text{（线性预热）} \\[6pt]
\alpha_{\min} + \dfrac{1}{2}\left(1 + \cos\left(\dfrac{t - T_w}{T_c - T_w}\pi\right)\right)(\alpha_{\max} - \alpha_{\min}), & T_w \le t \le T_c \quad\text{（余弦退火）} \\[6pt]
\alpha_{\min}, & t > T_c \quad\text{（尾部保持）}
\end{cases}
$$

线性预热避免了训练最初几步二阶动量尚未稳定时过大的步长冲毁初始表示，随后的余弦退火让模型在后期平滑收敛到高曲率极小值区域。

## 随机窗口采样 `get_batch`、Checkpoint 与 Top-$p$ 解码

### 1. 向量化滑动窗口采样 `get_batch`

给定长度为 $N$ 的一维 token 数组 `dataset` 和上下文长度 `context_length` $m$，合法的起点下标范围是 $[0, N - m - 1]$（因为标签 `y` 需要向右偏移 1 位取到 `start + m`）：

```python
start_indices = np.random.randint(0, len(dataset) - context_length, size=(batch_size,))
offsets = np.arange(context_length)
window_indices = start_indices[:, None] + offsets[None, :]
x = torch.as_tensor(dataset[window_indices], dtype=torch.long).to(device)
y = torch.as_tensor(dataset[window_indices + 1], dtype=torch.long).to(device)
```

配合 `np.memmap(path, dtype=np.uint16, mode="r")`，即使面对数十 GB 的 OpenWebText 词元文件，操作系统也只会按页调入当前 batch 采样的切片，实现恒定内存开销的数据加载。

### 2. Checkpoint 保存与恢复

`save_checkpoint` 将 `model.state_dict()`、`optimizer.state_dict()` 和当前迭代步数 `iteration` 打包为字典写入磁盘或二进制流；`load_checkpoint` 恢复模型与优化器动量状态并返回 `iteration`，通过了官方 `test_checkpointing` 对所有权重与动量张量的精确比对。

### 3. Temperature 缩放与 Top-$p$（Nucleus）采样解码

在 `cs336_basics/decoding.py` 中，`generate` 接收前缀 prompt，每步取最后一个位置的 logits：
1. 先除以温度 $\tau$ 并做 Softmax 得到概率分布 $q$；
2. 将概率按从大到小排序得到 $\tilde{q}$，计算累积和 $\operatorname{cumsum}(\tilde{q})$，把进入当前 token 之前累积概率已经达到阈值 $p$ 的尾部低概率词置零（`cumulative_probs - sorted_probs >= top_p`），重新归一化后调用 `torch.multinomial` 采样。

## 全量测试套件与最小训练闭环验证

在仓库根目录执行 `bash scripts/check_completed.sh`，运行官方全部 7 个测试文件：

```text
tests/test_data.py::test_get_batch PASSED
tests/test_model.py (13 passed)
tests/test_nn_utils.py (3 passed)
tests/test_optimizer.py (2 passed)
tests/test_serialization.py::test_checkpointing PASSED
tests/test_tokenizer.py (23 passed, 2 skipped)
tests/test_train_bpe.py (3 passed)
======================== 46 passed, 2 skipped in 4.66s =========================
```

同时执行 `python -m cs336_basics.train` 完成从合成词元加载、Warmup 余弦学习率调度、前向/反向传播、梯度裁剪、AdamW 更新、验证集困惑度评估到 Top-$p$ 文本生成的完整端到端闭环验证。

*封面使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 辅助生成，属于概念插画；准确的公式、代码与测试结果以正文为准。*

## 参考资料

- [Loshchilov & Hutter, 2019: Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [Holtzman et al., 2020: The Curious Case of Neural Text Degeneration (Top-p Sampling)](https://arxiv.org/abs/1904.09751)
- [Stanford CS336 Assignment 1 官方仓库](https://github.com/stanford-cs336/assignment1-basics)
