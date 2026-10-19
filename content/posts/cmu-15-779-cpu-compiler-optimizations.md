---
title: 'CMU 15-779 学习记录 09：CPU 编译优化、合法重写与候选搜索'
date: '2026-10-19'
description: '用 CPU 实验验证 BatchNorm 折叠、RMSNorm 重排与非线性反例，测量分块候选，并运行 TorchInductor，分别记录首调用与预热后的时间。'
series: CMU 15-779
seriesOrder: 9
cover: /images/posts/cmu-15-779-cpu-compiler-optimizations/cmu-cpu-compiler-cover-20261019.webp
coverAlt: 淡紫计算积木经过薄荷绿折叠模具变成浅蓝积木，前方摆放四种候选托盘的浅色手绘插画
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs/tree/9261bea6aedf31cdccbbab2541ca693fc97b3743
attachmentUrl: /materials/cmu-15-779/experiments/cpu-compiler-optimizations-20261019.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - ML Compiler
  - TorchInductor
  - PyTorch
draft: false
hidden: false
haloSync: false
haloPublished: false
---

> 编译章节的个人配套实验，编号 30 不对应官方作业。图重写、分块候选和 TorchInductor 均在 CPU 上运行；真实 GPU 实验按学习安排跳过。

[上一节](/posts/cmu-15-779-attention-online-softmax/)记录了分块 Attention 和 KV cache。这次转到编译优化：先检查计算改写是否成立，再测量不同执行方式。源码和原始报告冻结于 2026 年 10 月 19 日，整理时从独立副本复跑。

[查看代码版本](https://github.com/liangqianxing/cmu-15-779-labs/tree/9261bea6aedf31cdccbbab2541ca693fc97b3743) · [下载代码与结果 ZIP](/materials/cmu-15-779/experiments/cpu-compiler-optimizations-20261019.zip)

## Linear 后的 BatchNorm 什么时候能折叠

PyTorch 的 Linear 使用 `Y = X @ W.T + b`。如果 BatchNorm 处于 eval 模式，使用固定的运行均值 μ、运行方差 v 和参数 γ、β，可以先计算：

$$
\begin{aligned}
s&=\frac{\gamma}{\sqrt{v+\epsilon}},\\
W'&=\operatorname{diag}(s)W,\\
b'&=s\odot(b-\mu)+\beta.
\end{aligned}
$$

随后只运行 `X @ W'.T + b'`。训练模式中的统计量随输入变化，不能直接套用这个固定参数折叠。脚本拒绝训练模式、未跟踪运行统计量和特征维度不匹配的输入。

原始 float64 例子的最大绝对误差为 `8.88e-16`。测试也覆盖了 Linear 无偏置、BatchNorm 无 affine 参数以及不同 batch size。报告里的模块操作数由 2 变为 1，只表示显式模块结构变化，没有统计底层 kernel 数量或证明运行时间下降。

## RMSNorm 的行标量可以跨过无偏置线性映射

脚本中的 RMSNorm 为每行计算一个逆均方根 r，再乘固定的 γ。对无偏置矩阵映射 W：

$$
\begin{aligned}
r_i&=\left(\frac{1}{d}\sum_j x_{ij}^2+\epsilon\right)^{-1/2},\\
W_\gamma&=\operatorname{diag}(\gamma)W,\\
((x_i r_i)\odot\gamma)W&=(x_iW_\gamma)r_i.
\end{aligned}
$$

因此可预先得到 `preweighted = gamma[:, None] * weight`，先做矩阵乘法，最后乘每行的 r。r 仍依赖当前输入，不能预先变成一个固定常量。这里也没有中间非线性或待处理的线性层偏置。

原始例子的最大绝对误差为 `1.33e-15`，零输入行、带负号或为零的 γ 也通过了测试。浮点计算顺序变化后，结果按容差比较。

并非相邻操作都能移动。脚本保留了一个具体反例：

```python
x = torch.tensor([[-1.0, 1.0]])
w = torch.tensor([[-1.0], [1.0]])
torch.relu(x @ w)       # 2.0
torch.relu(x) @ w       # 1.0
```

测试给出数值对照和规则成立的前提；这些有限输入的检查本身不是形式化等价证明。

## 搜索候选前，每个候选都要通过正确性检查

矩阵尺寸固定为 `A[97,65] @ B[65,81]`，float64、单 CPU 线程。候选分块尺寸为 16、32、64、128，按固定种子打乱为 32、16、128、64，再逐一与 `a @ b` 对照并计时。

原始 `30_cpu_compiler.json` 中的中位时间如下，单位为毫秒：

| 执行方式 | 中位时间 | 输出容差检查 |
| --- | ---: | --- |
| 分块 T=16 | 1.445975 | 通过 |
| 分块 T=32 | 0.288637 | 通过 |
| 分块 T=64 | 0.076659 | 通过 |
| 分块 T=128 | 0.017947 | 通过 |
| 直接 a @ b | 0.007439 | 基准 |

这份测量选择 T=128；直接调用 matmul 仍更快。T=128 已覆盖三个矩阵维度，Python 循环只需一次局部矩阵乘法，但仍包含输出分配、切片和累加。小块会增加这些开销，不能据此推断 GPU 共享内存分块应选最大的块。

候选使用普通 Python 循环调用 PyTorch CPU matmul，没有生成 Triton 或 TVM kernel。每个 Timer 的最短测量窗口为 0.04 秒，独立批次只有 1 至 3 个，JSON 同时保存 IQR 和每批调用次数。这个结果只适用于当时的形状、机器、线程与候选集合，不能作为稳定排名或全局最优的证据。

## TorchInductor 的首调用需要单独记录

实际编译的是这个 float32 函数，输入 `x[1024,64]` 和 `bias[64]`：

```python
def elementwise(x, bias):
    return torch.relu(x + bias) * 0.5

compiled = torch.compile(elementwise, backend="inductor", fullgraph=True)
```

原始运行使用 Python 3.11.15、PyTorch 2.14.1、arm64、单 CPU 线程，`inductor.status` 为 `executed`。第一次调用返回后，脚本检查结果；另换一组输入再次比较，两组均通过 `rtol=atol=1e-6` 的检查。

| 原始记录的阶段 | 时间 |
| --- | ---: |
| 编译路径首调用 | 8628.540 ms |
| eager 预热后中位数 | 0.021665 ms |
| compiled 预热后中位数 | 0.018640 ms |

首调用包含编译或编译缓存查询及执行；后两项只测预热后的调用。整理时使用单独的编译缓存目录复跑，首调用为 5471.626 ms，eager 与 compiled 的预热后中位数分别为 0.022108 和 0.017297 ms，正确性再次通过。编译成本和短窗口计时都会波动，这两次结果不足以证明普遍的加速比例。

本实验确认了 CPU TorchInductor 编译路径执行成功。脚本没有检查生成代码或统计 kernel，因此不额外声称某几项操作必然融合成了一个 kernel。

## 复现与课程位置

ZIP 包含冻结脚本、原始 JSON、5 项编译测试、30 项测试通过的日志和依赖说明。完整发布仓库中的 30 项测试也在本次冻结副本中全部通过，涵盖此前的矩阵、归约、Attention 和 KV cache。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 30_cpu_compiler.py --compile \
  --output results/compiler-rerun.json
.venv/bin/python -m unittest discover -s tests -v
```

运行 `--compile` 还需要 TorchInductor 可用的本机编译工具链。不加该选项时，脚本只运行重写与候选搜索，并把编译状态记为 `not_requested`，不能把这种输出当作已执行编译。

可对照原课的 [Tile-based DSL](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/08-ML-compilers-part-1.pdf)、[图级优化](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/10-graph-level-optimizations.pdf)和 [ML Superoptimization](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/11-kernel-superoptimization.pdf)材料。本篇提供几个可运行的配套例子，没有实现这些课程系统的完整优化器，也不表示论文阅读、GPU 调优或整门课程已经完成。

[本篇 GitHub 代码](https://github.com/liangqianxing/cmu-15-779-labs/tree/9261bea6aedf31cdccbbab2541ca693fc97b3743) · [下载代码与结果 ZIP](/materials/cmu-15-779/experiments/cpu-compiler-optimizations-20261019.zip)
