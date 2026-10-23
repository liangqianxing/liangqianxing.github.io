---
title: 'CMU 15-779 学习记录 10：加权梯度、状态分片与同步流水线'
date: '2026-10-04'
description: '用双 CPU 进程 Gloo 检查不等数据分片的梯度，推导持久状态字节数，再用依赖模型比较 GPipe 与同步 1F1B 的激活存活。'
series: CMU 15-779
seriesOrder: 10
cover: /images/posts/cmu-15-779-cpu-parallelism-pipeline/cmu-parallel-cover-20261019.webp
coverAlt: 浅色手绘的四阶段桌面流水线与按数量汇总的托盘
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs/tree/c4851e0cedadf13af7c421867e83c8ff57244fa0
attachmentUrl: /materials/cmu-15-779/experiments/cpu-parallel-serving-lora-moe-20261019.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - Parallel Training
  - Gloo
  - Pipeline
draft: false
hidden: false
haloSync: false
haloPublished: false
---

> 第 12 至 14 讲的个人配套实验，脚本编号 31、32 不对应官方作业。Gloo 实际运行两个 CPU 进程；张量切分、状态字节数和流水线分别按下文的数学或成本模型解释。真实 GPU 实测按学习安排跳过。

[上一节](/posts/cmu-15-779-cpu-compiler-optimizations/)检查单个计算图的改写。这次把问题放到多个分片：梯度怎样汇总，状态怎样保存，前向和反向怎样排队。源码与原始结果保存在实验快照中，整理时从独立副本复跑，完整仓库的 52 项测试通过。

[查看代码版本](https://github.com/liangqianxing/cmu-15-779-labs/tree/c4851e0cedadf13af7c421867e83c8ff57244fa0) · [下载 31 至 37 号代码与结果 ZIP](/materials/cmu-15-779/experiments/cpu-parallel-serving-lora-moe-20261019.zip)

## 两个局部均值不能总是各占一半

31 号使用 float64 的 `Linear(3, 2)`，7 个样本分成 4 和 3 个。全批次参考使用 MSE 的元素均值；两个分片的局部 loss 也分别取元素均值。由于每个样本都有 2 个输出，全局梯度应为：

$$
g=\frac{4}{7}g_0+\frac{3}{7}g_1.
$$

如果直接计算 `(g0 + g1) / 2`，3 个样本的分片会获得过大的权重。本例的最大绝对梯度误差为 `0.0711947378`；按样本数加权后，误差为 `1.11e-16`。

实际 Gloo 路径在同一台机器启动两个进程，分别计算局部梯度，再执行手工 SUM：

```python
local_loss = nn.functional.mse_loss(model(shard_x), shard_y,
                                    reduction="mean")
(local_loss * len(shard_x) / len(x)).backward()
dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
```

两个 rank 的聚合梯度都通过对照；各自执行学习率 `0.1` 的一步 SGD 后，参数也与全批次更新一致。原始报告和整理时的复跑均记为 `gloo.status: executed`。

这里没有使用 DDP 的默认梯度平均。若换成 DDP，需要把它自身的 world-size 除法计入推导。若 loss 按有效 token 平均，权重也要对应有效 token 数；先明确全局 loss 的分母，再写聚合代码。

## 张量切分和状态分片处理不同问题

在 `A[7,5] @ B[5,9]` 的串行 CPU 对照中，按 B 的输出列切分后拼接，最大误差为 `2.22e-16`；沿归约维同时切分 A、B，再相加局部乘积，误差为 `8.88e-16`。这验证矩阵关系，没有测量并行通信速度。

持久状态模型则假设每参数占用 2B 权重、2B 梯度，以及合计 12B 的主权重和 Adam 两个矩。令参数量为 P、rank 数为 D，等长填充分片大小为 S：

$$
\begin{aligned}
S&=\lceil P/D\rceil,\\
M_{\mathrm{replicated}}&=16P,\\
M_{\mathrm{ZeRO1}}&=4P+12S,\\
M_{\mathrm{ZeRO2}}&=2P+14S,\\
M_{\mathrm{ZeRO3}}&=16S.
\end{aligned}
$$

P 为 100 万、D 为 4 时，每个 rank 的模型字节数如下，MB 使用十进制：

| 保存方式 | 每 rank 持久状态 |
| --- | ---: |
| 完整复制 | 16 MB |
| ZeRO-1 模型 | 7 MB |
| ZeRO-2 模型 | 5.5 MB |
| ZeRO-3 模型 | 4 MB |

这些数字排除了 activation、临时参数 gather、通信缓冲、分配器和 offload。脚本没有执行 ZeRO 或 FSDP，因此表格不能当作真实进程的峰值内存。

## 同步 1F1B 改变了激活需要保存多久

32 号为每个 `(stage, F/B, microbatch)` 建立依赖：上游前向、下游反向、同阶段的前一个任务都可能限制开始时间；每个阶段同一时刻只能运行一个任务。

模型使用 4 个阶段、8 个微批次，每阶段前向成本为 1、反向为 2。通信成本为零，所有任务使用同一参数版本，完整批次结束后才允许更新。

| 策略 | 总时长，模拟单位 | 各阶段存活激活峰值，份 |
| --- | ---: | --- |
| GPipe，先全部 F 再 B | 33 | `[8,8,8,8]` |
| 非交错同步 1F1B | 33 | `[4,3,2,1]` |

这里每份激活从前向完成一直保存到对应反向完成，假设各份大小相同，未计反向临时内存。1F1B 让较晚阶段更早消耗激活，因此需要同时保留的份数减少；这次的总时长没有减少。

均匀 GPipe 的总时长为 `(微批次数 + 阶段数 - 1) × (F + B)`，代入即为 33。所有时间来自给定成本，并非毫秒或硬件测量。同步 1F1B 也没有复现原始 PipeDream 的异步更新和 weight stashing，不能仅凭名称混用结论。

## 阶段划分依赖成本

给定 8 层成本 `[1,2,1,4,1,3,1,2]`，脚本穷举 21 种三段连续划分，以最大阶段成本最小为目标。选中的切分位置为 `[2,5]`，即 `[1,2]`、`[1,4,1]`、`[3,1,2]`，阶段成本为 `[3,6,6]`。

这个搜索空间没有通信、设备拓扑、重计算或内存限制。它提供一个可检查的小例子，没有运行 FlexFlow 或 Alpa。现有 8 项相关测试检查了分片梯度、张量边界、状态公式、流水线依赖、阶段互斥及划分覆盖。

## 复现与课程材料

ZIP 包含 31 至 37 号冻结脚本、原始报告及这批实验的 22 项测试。只复跑本篇时：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 31_cpu_parallelism.py --gloo --output results/31-rerun.json
.venv/bin/python 32_pipeline_schedule.py --output results/32-rerun.json
.venv/bin/python -m unittest discover -s tests -p 'test_cpu_parallelism.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_pipeline_schedule.py' -v
```

`--gloo` 需要 PyTorch 构建支持 Gloo；不加时仅执行本地对照并记录 `not_requested`。可对照原课的[并行训练一](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/12-ML-parallelization-part1.pdf)、[并行训练二](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/13-ML-parallelization-part2.pdf)和[自动并行化](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/14-auto-parallelization.pdf)。本篇的运行与测试不代替论文阅读确认或课程项目验收。
