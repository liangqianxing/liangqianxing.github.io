---
title: 'CMU 15-779 学习记录 07：矩阵分块、数据复用与树形归约'
date: '2026-10-16'
description: '用 CPU 模型统计矩阵分块的输入读取次数，再比较串行求和和树形归约的依赖深度，记录补零、同步与浮点加法顺序的边界。'
series: CMU 15-779
seriesOrder: 7
cover: /images/posts/cmu-15-779-matmul-tiling-tree-reduction/cmu-matmul-reduction-cover-20261016.webp
coverAlt: 矩阵积木在共享托盘中复用，旁边的小珠沿树形管道逐层合并的浅色手绘插画
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs/tree/201587e7b00637567e969368d3085a6ce490154c
attachmentUrl: /materials/cmu-15-779/experiments/matmul-tiling-tree-reduction-20261016.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - CUDA
  - 矩阵乘法
  - 数据复用
  - 归约
draft: false
hidden: false
haloSync: false
haloPublished: false
---

> 对应 GPU Architecture & CUDA Programming 的个人配套实验。矩阵分块和归约均在 CPU 上模拟；CUDA 矩阵乘法源码尚未编译、运行，也没有 GPU 性能结果。

[上一节](/posts/cmu-15-779-cuda-thread-indexing/)解决线程如何找到自己的元素。这次沿着[原课 GPU/CUDA 幻灯片](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/03-CUDA-programming.pdf)的矩阵乘法与归约部分，观察两件事：多个输出怎样共用输入，以及求和怎样缩短依赖链。

对应脚本为 `26_matmul_tiling.py`、`27_tree_reduction.py`，原始结果保存于 2026 年 10 月 16 日。脚本编号是个人练习编号，不对应官方作业。

[查看代码版本](https://github.com/liangqianxing/cmu-15-779-labs/tree/201587e7b00637567e969368d3085a6ce490154c) · [下载代码与结果 ZIP](/materials/cmu-15-779/experiments/matmul-tiling-tree-reduction-20261016.zip)

## 相邻输出会反复读取同一份输入

矩阵乘法中，一个输出元素是：

$$
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}.
$$

如果每个输出独立从输入矩阵取数，计算 `C[i, j]` 和 `C[i, j+1]` 时，都会读同一行 `A[i, :]`。共享内存分块先把一小块 A 和 B 加载进来，再让同一 block 中的多个输出使用它们。

CPU 模型把这个过程分成加载、计算、写回三个阶段，并在实际执行的循环中累计读取次数。它统计的是**算法层面的 A/B 标量读取**，不包含 C 写回、缓存命中、合并访存或真实 DRAM 事务。

对两个 `8×8` 矩阵，原始结果如下：

| 计算方式 | 输入读取 | 写入 | 乘加 |
| --- | ---: | ---: | ---: |
| 朴素方式 | 1024 | 64 | 512 |
| 分块 T=1 | 1024 | 64 | 512 |
| 分块 T=2 | 512 | 64 | 512 |
| 分块 T=4 | 256 | 64 | 512 |
| 分块 T=8 | 128 | 64 | 512 |

各配置的输出与 PyTorch 对照完全一致，每个有效输出也只写入一次。这里的输入采用可精确表示的二进制分数，不能把这次完全一致推广成所有浮点矩阵都逐位相同。

当方阵边长 N 能被 T 整除时，输出块有 `(N/T)²` 个，每个输出块沿 K 方向加载 `N/T` 轮，每轮读取两块 `T²` 个元素。因此：

$$
\text{输入读取数}=\left(\frac{N}{T}\right)^2\frac{N}{T}\,2T^2=\frac{2N^3}{T}.
$$

这解释了表里的读取变化。它没有给出 GPU 加速比：同步、缓存、共享内存用量和占用率还会影响真实运行时间。

## 非整除尺寸不能只套方阵公式

另一组原始结果使用 `A[3,5] @ B[5,7]`，T=4。输出是 `3×7`，但模型启动两个 `4×4` 输出块，最后的 K 分块也不完整。

模型对越界加载补零，只写有效输出，实测计数为：

| 指标 | 朴素方式 | 分块方式 |
| --- | ---: | ---: |
| A/B 输入读取 | 210 | 65 |
| 输出写入 | 21 | 21 |
| 执行的乘加 | 105 | 256 |

分块模型的 256 次乘加包含补零位置和越界输出线程所做的计算；有效结果需要的乘加仍是 105 次。模型另外记录了 63 次共享槽位补零。读取减少的同时，边缘块可能引入额外计算。

附带的 CUDA 文件使用相同的阶段划分：所有线程协作加载，`__syncthreads()` 后使用共享块；用完再同步，下一轮才能覆盖它。边界线程不能因为不写 C 就提前离开整个同步流程。[NVIDIA 的共享内存说明](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)可作为对照。

CPU 循环按阶段执行，验证不了真实 GPU 的竞争和同步安全。`cuda/26_matmul_tiling.cu` 仍保留为未编译、未执行的参考源码。

## 树形归约缩短的是依赖链

接下来把问题从数据复用换成求和。`27_tree_reduction.py` 保存了这棵小树：

```text
3  1  7  0  4  1  6  3
 4     7     5     9
    11          14
          25
```

8 个数一共做 7 次加法。串行累加需要依次等待前一个和；树形结构中，同层的独立加法可以并行，依赖深度只有 3。

默认的 257 个数得到：

| 指标 | 串行结构 | 树形结构 |
| --- | ---: | ---: |
| 加法总数 | 256 | 256 |
| 依赖深度 | 256 | 9 |
| 最终和 | -3.75 | -3.75 |

脚本还按每组 32 个输入分阶段求和：先得到 9 个局部和，再把 9 个局部和合成 1 个结果，总加法数仍为 256。这个结构可以帮助理解分阶段归约，但本次普通 CPU 循环没有并行执行，依赖深度不是计时结果。

## 加法顺序会改变浮点结果

同一份 JSON 还保留了一个不能忽略的反例：

```text
输入：[1e16, 1.0, -1e16, 1.0]
串行求和：1.0
树形求和：0.0
math.fsum：2.0
```

有限精度下，大数和小数相加会丢失信息。改变加法顺序，结果可能不同；归约正确性检查需要结合数据类型、数值尺度和容差，而不能要求任意浮点输入都逐位等于串行求和。

## 复现与证据

整理时从冻结副本复跑了两份矩阵报告和归约报告，数值与计数均复现。矩阵测试有 7 项，覆盖 500 组小尺寸配置、随机输入、补零、计数与非法参数；归约测试还检查了多种长度、块大小及上述浮点反例。完整代码仓库中的矩阵、归约、Attention 测试共 21 项，整理时全部通过。

ZIP 包含本篇两份脚本、CUDA 矩阵源码、三份原始 JSON、原始矩阵测试输出、矩阵测试和依赖说明。复跑结果另存：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 26_matmul_tiling.py --output results/matmul-rerun.json
.venv/bin/python 26_matmul_tiling.py --m 3 --k 5 --n 7 --tile 4 \
  --output results/rectangular-rerun.json
.venv/bin/python 27_tree_reduction.py --output results/reduction-rerun.json
.venv/bin/python -m unittest discover -s tests -v
```

本篇完成的是数据复用、计数和归约依赖结构的 CPU 验证。GPU 时间、实际显存流量，以及 CUDA 内存和同步检查均未测量；原课寄存器分块也没有新增独立实验结果。

[本篇 GitHub 代码](https://github.com/liangqianxing/cmu-15-779-labs/tree/201587e7b00637567e969368d3085a6ce490154c) · [下载代码与结果 ZIP](/materials/cmu-15-779/experiments/matmul-tiling-tree-reduction-20261016.zip)
