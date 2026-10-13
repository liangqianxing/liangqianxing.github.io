# TVM：面向深度学习的自动化端到端优化编译器
## OSDI 2018 幻灯片中文翻译

- **来源文件：** `osdi18_slides_chen.pdf`
- **页数：** 110 页
- **说明：** 本译稿按 PDF 页码逐页对应。Keynote 动画导出会把一张幻灯片拆成多个页面；对于仅新增少量元素的构建页，译文明确写出新增内容，同时保留该页可见的全部标签、公式、代码和图表说明。
- **术语约定：** TVM、VTA、CUDA、LLVM、cuDNN、Halide、Loopy、GEMM、JIT 等专有名词保留英文；首次出现时给出中文释义。
- **图表说明：** 源 PDF 中的柱线、流程框和加速器示意图属于绘图对象，译稿翻译其可提取文字并描述含义；具体柱高和布局请结合原 PDF 对照。

## 第 1 页｜TVM：面向深度学习的自动化端到端优化编译器

![TVM 原始幻灯片第 1 页：TVM：面向深度学习的自动化端到端优化编译器](/images/materials/cmu-15-779/tvm-slides/page-001.webp)

**作者：** Tianqi Chen、Thierry Moreau、Ziheng Jiang、Lianmin Zheng、Eddie Yan、Meghan Cowan、Haichen Shen、Leyuan Wang、Yuwei Hu、Luis Ceze、Carlos Guestrin、Arvind Krishnamurthy。

本页是 OSDI 2018 报告的标题页。

## 第 2 页｜合作人员

![TVM 原始幻灯片第 2 页：合作人员](/images/materials/cmu-15-779/tvm-slides/page-002.webp)

合作实验室：**SAMPL Lab**（<https://sampl.cs.washington.edu>）。

## 第 3 页｜故事的开端

![TVM 原始幻灯片第 3 页：故事的开端](/images/materials/cmu-15-779/tvm-slides/page-003.webp)

本页仅显示章节标题“故事的开端”，引出专用深度学习加速器与编译器之间的关系。

## 第 4 页｜故事的开端：加速器

![TVM 原始幻灯片第 4 页：故事的开端：加速器](/images/materials/cmu-15-779/tvm-slides/page-004.webp)

页面中央标出“加速器（Accelerator）”，表示从专用硬件的演进开始叙述。

## 第 5 页｜故事的开端：加速器

![TVM 原始幻灯片第 5 页：故事的开端：加速器](/images/materials/cmu-15-779/tvm-slides/page-005.webp)

动画构建页，继续突出“加速器（Accelerator）”这一主题。

## 第 6 页｜故事的开端：VIA 加速器微代码

![TVM 原始幻灯片第 6 页：故事的开端：VIA 加速器微代码](/images/materials/cmu-15-779/tvm-slides/page-006.webp)

本页展示 VIA 加速器的多线程分块卷积微代码，以及最大池化、批归一化和激活函数的访问模式。寄存器索引由二维仿射函数计算。

**(a) 多线程上下文的分块卷积程序（伪代码）**

```text
// 虚拟线程 0
0x00: LOAD(PARAM[ 0-71])          // LD@TID0
0x01: LOAD(ACTIV[ 0-24])          // LD@TID0
0x02: LOAD(LDBUF[ 0-31])           // LD@TID0
0x03: PUSH(LD->EX)                 // LD@TID0
0x04: POP (LD->EX)                 // EX@TID0
0x05: EXE (ACTIV[0-24], PARAM[0-71], LDBUF[0-31], STBUF[0-7]) // EX@TID0
0x06: PUSH(EX->LD)                 // EX@TID0
0x07: PUSH(EX->ST)                 // EX@TID0
0x08: POP (EX->ST)                  // ST@TID0
0x09: STOR(STBUF[0-7])              // ST@TID0
0x0A: PUSH(ST->EX)                  // ST@TID0

// 虚拟线程 1
0x0B: LOAD(ACTIV[25-50])            // LD@TID1
0x0C: LOAD(LDBUF[32-63])            // LD@TID1
0x0D: PUSH(LD->EX)                  // LD@TID1
0x0E: POP (LD->EX)                  // EX@TID1
0x0F: EXE (ACTIV[25-50], PARAM[0-71], LDBUF[32-63], STBUF[32-39]) // EX@TID1
0x10: PUSH(EX->LD)                  // EX@TID1
0x11: PUSH(EX->ST)                  // EX@TID1
0x12: POP (EX->ST)                  // ST@TID1
0x13: STOR(STBUF[32-39])             // ST@TID1
0x14: PUSH(ST->EX)                  // ST@TID1

// 虚拟线程 2
0x15: POP (EX->LD)                  // LD@TID2
0x16: LOAD(PARAM[0-71])              // LD@TID2
0x17: LOAD(ACTIV[0-24])              // LD@TID2
0x18: LOAD(LDBUF[0-31])              // LD@TID2
0x19: PUSH(LD->EX)                  // LD@TID2
0x1A: POP (LD->EX)                  // EX@TID2
0x1B: POP (ST->EX)                  // EX@TID2
0x1C: EXE (ACTIV[0-24], PARAM[0-71], LDBUF[0-31], STBUF[0-7]) // EX@TID2
0x1D: PUSH(EX->ST)                  // EX@TID2
0x1E: POP (EX->ST)                  // ST@TID2
0x1F: STOR(STBUF[0-7])               // ST@TID2

// 虚拟线程 3
0x20: POP (EX->LD)                  // LD@TID3
0x21: LOAD(ACTIV[25-50])             // LD@TID3
0x22: LOAD(LDBUF[32-63])             // LD@TID3
0x23: PUSH(LD->EX)                   // LD@TID3
0x24: POP (LD->EX)                   // EX@TID3
0x25: POP (ST->EX)                   // EX@TID2
0x26: EXE (ACTIV[25-50], PARAM[0-71], LDBUF[32-63], STBUF[32-39]) // EX@TID3
0x27: PUSH(EX->ST)                   // EX@TID3
0x28: POP (EX->ST)                   // ST@TID3
0x29: STOR(STBUF[32-39])             // ST@TID3
```

**(b) 卷积访问模式：** 对 `y∈[0,i)`、`x∈[0,j)`，例如 `idxrf = a_rf*y + b_rf*x + c_rf0`；其中 `c_rf0` 由微操作字段给出，并执行 `rf[idxrf] += GEVM(act[idxact], par[idxpar])`。

**(c) 最大池化、批归一化和激活：**

```text
// 最大池化
rf[idxdst0] = MAX(rf[idxdst0], rf[idxsrc0])
rf[idxdst1] = MAX(rf[idxdst1], rf[idxsrc1])

// 批归一化（示意）
rf[idxdstm]   = MUL(rf[idxdstm],   rf[idxsrcm])
rf[idxdstm+1] = ADD(rf[idxdstm+1], rf[idxsrcm+1])
rf[idxdstm+2] = MUL(rf[idxdstm+2], rf[idxsrcm+2])
rf[idxdstm+3] = ADD(rf[idxdstm+3], rf[idxsrcm+3])

// 激活
rf[idxdstn-1] = RELU(rf[idxdstn-1], rf[idxsrcn-1])
rf[idxdstn]   = RELU(rf[idxdstn],   rf[idxsrcn])
```

## 第 7 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 7 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-007.webp)

**框架（Frameworks）**。本页开始讨论将深度学习部署到各种设备的目标。

## 第 8 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 8 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-008.webp)

**模型和框架的爆炸式增长（Explosion of models and frameworks）。** 随着网络结构和软件框架不断增加，手工适配成本迅速上升。

## 第 9 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 9 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-009.webp)

在模型/框架爆炸之外，又出现了**硬件后端的爆炸式增长（Explosion of hardware backends）**。

## 第 10 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 10 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-010.webp)

动画构建页：继续显示“模型和框架的爆炸式增长”与“硬件后端的爆炸式增长”。图中逐步增加不同后端/加速器，强调组合数量持续增长。

## 第 11 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 11 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-011.webp)

动画构建页：继续显示“模型和框架的爆炸式增长”与“硬件后端的爆炸式增长”。图中逐步增加不同后端/加速器，强调组合数量持续增长。

## 第 12 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 12 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-012.webp)

动画构建页：继续显示“模型和框架的爆炸式增长”与“硬件后端的爆炸式增长”。图中逐步增加不同后端/加速器，强调组合数量持续增长。

## 第 13 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 13 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-013.webp)

动画构建页：继续显示“模型和框架的爆炸式增长”与“硬件后端的爆炸式增长”。图中逐步增加不同后端/加速器，强调组合数量持续增长。

## 第 14 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 14 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-014.webp)

动画构建页：继续显示“模型和框架的爆炸式增长”与“硬件后端的爆炸式增长”。图中逐步增加不同后端/加速器，强调组合数量持续增长。

## 第 15 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 15 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-015.webp)

动画构建页：继续显示“模型和框架的爆炸式增长”与“硬件后端的爆炸式增长”。图中逐步增加不同后端/加速器，强调组合数量持续增长。

## 第 16 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 16 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-016.webp)

动画构建页：继续显示“模型和框架的爆炸式增长”与“硬件后端的爆炸式增长”。图中逐步增加不同后端/加速器，强调组合数量持续增长。

## 第 17 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 17 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-017.webp)

页面同时列出：

- 模型和框架的爆炸式增长；
- 硬件后端的爆炸式增长；
- 加速器（Accelerator）。

## 第 18 页｜目标：让深度学习无处不在

![TVM 原始幻灯片第 18 页：目标：让深度学习无处不在](/images/materials/cmu-15-779/tvm-slides/page-018.webp)

页面明确指出：模型/框架与硬件后端之间存在**巨大鸿沟（Huge gap between model/frameworks and hardware backends）**。

## 第 19 页｜现有方法

![TVM 原始幻灯片第 19 页：现有方法](/images/materials/cmu-15-779/tvm-slides/page-019.webp)

框架（Frameworks）与硬件（Hardware）之间需要衔接。

## 第 20 页｜现有方法：高层数据流图

![TVM 原始幻灯片第 20 页：现有方法：高层数据流图](/images/materials/cmu-15-779/tvm-slides/page-020.webp)

框架生成**高层数据流图（High-level data flow graph）**，再交给硬件侧执行。

## 第 21 页｜现有方法：原语张量算子

![TVM 原始幻灯片第 21 页：现有方法：原语张量算子](/images/materials/cmu-15-779/tvm-slides/page-021.webp)

高层图由诸如 `Conv2D` 的**原语张量算子（Primitive Tensor operators）**组成。

## 第 22 页｜现有方法：调用优化算子库

![TVM 原始幻灯片第 22 页：现有方法：调用优化算子库](/images/materials/cmu-15-779/tvm-slides/page-022.webp)

示例：`cuDNN`。将算子卸载（offload）到高度优化的 DNN 算子库。

这种方法依赖为每个硬件平台预先手工实现和调优算子。

## 第 23 页｜现有方法：工程化优化张量算子

![TVM 原始幻灯片第 23 页：现有方法：工程化优化张量算子](/images/materials/cmu-15-779/tvm-slides/page-023.webp)

章节标题：通过工程手段为张量算子编写优化实现。

## 第 24 页｜现有方法：工程化优化张量算子

![TVM 原始幻灯片第 24 页：现有方法：工程化优化张量算子](/images/materials/cmu-15-779/tvm-slides/page-024.webp)

动画构建页，继续展示“工程化优化张量算子”的流程。

## 第 25 页｜矩阵乘：算子规格

![TVM 原始幻灯片第 25 页：矩阵乘：算子规格](/images/materials/cmu-15-779/tvm-slides/page-025.webp)

**Matmul：算子规格（Operator Specification）**。

```python
C = tvm.compute((m, n),
    lambda y, x: tvm.sum(A[k, y] * B[k, x], axis=k))
```

该声明描述输出张量 `C[y,x]` 沿归约轴 `k` 对 `A[k,y]` 与 `B[k,x]` 求和。

## 第 26 页｜矩阵乘：算子规格

![TVM 原始幻灯片第 26 页：矩阵乘：算子规格](/images/materials/cmu-15-779/tvm-slides/page-026.webp)

动画构建页：重复显示矩阵乘规格，强调规格与具体实现策略相互独立。

```python
C = tvm.compute((m, n),
    lambda y, x: tvm.sum(A[k, y] * B[k, x], axis=k))
```

## 第 27 页｜矩阵乘：朴素代码

![TVM 原始幻灯片第 27 页：矩阵乘：朴素代码](/images/materials/cmu-15-779/tvm-slides/page-027.webp)

**朴素实现（Vanilla Code）：**

```python
for y in range(1024):
  for x in range(1024):
    C[y][x] = 0
    for k in range(1024):
      C[y][x] += A[k][y] * B[k][x]
```

这是规格的直接三重循环实现，没有利用局部性或专用硬件。

## 第 28 页｜矩阵乘：循环分块提升局部性

![TVM 原始幻灯片第 28 页：矩阵乘：循环分块提升局部性](/images/materials/cmu-15-779/tvm-slides/page-028.webp)

**循环分块（Loop Tiling for Locality）：**

```python
for yo in range(128):
  for xo in range(128):
    C[yo*8:yo*8+8][xo*8:xo*8+8] = 0
    for ko in range(128):
      for yi in range(8):
        for xi in range(8):
          for ki in range(8):
            C[yo*8+yi][xo*8+xi] += \
              A[ko*8+ki][yo*8+yi] * B[ko*8+ki][xo*8+xi]
```

把 1024×1024 的循环切成 8×8 tile，使数据可在缓存/片上存储中复用。

## 第 29 页｜矩阵乘：映射到加速器

![TVM 原始幻灯片第 29 页：矩阵乘：映射到加速器](/images/materials/cmu-15-779/tvm-slides/page-029.webp)

**映射到加速器（Map to Accelerators）：**

```python
inp_buffer AL[8][8], BL[8][8]
acc_buffer CL[8][8]
for yo in range(128):
  for xo in range(128):
    vdla.fill_zero(CL)
    for ko in range(128):
      vdla.dma_copy2d(AL, A[ko*8:ko*8+8][yo*8:yo*8+8])
      vdla.dma_copy2d(BL, B[ko*8:ko*8+8][xo*8:xo*8+8])
      vdla.fused_gemm8x8_add(CL, AL, BL)
    vdla.dma_copy2d(C[yo*8:yo*8+8, xo*8:xo*8+8], CL)
```

这里使用 DMA 复制、8×8 融合 GEMM 和累加缓冲区。页脚指出：优化代码需要**人工探索（Human exploration of optimized code）**。

## 第 30 页｜现有方法的局限

![TVM 原始幻灯片第 30 页：现有方法的局限](/images/materials/cmu-15-779/tvm-slides/page-030.webp)

框架侧调用 `cuDNN` 等库。页面开始展示现有方法的问题。

## 第 31 页｜现有方法的局限

![TVM 原始幻灯片第 31 页：现有方法的局限](/images/materials/cmu-15-779/tvm-slides/page-031.webp)

动画构建页（第 31 页）：继续显示框架到 `cuDNN` 的调用路径，说明现有方案依赖预先工程化的库算子。

## 第 32 页｜现有方法的局限

![TVM 原始幻灯片第 32 页：现有方法的局限](/images/materials/cmu-15-779/tvm-slides/page-032.webp)

动画构建页（第 32 页）：继续显示框架到 `cuDNN` 的调用路径，说明现有方案依赖预先工程化的库算子。

## 第 33 页｜现有方法的局限

![TVM 原始幻灯片第 33 页：现有方法的局限](/images/materials/cmu-15-779/tvm-slides/page-033.webp)

动画构建页（第 33 页）：继续显示框架到 `cuDNN` 的调用路径，说明现有方案依赖预先工程化的库算子。

## 第 34 页｜现有方法的局限

![TVM 原始幻灯片第 34 页：现有方法的局限](/images/materials/cmu-15-779/tvm-slides/page-034.webp)

动画构建页（第 34 页）：继续显示框架到 `cuDNN` 的调用路径，说明现有方案依赖预先工程化的库算子。

## 第 35 页｜现有方法的局限

![TVM 原始幻灯片第 35 页：现有方法的局限](/images/materials/cmu-15-779/tvm-slides/page-035.webp)

新增说明：**算子融合优化引入了新算子（New operator introduced by operator fusion optimization）**，理论上可能带来 **1.5 倍加速（potentially benefit: 1.5x speedup）**。

## 第 36 页｜现有方法的局限

![TVM 原始幻灯片第 36 页：现有方法的局限](/images/materials/cmu-15-779/tvm-slides/page-036.webp)

新增说明：**算子融合优化引入了新算子（New operator introduced by operator fusion optimization）**，理论上可能带来 **1.5 倍加速（potentially benefit: 1.5x speedup）**。

## 第 37 页｜现有方法的局限

![TVM 原始幻灯片第 37 页：现有方法的局限](/images/materials/cmu-15-779/tvm-slides/page-037.webp)

新增说明：**算子融合优化引入了新算子（New operator introduced by operator fusion optimization）**，理论上可能带来 **1.5 倍加速（potentially benefit: 1.5x speedup）**。

## 第 38 页｜现有方法的局限

![TVM 原始幻灯片第 38 页：现有方法的局限](/images/materials/cmu-15-779/tvm-slides/page-038.webp)

算子融合可能获得 1.5 倍加速，但页面强调：**工程工作量很大（Engineering intensive）**。每种模型、形状、硬件组合都可能需要新的手工实现。

## 第 39 页｜基于学习的学习系统

![TVM 原始幻灯片第 39 页：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-039.webp)

框架提供**高层数据流图和优化（High-level data flow graph and optimizations）**，硬件执行优化后的程序。

## 第 40 页｜基于学习的学习系统

![TVM 原始幻灯片第 40 页：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-040.webp)

动画构建页：继续显示高层图优化与硬件之间的连接。

## 第 41 页｜基于学习的学习系统

![TVM 原始幻灯片第 41 页：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-041.webp)

加入**面向硬件的优化张量程序搜索空间（Hardware-aware Search Space of Optimized Tensor Programs）**。

## 第 42 页｜基于学习的学习系统

![TVM 原始幻灯片第 42 页：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-042.webp)

加入**基于机器学习的程序优化器（Machine Learning based Program Optimizer）**。

## 第 43 页｜基于学习的学习系统

![TVM 原始幻灯片第 43 页：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-043.webp)

学习型优化器可以针对新的算子工作负载和新硬件，直接生成优化程序（directly generate optimized program for new operator workloads and hardware）。

## 第 44 页｜基于学习的学习系统

![TVM 原始幻灯片第 44 页：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-044.webp)

动画构建页：完整流程为“框架 → 高层数据流图与优化 → 面向硬件的搜索空间 → 机器学习程序优化器 → 硬件”。

## 第 45 页｜基于学习的学习系统

![TVM 原始幻灯片第 45 页：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-045.webp)

与第 44 页相同的完整流程，作为章节过渡页。

## 第 46 页｜面向硬件的搜索空间：张量表达式语言

![TVM 原始幻灯片第 46 页：面向硬件的搜索空间：张量表达式语言](/images/materials/cmu-15-779/tvm-slides/page-046.webp)

**张量表达式语言（Tensor Expression Language，规格层）**描述矩阵乘：

```python
C = tvm.compute((m, n),
    lambda y, x: tvm.sum(A[k, y] * B[k, x], axis=k))
```

下方连接到硬件，表示表达式最终要映射到具体硬件程序。

## 第 47 页｜面向硬件的搜索空间：从表达式到硬件

![TVM 原始幻灯片第 47 页：面向硬件的搜索空间：从表达式到硬件](/images/materials/cmu-15-779/tvm-slides/page-047.webp)

定义从表达式到硬件程序的**面向硬件映射搜索空间**。设计基于 Halide 的“计算/调度分离”（compute/schedule separation）：计算定义语义，调度选择实现方式。

## 第 48 页｜面向硬件的搜索空间：CPU

![TVM 原始幻灯片第 48 页：面向硬件的搜索空间：CPU](/images/materials/cmu-15-779/tvm-slides/page-048.webp)

**CPU 的计算原语与内存层次：**

- 计算原语：标量（scalar）、向量（vector）；
- 内存子系统：L1D、L1I、L2、L3，通常由硬件隐式管理（implicitly managed）；
- 调度原语：循环变换（Loop Transformations）、缓存局部性（Cache Locality）、向量化（Vectorization）。

这些原语复用了 Halide、Loopy 等先前工作的思想。

## 第 49 页｜支持多样硬件后端的挑战

![TVM 原始幻灯片第 49 页：支持多样硬件后端的挑战](/images/materials/cmu-15-779/tvm-slides/page-049.webp)

需要同时支持 **CPU、GPU 和类似 TPU 的专用加速器**。三类设备的计算原语、内存层次和管理方式差异很大。

## 第 50 页｜面向硬件的搜索空间：GPU

![TVM 原始幻灯片第 50 页：面向硬件的搜索空间：GPU](/images/materials/cmu-15-779/tvm-slides/page-050.webp)

**GPU 的计算原语与内存层次：**

- 设备级 L2；
- 多个 SM（流式多处理器）；
- TX/L1（纹理/一级缓存）；
- RF（寄存器文件）；
- 计算同时包含标量、向量和混合模式；
- 片上资源在一定程度上由程序显式管理。

## 第 51 页｜面向硬件的搜索空间：GPU

![TVM 原始幻灯片第 51 页：面向硬件的搜索空间：GPU](/images/materials/cmu-15-779/tvm-slides/page-051.webp)

新增标注：计算核心之间可共享共享内存（Shared memory among compute cores）。

## 第 52 页｜面向硬件的搜索空间：GPU

![TVM 原始幻灯片第 52 页：面向硬件的搜索空间：GPU](/images/materials/cmu-15-779/tvm-slides/page-052.webp)

新增调度原语：使用共享内存（Use of Shared Memory）和线程协作（Thread Cooperation），以实现片上数据复用和协同计算。

## 第 53 页｜面向硬件的搜索空间：类似 TPU 的专用加速器

![TVM 原始幻灯片第 53 页：面向硬件的搜索空间：类似 TPU 的专用加速器](/images/materials/cmu-15-779/tvm-slides/page-053.webp)

**类似 TPU 的专用加速器：**

- 计算原语是张量（tensor）；
- 内存子系统包含 FIFO、统一缓冲区（Unified Buffer）和累加器（Acc）；
- 内存由软件显式管理（explicitly managed）。

## 第 54 页｜面向硬件的搜索空间：类似 TPU 的专用加速器

![TVM 原始幻灯片第 54 页：面向硬件的搜索空间：类似 TPU 的专用加速器](/images/materials/cmu-15-779/tvm-slides/page-054.webp)

与第 53 页相同的硬件抽象，强调专用加速器的 FIFO、统一缓冲区和累加器。

## 第 55 页｜张量化挑战

![TVM 原始幻灯片第 55 页：张量化挑战](/images/materials/cmu-15-779/tvm-slides/page-055.webp)

页面只显示计算原语（Compute primitives），开始从标量到向量、张量的逐级抽象。

## 第 56 页｜张量化挑战：标量

![TVM 原始幻灯片第 56 页：张量化挑战：标量](/images/materials/cmu-15-779/tvm-slides/page-056.webp)

新增计算原语：**标量（scalar）**。

## 第 57 页｜张量化挑战：向量

![TVM 原始幻灯片第 57 页：张量化挑战：向量](/images/materials/cmu-15-779/tvm-slides/page-057.webp)

新增计算原语：**向量（vector）**，形成“标量 + 向量”的层次。

## 第 58 页｜张量化挑战：张量

![TVM 原始幻灯片第 58 页：张量化挑战：张量](/images/materials/cmu-15-779/tvm-slides/page-058.webp)

新增计算原语：**张量（tensor）**，形成“标量 + 向量 + 张量”的层次。

## 第 59 页｜张量化挑战：声明张量指令接口

![TVM 原始幻灯片第 59 页：张量化挑战：声明张量指令接口](/images/materials/cmu-15-779/tvm-slides/page-059.webp)

硬件设计者使用张量表达式声明张量指令的行为与降低规则。

```python
w, x = t.placeholder((8, 8)), t.placeholder((8, 8))
k = t.reduce_axis((0, 8))                 # 声明归约轴
y = t.compute((8, 8), lambda i, j:
               t.sum(w[i, k] * x[j, k], axis=k))

def gemm_intrin_lower(inputs, outputs):
    ww_ptr = inputs[0].access_ptr("r")
    xx_ptr = inputs[1].access_ptr("r")
    zz_ptr = outputs[0].access_ptr("w")
    compute = t.hardware_intrin("gemm8x8", ww_ptr, xx_ptr, zz_ptr)
    reset = t.hardware_intrin("fill_zero", zz_ptr)
    update = t.hardware_intrin("fuse_gemm8x8_add", ww_ptr, xx_ptr, zz_ptr)
    return compute, reset, update

gemm8x8 = t.decl_tensor_intrin(y.op, gemm_intrin_lower)
```

右侧说明：硬件设计者声明张量指令接口；降低规则生成用于执行计算的硬件内建指令。

## 第 60 页｜张量化挑战：将程序转换为张量指令

![TVM 原始幻灯片第 60 页：张量化挑战：将程序转换为张量指令](/images/materials/cmu-15-779/tvm-slides/page-060.webp)

在第 59 页接口之上，**张量化（Tensorize）**把程序转换为使用张量指令的实现。

```python
# 计算行为
w, x = t.placeholder((8, 8)), t.placeholder((8, 8))
k = t.reduce_axis((0, 8))
y = t.compute((8, 8), lambda i, j:
               t.sum(w[i, k] * x[j, k], axis=k))

# 硬件内建：清零、计算、融合累加
compute = t.hardware_intrin("gemm8x8", ww_ptr, xx_ptr, zz_ptr)
reset = t.hardware_intrin("fill_zero", zz_ptr)
update = t.hardware_intrin("fuse_gemm8x8_add", ww_ptr, xx_ptr, zz_ptr)
```

图示从标量直接跳到张量，表示调度器可以把一段标量循环整体替换成张量指令。

## 第 61 页｜面向硬件的搜索空间：类似 TPU 的专用加速器

![TVM 原始幻灯片第 61 页：面向硬件的搜索空间：类似 TPU 的专用加速器](/images/materials/cmu-15-779/tvm-slides/page-061.webp)

回到类似 TPU 的加速器抽象：张量计算、FIFO、统一缓冲区、累加器，以及显式内存管理。

## 第 62 页｜面向硬件的搜索空间：类似 TPU 的专用加速器

![TVM 原始幻灯片第 62 页：面向硬件的搜索空间：类似 TPU 的专用加速器](/images/materials/cmu-15-779/tvm-slides/page-062.webp)

动画构建页，继续显示专用加速器的计算/内存层次。

## 第 63 页｜软件支持隐藏延迟：无任务流水线

![TVM 原始幻灯片第 63 页：软件支持隐藏延迟：无任务流水线](/images/materials/cmu-15-779/tvm-slides/page-063.webp)

**单模块、无任务流水线（Single Module / No Task-Pipelining）：**

时间线依次执行：加载输入 → 加载权重 → 矩阵乘 → 写回输出，然后重复下一模块。由于加载、计算、存储串行，内存延迟暴露在关键路径上。

## 第 64 页｜软件支持隐藏延迟：任务级流水线

![TVM 原始幻灯片第 64 页：软件支持隐藏延迟：任务级流水线](/images/materials/cmu-15-779/tvm-slides/page-064.webp)

**多模块、任务级流水线（Multiple-Module / Task-Level Pipelining）：**

多个模块交错执行：先并行加载模块 1/2 的输入和权重，再让矩阵乘阶段与下一组加载重叠，最后分别存储输出。这样可用计算工作覆盖内存访问延迟。

## 第 65 页｜软件支持隐藏延迟：显式依赖跟踪

![TVM 原始幻灯片第 65 页：软件支持隐藏延迟：显式依赖跟踪](/images/materials/cmu-15-779/tvm-slides/page-065.webp)

任务级流水线的依赖关系由软件显式跟踪（Explicit dependency tracking managed by software），从而隐藏内存延迟。

## 第 66 页｜面向硬件的搜索空间：张量表达式语言

![TVM 原始幻灯片第 66 页：面向硬件的搜索空间：张量表达式语言](/images/materials/cmu-15-779/tvm-slides/page-066.webp)

再次给出张量表达式规格：

```python
C = tvm.compute((m, n),
    lambda y, x: tvm.sum(A[k, y] * B[k, x], axis=k))
```

表达式本身保持与硬件无关。

## 第 67 页｜面向硬件的搜索空间：既有原语

![TVM 原始幻灯片第 67 页：面向硬件的搜索空间：既有原语](/images/materials/cmu-15-779/tvm-slides/page-067.webp)

先前工作（Halide、Loopy）中的原语包括：

- 循环变换（Loop Transformations）；
- 线程绑定（Thread Bindings）；
- 缓存局部性（Cache Locality）。

## 第 68 页｜面向硬件的搜索空间：新增 GPU/加速器原语

![TVM 原始幻灯片第 68 页：面向硬件的搜索空间：新增 GPU/加速器原语](/images/materials/cmu-15-779/tvm-slides/page-068.webp)

为 GPU 和类似 TPU 的加速器加入新原语：

- 线程协作（Thread Cooperation）；
- 张量化（Tensorization）；
- 延迟隐藏（Latency Hiding）；
- 以及更多原语（…）。

## 第 69 页｜面向硬件的搜索空间：调度原语集合

![TVM 原始幻灯片第 69 页：面向硬件的搜索空间：调度原语集合](/images/materials/cmu-15-779/tvm-slides/page-069.webp)

完整调度原语集合包括循环变换、线程绑定、缓存局部性、线程协作、张量化和延迟隐藏。

## 第 70 页｜面向硬件的搜索空间：调度原语集合

![TVM 原始幻灯片第 70 页：面向硬件的搜索空间：调度原语集合](/images/materials/cmu-15-779/tvm-slides/page-070.webp)

动画构建页：继续呈现同一组调度原语与硬件的连接。

## 第 71 页｜面向硬件的搜索空间：数十亿优化选择

![TVM 原始幻灯片第 71 页：面向硬件的搜索空间：数十亿优化选择](/images/materials/cmu-15-779/tvm-slides/page-071.webp)

这些组合带来**数十亿种可能的优化选择（Billions of possible optimization choices）**，人工搜索无法扩展，因此需要学习型程序优化器。

## 第 72 页｜基于学习的学习系统

![TVM 原始幻灯片第 72 页：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-072.webp)

完整系统由四层构成：框架中的高层数据流图与优化、面向硬件的优化张量程序搜索空间、机器学习程序优化器，以及硬件。

## 第 73 页｜基于学习的学习系统

![TVM 原始幻灯片第 73 页：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-073.webp)

动画构建页，重复显示完整四层系统。

## 第 74 页｜基于学习的程序优化器

![TVM 原始幻灯片第 74 页：基于学习的程序优化器](/images/materials/cmu-15-779/tvm-slides/page-074.webp)

章节标题：基于学习的程序优化器（Learning-based Program Optimizer）。

## 第 75 页｜基于学习的程序优化器：程序生成

![TVM 原始幻灯片第 75 页：基于学习的程序优化器：程序生成](/images/materials/cmu-15-779/tvm-slides/page-075.webp)

优化器接收候选**程序（Program）**，通过**代码生成器（Code Generator）**产生可执行代码。

## 第 76 页｜基于学习的程序优化器：运行时测量

![TVM 原始幻灯片第 76 页：基于学习的程序优化器：运行时测量](/images/materials/cmu-15-779/tvm-slides/page-076.webp)

代码在目标硬件上运行，并收集**运行时测量结果（Runtime Measurements）**，反馈给优化器。

## 第 77 页｜基于学习的程序优化器：试验成本

![TVM 原始幻灯片第 77 页：基于学习的程序优化器：试验成本](/images/materials/cmu-15-779/tvm-slides/page-077.webp)

每轮试验都要生成代码并在硬件上运行。页面指出试验成本很高：**每次试验约需 1 秒（each trial costs ~1 second）**。

## 第 78 页｜基于学习的程序优化器：代价模型

![TVM 原始幻灯片第 78 页：基于学习的程序优化器：代价模型](/images/materials/cmu-15-779/tvm-slides/page-078.webp)

为减少真实运行次数，引入代价模型（Cost Model）。

## 第 79 页｜基于学习的程序优化器：代价模型

![TVM 原始幻灯片第 79 页：基于学习的程序优化器：代价模型](/images/materials/cmu-15-779/tvm-slides/page-079.webp)

代价模型用于预测候选程序性能，再由优化器选择下一组搜索点。

## 第 80 页｜基于学习的程序优化器：硬件相关代价模型

![TVM 原始幻灯片第 80 页：基于学习的程序优化器：硬件相关代价模型](/images/materials/cmu-15-779/tvm-slides/page-080.webp)

页面强调：需要针对每种硬件拥有可靠的代价模型（Need reliable cost model per hardware）。

## 第 81 页｜基于学习的程序优化器：学习闭环

![TVM 原始幻灯片第 81 页：基于学习的程序优化器：学习闭环](/images/materials/cmu-15-779/tvm-slides/page-081.webp)

动画构建页：程序优化器、代码生成器、运行时测量和代价模型形成闭环。

## 第 82 页｜基于学习的程序优化器：训练数据

![TVM 原始幻灯片第 82 页：基于学习的程序优化器：训练数据](/images/materials/cmu-15-779/tvm-slides/page-082.webp)

从程序在硬件上的测量结果获得训练数据（Training data），用于拟合统计代价模型。图中 `D` 表示收集的数据集。

## 第 83 页｜基于学习的程序优化器：统计代价模型

![TVM 原始幻灯片第 83 页：基于学习的程序优化器：统计代价模型](/images/materials/cmu-15-779/tvm-slides/page-083.webp)

加入**学习（Learning）**与**统计代价模型（Statistical Cost Model）**：用训练数据拟合程序特征到运行时间/代价的预测函数。

## 第 84 页｜基于学习的程序优化器：适配不同硬件

![TVM 原始幻灯片第 84 页：基于学习的程序优化器：适配不同硬件](/images/materials/cmu-15-779/tvm-slides/page-084.webp)

统计代价模型通过学习适配硬件类型（Adapt to hardware type by learning），并可在约 **1 ms 量级（Make prediction in 1ms level）**完成预测，远快于实际运行一次程序。

## 第 85 页｜机器学习模型的有效性

![TVM 原始幻灯片第 85 页：机器学习模型的有效性](/images/materials/cmu-15-779/tvm-slides/page-085.webp)

图表逐步绘制中，纵轴范围约为 0.00–1.50，横轴为试验次数（0–800）。

## 第 86 页｜机器学习模型的有效性

![TVM 原始幻灯片第 86 页：机器学习模型的有效性](/images/materials/cmu-15-779/tvm-slides/page-086.webp)

实验对象标注为：**Titan X 上 ResNet18 的一个 Conv2D 层（One Conv2D Layer of ResNet18 on Titan X）**。

## 第 87 页｜机器学习模型的有效性

![TVM 原始幻灯片第 87 页：机器学习模型的有效性](/images/materials/cmu-15-779/tvm-slides/page-087.webp)

横轴明确为**试验次数（Number of Trials）**，纵轴为相对性能/加速比。

## 第 88 页｜机器学习模型的有效性：cuDNN 基线

![TVM 原始幻灯片第 88 页：机器学习模型的有效性：cuDNN 基线](/images/materials/cmu-15-779/tvm-slides/page-088.webp)

纵轴为**相对加速比（Relative Speedup）**；图例加入 **基线：cuDNN（Baseline: CuDNN）**。实验仍是 Titan X 上 ResNet18 的一个 Conv2D 层。

## 第 89 页｜机器学习模型的有效性：随机搜索

![TVM 原始幻灯片第 89 页：机器学习模型的有效性：随机搜索](/images/materials/cmu-15-779/tvm-slides/page-089.webp)

图例加入 **TVM：随机搜索（TVM: Random Search）**，与 cuDNN 基线比较。随机搜索需要较多试验才能找到高性能程序。

## 第 90 页｜机器学习模型的有效性：基于机器学习的模型

![TVM 原始幻灯片第 90 页：机器学习模型的有效性：基于机器学习的模型](/images/materials/cmu-15-779/tvm-slides/page-090.webp)

图例加入 **TVM：基于机器学习的模型（TVM: ML-based Model）**。学习型代价模型比随机搜索更快接近高性能区域，并超过 cuDNN 基线。

## 第 91 页｜基于学习的学习系统

![TVM 原始幻灯片第 91 页：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-091.webp)

完整方案总结：

1. 高层数据流图和优化；
2. 面向硬件的优化张量程序搜索空间；
3. 基于机器学习的程序优化器；
4. 目标硬件。

## 第 92 页｜端到端推理性能：NVIDIA Titan X

![TVM 原始幻灯片第 92 页：端到端推理性能：NVIDIA Titan X](/images/materials/cmu-15-779/tvm-slides/page-092.webp)

在 NVIDIA Titan X 上比较 TensorFlow、Apache MXNet 和 TensorFlow-XLA。

图表：纵轴为时间（ms）；模型横轴包括 ResNet-18、MobileNet、LSTM LM、DQN、DCGAN。

## 第 93 页｜端到端推理性能：cuDNN 支持

![TVM 原始幻灯片第 93 页：端到端推理性能：cuDNN 支持](/images/materials/cmu-15-779/tvm-slides/page-093.webp)

新增说明：各框架由 **cuDNN 支持（Backed by cuDNN）**。图表比较 TensorFlow、Apache MXNet 与 TensorFlow-XLA 的端到端推理时间。

## 第 94 页｜端到端推理性能：TVM（无图优化）

![TVM 原始幻灯片第 94 页：端到端推理性能：TVM（无图优化）](/images/materials/cmu-15-779/tvm-slides/page-094.webp)

加入 **TVM：无图优化（TVM: without graph optimizations）**，用于区分算子级优化与整图优化的贡献。

## 第 95 页｜端到端推理性能：TVM 全部优化

![TVM 原始幻灯片第 95 页：端到端推理性能：TVM 全部优化](/images/materials/cmu-15-779/tvm-slides/page-095.webp)

加入 **TVM：全部优化（TVM: all optimizations）**，即同时使用算子调优和图级优化。

## 第 96 页｜端到端推理性能：标准模型

![TVM 原始幻灯片第 96 页：端到端推理性能：标准模型](/images/materials/cmu-15-779/tvm-slides/page-096.webp)

标注结论：在标准模型（standard models）上，TVM 具有竞争力（Competitive on standard models）。

## 第 97 页｜端到端推理性能：非传统模型

![TVM 原始幻灯片第 97 页：端到端推理性能：非传统模型](/images/materials/cmu-15-779/tvm-slides/page-097.webp)

补充结论：在不太传统的模型（less conventional models）上，TVM 与现有方案之间的性能差距更大（Bigger gap）。

## 第 98 页｜端到端性能：ARM Cortex-A53

![TVM 原始幻灯片第 98 页：端到端性能：ARM Cortex-A53](/images/materials/cmu-15-779/tvm-slides/page-098.webp)

在 ARM Cortex-A53 上比较 TensorFlow Lite、TVM（无图优化）和 TVM。

- 模型：ResNet-18、MobileNet、DQN；
- 纵轴：时间（ms），不同模型使用不同刻度；
- 目标是嵌入式 ARM 平台的端到端性能。

## 第 99 页｜端到端性能：ARM Cortex-A53

![TVM 原始幻灯片第 99 页：端到端性能：ARM Cortex-A53](/images/materials/cmu-15-779/tvm-slides/page-099.webp)

图中注明 TensorFlow Lite **专门针对嵌入式 ARM 系统优化（Specially optimized for Embedded system (ARM)）**，并与 TVM 的无图优化/全部优化版本比较。

## 第 100 页｜端到端性能：ARM GPU

![TVM 原始幻灯片第 100 页：端到端性能：ARM GPU](/images/materials/cmu-15-779/tvm-slides/page-100.webp)

在 ARM GPU 上比较 ARMComputeLib、TVM（无图优化）和 TVM。

- 模型：ResNet-18、MobileNet、DQN；
- 数据类型：float32 与 float16；
- 纵轴：时间（ms）。

## 第 101 页｜支持新的专用加速器

![TVM 原始幻灯片第 101 页：支持新的专用加速器](/images/materials/cmu-15-779/tvm-slides/page-101.webp)

在框架、高层图优化、面向硬件的张量程序搜索空间和机器学习优化器之下，可通过 **LLVM** 与 **CUDA** 连接不同后端。

## 第 102 页｜支持新的专用加速器：VTA

![TVM 原始幻灯片第 102 页：支持新的专用加速器：VTA](/images/materials/cmu-15-779/tvm-slides/page-102.webp)

新增 **VTA：开放且可定制的深度学习加速器（Open, Customizable Deep Learning Accelerator）**。目标硬件包括边缘 FPGA（Edge FPGA）、数据中心 FPGA（Data Center FPGA）和 ASIC。

## 第 103 页｜TVM/VTA：全栈开源系统

![TVM 原始幻灯片第 103 页：TVM/VTA：全栈开源系统](/images/materials/cmu-15-779/tvm-slides/page-103.webp)

系统栈的顶层：

- 高层优化（High-level Optimizations）；
- 张量程序搜索空间（Tensor Program Search Space）；
- 基于机器学习的优化器（ML-based Optimizer）。

## 第 104 页｜TVM/VTA：全栈开源系统

![TVM 原始幻灯片第 104 页：TVM/VTA：全栈开源系统](/images/materials/cmu-15-779/tvm-slides/page-104.webp)

在上述软件栈下加入 **VTA 微架构（VTA MicroArchitecture）**。

## 第 105 页｜TVM/VTA：全栈开源系统

![TVM 原始幻灯片第 105 页：TVM/VTA：全栈开源系统](/images/materials/cmu-15-779/tvm-slides/page-105.webp)

继续加入 **VTA 硬件/软件接口（ISA）**，连接编译器生成的程序与微架构。

## 第 106 页｜TVM/VTA：全栈开源系统

![TVM 原始幻灯片第 106 页：TVM/VTA：全栈开源系统](/images/materials/cmu-15-779/tvm-slides/page-106.webp)

加入 **VTA 运行时与 JIT 编译器（VTA Runtime & JIT Compiler）**，负责运行时装载和即时编译。

## 第 107 页｜TVM/VTA：全栈开源系统

![TVM 原始幻灯片第 107 页：TVM/VTA：全栈开源系统](/images/materials/cmu-15-779/tvm-slides/page-107.webp)

加入 **VTA 模拟器（VTA Simulator）**，可在没有真实硬件时验证指令和编译流程。

## 第 108 页｜TVM/VTA：全栈开源系统

![TVM 原始幻灯片第 108 页：TVM/VTA：全栈开源系统](/images/materials/cmu-15-779/tvm-slides/page-108.webp)

页面列出系统目标：

- JIT 编译加速器微代码（JIT compile accelerator micro code）；
- 支持异构设备，在同一开发板上比 CPU 快 10 倍（Support heterogeneous devices, 10x better than CPU on the same board）；
- 将硬件复杂性转移到软件（Move hardware complexity to software）。

## 第 109 页｜TVM/VTA：全栈开源系统

![TVM 原始幻灯片第 109 页：TVM/VTA：全栈开源系统](/images/materials/cmu-15-779/tvm-slides/page-109.webp)

完整结论：把硬件复杂性转移到软件、编译器、驱动和硬件设计；形成**全栈开源（full stack open source）**系统。

- JIT 编译加速器微代码；
- 支持异构设备，在同一开发板上达到 CPU 的 10 倍性能；
- 开放 VTA 模拟器、ISA、微架构、运行时和 JIT 编译器。

## 第 110 页｜TVM：基于学习的学习系统

![TVM 原始幻灯片第 110 页：TVM：基于学习的学习系统](/images/materials/cmu-15-779/tvm-slides/page-110.webp)

总结架构：

- 高层优化；
- 张量程序搜索空间；
- 基于机器学习的优化器；
- LLVM、CUDA、VTA 后端。

页面号召：**欢迎体验（Check it out!）**。
