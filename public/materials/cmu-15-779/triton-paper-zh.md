# Triton：用于分块神经网络计算的中间语言和编译器

**作者：**Philippe Tillet、H. T. Kung、David Cox

**来源：**MAPL ’19，2019 年 6 月 22 日，第 3 届 ACM SIGPLAN 机器学习与编程语言国际研讨会。

> 说明：本文根据用户提供的论文 PDF 翻译。Triton/C 代码、LLVM IR、公式和引用编号保留，便于与原文对照。

## 摘要

在深度学习领域，新研究思想的验证和部署经常受限于某些基础原语缺少高效计算 kernel。尤其是无法利用厂商库（例如 cuBLAS、cuDNN）的操作，如果没有专家编写自定义实现，就可能无法充分利用设备；而手工实现通常又会牺牲可移植性。因此，在性能代价很小的前提下，为指定自定义深度学习工作负载提供新的编程抽象，已经变得非常重要。

我们提出 Triton：一种围绕 tile（分块）概念设计的语言和编译器。tile 指静态形状的多维子数组。我们的方法包括两部分：

1. 一种基于 C 的语言和基于 LLVM 的中间表示（IR），用参数化 tile 变量上的操作来表达张量程序；
2. 一组新的 tile 级优化 pass，把这些程序编译为高效 GPU 代码。

我们展示了 Triton 如何构造性能接近手工调优厂商库（cuBLAS/cuDNN）的矩阵乘法和卷积 kernel，并高效实现 shift convolution 等近期研究思想。

## 1 引言

近年来，深度神经网络（DNN）的复兴在很大程度上得益于可编程并行计算设备的普及。尤其是多核架构（如 GPU）性能持续提升，使研究人员和工程师能够使用越来越多的数据探索规模越来越大的模型。

厂商库 cuBLAS 和 cuDNN 也支持了这一过程，它们把最新硬件创新快速带给使用者。然而，这些库只支持有限的一组张量操作，使新原语的实现仍然需要专家完成。

![不同实现的矩阵乘与卷积性能（原文图 1）](/images/materials/cmu-15-779/triton/figure-01.webp)

这一观察推动了多种 DNN 领域专用语言（DSL）的发展，包括基于多面体模型的 Tensor Comprehensions，以及基于循环综合技术的 Halide、TVM 和 PlaidML。这些系统在深度可分离卷积（如 MobileNet）等部分问题上通常表现良好，但实践中往往比厂商库慢很多，而且缺少表达结构化稀疏模式所需的表达能力；这类模式无法直接用嵌套循环中的仿射数组索引指定。

人们经常使用 micro-kernel（手写的 tile 级内核指令）解决上述问题，但这需要大量人工工作且缺少可移植性。近来也有人提出高层次的分块编程抽象，但底层编译器后端仍不支持 tile 级操作和优化。

因此，我们提出 Triton（图 2）：一种开源的中间语言和编译器，用于指定并把 tile 程序编译为高效 GPU 代码。

![Triton 编译流程概览（原文图 2）](/images/materials/cmu-15-779/triton/figure-02.webp)

本文主要贡献如下：

- **Triton-C（第 3 节）：**一种类似 C 的语言，用参数化 tile 变量表达张量程序。它为现有 DNN 转译编译器（如 PlaidML、Tensor Comprehensions）以及熟悉 CUDA 的程序员提供稳定接口。
- **Triton-IR（第 4 节）：**一种基于 LLVM 的中间表示，为 tile 级程序分析、变换和优化提供环境。
- **Triton-JIT（第 5 节）：**一个即时编译器和代码生成后端，把 Triton-IR 编译为高效 LLVM bitcode。它包含与机器无关的 tile 级优化 pass、与机器相关的 pass，以及自动调优器。
- **数值实验（第 6 节）：**实验表明 Triton 能生成性能接近 cuBLAS 的矩阵乘法，在循环和 Transformer 网络上最多比其他 DSL 快 3 倍；能够无性能损失地重新实现 cuDNN 的 `IMPLICIT_GEMM` 稠密卷积算法；还可以为 shift-conv 模块等新研究思想生成高效实现。

## 2 相关工作

深度学习框架和库对新神经网络架构与算法的出现至关重要。但尽管线性代数编译器在分析式启发式和经验式启发式方面不断进步，它们仍然依赖手工优化子程序，例如 cuBLAS 和 cuDNN。因此，人们发展了多种 DNN DSL 和编译器，主要基于三种方法：

- **张量级 IR：**XLA [16] 和 Glow [38] 使用张量级 IR，通过模式匹配把张量程序转换为预定义的 LLVM-IR 和 CUDA-C 操作模板（如张量收缩和逐元素运算）。
- **多面体模型：**Tensor Comprehensions（TC）[43] 和 Diesel [14] 使用多面体模型，对一个或多个 DNN 层进行参数化，并自动编译为 LLVM-IR 和 CUDA-C 程序。
- **循环综合器：**Halide [37] 和 TVM [10] 把张量计算转换为循环嵌套，之后由用户定义的、可能带参数的 schedule 手工优化。

与这些方法相比，Triton 把 tile 级操作与 tile 级优化加入传统编译流水线。这种方法相比 XLA 和 Glow 更灵活；支持非仿射张量索引，克服 TC 和 Diesel 的限制；并且可以自动推断执行 schedule，而 Halide 和 TVM 通常需要手工指定。代价是编程工作量增加。

对于 `C = A × B^T`，不同 DSL 的表达形式例如：

```text
C = tf.matmul(A, tf.transpose(B))       // TensorFlow
C[i, j:J] = +(A[i, k] * B[j, k])        // PlaidML
C(i, j) +=! A(i, k) * B(j, k)            // Tensor Comprehensions
tvm.sum(A[i, k] * B[j, k], axis=k)       // TVM
```

## 3 Triton-C 语言

Triton-C 的目标是为现有和未来的 DNN 转译编译器，以及熟悉底层 GPU 编程的程序员，提供稳定前端。本节介绍 Triton-C 类 CUDA 的语法、类似 NumPy 的语义，以及“单程序多数据”（SPMD）编程模型。

### 3.1 语法

Triton-C 的语法基于 ANSI C，更具体地说基于 CUDA-C，但为了适应 tile 语义和编程模型进行了修改和扩展。

**Tile 声明：**我们增加了声明多维数组的特殊语法，例如 `int tile[16, 16]`，以强调它与 ANSI C 中嵌套数组（例如 `int tile[16][16]`）的语义差异。tile 的形状必须是常量，但使用 `tunable` 关键字后也可以参数化。一维整数 tile 可以用省略号初始化，例如 `int range[8] = 0 ... 8`。

**内置函数：**普通 C 语法保留给逐元素数组运算（`+`、`-`、`&&`、`*` 等）；同时增加 `dot`、`trans` 和 `get_global_range` 等内置函数来支持 tile 语义和 SPMD 模型。

**广播：**N 维 tile 可以使用 `newaxis` 关键字和通常的切片语法沿任意轴广播，例如 `int broadcast[8, 8] = range[:, newaxis]`。不过，切片 tile 以取得标量或子数组的做法被禁止。

**谓词：**tile 操作内部的基本控制流通过带 `@` 前缀的谓词语句实现。

Triton-C 的语法扩展可概括为：

```text
slice           : ':' | 'newaxis'
slice_list      : slice | slice_list ',' slice
slice_expr      : postfix_expr | expr '[' slice_list ']'
constant_range  : expr '...' expr
global_range    : 'get_global_range' '(' constant ')'
dot             : 'dot' '(' expr ',' expr ')'
trans           : 'trans' '(' expr ',' expr ')'
predicate_expr  : '@' expr
abstract_decl   : abstract_decl | '[' constant_list ']'
expr            : expr | constant_range | slice_expr | intrinsic_expr
storage_spec    : storage_spec | 'kernel'
type_spec       : type_spec | 'tunable'
```

### 3.2 语义

#### 3.2.1 Tile 语义

Triton-C 的内置 tile 类型和操作有两个主要好处。首先，它隐藏了 tile 内存合并、缓存管理和专用硬件利用等性能细节，从而简化张量程序结构。其次，它为编译器自动执行这些优化打开了空间。

#### 3.2.2 广播语义

Triton-C 中的 tile 是强类型的：某些指令要求操作数满足严格形状约束。例如，标量不能直接加到数组上，除非它先被正确广播。广播语义提供了一组隐式转换规则：

1. **填充：**向形状较短的操作数左侧补 1，直到两个操作数维度相同。
2. **广播：**沿大小为 1 的维度复制两个操作数的内容，直到形状完全相同；如果无法完成则报告错误。

示例：

```text
int a[16], b[32, 16], c[16, 1];
int x_1[32, 16] = a[newaxis, :] + b;
int x_2[32, 16] = a + b;
int y[16, 16] = a + c;
```

#### 3.3 编程模型

CUDA [33] 在 GPU 上使用 SPMD [4] 编程模型：每个 kernel 与 launch grid 中一个可识别的线程块关联。Triton 的编程模型类似，但每个 kernel 是单线程的（编译器会自动并行化），并与一组会随实例变化的全局范围关联，如图 3 所示。

![CUDA 与 Triton 编程模型的差异（原文图 3）](/images/materials/cmu-15-779/triton/figure-03.webp)

这种方法产生更简单的 kernel，因为不存在 CUDA 风格的并发原语，如共享内存同步和线程间通信。

kernel 可以通过 `get_global_range(axis)` 内置函数查询与之关联的全局范围，从而构造指针 tile，如前面的矩阵乘法示例所示。

## 4 Triton IR

Triton-IR 是一种基于 LLVM 的中间表示，目的是为 tile 级程序分析、变换和优化提供合适环境。本文中 Triton-IR 程序直接从 Triton-C 程序构造，但未来也可以从嵌入式 DSL 或 TVM 等更高层 DNN 编译器自动生成。

Triton-IR 与 LLVM-IR 共享相同的高层结构，但增加了 tile 级数据流和控制流分析所需的扩展。这些扩展对第 5 节中的优化至关重要，也能安全访问任意形状的张量。

### 4.1 结构

#### 4.1.1 模块

在最高层，Triton-IR 程序由一个或多个称为模块的基本编译单元组成。模块彼此独立编译，最后由链接器合并；链接器负责解析前向声明并正确合并全局定义。

每个模块由函数、全局变量、常量和其他符号（例如元数据和函数属性）组成。

#### 4.1.2 函数

Triton-IR 函数定义包含返回类型、名称和可能为空的参数列表。根据需要可以添加可见性、对齐方式和链接属性。还可以指定函数属性（例如内联提示）和参数属性（例如只读、别名提示），让编译器后端能够执行更激进的优化，例如更好地利用只读内存缓存。

函数头部之后是由基本块列表组成的函数体；基本块之间的依赖构成函数的控制流图（CFG）。

#### 4.1.3 基本块

基本块是直线代码序列，末尾只能包含终结指令，例如分支和返回。

Triton-IR 使用静态单赋值（SSA）形式：每个基本块中的每个变量必须只被赋值一次，并且必须先定义后使用。这样，每个基本块隐式定义一个数据流图；不同路径对应 SSA 表示中的使用-定义链。

### 4.2 对 tile 级数据流分析的支持

#### 4.2.1 类型

多维 tile 是 Triton-IR 数据流分析的中心，可以使用类似 LLVM-IR 向量声明的语法。例如，`i32<8, 8>` 表示 `8 × 8` 的 32 位整数 tile。

Triton-IR 中没有 `tunable` 关键字，因此参数化形状必须在生成程序之前解析。在本文实现中，这是由 Triton-JIT 的自动调优器完成的。

#### 4.2.2 指令

Triton-IR 引入了一组重新分块指令，用于支持第 3.2.2 节描述的广播语义：

- `reshape` 根据输入数据创建指定形状的 tile。它特别适合通过在输入形状中补 1，把变量重新解释为更高维数组，以准备隐式或显式广播。
- `broadcast` 沿大小为 1 的维度复制输入参数，创建指定形状的 tile。

![Triton-IR 的广播指令（原文图 4）](/images/materials/cmu-15-779/triton/figure-04.webp)

普通标量指令（`cmp`、`getelementptr`、`add`、`load` 等）被保留并扩展，以表示对 tile 操作数的逐元素操作。Triton-IR 还公开了处理转置（`trans`）和矩阵乘法（`dot`）的专用算术指令。

### 4.3 对 tile 级控制流分析的支持

Triton-IR 引入 tile 级操作后，一个问题是：无法在 tile 内表达分歧控制流。例如，程序可能需要部分保护 tile 级 load，避免内存访问越界，但不能通过普通分支实现，因为不能单独访问 tile 中的元素。

我们使用 Predicated SSA（PSSA）形式 [8] 和 ψ 函数 [39] 解决这一问题。Triton-IR 增加两类指令：

- `cmpp` 指令类似普通比较指令，但会返回两个相反的谓词，而不是一个。
- `psi` 指令合并来自不同谓词指令流的结果。

示例：

```text
; 若 x[i,j] < 5，则 pt[i,j], pf[i,j] = (true, false)
; 否则为 (false, true)
% pt, % pf = icmpp slt %x, 5
@%pt %x1 = add %y, 1
@%pf %x2 = sub %y, 1
; 合并不同谓词产生的值
%x = psi i32<8,8> [%pt, %x1], [%pf, %x2]
%z = mul i32<8,8> %x, 2
```

## 5 Triton-JIT 编译器

Triton-JIT 的目标是通过一组与机器无关的 pass（第 5.1 节）和与机器相关的 pass（第 5.2 节），并由自动调优引擎支持（第 5.3 节），把 Triton-IR 程序简化并编译为高效机器代码。

### 5.1 与机器无关的 pass

#### 5.1.1 预取

循环中的 tile 级内存操作可能产生严重延迟；如果没有足够多的独立指令，就难以隐藏这种延迟。不过，可以直接在 Triton-IR 中检测循环，并在需要的位置加入合适的预取代码。

例如，原本的循环可能在当前迭代中 load 指针 `p` 指向的数据；自动预取后，会在当前迭代中计算并 load 下一次迭代需要的 `p1`，下一轮直接使用已经加载的 `x1`。

#### 5.1.2 Tile 级窥孔优化

Triton-IR 中存在 tile 级操作，为窥孔优化器 [29] 带来了新的机会。例如，对任意 tile `X`，恒等式 `X = (X^T)^T` 可以简化连续的转置操作。未来还可以利用与对角 tile 等结构相关的其他代数性质。

### 5.2 与机器相关的 pass

现在考虑遵循图 5 所示高层模型的机器。Triton-JIT 执行的优化包括：

1. 分层分块；
2. 内存合并；
3. 共享内存分配；
4. 共享内存同步。

#### 5.2.1 分层分块

嵌套分块策略（图 5）把 tile 分解为 micro-tile，最终分解为 nano-tile，使其尽可能紧密地适配机器的计算能力和内存层次结构。

![Triton-IR 机器模型中的分层分块（原文图 5）](/images/materials/cmu-15-779/triton/figure-05.webp)

虽然自动调优框架经常使用这一技术，但 Triton-IR 的结构使编译器能够自动枚举并优化任意可表达程序的有效嵌套分块配置，而不需要多面体工具。

#### 5.2.2 内存合并

当相邻线程同时访问相邻内存位置时，内存访问称为合并访问。这一点很重要，因为内存通常以大块形式从 DRAM 取出。

由于 Triton-IR 程序是单线程的、由编译器自动并行化，后端可以在每个 micro-tile 内部安排线程顺序，尽可能避免未合并访问。这会减少加载 tile 列所需的内存事务数量，如图 6 所示。

图 6 中，未合并访问会让不同线程从 DRAM 读取彼此分散的位置；合并访问则把访问组织为相邻块，减少传输开销。

![未合并与合并 DRAM 访问（原文图 6）](/images/materials/cmu-15-779/triton/figure-06.webp)

#### 5.2.3 共享内存分配

具有高算术强度的 tile 级操作（例如 `dot`）可以把操作数暂时存储在快速共享内存中。共享内存分配 pass 的目标是确定何时、何地把 tile 放入这块空间。

如图 7 所示，首先计算每个相关变量的活跃区间，再使用 [15] 提出的线性时间存储分配算法。

![共享内存分配与变量活跃区间（原文图 7）](/images/materials/cmu-15-779/triton/figure-07.webp)

#### 5.2.4 共享内存同步

在本文机器模型中，对共享内存的读写是异步的。共享内存同步 pass 会自动在生成的 GPU 源代码中插入屏障，以保持程序正确性。

它使用前向数据流分析检测读后写（RAW）和写后读（WAR）冒险。若当前指令读取的数据与尚未解决的写集合相交，就插入屏障；若当前指令写入的数据与尚未解决的读集合相交，也插入屏障；否则将当前读写操作加入相应集合。

### 5.3 自动调优器

传统自动调优器 [42, 45] 通常依赖手工编写的参数化代码模板，在预定义工作负载上获得良好性能。相比之下，Triton-JIT 可以直接从 Triton-IR 程序中提取优化空间：把各优化 pass 关联的元参数简单拼接起来即可。

本文只考虑分层分块 pass，因此每个 tile 的每个维度最多有 3 个分块参数。这些参数通过穷举搜索优化：

- tile size 在 32 到 128 之间搜索 2 的幂；
- micro-tile size 在 8 到 32 之间搜索 2 的幂；
- nano-tile size 在 1 到 4 之间搜索 2 的幂。

未来可以使用更好的自动调优方法。

## 6 数值实验

本节在深度学习文献中的多种工作负载上评估 Triton。实验使用 NVIDIA GeForce GTX1070，并与最新厂商库 cuBLAS 10.0、cuDNN 7.0，以及相关编译技术 Auto-TVM、TC 和 PlaidML 比较。在适用时，我们按照官方文档建议对这些 DSL 针对每个问题规模进行自动调优。

### 6.1 矩阵乘法

矩阵乘法任务 `A = D × W^T`（`D ∈ R^(M×K)`，`W ∈ R^(N×K)`）是神经网络计算的核心。我们从循环网络（DeepSpeech2 [3]）和 Transformer [44] 中选择多种任务，并在图 8 中报告性能。

Triton 与 cuBLAS 通常性能相当，达到设备峰值性能的 90% 以上；在某些任务上，Triton 还比其他 DSL 快最多 3 倍。

![矩阵乘法性能（原文图 8）](/images/materials/cmu-15-779/triton/figure-08.webp)

不过，在浅层 Transformer 网络上，cuBLAS 仍然更快，因为它使用了三维算法 [2]，把较深的归约拆分为独立块，在 `M` 和 `N` 较小时提供更多并行性。除此之外，现有 DSL 通常比 Triton 慢 2–3 倍；当输入形状是 32 的倍数时，TVM 的差距小于 2 倍。

### 6.2 卷积

卷积神经网络是机器学习模型的重要类别，DSL 和编译器应当很好地支持它们。CNN 以卷积层为基础（图 9a）；为了利用专用张量处理硬件，必须把卷积实现为矩阵乘法（图 9b），但现有 DSL 通常不支持这一点。

![稠密和移位卷积的实现（原文图 9）](/images/materials/cmu-15-779/triton/figure-09.webp)

我们测试了 Triton 对 cuDNN `IMPLICIT_GEMM` 算法的重新实现（第 6.2.1 节），并提供了第一个可用的 shift convolution 融合 kernel（第 6.2.2 节）。这些程序使用指针增量查找表实现，如代码清单 8 所示。

#### 6.2.1 稠密卷积

本小节考虑深度学习文献中的卷积层，任务列在表 1 中。图 10 显示，Triton 在 ResNet 上超过了 cuDNN 的 `IMPLICIT_GEMM` 实现。

![隐式矩阵乘法性能（原文图 10）](/images/materials/cmu-15-779/triton/figure-10.webp)

可能的原因是 cuDNN 对 3×3 卷积还维护了更好的算法，例如 Winograd [25]，因此没有投入很多工程资源去优化重要性较低的 kernel。在没有快速算法可用的任务（如 DeepSpeech2）上，cuDNN 与 Triton 性能相当。

#### 6.2.2 Shift convolution

最后，我们把表 1 中的 Task 1–5 作为 shift convolution（图 9a）进行实现。这是 CNN 的一种新方法。我们将 Triton 中融合的 shift-conv 模块与一种朴素实现比较：后者依赖手写 shift kernel，再单独调用 cuBLAS。

我们还报告了不执行 shift 时的最大可达到性能，即 1×1 卷积的性能。如图 11 所示，Triton 实现几乎完全隐藏了 shift 操作的代价。

![Triton 移位卷积性能（原文图 11）](/images/materials/cmu-15-779/triton/figure-11.webp)

## 7 结论

本文提出 Triton：一种开源语言和编译器，用于表达并把分块神经网络计算编译为高效机器代码。

我们展示了，只需向 LLVM-IR 增加少量数据流和控制流扩展，就可以实现多种 tile 级优化 pass；这些 pass 联合起来，可以达到接近厂商库的性能。我们还提出了更高层的 Triton-C 语言，并使用它为 CNN 中的新型神经网络架构简洁地实现了高效 kernel。

未来工作包括支持 tensor core、实现量化 kernel [26]，以及集成到更高层 DSL 中。

## 8 致谢

本工作得到 NVIDIA Graduate Fellowship 支持。

## 参考文献

参考文献 [1]–[48] 的书目信息保留原文，便于检索原始论文、库和工具；正文中的引用编号与原文一致。
