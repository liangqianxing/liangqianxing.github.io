# TensorFlow：面向大规模机器学习的系统

> 翻译说明：源 PDF 共 21 页（第 1 页为会议封面/版权页，第 2–15 页为正文，第 16–20 页为参考文献，第 21 页为空白）。本文按原文顺序翻译正文、图表说明、公式、代码、致谢和全部 76 条参考文献；引用编号保持不变。

马丁·阿巴迪、保罗·巴勒姆、简敏·陈、志峰·陈、安迪·戴维斯、杰弗里·迪恩、马蒂厄·德温、桑杰·格玛瓦特、杰弗里·欧文、迈克尔·伊萨德、曼努吉·库德勒、乔什·莱文伯格、拉贾特·蒙加、雪莉·摩尔、德里克·G·默里、贝努瓦·施泰纳、保罗·塔克、维杰·瓦苏代万、皮特·沃登、马丁·维克、袁宇、晓强·郑（Google Brain）

论文发表于第 12 届 USENIX 操作系统设计与实现研讨会（OSDI ’16），2016 年 11 月 2–4 日，美国萨凡纳。原文软件地址：https://tensorflow.org。

本文收录于第 12 届 USENIX 操作系统设计与实现研讨会论文集；论文集的开放获取由 USENIX 赞助。

## 摘要

TensorFlow 是一个面向大规模和异构环境运行的机器学习系统。TensorFlow 使用数据流图表示计算、共享状态以及修改这些状态的操作。它把数据流图中的节点映射到集群中的多台机器上，并在单台机器内映射到多种计算设备，包括多核 CPU、通用 GPU，以及称为张量处理单元（Tensor Processing Unit，TPU）的专用 ASIC。这种架构给应用开发者带来了灵活性：以往“参数服务器”架构把共享状态的管理固化在系统中，而 TensorFlow 允许开发者尝试新的优化和训练算法。

TensorFlow 支持多种应用，重点是深度神经网络的训练和推理。Google 的多项服务已经在生产环境使用 TensorFlow；我们将其作为开源项目发布后，它也被机器学习研究广泛采用。本文介绍 TensorFlow 的数据流模型，并展示它在多个真实应用中取得的出色性能。

## 1 引言

近年来，机器学习推动了许多领域的发展 [3, 5, 24, 25, 29, 31, 42, 47, 50, 52, 57, 67, 68, 72, 76]。我们将这种成功归因于几个因素：更复杂机器学习模型的发明 [44, 54]；用于解决这些领域问题的大规模数据集 [9, 64]；以及能够在这些数据集上训练模型、并方便地使用大量计算资源的软件平台 [14, 20]。

我们开发 TensorFlow，是为了在大规模数据集上试验新模型、进行训练，并把模型部署到生产环境。这个系统建立在我们使用第一代系统 DistBelief [20] 多年的经验之上，同时对其进行了简化和泛化，使研究人员能够相对容易地探索更广泛的想法。TensorFlow 同时支持大规模训练和推理：它可以高效利用数百台带 GPU 的强大服务器进行快速训练，也可以在多种生产平台上运行训练好的模型，范围从数据中心的大型分布式集群一直到移动设备本地。与此同时，它保持了足够的灵活性，能够支持新机器学习模型和系统级优化的实验。

TensorFlow 使用统一的数据流图表示算法中的计算以及算法所操作的状态。我们借鉴了数据流系统的高级编程模型 [2, 21, 34] 和参数服务器的底层效率 [14, 20, 49]。传统数据流系统中的图顶点表示在不可变数据上执行的函数式计算，而 TensorFlow 允许顶点表示拥有或更新可变状态的计算。边在节点之间传递张量（多维数组），TensorFlow 会透明地插入分布式子计算之间所需的通信。把计算与状态管理统一到同一种编程模型中，使程序员能够尝试不同的并行化方案，例如把计算卸载到保存共享状态的服务器上，从而减少网络流量。我们还构建了多种协调协议，并在同步复制方面取得了令人鼓舞的结果；这些结果与“可扩展学习必须使用异步复制”的传统看法相矛盾 [10, 18]。

过去一年中，Google 有 150 多个团队使用了 TensorFlow，我们也将其发布为开源项目。¹ 通过庞大的用户社区，我们积累了不同机器学习应用的经验。本文把神经网络训练视为一个具有挑战性的系统问题，并从中选择两个有代表性的应用：图像分类和语言建模。这两个应用分别对计算吞吐量和模型总大小构成压力。我们用它们来展示 TensorFlow 的可扩展性，并评估当前实现的效率和可扩展能力。

¹ 软件地址：https://tensorflow.org。

## 2 背景与动机

我们首先描述之前系统的局限（§2.1），然后概述开发 TensorFlow 时采用的设计原则（§2.2）。

### 2.1 之前的系统：DistBelief

TensorFlow 是 DistBelief 的后继者。DistBelief 是 Google 自 2011 年起用于训练神经网络的分布式系统 [20]。DistBelief 使用参数服务器架构；下面我们批评这种架构在 DistBelief 中表现出的局限。不过，其他基于参数服务器的系统也以不同方式解决了这些局限 [11, 14, 49]；我们将在 §2.3 讨论这些系统。

在参数服务器架构中，一个作业由两组互不相交的进程组成：无状态的 worker 进程执行训练模型时的大部分计算；有状态的参数服务器进程维护模型参数的当前版本。DistBelief 的编程模型与 Caffe [38] 类似：用户把神经网络定义为一个以损失函数结束的有向无环层图。一个层是数学运算符的组合。例如，全连接层把输入乘以权重矩阵，加上偏置向量，再对结果应用非线性函数（例如 sigmoid）。损失函数是一个标量函数，用于度量预测值（对于给定输入数据点）与真实标签之间的差异。在全连接层中，权重矩阵和偏置向量是参数，学习算法会更新它们，以最小化损失函数。DistBelief 利用 DAG 结构以及对层语义的了解，通过反向传播 [63] 计算模型中每个参数的梯度。由于许多算法中的参数更新具有交换性，且只要求较弱的一致性 [61]，worker 进程可以独立计算更新，然后把“增量”更新写回参数服务器，由参数服务器将更新与当前状态合并。

尽管 DistBelief 让许多 Google 产品能够使用深度神经网络，并成为多个机器学习研究项目的基础，但我们很快感到它存在局限。它基于 Python 的脚本接口用于组合预定义层，对于需求简单的用户已经足够；然而，更高级的用户还需要三类灵活性：

**定义新层。** 为了提高效率，我们把 DistBelief 的层实现为 C++ 类。使用另一种并不熟悉的语言来实现层，会阻碍机器学习研究人员试验新的层结构，例如 sampled softmax 分类器 [37] 和注意力模块 [53]。

**改进训练算法。** 许多神经网络使用随机梯度下降（SGD）训练：算法迭代地沿着使损失函数下降最快的方向移动网络参数。SGD 的若干改进通过改变更新规则来加速收敛 [23, 66]。研究人员经常希望试验新的优化方法，但在 DistBelief 中这需要修改参数服务器实现。此外，参数服务器的 get()/put() 接口并不适合所有优化方法：有时必须原子地更新一组相关参数；很多情况下，把计算卸载到参数服务器上会更高效，因为这样可以减少网络流量。

**定义新的训练算法。** DistBelief 的 worker 遵循固定执行模式：读取一批输入数据和当前参数值，计算损失函数（对网络做前向传播），计算各个参数的梯度（反向传播），再把梯度写回参数服务器。这个模式适用于简单的前馈神经网络，却无法处理更高级的模型，例如包含循环的循环神经网络 [39]；需要交替训练两个相关网络的对抗网络 [26]；以及损失函数由另一个系统中的智能体（如电子游戏模拟器）计算的强化学习模型 [54]。此外，还有许多不符合神经网络训练模式的机器学习算法——例如期望最大化、决策树训练和隐狄利克雷分配——它们同样可以从一个通用且优化良好的分布式运行时中受益。

我们还把 DistBelief 设计成只面向一种平台：大规模的多核服务器分布式集群。当 GPU 加速对高效执行卷积核变得至关重要时 [44]，我们为 DistBelief 增加了 GPU 支持；但是 DistBelief 仍然是一个重量级系统，面向在海量数据集上训练深度神经网络，很难扩展到其他环境。许多用户希望先在带 GPU 的工作站上本地打磨模型，然后用同一份代码扩展到更大的数据集上训练。模型在集群上训练完成后，下一步通常是投入生产：可能是把模型集成到在线服务中，也可能是部署到移动设备进行离线执行。这些任务在计算结构上有共同点，但我们的同事发现，有必要使用或创建满足不同平台性能和资源要求的独立系统。TensorFlow 为所有这些环境提供了统一的编程模型和运行时系统。

### 2.2 设计原则

我们设计 TensorFlow 时，希望它比 DistBelief 灵活得多，同时保留满足 Google 生产机器学习工作负载的能力。TensorFlow 提供了简单的、基于数据流的编程抽象，使用户可以把应用部署到分布式集群、本地工作站、移动设备和专用加速器上。一个高级脚本接口（图 1）封装了数据流图的构建，用户可以试验不同的模型结构和优化算法，而无需修改核心系统。本小节简要介绍 TensorFlow 的核心设计原则。

**由基本运算符构成的数据流图。** TensorFlow 和 DistBelief 都用数据流表示模型，但最显著的差异在于：DistBelief 模型由少量复杂的“层”组成，而对应的 TensorFlow 模型把单个数学运算符（例如矩阵乘法、卷积等）作为数据流图中的节点。这种方式让用户更容易通过高级脚本接口组合出新层。许多优化算法要求每一层都定义梯度；用简单运算符构建层，可以自动对模型求导（§4.1）。除了函数式运算符外，我们还把可变状态及更新状态的操作表示为数据流图节点，从而可以试验不同的更新规则。

**延迟执行。** 一个典型的 TensorFlow 应用有两个不同阶段：第一阶段把程序（例如要训练的神经网络及其更新规则）定义成符号数据流图，其中包含输入数据的占位符和表示状态的变量；第二阶段在一组可用设备上执行程序的优化版本。把执行延迟到整个程序都可用后，TensorFlow 可以利用关于计算的全局信息优化执行。例如，TensorFlow 利用图的依赖关系向 GPU 发出一连串内核，在等待中间结果之前继续发射，从而实现较高的 GPU 利用率。这个设计选择也意味着，我们必须把动态控制流等更复杂特性（§3.4）放入数据流图中，使使用这些特性的模型同样受益于这些优化。

**异构加速器的统一抽象。** 除多核 CPU、GPU 等通用设备外，深度学习专用加速器也能显著提升性能并降低功耗。在 Google，我们的同事专门为机器学习构建了张量处理单元（TPU）；与当时其他先进技术相比，TPU 的每瓦性能提高了一个数量级 [40]。为了在 TensorFlow 中支持这些加速器，我们定义了统一的设备抽象。设备至少必须实现以下方法：（i）发出内核执行请求；（ii）为输入和输出分配内存；（iii）在主机内存与设备之间传输缓冲区。每个运算符（例如矩阵乘法）可以为不同设备提供多个专门实现。因此，同一程序可以根据训练、服务和离线推理的需要，轻松地面向 GPU、TPU 或移动 CPU。

TensorFlow 使用基本类型值组成的张量作为所有设备都能理解的通用交换格式。在最低层，TensorFlow 中的所有张量都是稠密的；稀疏张量可以用稠密张量表示（§3.1）。这一决定让系统底层的内存分配和序列化实现保持简单，从而减少框架开销。张量还支持其他内存管理和通信优化，例如 RDMA 以及 GPU 到 GPU 的直接传输。

这些原则的主要结果是：TensorFlow 中不存在所谓的参数服务器。在集群上，我们把 TensorFlow 部署为一组 task（能够通过网络通信的进程）；每个 task 都导出相同的图执行 API，并包含一个或多个设备。通常，一部分 task 承担其他系统中参数服务器的角色 [11, 14, 20, 49]，因此称为 PS task；其余 task 称为 worker task。不过，由于 PS task 能够运行任意 TensorFlow 图，它比传统参数服务器更灵活：用户可以用定义模型时使用的同一个脚本接口对它进行编程。这种灵活性是 TensorFlow 与当时系统之间的关键差异；本文后面将讨论这种灵活性带来的应用。

### 2.3 相关工作

**单机框架。** 许多机器学习研究人员在单台（通常带 GPU 的）计算机上开展工作 [43, 44]，有多个单机框架支持这种场景。Caffe [38] 是一个高性能框架，用于在多核 CPU 和 GPU 上训练以声明方式指定的神经网络。如 §2.1 所述，它的编程模型与 DistBelief 相似，因此可以很容易地用已有层组合模型，但添加新层或新优化器相对困难。Theano [2] 允许程序员把模型表示为由基本运算符组成的数据流图，并生成训练该模型的高效编译代码。它的编程模型与 TensorFlow 最接近，在单机环境中也提供了类似的灵活性。

与 Caffe、Theano 和 TensorFlow 不同，Torch [17] 为科学计算和机器学习提供了强大的命令式编程模型。它允许细粒度地控制执行顺序和内存使用，使高级用户可以优化程序性能。这种灵活性对研究很有用，但 Torch 缺少数据流图作为一种可移植表示所带来的优势，因而无法自然地贯穿小规模实验、生产训练和部署。

**批处理数据流系统。** 从 MapReduce [21] 开始，批处理数据流系统被应用于大量机器学习算法 [70]；较新的系统则侧重于提高表达能力和性能。DryadLINQ [74] 增加了高级查询语言，支持比 MapReduce 更复杂的算法。Spark [75] 在 DryadLINQ 的基础上增加了把已经计算出的数据集缓存到内存中的能力，因此当输入数据能放入内存时，更适合迭代式机器学习算法（例如 k-means 聚类和逻辑回归）。Dandelion 为 DryadLINQ 增加了针对 GPU [62] 和 FPGA [16] 的代码生成能力。

批处理数据流系统的主要局限是：它要求输入数据不可变，并且所有子计算都必须是确定性的，这样系统才能在集群机器发生故障时重新执行子计算。对于很多传统工作负载，这个特性很有益；但它让更新机器学习模型变得昂贵。例如，Spark 上的深度神经网络训练系统 SparkNet 每次向五个 worker 广播权重并收集更新需要 20 秒 [55]。因此，这类系统中的每一步模型更新都必须处理更大的批次，从而减慢收敛 [8]。我们在 §6.3 中展示，TensorFlow 可以在更大的集群上训练更大的模型，步时间最低可达 2 秒。

**参数服务器。** 如 §2.1 所述，参数服务器架构使用一组服务器管理共享状态，多个并行 worker 对其进行更新。这种架构源自可扩展主题模型的研究 [65]；DistBelief 说明它可以用于深度神经网络训练。Project Adam [14] 进一步使用该架构高效训练卷积神经网络；Li 等人的“Parameter Server”[49] 在一致性模型、容错和弹性扩展方面进行了创新。尽管早期有人怀疑参数服务器与 GPU 加速不兼容 [14]，Cui 等人最近表明，专门面向 GPU 的参数服务器可以在小型集群上获得加速 [18]。

MXNet [11] 也许是设计上与 TensorFlow 最接近的系统。它使用数据流图表示每个 worker 上的计算，并使用参数服务器扩展到多台机器。MXNet 参数服务器导出键值存储接口，支持聚合 worker 中多个设备发来的更新，并使用任意用户提供的函数把传入更新与当前值合并。MXNet 的键值存储接口 [22] 目前不允许在单个值内进行稀疏梯度更新，而这对于大模型分布式训练至关重要（§4.2）；增加该功能需要修改核心系统。

参数服务器架构满足了我们的许多要求；只要投入足够的工程工作，就可以把本文描述的大部分功能加入参数服务器。不过，在设计 TensorFlow 时，我们寻找的是一种高级编程模型，让用户能够定制系统各个部分中运行的代码，从而降低试验新优化算法和模型结构的成本。下一节将更详细地介绍 TensorFlow 程序的构件。

## 3 TensorFlow 执行模型

TensorFlow 使用单个数据流图表示机器学习算法中的全部计算和状态，包括单个数学运算、参数及参数更新规则，以及输入预处理（图 2）。数据流图显式表达子计算之间的通信，因此可以很容易地并行执行相互独立的计算，并把计算划分到多个设备上。TensorFlow 与批处理数据流系统（§2.3）有两点不同：

* 模型支持在整个图的重叠子图上进行多次并发执行。
* 单个顶点可以包含可变状态，并可在图的不同执行之间共享。

参数服务器架构 [14, 20, 49] 的关键观察是：训练超大模型时，可变状态至关重要，因为它可以对超大参数进行原地更新，并尽快把更新传播到并行训练步骤。带可变状态的数据流让 TensorFlow 能够模拟参数服务器的功能，同时提供额外灵活性：可以在保存共享模型参数的机器上执行任意数据流子图。因此，用户能够试验不同的优化算法、一致性方案和并行化策略。

### 图 1：使用 TensorFlow Python API 编写的图像分类器

该程序是 MNIST 手写数字分类问题 [48] 的简单解决方案，输入为 784 像素图像，输出为 10 个类别。

```python
# 1. 构造表示模型的图。
x = tf.placeholder(tf.float32, [BATCH_SIZE, 784])                     # 输入占位符。
y = tf.placeholder(tf.float32, [BATCH_SIZE, 10])                      # 标签占位符。

W_1 = tf.Variable(tf.random_uniform([784, 100]))                      # 784x100 权重矩阵。
b_1 = tf.Variable(tf.zeros([100]))                                    # 100 元素偏置向量。
layer_1 = tf.nn.relu(tf.matmul(x, W_1) + b_2)                         # 隐藏层输出。

W_2 = tf.Variable(tf.random_uniform([100, 10]))                       # 100x10 权重矩阵。
b_2 = tf.Variable(tf.zeros([10]))                                     # 10 元素偏置向量。
layer_2 = tf.matmul(layer_1, W_2) + b_2                               # 线性层输出。

# 2. 添加表示优化算法的节点。
loss = tf.nn.softmax_cross_entropy_with_logits(layer_2, y)
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

# 3. 在输入数据批次上执行图。
with tf.Session() as sess:                                            # 连接到 TensorFlow 运行时。
  sess.run(tf.initialize_all_variables())                             # 随机初始化权重。
  for step in range(NUM_STEPS):                                       # 迭代 NUM_STEPS 次训练。
    x_data, y_data = ...                                              # 加载一批输入数据。
    sess.run(train_op, {x: x_data, y: y_data})                        # 执行一次训练步骤。
```

图 2 给出了一个典型训练应用的数据流图，其中有多个并发执行并通过共享变量和队列交互的子图。核心训练子图依赖一组模型参数以及来自队列的输入批次。训练子图的多个并发步骤根据不同输入批次更新模型，从而实现数据并行训练。为了填充输入队列，并发的预处理步骤会变换单条输入记录（例如解码图像并施加随机失真）；独立的 I/O 子图从分布式文件系统中读取记录。检查点子图会定期运行，以提供容错能力（§4.3）。

## 图 2：训练流水线的数据流图

该图包含读取输入数据、预处理、训练和检查点状态等子图。图中的 `Input data`、`Reader`、`Preprocessing`、`Training`、`Periodic checkpoint`、`Dist. FS` 等标签分别表示输入数据、读取器、预处理、训练、周期性检查点和分布式文件系统；`Queue`、`Parameters`、`Read params`、`Apply grads`、`Fwd`、`Back`、`Mod`、`Part`、`Stitch`、`Sum` 等为相应操作或张量。

### 3.1 数据流图元素

在 TensorFlow 图中，每个顶点表示一个本地计算单元，每条边表示顶点的输出或输入。我们把顶点处的计算称为**操作**，沿边流动的值称为**张量**。本小节介绍常见的操作和张量类型。

**张量。** 在 TensorFlow 中，所有数据都建模为张量（n 维数组），元素采用少数几种基本类型之一，例如 `int32`、`float32` 或 `string`（其中 `string` 可以表示任意二进制数据）。张量自然地表示许多机器学习算法中常见数学运算的输入和结果：例如，矩阵乘法接收两个二维张量并生成一个二维张量；批量二维卷积接收两个四维张量并生成另一个四维张量。

在最低层，所有 TensorFlow 张量都是稠密的，原因见 §2.2。TensorFlow 提供两种表示稀疏数据的替代方式：要么把数据编码成稠密张量中的变长字符串元素，要么使用稠密张量的元组。例如，一个有 m 个非零元素的 n 维稀疏张量，可以用坐标列表格式表示为一个 m×n 的坐标矩阵和一个长度为 m 的值向量。张量的形状可以在一个或多个维度上变化，因此能够表示元素数量不同的稀疏张量。

**操作。** 一个操作接收 m≥0 个张量作为输入，并产生 n≥0 个张量作为输出。操作有一个命名的“类型”（例如 `Const`、`MatMul` 或 `Assign`），还可以有零个或多个在编译时确定行为的属性。操作可以在编译时具有多态性和可变参数个数：其属性决定输入和输出的类型以及数量。

例如，最简单的操作 `Const` 没有输入且只有一个输出；它的值是一个编译时属性。`AddN` 对多个元素类型相同的张量求和，它有一个类型属性 `T` 和一个整数属性 `N`，共同定义其类型签名。

**有状态操作：变量。** 每次执行操作时，它都可以读取或写入可变状态。`Variable` 操作拥有一个可变缓冲区，可以在训练模型时存储共享参数。`Variable` 没有输入，并产生一个引用句柄；这个句柄是一种带类型的能力，用来读取和写入缓冲区。`Read` 操作接收引用句柄 `r`，并把变量的值 `State[r]` 作为稠密张量输出。其他操作会修改底层缓冲区。例如，`AssignAdd` 接收引用句柄 `r` 和张量值 `x`，执行时进行更新：

\[
State'[r] \leftarrow State[r] + x。
\]

后续的 `Read(r)` 操作将产生值 `State'[r]`。

**有状态操作：队列。** TensorFlow 包含多种队列实现，支持更高级的协调形式。最简单的队列是 `FIFOQueue`：它拥有一个内部张量队列，并按照先进先出顺序允许并发访问。其他队列类型按随机顺序或优先级顺序出队，从而保证以适当方式采样输入数据。与 `Variable` 类似，`FIFOQueue` 操作产生一个引用句柄；这个句柄可以被标准队列操作（例如 `Enqueue` 和 `Dequeue`）消费。`Enqueue` 把输入推入队列尾部，`Dequeue` 弹出队列头部元素并输出。当给定队列已满时，`Enqueue` 会阻塞；当队列为空时，`Dequeue` 会阻塞。

在输入预处理流水线中使用队列时，这种阻塞提供了背压；它还支持同步（§4.4）。队列与动态控制流（§3.4）结合后，也可以在子图之间实现一种流式计算。

### 3.2 部分执行与并发执行

TensorFlow 使用数据流图表示一个应用中所有可能的计算。执行图的 API 允许客户端以声明方式指定应执行的子图。客户端选择一条或多条边，把输入张量送入数据流；同时选择一条或多条边，从数据流中取出输出张量。运行时随后裁剪图，只保留必要的操作。API 的每次调用称为一个**step**，TensorFlow 支持在同一图上并发执行多个 step。有状态操作允许不同 step 在需要时共享数据并进行同步。

部分执行和并发执行是 TensorFlow 灵活性的主要来源。通过在用户级代码中加入可变状态和利用队列进行协调，可以指定多种模型结构；高级用户无需修改 TensorFlow 运行时内部实现即可进行试验。默认情况下，TensorFlow 子图的并发执行彼此异步。这种异步性使实现具有弱一致性要求的机器学习算法变得直接 [61]，其中包括许多神经网络训练算法 [20]。如后文所述，TensorFlow 也提供了训练期间同步 worker 所需的原语（§4.4），并在一些学习任务上取得了有希望的结果（§6.3）。

### 3.3 分布式执行

数据流使分布式执行变得简单，因为它显式表达了子计算之间的通信。同一个 TensorFlow 程序可以部署到 GPU 集群上进行训练、TPU 集群上提供服务，也可以部署到手机上进行移动推理。

每个操作驻留在特定设备上，例如某个 task 中的 CPU 或 GPU。设备负责为分配给它的每个操作执行一个内核。TensorFlow 允许为单个操作注册多个内核，为特定设备或数据类型提供专门实现（详见 §5）。对于许多操作（例如逐元素运算 `Add`、`Sub` 等），我们可以使用不同编译器为 CPU 和 GPU 编译同一个内核实现。

TensorFlow 运行时会把操作放置到设备上，同时遵守图中的隐式或显式约束。放置算法为每个操作计算可行设备集合，计算必须放置在同一设备上的操作集合，并为每个共置组选择一个满足约束的设备。每个有状态操作及其状态必须位于同一设备上，这是算法必须遵守的隐式共置约束。此外，用户可以指定部分设备偏好，例如“某个 task 中的任意设备”或“任意 task 中的 GPU”，运行时会遵循这些约束。典型训练应用会使用客户端编程结构增加约束，例如把参数分布到一组“PS” task 上（§4.2）。

因此，TensorFlow 允许以很大的灵活性把数据流图中的操作映射到设备上。简单启发式算法对普通用户可以提供足够性能；而专家用户则可以手动放置操作，在多个 task 之间以及每个 task 内的多个设备之间平衡计算、内存和网络需求。一个开放问题是：如何让 TensorFlow 自动确定给定设备集合上的近似最优放置，从而让用户无需处理这一问题。即使没有这种自动化，把放置指令与模型定义的其他部分分离也可能有价值，例如训练完模型后可以很容易地修改放置位置。

图中的操作完成放置、并针对一个 step 计算出部分子图（§3.2）后，TensorFlow 会把操作划分为每个设备一个子图。设备 d 的子图包含所有分配给 d 的操作，并用额外的 `Send` 和 `Recv` 操作替换跨设备边。`Send` 在张量可用时立即把其唯一输入传输到指定设备，并使用 rendezvous key 命名该值。`Recv` 有一个输出，在本地指定 rendezvous key 对应的值可用之前阻塞；值到达后，它才产生输出。`Send` 和 `Recv` 针对若干设备类型对有专门实现；第 5 节将介绍其中一些。

我们针对低延迟、重复执行大型子图优化了 TensorFlow。一个 step 的图被裁剪、放置并分区后，其子图会缓存在相应设备上。客户端 session 维护 step 定义到缓存子图的映射，因此在大型图上启动一次分布式 step，只需向每个参与的 task 发送一条小消息。这种模型偏好静态、可复用的图，但通过动态控制流仍然能够支持动态计算，下一小节将对此进行说明。

### 3.4 动态控制流

TensorFlow 支持包含条件和迭代控制流的高级机器学习算法。例如，循环神经网络（RNN）[39]，包括 LSTM [32]，可以根据序列数据生成预测。Google 的神经机器翻译系统使用 TensorFlow 训练深层 LSTM，在多个翻译任务上达到最先进性能 [73]。RNN 的核心是一个递推关系：序列元素 i 的输出是某种状态的函数，而该状态沿序列累积（图 3）。动态控制流使得模型能够遍历长度可变的序列，而无需把计算展开到最长序列的长度。

如 §2.2 所述，TensorFlow 通过数据流图延迟执行，把更大的工作块卸载到加速器。因此，为了实现 RNN 和其他高级算法，我们把条件（if 语句）和迭代（while 循环）编程结构本身加入数据流图。我们利用这些原语构建 `map()`、`fold()` 和 `scan()` 等高阶结构 [2]。

为此，我们借用了经典动态数据流架构中的 `Switch` 和 `Merge` 原语 [4]。`Switch` 是一个分路器：它接收一个数据输入和一个控制输入，并用控制输入选择两个输出中的哪一个产生值。未被选择的 `Switch` 输出会收到一个特殊的 dead 值，该值递归地沿图传播，直到到达 `Merge` 操作。`Merge` 是一个多路复用器：它把至多一个非 dead 输入转发到输出；如果两个输入都是 dead，则产生 dead 输出。条件运算符使用 `Switch`，根据布尔张量的运行时值执行两个分支之一，再使用 `Merge` 合并分支输出。while 循环更加复杂，它使用 `Enter`、`Exit` 和 `NextIteration` 操作保证循环格式良好 [56]。

迭代的执行可以重叠，TensorFlow 还可以把条件分支和循环体划分到多个设备和进程上。分区步骤增加逻辑，用于协调每台设备上每次迭代的开始和结束，并决定循环终止条件。如 §4.1 所述，TensorFlow 也支持对控制流结构自动求导。自动求导会把计算梯度的子图加入数据流图；TensorFlow 可以把这些子图划分到潜在的分布式设备上，并行计算梯度。

### 图 3：抽象 RNN 的伪代码（§3.4）

```text
input = ...        # 张量序列
state = 0          # 初始状态
w = ...            # 可训练权重
for i in range(len(input)):
    state, out[i] = f(state, w, input[i])
```

函数 `f` 通常由可微运算组成，例如矩阵乘法和卷积 [32]。TensorFlow 在数据流图中实现这一循环。

## 4 可扩展性案例

通过为 TensorFlow 中的所有计算选择统一表示，我们让用户能够试验 DistBelief 运行时中原本硬编码的功能。本节讨论我们使用数据流原语和“用户级”代码构建的四种扩展。

### 4.1 求导与优化

许多学习算法使用某种 SGD 变体训练一组参数：先计算损失函数相对于这些参数的梯度，再根据梯度更新参数。TensorFlow 包含一个用户级库，可以对损失函数的符号表达式求导，并生成表示梯度的新符号表达式。例如，给定由多层组合而成的神经网络和一个损失函数，该库会自动推导出反向传播代码。

求导算法执行广度优先搜索，识别从目标操作（例如损失函数）到参数集合的所有反向路径，并把每条路径贡献的部分梯度相加。我们的用户经常为某些操作专门定制梯度，并实现了批归一化 [33] 和梯度裁剪 [60] 等优化，以加速训练并提高鲁棒性。我们扩展了该算法，使其能够对条件和迭代子计算求导（§3.4）：具体做法是在图中添加节点，记录前向传播中的控制流决策，并在反向传播中反向重放这些决策。对长序列上的迭代计算求导，可能导致大量中间状态累积在内存中；针对这类计算，我们开发了管理有限 GPU 内存的技术。

TensorFlow 用户还可以试验多种优化算法，这些算法在每个训练步骤中计算参数的新值。在参数服务器中实现 SGD 很简单：对于参数 W、梯度 ∂L/∂W 和学习率 α，更新规则为

\[
W' \leftarrow W - \alpha \times \frac{\partial L}{\partial W}。
\]

参数服务器可以使用 `-=` 作为写操作，并在每个训练步骤后把 α×∂L/∂W 写入每个 W。

但许多更高级的优化方案很难表示为一次写操作。例如，Momentum 算法根据参数在多个迭代中的梯度为每个参数累积一个“速度”，再根据该累积量计算参数更新；人们还提出了该算法的许多改进 [66]。在 DistBelief [20] 中实现 Momentum，需要修改参数服务器实现，改变参数数据的表示方式，并在写操作中执行复杂逻辑；这对许多用户来说很有挑战。优化算法是一个活跃的研究领域，研究人员已经在 TensorFlow 上实现了 Momentum、AdaGrad、AdaDelta、RMSProp、Adam 和 L-BFGS 等算法。这些算法可以通过 `Variable` 操作和基本数学运算在 TensorFlow 中构建，而无需修改底层系统，因此当新算法出现时，可以很容易地进行试验。

### 4.2 训练超大模型

为了在文本语料库中的单词等高维数据上训练模型，通常使用分布式表示：一个训练样本被表示为多个神经元上的活动模式，并通过反向传播学习这种表示 [30]。例如，在语言模型中，一个训练样本可能是一个稀疏向量，词表中对应单词 ID 的位置为非零值；每个单词的分布式表示是一个低维向量 [6]。“宽而深学习”从类别特征的交叉积变换中创建分布式表示；基于 TensorFlow 的实现被用于 Google Play 应用商店推荐系统 [12]。

推理过程把一批 b 个稀疏向量与 n×d 嵌入矩阵相乘，其中 n 是词表中的单词数、d 是期望维度，结果是一个更小的 b×d 稠密矩阵表示。训练时，大多数优化算法只会修改稀疏乘法所读取的嵌入矩阵行。在 TensorFlow 中，处理稀疏数据的模型可能有 n×d GB 级参数：例如，大型语言模型可能使用含 80 万个单词的词表，并包含超过 10^9 个参数 [41]；我们还处理过参数占用数 TB 的文档模型 [19]。这类模型太大，既无法每次使用时复制到 worker，甚至无法存放在单台主机的 RAM 中。

我们把稀疏嵌入层实现为基本操作的组合。图 4 展示了一个简化图：嵌入层被划分到两个参数服务器 task。该子图的核心操作是 `Gather`，它从张量中提取一组稀疏行；TensorFlow 会把这一操作与其操作的变量共置。动态分区（`Part`）操作把输入索引划分成大小可变的张量，每个张量包含发往一个分片的索引；动态合并拼接（`Stitch`）操作把各分片的部分结果重新组装成一个结果张量。这些操作分别有对应的梯度，因此支持自动求导（§4.1）；最终得到的一组稀疏更新操作只作用于最初从各分片提取的那些值。

用户通常不会手动构造类似图 4 的图。TensorFlow 提供了表示分片参数的抽象库，并根据期望的分布程度构建由基本操作组成的适当图。

虽然参数服务器也可以实现稀疏读取和更新 [49]，但 TensorFlow 增加了把任意计算卸载到保存共享参数的设备上的灵活性。例如，分类模型通常使用 softmax 分类器，把最终输出乘以一个有 c 列的权重矩阵，其中 c 是可能类别数；语言模型中 c 就是词表大小，可能非常大。我们的用户试验了多种加速 softmax 计算的方案。第一种与 Project Adam [14] 中的优化类似：把权重分片到多个 task，并把乘法和梯度计算与分片共置。使用 sampled softmax 可以更高效地训练 [37]：它根据样本的真实类别和一组随机采样的错误类别执行稀疏乘法。我们在 §6.4 比较了这两种方案的性能。

### 图 4：采用双向分片嵌入矩阵的嵌入层稀疏更新数据流

图中 `Shard 0`、`Shard 1` 表示两个分片，`Gather` 表示提取行，`Mod`、`Part`、`Stitch`、`Sum` 为相应基本操作；`True branch` 与 `False branch` 表示条件分支。

### 4.3 容错

即使使用大量机器，训练一个模型也可能需要数小时或数天 [14, 20]。我们经常需要使用非专用资源训练模型，例如使用 Borg 集群管理器 [71]；Borg 不保证在整个训练期间都能提供同一批资源。因此，一个运行时间很长的 TensorFlow 作业很可能遭遇故障或抢占，我们需要某种容错机制。

task 不太可能频繁失败到需要让单个操作具备容错能力，因此 Spark 的 RDD [75] 那样的机制会带来很大开销，却几乎没有收益。也没有必要让参数状态的每次写入都持久化，因为我们可以根据输入数据重新计算任何更新，而且许多学习算法并不要求强一致性 [61]。

我们在用户层实现了基于检查点的容错，使用图中的两个操作（图 2）：`Save` 把一个或多个张量写入检查点文件，`Restore` 从检查点文件读取一个或多个张量。典型配置是：把一个 task 中的每个 `Variable` 连接到同一个 `Save` 操作；每个 task 使用一个 `Save`，以最大化对分布式文件系统的 I/O 带宽。`Restore` 操作从文件读取命名张量，标准的 `Assign` 操作再把恢复值存入相应变量。训练期间，典型客户端定期运行所有 `Save` 操作，生成新的检查点；客户端启动时，则尝试恢复最新检查点。

TensorFlow 包含一个客户端库，用于构造所需的图结构并在需要时调用 `Save` 和 `Restore`。这种行为可以定制：用户可以对模型中的变量子集应用不同策略，也可以自定义检查点保留方案。例如，许多用户会保留自定义评估指标得分最高的检查点。该实现还可以复用到模型微调和无监督预训练 [45, 47] 中；这两者都是迁移学习形式，即把在一个任务上训练的模型参数作为另一个任务的起点（例如，先训练识别一般图像，再训练识别不同犬种）。把检查点和参数管理作为图中的可编程操作，使用户能够实现这类方案以及我们尚未预见的其他方案。

检查点库不尝试生成一致的检查点：如果训练与检查点操作并发执行，检查点可能包含训练步骤中的零个、全部或部分更新。这种行为与异步 SGD 的宽松保证兼容 [20]。一致检查点需要额外同步，确保更新操作不会干扰检查点；如果需要，可以使用下一小节的方案，在同步更新步骤之后保存检查点。

### 4.4 同步副本协调

SGD 对异步性具有鲁棒性 [61]，许多系统使用异步参数更新训练深度神经网络 [14, 20]。人们认为异步训练具有可扩展性，因为即使存在落后 worker（straggler），也能维持较高吞吐量。提高吞吐量的代价是：训练步骤使用过时的参数值。最近的一些工作重新考察了“同步训练不可扩展”这一假设 [10, 18]。由于 GPU 让训练所需机器数量从数千台降至数百台 [47]，在同一平台上，同步训练达到目标质量的时间可能比异步训练更短。

尽管我们最初为异步训练设计 TensorFlow，但也开始试验同步方法。TensorFlow 图允许用户改变训练期间读取和写入参数的方式；我们实现了三种替代方案。在异步方案中（图 5(a)），每个 worker 在每个 step 开始时读取参数当前值，并在结束时把梯度应用到（可能已经不同的）当前值上。这种方式保持较高利用率，但每个 step 使用过时参数值，因而单个 step 的效果较差。

我们使用队列（§3.1）实现同步版本来协调执行：阻塞队列充当屏障，确保所有 worker 读取相同的参数值；每个变量对应一个队列，累积所有 worker 的梯度更新，再原子地应用这些更新。简单同步版本（图 5(b)）会先累积所有 worker 的更新再应用，但慢 worker 会限制总体吞吐量。

为减轻落后 worker 的影响，我们实现了备用 worker（图 5(c)，见 [10]），其思想类似于 MapReduce 的备用任务 [21]。MapReduce 在检测到落后任务后才反应式地启动备用任务；我们的备用 worker 主动运行，聚合器接收 n 个更新中的前 m 个。我们利用了 SGD 每一步都会随机采样训练数据这一事实：每个 worker 处理不同的随机批次，因此丢弃某一个批次不会造成问题。§6.3 展示了备用 worker 如何把吞吐量提高最多 10%。

### 图 5：并行 SGD 的三种同步方案

（a）异步复制；（b）同步复制；（c）带备用 worker 的同步复制。每种颜色代表不同的起始参数值；白色方格表示参数更新。在（c）中，虚线矩形表示其结果会被丢弃的备用 worker。

## 5 实现

TensorFlow 运行时是一个跨平台库。图 6 展示了其架构：C API 把不同语言的用户级代码与核心运行时分开。

核心 TensorFlow 库用 C++ 实现，以兼顾可移植性和性能。它运行在 Linux、Mac OS X、Windows、Android 和 iOS 等操作系统上，支持 x86 和多种基于 ARM 的 CPU 架构，以及 NVIDIA 的 Kepler、Maxwell 和 Pascal GPU 微架构。该实现是开源的，我们已经接收了多个外部贡献，使 TensorFlow 能够运行在其他架构上。

分布式 master 把用户请求转换为一组 task 上的执行。给定一个图和 step 定义，它会裁剪（§3.2）并划分（§3.3）该图，为每个参与的设备生成子图，并缓存这些子图以便后续 step 复用。由于 master 能看到一个 step 的整体计算，它可以应用公共子表达式消除和常量折叠等标准优化；裁剪是一种死代码消除。然后，master 协调一组 task 上优化子图的执行。

每个 task 中的数据流执行器处理 master 发来的请求，并调度组成局部子图的内核执行。我们针对低开销执行大型图优化了数据流执行器。当前实现每秒可以执行 10,000 个子图（§6.2），从而支持大量副本以快速执行细粒度训练步骤。数据流执行器把内核派发到本地设备，并在可能时并行运行内核，例如使用多个 CPU 核心或多个 GPU stream。

运行时包含 200 多个标准操作，包括数学、数组操作、控制流和状态管理操作。许多操作内核使用 Eigen::Tensor [36] 实现；它使用 C++ 模板为多核 CPU 和 GPU 生成高效并行代码。不过，在有更高效内核实现的场合，我们也广泛使用 cuDNN [13] 等库。我们还实现了量化，使移动设备和数据中心高吞吐应用等环境能够更快地进行推理；并使用 gemmlowp 低精度矩阵库 [35] 加速量化计算。

我们针对每一种源设备类型和目标设备类型组合，专门实现 `Send` 和 `Recv` 操作。本地 CPU 与 GPU 设备之间的传输使用 `cudaMemcpyAsync()` API，使计算和数据传输重叠；两个本地 GPU 之间的传输使用 DMA，减轻主机压力。task 之间的传输使用多种协议，包括基于 TCP 的 gRPC，以及基于融合以太网的 RDMA。我们还在研究使用集合通信操作优化 GPU 到 GPU 的通信 [59]。

第 4 节描述的功能完全构建在 C API 之上的用户级代码中。用户通常组合标准操作，构建更高级的抽象，例如神经网络层、优化算法（§4.1）和分片嵌入计算（§4.2）。TensorFlow 支持多种客户端语言；由于内部用户最熟悉 Python 和 C++，我们优先支持这两种语言。某个功能变得稳定后，我们通常会把它移植到 C++，使所有客户端语言都能访问优化后的实现。

如果某个子计算难以或低效地表示为操作组合，用户可以注册额外内核，用 C++ 提供高效实现。我们发现，为性能关键操作手工实现融合内核很有价值，例如 ReLU 和 Sigmoid 激活函数及其对应梯度。我们目前正在研究基于编译的自动内核融合方法。

除核心运行时外，我们的同事还构建了多个帮助 TensorFlow 用户的工具。这些工具包括生产环境推理服务基础设施 [27]；用于跟踪训练运行进度的可视化仪表板；帮助用户理解模型连接关系的图可视化器；以及能够跨多个设备和 task 跟踪计算执行的分布式分析器。我们在扩展白皮书 [1] 中描述了这些工具。

### 图 6：分层 TensorFlow 架构

从上到下的主要组件包括训练库、推理库、Python/C++ 客户端、C API、分布式 master、数据流执行器、基本操作及其内核实现、RPC/RDMA 与 CPU/GPU 设备、网络层和设备层。

## 6 评测

本节在若干合成工作负载和真实工作负载上评估 TensorFlow 性能。除非另有说明，所有实验都在共享生产集群上运行；图中绘制中位数，误差条表示第 10 和第 90 百分位。

本文关注系统性能指标，而不是达到准确率所需时间等学习目标。TensorFlow 是一个允许机器学习实践者和研究人员试验新技术的系统；本评测展示了：（i）系统开销很小；（ii）系统可以利用大量计算来加速真实应用。同步复制等技术有时可以让模型用更少的总步骤收敛，但对这类改进的分析留给其他论文。

### 6.1 单机基准

虽然 TensorFlow 是“面向大规模”机器学习的系统，但可扩展性不能掩盖小规模下的糟糕性能 [51]。表 1 汇报了 Chintala 对 TensorFlow 及三个单机框架的卷积模型基准测试结果 [15]。所有框架都使用六核 3.5 GHz Intel Core i7-5930K CPU 和 NVIDIA Titan X GPU。

#### 表 1：使用不同库、单 GPU 训练四种卷积模型的步时间（毫秒）

| 库 | AlexNet | Overfeat | OxfordNet | GoogleNet |
|---|---:|---:|---:|---:|
| Caffe [38] | 324 | 823 | 1068 | 1935 |
| Neon [58] | 87 | 211 | 320 | 270 |
| Torch [17] | 81 | 268 | 529 | 470 |
| TensorFlow | 81 | 279 | 540 | 445 |

所有结果均使用 32 位浮点数训练；每个模型的最快时间在原表中以粗体标出。

表 1 表明，TensorFlow 的步时间比 Caffe [38] 更短，性能与最新版本 Torch [17] 相差不超过 6%。我们把 TensorFlow 与 Torch 的相近性能归因于两者使用相同版本的 cuDNN 库 [13]；该库实现了训练关键路径上的卷积和池化操作。Caffe 使用这些操作的开源实现，代码更简单，但效率低于 cuDNN。Neon 库 [58] 在其中三个模型上优于 TensorFlow，这是因为它使用汇编语言实现的手工优化卷积内核 [46]。原则上，我们也可以在 TensorFlow 中采用相同方法，但目前尚未这样做。

### 6.2 同步副本微基准

我们的协调实现（§4.4）是使用更多机器进行扩展时的主要限制因素。图 7 展示了不同模型大小和不同同步 worker 数量下，TensorFlow 每秒执行的空训练步骤数。在一个空训练步骤中，worker 从 16 个 PS task 获取共享模型参数，执行简单计算，再向参数发送更新。

图 7 中的 Scalar 曲线显示了同步训练步骤理论上能够达到的最佳性能，因为它只从每个 PS task 获取一个 4 字节值。单 worker 时，中位数步时间为 1.8 ms；100 个 worker 时增加到 8.8 ms。这些时间衡量了同步机制的开销，也包含在共享集群上运行时预期会出现的一些噪声。

Dense 曲线显示 worker 获取整个模型时空步骤的性能。我们分别使用大小为 100 MB 和 1 GB 的模型进行实验，并把参数均匀分片到 16 个 PS task。对于 100 MB 模型，中位数步时间从单 worker 的 147 ms 增加到 100 worker 的 613 ms；对于 1 GB 模型，则从单 worker 的 1.01 s 增加到 100 worker 的 7.16 s。

对于大型模型，典型训练步骤只访问参数子集；Sparse 曲线展示了 §4.2 中嵌入查找操作的吞吐量。每个 worker 从包含 1 GB 或 16 GB 数据的大型嵌入矩阵中随机选择 32 个条目。正如预期，步时间不随嵌入矩阵大小变化，TensorFlow 的步时间范围为 5–20 ms。

### 图 7：空模型同步复制的基线吞吐量

稀疏访问使 TensorFlow 能够处理更大的模型，例如嵌入矩阵（§4.2）。横轴为 worker 数量，纵轴为 batches/second；曲线分别表示 Scalar、Sparse 1GB、Sparse 16GB、Dense 100M 和 Dense 1GB。

### 6.3 图像分类

深度神经网络在图像中的目标识别等计算机视觉任务上取得了突破性性能 [44]；这些任务也是 TensorFlow 在 Google 的主要应用。训练一个高准确率网络需要大量计算，我们使用 TensorFlow 把计算扩展到由 GPU 服务器组成的集群上。在这些实验中，我们重点研究 Google 的 Inception-v3 模型。该模型在 ILSVRC 2012 图像分类挑战赛中达到了 78.8% 的准确率 [69]；同样的技术也适用于在 TensorFlow 上实现的其他深度卷积模型，例如 ResNet [28]。我们使用多个副本研究 Inception-v3 的训练可扩展性。TensorFlow 配置为 7 个 PS task，并在两个不同集群上改变 worker task 数量。

第一个实验比较 TensorFlow 和 MXNet（一个使用参数服务器架构的同期系统）在使用异步 SGD 训练 Inception 时的性能。实验使用 Google Compute Engine 虚拟机；虚拟机运行在配备 NVIDIA K80 GPU 的 Intel Xeon E5 服务器上，配置为 8 个 vCPU、16 Gbps 网络带宽，每台虚拟机一个 GPU。两个系统都使用运行在独立、无 GPU 虚拟机上的 7 个 PS task。图 8(a) 显示 TensorFlow 的性能略优于 MXNet。正如预期，结果主要由单 GPU 性能决定；两个系统都使用 cuDNN 5.1，因此拥有相同的优化 GPU 内核。

我们使用更大的内部集群（配备 NVIDIA K40 GPU，并通过共享数据中心网络连接）研究协调机制（§4.4）对训练性能的影响。理想情况下，如果同步训练高效，Inception-v3 这样的模型会用更少步骤训练，并且比异步训练达到更高准确率 [10]。当 worker 数量增加到 200 时，训练吞吐量提高到每秒 2,300 张图像，但收益逐渐递减（图 8(b)）。worker 越多，step 时间越长，因为 PS task 的网络接口以及更新聚合都会产生更多竞争。正如预期，在所有配置下，同步 step 都比异步 step 长，因为所有 worker 必须等待最慢 worker 赶上后才能开始下一步。

在 worker 数量相同的情况下，中位数同步 step 比异步 step 大约长 10%；但超过第 90 百分位后，同步性能会急剧下降，因为落后 worker 对尾延迟的影响很大。

为了降低尾延迟，我们加入备用 worker，使得某个 step 在 n 个 task 中的前 m 个产生梯度后就完成。图 8(c) 展示了在 50 个 worker 的 Inception 训练作业中加入备用 worker 的效果。增加备用 worker（包括第 4 个）会降低中位数 step 时间，因为落后 worker 影响 step 的概率变小。加入第 5 个备用 worker 后性能略有下降，因为第 51 个 worker（其结果会被丢弃的第一个 worker）更可能不是落后 worker，从而为 PS task 产生更多输入流量。

图 8(c) 还绘制了归一化加速比，其定义为

\[
\frac{t(b)}{t(0)} \times \frac{50}{50+b},
\]

其中 t(b) 是加入 b 个备用 worker 时的中位数 step 时间；该指标根据额外资源消耗的比例折算加速比。加入 4 个备用 worker 可获得最短总 step 时间（1.93 s）；加入 3 个备用 worker 则取得最高归一化加速比（9.5%），因此到达同样质量所消耗的 GPU 时间更少。

### 图 8：Inception-v3 训练性能评测结果（§6.3）

（a）TensorFlow 的异步训练吞吐量略高于 MXNet；（b）异步和同步训练的吞吐量最多可扩展到 200 个 worker；（c）向 50-worker 训练作业加入备用 worker 可以降低总 step 时间，即使按资源消耗归一化后仍能提升性能。

### 6.4 语言建模

给定一个单词序列，语言模型预测最可能出现的下一个单词 [6]。因此，语言模型是预测文本、语音识别和机器翻译应用的重要组成部分。本实验研究 TensorFlow 如何训练循环神经网络（LSTM-512-512 [41]），以对 One Billion Word Benchmark 文本 [9] 建模。

词表大小 |V| 限制了训练性能，因为最后一层必须把输出状态解码成 |V| 个类别的概率 [37]。相应参数可能很大（输出状态维度为 d 时，大小为 |V|×d），因此我们使用 §4.2 中处理大模型的技术。为了实验较小配置，我们使用最常见的 40,000 个单词组成的受限词表，而不是完整的 800,000 个单词词表 [9]。

图 9 展示了不同 PS task 数量和 worker task 数量下的训练吞吐量（以每秒处理的单词数计），以及两种 softmax 实现。

完整 softmax（图 9(a)）把每个输出乘以一个 512×40,000 的权重矩阵，该矩阵被分片到 PS task 上。增加 PS task 会提高吞吐量，因为 TensorFlow 可以利用分布式模型并行 [20, 43]，并在 PS task 上执行乘法和梯度计算，就像 Project Adam [14] 一样。从 1 个 PS task 增加到 2 个的效果，比把 worker 从 4 增加到 32，或从 32 增加到 256 更明显。最终吞吐量会饱和，因为 LSTM 计算成为训练 step 的主导部分。

采样 softmax（图 9(b)）减少了 PS task 上的数据传输和计算量 [37]。它不使用稠密权重矩阵，而是把输出乘以一个随机稀疏矩阵；该矩阵包含真实类别和随机采样错误类别的权重。我们为每个批次采样 512 个类别，使 softmax 的数据传输和计算量减少 78 倍。

### 图 9：语言模型训练中的 softmax 吞吐量

增加 PS task 数量可以并行化 softmax 计算，从而提高吞吐量。采样 softmax 通过减少计算量进一步提高吞吐量。（a）完整 softmax；（b）采样 softmax。纵轴为每秒处理单词数，曲线对应 4、32、256 个 worker；横轴为 PS task 数量。

## 7 结论

本文介绍了 TensorFlow 系统及其编程模型。TensorFlow 的数据流表示包含了已有参数服务器系统的能力，并提供一组统一抽象，使用户能够利用大规模异构系统，既完成生产任务，也能试验新方法。我们展示了 TensorFlow 编程模型如何促进试验（§4），并证明由此得到的实现具备性能和可扩展性（§6）。

我们对 TensorFlow 的初步经验令人鼓舞。Google 的许多团队已经将 TensorFlow 部署到生产环境，TensorFlow 也帮助研究同事在机器学习方面取得新进展。自从我们把 TensorFlow 作为开源软件发布以来，已有超过 14,000 人 fork 源代码仓库，二进制发行版下载量超过一百万次，数十个使用 TensorFlow 的机器学习模型已经发表。

TensorFlow 仍在不断发展。它灵活的数据流表示让高级用户能够达到出色性能，但我们还没有确定适用于所有用户的默认策略。对自动优化的进一步研究应当弥合这一差距。在系统层面，我们正在积极开发自动放置、内核融合、内存管理和调度算法。当前的可变状态和容错实现足以支持具有弱一致性要求的应用，但我们预计某些 TensorFlow 应用会需要更强的一致性，并正在研究如何在用户层构建这些策略。最后，一些用户已经开始感到静态数据流图存在限制，尤其是在深度强化学习等算法中 [54]。因此，我们面临一个有趣的问题：即使计算结构是动态展开的，如何仍然提供一个能够透明、高效地使用分布式资源的系统。

## 致谢

我们衷心感谢 Google 同事以及更广泛机器学习社区成员的贡献。特别感谢 Google Brain 团队其他成员以及 DistBelief 和 TensorFlow 众多用户给予的反馈。我们也感谢匿名 OSDI 审稿人和 shepherd KyoungSoo Park 提出的建议，这些建议大幅改进了本文的呈现。

## 参考文献

作者姓名、会议/期刊名称、年份、页码和网址保留原文格式；论文题名附中文翻译，引用编号与原文一致。

[1] M. Abadi 等：**TensorFlow：异构分布式系统上的大规模机器学习**（TensorFlow: Large-scale machine learning on heterogeneous distributed systems）。arXiv 预印本 1603.04467，2016。https://arxiv.org/abs/1603.04467。软件地址：https://tensorflow.org。

[2] R. Al-Rfou 等：**Theano：用于快速计算数学表达式的 Python 框架**（Theano: A Python framework for fast computation of mathematical expressions）。arXiv 预印本 1605.02688，2016。https://arxiv.org/abs/1605.02688。

[3] A. Angelova、A. Krizhevsky、V. Vanhoucke：**使用大视野深度网络进行行人检测**（Pedestrian detection with a large-field-of-view deep network）。ICRA 论文集，第 704–711 页，IEEE，2015。www.vision.caltech.edu/anelia/publications/Angelova15LFOV.pdf。

[4] Arvind、D. E. Culler：**数据流架构**（Dataflow architectures）。《Annual Review of Computer Science》第 1 卷，225–253 页，Annual Reviews Inc.，1986。www.dtic.mil/cgi-bin/GetTRDoc?Location=U2&doc=GetTRDoc.pdf&AD=ADA166235。

[5] J. Ba、V. Mnih、D. Kavukcuoglu：**利用视觉注意力识别多个目标**（Multiple object recognition with visual attention）。arXiv 预印本 1412.7755，2014。https://arxiv.org/abs/1412.7755。

[6] Y. Bengio、R. Ducharme、P. Vincent、C. Jauvin：**神经概率语言模型**（A neural probabilistic language model）。《Journal of Machine Learning Research》3:1137–1155，2003。jmlr.org/papers/volume3/bengio03a/bengio03a.pdf。

[7] T. Brants、A. Franz：**Web 1T 五元语法第 1 版**（Web 1T 5-gram version 1），2006。catalog.ldc.upenn.edu/LDC2006T13。

[8] R. H. Byrd、G. M. Chin、J. Nocedal、Y. Wu：**机器学习优化方法中的样本规模选择**（Sample size selection in optimization methods for machine learning）。《Mathematical Programming》134(1):127–155，2012。https://doi.org/10.1007/s10107-012-0572-5。

[9] C. Chelba、T. Mikolov、M. Schuster、Q. Ge、T. Brants、P. Koehn：**用于衡量统计语言模型进展的十亿词基准**（One billion word benchmark for measuring progress in statistical language modeling）。arXiv 预印本 1312.3005，2013。https://arxiv.org/abs/1312.3005。

[10] J. Chen、R. Monga、S. Bengio、R. Jozefowicz：**重新审视分布式同步 SGD**（Revisiting distributed synchronous SGD）。ICLR Workshop Track 论文集，2016。https://arxiv.org/abs/1604.00981。

[11] T. Chen、M. Li、Y. Li、M. Lin、N. Wang、M. Wang、T. Xiao、B. Xu、C. Zhang、Z. Zhang：**MXNet：面向异构分布式系统的灵活高效机器学习库**（MXNet: A flexible and efficient machine learning library for heterogeneous distributed systems）。LearningSys 论文集，2015。www.cs.cmu.edu/~muli/file/mxnet-learning-sys.pdf。

[12] H.-T. Cheng 等：**用于推荐系统的宽而深学习**（Wide & deep learning for recommender systems）。arXiv 预印本 1606.07792，2016。

[13] S. Chetlur 等：**cuDNN：深度学习的高效原语**（cuDNN: Efficient primitives for deep learning）。arXiv 预印本 1410.0759，2014。https://arxiv.org/abs/1410.0759。

[14] T. Chilimbi、Y. Suzue、J. Apacible、K. Kalyanaraman：**Project Adam：构建高效且可扩展的深度学习训练系统**（Project Adam: Building an efficient and scalable deep learning training system）。OSDI 论文集，571–582 页，2014。www.usenix.org/system/files/conference/osdi14/osdi14-paper-chilimbi.pdf。

[15] S. Chintala：**convnet-benchmarks**，2016。github.com/soumith/convnet-benchmarks。

[16] E. S. Chung、J. D. Davis、J. Lee：**LINQits：小型客户端上的大数据**（LINQits: Big data on little clients）。ISCA 论文集，261–272 页，2013。www.microsoft.com/en-us/research/wp-content/uploads/2013/06/ISCA13-linqits.pdf。

[17] R. Collobert、S. Bengio、J. Marithoz：**Torch：模块化机器学习软件库**（Torch: A modular machine learning software library）。IDIAP 技术报告，2002。infoscience.epfl.ch/record/82802/files/rr02-46.pdf。

[18] H. Cui、H. Zhang、G. R. Ganger、P. B. Gibbons、E. P. Xing：**GeePS：使用 GPU 专用参数服务器的可扩展分布式 GPU 深度学习**（GeePS: Scalable deep learning on distributed GPUs with a GPU-specialized parameter server）。EuroSys 论文集，2016。www.pdl.cmu.edu/PDL-FTP/CloudComputing/GeePS-cui-eurosys16.pdf。

[19] A. Dai、C. Olah、Q. V. Le：**使用段落向量进行文档嵌入**（Document embedding with paragraph vectors）。arXiv 预印本 1507.07998，2015。https://arxiv.org/abs/1507.07998。

[20] J. Dean 等：**大规模分布式深度网络**（Large scale distributed deep networks）。NIPS 论文集，1232–1240 页，2012。research.google.com/archive/large_​​deep_networks_nips2012.pdf。

[21] J. Dean、S. Ghemawat：**MapReduce：简化大规模集群上的数据处理**（MapReduce: Simplified data processing on large clusters）。OSDI 论文集，137–149 页，2004。research.google.com/archive/mapreduce-osdi04.pdf。

[22] DMLC：**用于深度学习的 MXNet**（MXNet for deep learning），2016。github.com/dmlc/mxnet。

[23] J. Duchi、E. Hazan、Y. Singer：**在线学习和随机优化的自适应次梯度方法**（Adaptive subgradient methods for online learning and stochastic optimization）。《Journal of Machine Learning Research》12:2121–2159，2011。jmlr.org/papers/volume12/duchi11a/duchi11a.pdf。

[24] A. Frome 等：**DeVISE：深度视觉—语义嵌入模型**（DeVISE: A deep visual-semantic embedding model）。NIPS 论文集，2121–2129 页，2013。research.google.com/pubs/archive/41473.pdf。

[25] J. Gonzalez-Dominguez、I. Lopez-Moreno、P. J. Moreno、J. Gonzalez-Rodriguez：**使用深度神经网络对短语音逐帧进行语言识别**（Frame-by-frame language identification in short utterances using deep neural networks）。《Neural Networks》64:49–58，2015。research.google.com/pubs/archive/42929.pdf。

[26] I. J. Goodfellow 等：**生成对抗网络**（Generative adversarial nets）。NIPS 论文集，2672–2680 页，2014。papers.nips.cc/paper/5423-generative-adversarial-nets.pdf。

[27] Google Research：**TensorFlow Serving**，2016。tensorflow.github.io/serving/。

[28] K. He、X. Zhang、S. Ren、J. Sun：**用于图像识别的深度残差学习**（Deep residual learning for image recognition）。CVPR 论文集，770–778 页，2016。https://arxiv.org/abs/1512.03385。

[29] G. Heigold 等：**使用分布式深度神经网络的多语言声学模型**（Multilingual acoustic models using distributed deep neural networks）。ICASSP 论文集，8619–8623 页，2013。research.google.com/pubs/archive/40807.pdf。

[30] G. E. Hinton：**学习概念的分布式表示**（Learning distributed representations of concepts）。第八届认知科学学会年会论文集，1–12 页，1986。www.cogsci.ucsd.edu/~ajyu/Teaching/Cogs202-sp13/Readings/hinton86.pdf。

[31] G. E. Hinton 等：**语音识别声学建模的深度神经网络：四个研究小组的共同观点**（Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups）。《IEEE Signal Processing Magazine》29(6):82–97，2012。www.cs.toronto.edu/~gdahl/papers/deepSpeechReviewSPM2012.pdf。

[32] S. Hochreiter、J. Schmidhuber：**长短期记忆**（Long short-term memory）。《Neural Computation》9(8):1735–1780，1997。deeplearning.cs.cmu.edu/pdfs/Hochreiter97-lstm.pdf。

[33] S. Ioffe、C. Szegedy：**批归一化：通过减少内部协变量偏移加速深度网络训练**（Batch normalization: Accelerating deep network training by reducing internal covariate shift）。ICML 论文集，448–456 页，2015。jmlr.org/proceedings/papers/v37/ioffe15.pdf。

[34] M. Isard、M. Budiu、Y. Yu、A. Birrell、D. Fetterly：**Dryad：由顺序构件组成的分布式数据并行程序**（Dryad: Distributed data-parallel programs from sequential building blocks）。EuroSys 论文集，59–72 页，2007。www.microsoft.com/en-us/research/wp-content/uploads/2007/03/eurosys07.pdf。

[35] B. Jacob 等：**gemmlowp：小型自包含低精度 GEMM 库**（gemmlowp: A small self-contained low-precision GEMM library），2015。github.com/google/gemmlowp。

[36] B. Jacob、G. Guennebaud 等：**Eigen 线性代数库**（Eigen library for linear algebra）。eigen.tuxfamily.org。

[37] S. Jean、K. Cho、R. Memisevic、Y. Bengio：**在神经机器翻译中使用超大目标词表**（On using very large target vocabulary for neural machine translation）。ACL-IJCNLP 论文集，1–10 页，2015 年 7 月。www.aclweb.org/anthology/P15-1001。

[38] Y. Jia 等：**Caffe：用于快速特征嵌入的卷积架构**（Caffe: Convolutional architecture for fast feature embedding）。ACM Multimedia 论文集，675–678 页，2014。https://arxiv.org/abs/1408.5093。

[39] M. I. Jordan：**串行顺序：一种并行分布式处理方法**（Serial order: A parallel distributed processing approach）。ICS 报告 8608，加州大学圣迭戈分校认知科学研究所，1986。cseweb.ucsd.edu/~gary/PAPER-SUGGESTIONS/Jordan-TR-8604.pdf。

[40] N. Jouppi：**Google 使用 TPU 定制芯片大幅加速机器学习任务**（Google supercharges machine learning tasks with TPU custom chip），2016。cloudplatform.googleblog.com/2016/05/Google-supercharges-machine-learning-tasks-with-custom-chip.html。

[41] R. Józefowicz、O. Vinyals、M. Schuster、N. Shazeer、Y. Wu：**探索语言建模的极限**（Exploring the limits of language modeling）。arXiv 预印本 1602.02410，2016。https://arxiv.org/abs/1602.02410。

[42] A. Karpathy、G. Toderici、S. Shetty、T. Leung、R. Sukthankar、L. Fei-Fei：**使用卷积神经网络进行大规模视频分类**（Large-scale video classification with convolutional neural networks）。CVPR 论文集，1725–1732 页，2014。research.google.com/pubs/archive/42455.pdf。

[43] A. Krizhevsky：**并行化卷积神经网络的一个奇妙技巧**（One weird trick for parallelizing convolutional neural networks）。arXiv 预印本 1404.5997，2014。https://arxiv.org/abs/1404.5997。

[44] A. Krizhevsky、I. Sutskever、G. E. Hinton：**使用深度卷积神经网络进行 ImageNet 分类**（ImageNet classification with deep convolutional neural networks）。NIPS 论文集，1106–1114 页，2012。papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf。

[45] H. Larochelle、Y. Bengio、J. Louradour、P. Lamblin：**探索训练深度神经网络的策略**（Exploring strategies for training deep neural networks）。《Journal of Machine Learning Research》10:1–40，2009。jmlr.org/papers/volume10/larochelle09a/larochelle09a.pdf。

[46] A. Lavin、S. Gray：**卷积神经网络的快速算法**（Fast algorithms for convolutional neural networks）。CVPR 论文集，4013–4021 页，2016。https://arxiv.org/abs/1509.09308。

[47] Q. Le、M. Ranzato、R. Monga、M. Devin、G. Corrado、K. Chen、J. Dean、A. Ng：**通过大规模无监督学习构建高级特征**（Building high-level features using large scale unsupervised learning）。ICML 论文集，81–88 页，2012。research.google.com/archive/unsupervised-icml2012.pdf。

[48] Y. LeCun、C. Cortes、C. J. Burges：**MNIST 手写数字数据库**（The MNIST database of handwritten digits），1998。yann.lecun.com/exdb/mnist/。

[49] M. Li、D. G. Andersen、J. Park、A. J. Smola、A. Ahmed、V. Josifovski、J. Long、E. J. Shekita、B.-Y. Su：**使用参数服务器扩展分布式机器学习**（Scaling distributed machine learning with the Parameter Server）。OSDI 论文集，583–598 页，2014。www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf。

[50] C. J. Maddison、A. Huang、I. Sutskever、D. Silver：**使用深度卷积神经网络进行围棋落子评估**（Move evaluation in Go using deep convolutional neural networks）。arXiv 预印本 1412.6564，2014。https://arxiv.org/abs/1412.6564。

[51] F. McSherry、M. Isard、D. G. Murray：**可扩展性！但代价是什么？**（Scalability! But at what COST?）。HotOS（HOTOS’15）论文集，2015。www.usenix.org/system/files/conference/hotos15/hotos15-paper-mcsherry.pdf。

[52] T. Mikolov、K. Chen、G. Corrado、J. Dean：**向量空间中词表示的高效估计**（Efficient estimation of word representations in vector space）。ICLR Workshops Track 论文集，2013。https://arxiv.org/abs/1301.3781。

[53] V. Mnih、N. Heess、A. Graves、K. Kavukcuoglu：**视觉注意力的循环模型**（Recurrent models of visual attention）。NIPS 论文集，2204–2212 页，2014。papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf。

[54] V. Mnih 等：**通过深度强化学习实现人类级控制**（Human-level control through deep reinforcement learning）。《Nature》518(7540):529–533，2015。https://doi.org/10.1038/nature14236。

[55] P. Moritz、R. Nishihara、I. Stoica、M. I. Jordan：**SparkNet：在 Spark 中训练深度网络**（SparkNet: Training deep networks in Spark）。ICLR 论文集，2016。https://arxiv.org/abs/1511.06051。

[56] D. G. Murray、F. McSherry、M. Isard、R. Isaacs、P. Barham、M. Abadi：**使用及时数据流进行增量式迭代数据处理**（Incremental, iterative data processing with timely dataflow）。《Communications of the ACM》59(10):75–83，2016。dl.acm.org/citation.cfm?id=2983551。

[57] A. Nair 等：**深度强化学习的大规模并行方法**（Massively parallel methods for deep reinforcement learning）。arXiv 预印本 1507.04296，2015。https://arxiv.org/abs/1507.04296。

[58] Nervana Systems：**Neon 深度学习框架**（Neon deep learning framework），2016。github.com/NervanaSystems/neon。

[59] NVIDIA Corporation：**NCCL：优化的多 GPU 集合通信原语**（NCCL: Optimized primitives for collective multi-GPU communication），2016。github.com/NVIDIA/nccl。

[60] R. Pascanu、T. Mikolov、Y. Bengio：**训练循环神经网络的困难**（On the difficulty of training recurrent neural networks）。ICML 论文集，1310–1318 页，2013。jmlr.org/proceedings/papers/v28/pascanu13.pdf。

[61] B. Recht、C. Re、S. Wright、F. Niu：**Hogwild：一种无锁并行随机梯度下降方法**（Hogwild: A lock-free approach to parallelizing stochastic gradient descent）。NIPS 论文集，693–701 页，2011。papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent.pdf。

[62] C. J. Rossbach、Y. Yu、J. Currey、J.-P. Martin、D. Fetterly：**Dandelion：面向异构系统的编译器和运行时**（Dandelion: A compiler and runtime for heterogeneous systems）。SOSP 论文集，49–68 页，2013。sigops.org/sosp/sosp13/papers/p49-rossbach.pdf。

[63] D. E. Rumelhart、G. E. Hinton、R. J. Williams：**通过反向传播误差学习表示**（Learning representations by back-propagating errors）。《Cognitive modeling》第 5 卷，213–220 页，MIT Press，1988。www.cs.toronto.edu/~hinton/absps/naturebp.pdf。

[64] O. Russakovsky 等：**ImageNet 大规模视觉识别挑战赛**（ImageNet Large Scale Visual Recognition Challenge）。《International Journal of Computer Vision》115(3):211–252，2015。https://arxiv.org/abs/1409.0575。

[65] A. Smola、S. Narayanamurthy：**并行主题模型的架构**（An architecture for parallel topic models）。《Proceedings of the VLDB Endowment》3(1–2):703–710，2010。vldb.org/pvldb/vldb2010/papers/R63.pdf。

[66] I. Sutskever、J. Martens、G. E. Dahl、G. E. Hinton：**初始化和动量在深度学习中的重要性**（On the importance of initialization and momentum in deep learning）。ICML 论文集，1139–1147 页，2013。jmlr.org/proceedings/papers/v28/sutskever13.pdf。

[67] I. Sutskever、O. Vinyals、Q. V. Le：**使用神经网络进行序列到序列学习**（Sequence to sequence learning with neural networks）。NIPS 论文集，3104–3112 页，2014。papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural.pdf。

[68] C. Szegedy 等：**使用卷积深入网络**（Going deeper with convolutions）。CVPR 论文集，1–9 页，2015。https://arxiv.org/abs/1409.4842。

[69] C. Szegedy、V. Vanhoucke、S. Ioffe、J. Shlens、Z. Wojna：**重新思考 Inception 视觉架构**（Rethinking the Inception architecture for computer vision）。arXiv 预印本 1512.00567，2015。https://arxiv.org/abs/1512.00567。

[70] C. tao Chu、S. K. Kim、Y. an Lin、Y. Yu、G. Bradski、K. Olukotun、A. Y. Ng：**多核上的机器学习 MapReduce**（Map-reduce for machine learning on multicore）。NIPS 论文集，281–288 页，2007。papers.nips.cc/paper/3150-map-reduce-for-machine-learning-on-multicore.pdf。

[71] A. Verma、L. Pedrosa、M. Korupolu、D. Oppenheimer、E. Tune、J. Wilkes：**Google 的大规模集群管理：Borg**（Large-scale cluster management at Google with Borg）。EuroSys 论文集，2015。research.google.com/pubs/archive/43438.pdf。

[72] O. Vinyals、L. Kaiser、T. Koo、S. Petrov、I. Sutskever、G. Hinton：**把语法当作外语**（Grammar as a foreign language）。arXiv 预印本，2014。https://arxiv.org/abs/1412.7449。

[73] Y. Wu 等：**Google 神经机器翻译系统：弥合人机翻译差距**（Google’s Neural Machine Translation system: Bridging the gap between human and machine translation）。arXiv 预印本 1609.08144，2016。https://arxiv.org/abs/1609.08144。

[74] Y. Yu、M. Isard、D. Fetterly、M. Budiu、U. Erlingsson、P. K. Gunda、J. Currey：**DryadLINQ：使用高级语言进行通用分布式数据并行计算的系统**（DryadLINQ: A system for general-purpose distributed data-parallel computing using a high-level language）。OSDI 论文集，1–14 页，2008。www.usenix.org/legacy/event/osdi08/tech/full_papers/yu_y/yu_y.pdf。

[75] M. Zaharia、M. Chowdhury、T. Das、A. Dave、J. Ma、M. McCauley、M. J. Franklin、S. Shenker、I. Stoica：**弹性分布式数据集：内存集群计算的容错抽象**（Resilient distributed datasets: A fault-tolerant abstraction for in-memory cluster computing）。NSDI 论文集，15–28 页，2012。www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf。

[76] M. D. Zeiler、M. Ranzato、R. Monga、M. Mao、K. Yang、Q. Le、P. Nguyen、A. Senior、V. Vanhoucke、G. E. Hinton：**用于语音处理的修正线性单元**（On rectified linear units for speech processing）。ICASSP 论文集，3517–3521 页，2013。research.google.com/pubs/archive/40811.pdf。



