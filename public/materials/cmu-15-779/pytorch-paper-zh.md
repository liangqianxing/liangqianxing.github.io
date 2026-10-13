# PyTorch：一种命令式风格的高性能深度学习库

**作者**：Adam Paszke，Sam Gross，Francisco Massa，Adam Lerer，James Bradbury，Gregory Chanan，Trevor Killeen，Zeming Lin，Natalia Gimelshein，Luca Antiga，Alban Desmaison，Andreas Kӧpf，Edward Yang，Zach DeVito，Martin Raison，Alykhan Tejani，Sasank Chilamkurthy，Benoit Steiner，Lu Fang，Junjie Bai，Soumith Chintala

**来源**：NeurIPS 2019，第 33 届神经信息处理系统大会；原文版本 arXiv:1912.01703v1（2019 年 12 月 3 日）

> 说明：这是根据用户提供的论文 PDF 制作的中文学习译稿。代码、API 名称、数学符号和引用编号保留原样，便于与原文对照。

## 摘要

深度学习框架通常只侧重易用性或速度，很少同时兼顾二者。PyTorch 这一机器学习库说明，这两个目标实际上可以兼容：它提供命令式、符合 Python 习惯的编程风格，将代码本身作为模型，便于调试，并且与其他流行的科学计算库保持一致；与此同时，它仍然高效，并支持 GPU 等硬件加速器。

本文详细介绍驱动 PyTorch 实现的设计原则，以及这些原则如何体现在系统架构中。我们强调，PyTorch 的每个部分都是一个完全由用户控制的普通 Python 程序。我们还说明，运行时关键组件经过谨慎而务实的实现后，如何协同工作以获得出色的性能。最后，我们在多个常见基准上展示各个子系统的效率，以及 PyTorch 的总体速度。

## 1 引言

近年来，随着人们对深度学习兴趣的不断增长，机器学习工具数量迅速增加。Caffe [1]、CNTK [2]、TensorFlow [3] 和 Theano [4] 等许多流行框架都会构造一个表示计算过程的静态数据流图，然后反复将该图应用于不同的数据批次。这种方法可以在执行前看到完整计算过程，理论上也能利用这些信息提升性能和可扩展性。然而，它牺牲了易用性、易调试性，以及对可表示计算类型的灵活性。

此前的研究已经认识到，动态即时执行对深度学习很有价值。一些较新的框架实现了这种“运行时定义”（define-by-run）方式，但有的牺牲了性能（如 Chainer [5]），有的使用表达能力较弱但速度较快的语言（如 Torch [6] 和 DyNet [7]），因此限制了适用范围。

通过仔细的实现和设计选择，动态即时执行可以在很大程度上不牺牲性能。本文介绍 PyTorch：一个对动态张量计算进行即时执行、支持自动微分和 GPU 加速的 Python 库，同时保持与当前最快深度学习库相当的性能。事实证明，这种组合在研究社区中非常受欢迎；例如，在 ICLR 2019 的投稿中，有 296 篇提到了 PyTorch。

## 2 背景

科学计算中的四个主要趋势对深度学习日益重要。

第一，从 20 世纪 60 年代开始，APL [8]、MATLAB [9]、R [10] 和 Julia [11] 等领域专用语言的发展，使多维数组（通常称为张量）成为一等对象，并提供了操作这些对象的大量数学原语。与此同时，NumPy [12]、Torch [6]、Eigen [13] 和 Lush [14] 等库让数组编程在 Python、Lisp、C++ 和 Lua 等通用语言中变得高效。

第二，自动微分 [15] 的发展使导数计算这一原本繁重的工作可以完全自动化。这显著降低了尝试不同机器学习方法的难度，同时仍能进行高效的基于梯度的优化。autograd [16] 软件包推动了这种技术在 NumPy 数组上的普及；Chainer [5]、DyNet [7]、Lush [14]、Torch [6]、Jax [17] 和 Flux.jl [18] 等框架也采用了类似方法。

第三，随着自由软件运动兴起，科学界逐渐从 Matlab [9] 等封闭的专有软件转向以 Python 为核心的开源生态，包括 NumPy [12]、SciPy [19] 和 Pandas [20]。这个生态基本满足了研究人员在数值分析方面的需求，同时让他们可以利用大量库来处理数据预处理、统计分析、绘图等任务。自由软件的开放性、互操作性和灵活性还促进了活跃社区的发展：社区能够快速响应新的或变化中的需求，通过扩展现有库，或者在必要时开发并发布全新的库来解决问题。

虽然其他语言也有丰富的神经网络开源软件，例如 Lisp 中的 Lush [14]，C++、Objective-C 和 Lua 中的 Torch [6]，C++ 中的 EBLearn [21] 和 Caffe [1]，但 Python 生态的网络效应使 Python 成为启动研究工作不可或缺的技能。因此，自 2014 年以来，大多数深度学习框架都把 Python 接口作为一项基本特性。

最后，GPU 等通用的大规模并行硬件的普及，为深度学习方法提供了所需的计算能力。cuDNN [22] 等专用库，以及学术界的一系列研究 [23, 24]，产生了高性能、可复用的深度学习 kernel，使 Caffe [1]、Torch7 [25] 和 TensorFlow [3] 等框架能够利用这些硬件加速器。

PyTorch 建立在这些趋势之上：它提供由 GPU 加速、通过自动微分实现可微分、并融入 Python 生态的数组编程模型。

## 3 设计原则

PyTorch 的成功来自把已有思想编织成一个兼顾速度与易用性的设计。我们的选择背后有四项主要原则。

### 3.1 遵循 Python 习惯

数据科学家熟悉 Python 语言、它的编程模型及其工具。PyTorch 应当成为这一生态中的一等成员。它遵循一些已经普遍建立的设计目标：接口简单且一致，理想情况下每件事都有一种符合习惯的做法；同时，它自然地集成标准的绘图、调试和数据处理工具。

### 3.2 优先服务研究人员

PyTorch 力求让模型、数据加载器和优化器的编写尽可能简单、高效。机器学习本身的复杂性应由 PyTorch 库在内部处理，并隐藏在直观、没有副作用和意外性能陷阱的 API 之后。

### 3.3 提供务实的性能

要有用，PyTorch 必须提供出色的性能，但不能以牺牲简单性和易用性为代价。为了让模型明显更容易使用，牺牲 10% 的速度是可以接受的；牺牲 100% 则不可接受。因此，PyTorch 的实现愿意增加内部复杂度来获得性能。此外，为研究人员提供可以手动控制代码执行的工具，能够帮助他们独立于库的自动优化，找到自己的性能改进方法。

### 3.4 “差一点但更好”（Worse is better）

在工程资源固定的情况下，其他条件相同，保持 PyTorch 内部实现简单所节省的时间，可以用来实现更多功能、适应新的情况，并跟上人工智能领域的快速发展。因此，一个简单但略有不完整的解决方案，往往优于一个全面但复杂且难以维护的设计。

## 4 以易用性为中心的设计

### 4.1 深度学习模型就是 Python 程序

在很短的时间内，机器学习从识别单个数字 [27] 发展到自主进行《星际争霸》游戏 [28]。相应地，神经网络也从由简单前馈层组成的序列，迅速演化为极其多样化的数值程序，其中往往包含大量循环和递归函数。

为了支持这种不断增长的复杂性，PyTorch 放弃了基于图元编程方法可能带来的部分收益，保留 Python 的命令式编程模型。这种设计最早由 Chainer [5] 和 DyNet [7] 在模型编写中推广。PyTorch 将它扩展到深度学习工作流的所有方面：定义层、组合模型、加载数据、运行优化器，以及并行化训练过程，都使用通用编程中熟悉的概念来表达。

这种方案确保任何新的神经网络架构都能在 PyTorch 中轻松实现。例如，层（在现代机器学习中，更准确地说应被理解为带有隐式参数的有状态函数）通常表示为 Python 类：构造函数创建并初始化参数，`forward` 方法处理输入激活值。类似地，模型通常表示为组合多个层的类，但再次强调，用户并没有被强制要求采用这种结构。

下面的代码展示了如何组合 PyTorch 提供的二维卷积、矩阵乘法、dropout 和 softmax 等功能，构造一个用于灰度图像分类的完整模型。线性层当然已经包含在库中，但这里给出一个简单实现，是为了说明它可以多么容易地被定义。

```python
class LinearLayer(Module):
   def __init__(self, in_sz, out_sz):
      super().__init__()
      t1 = torch.randn(in_sz, out_sz)
      self.w = nn.Parameter(t1)
      t2 = torch.randn(out_sz)
      self.b = nn.Parameter(t2)

   def forward(self, activations):
      t = torch.mm(activations, self.w)
      return t + self.b

class FullBasicModel(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv = nn.Conv2d(1, 128, 3)
      self.fc = LinearLayer(128, 10)

   def forward(self, x):
      t1 = self.conv(x)
      t2 = nn.functional.relu(t1)
      t3 = self.fc(t1)
      return nn.functional.softmax(t3)
```

**表 1：**作为简单完整神经网络构件的自定义层。

“所有东西都只是程序”这一理念并不局限于模型，也适用于优化器和数据加载器。这使新训练技术的实验更加容易。例如，实现非常流行的生成对抗网络，需要指定两个独立模型（生成器和判别器），以及同时依赖这两个模型的两个损失函数。僵化的 API 可能很难处理这种设置，而 PyTorch 的简单设计可以自然地适应它，如下所示：

```python
discriminator = create_discriminator()
generator = create_generator()
optimD = optim.Adam(discriminator.parameters())
optimG = optim.Adam(generator.parameters())

def step(real_sample):
  # (1) 更新判别器
  errD_real = loss(discriminator(real_sample), real_label)
  errD_real.backward()
  fake = generator(get_noise())
  errD_fake = loss(discriminator(fake.detach(), fake_label)
  errD_fake.backward()
  optimD.step()
  # (2) 更新生成器
  errG = loss(discriminator(fake), real_label)
  errG.backward()
  optimG.step()
```

**表 2：**生成对抗网络的简化训练过程。

由于 PyTorch 程序采用即时执行，Python 的所有功能在整个设计过程中都可用。打印语句、标准调试器，以及 matplotlib 等常见可视化工具都能按预期工作。用户不必等待漫长的编译过程才能开始运行程序；更重要的是，可以观察中间计算结果，以理解模型的工作方式并检查结果是否正确。

### 4.2 互操作性与可扩展性

简单而高效的互操作性是 PyTorch 的最高优先级之一，因为它让用户程序可以利用丰富的 Python 库生态。因此，PyTorch 允许与外部库双向交换数据。例如，它提供 `torch.from_numpy()` 函数和张量的 `.numpy()` 方法，在 NumPy 数组与 PyTorch 张量之间转换。对以 DLPack [29] 格式存储的数据，也提供类似功能。

这两种交换都不会复制数据：两边的对象只描述如何解释一段共享内存区域。因此，这些操作的代价非常低，无论转换数组有多大，时间复杂度都为常数。

许多关键系统也被专门设计成可扩展的。例如，自动微分系统允许用户为自定义可微函数增加支持。用户可以定义 `torch.autograd.Function` 的新子类，实现 `forward()` 和 `backward()` 方法，分别指定函数本身和它的导数，更严格地说是向量-雅可比积。

类似地，用户可以通过继承 `torch.utils.data.Dataset` 并实现两个方法来增加新数据集：`__getitem__`（索引操作）和 `__len__`（长度操作）。这样，数据集就表现得像一个（可能是惰性的）列表。具体实现完全由作者决定，许多用户会利用其他 Python 包来加载数据。

`DataLoader` 类可以消费符合这一接口的对象，并提供数据迭代器，负责打乱、批处理、并行化以及管理固定 CUDA 内存，从而提高吞吐量。

最重要的是，用户可以自由替换 PyTorch 中任何不符合项目需求或性能要求的组件。所有组件都被设计成可以完全互换，PyTorch 会非常谨慎地避免强加某一种特定解决方案。

### 4.3 自动微分

由于基于梯度的优化对深度学习至关重要，PyTorch 必须能够自动计算用户指定模型的梯度，而这些模型可以是任意 Python 程序。然而，Python 是一种动态编程语言，允许在运行时改变大多数行为，因此，提前进行源码到源码的微分非常困难。

PyTorch 采用运算符重载方法：每次执行时，都构建出已计算函数的表示。在当前实现中 [30]，PyTorch 执行反向模式自动微分，计算一个标量输出相对于多变量输入的梯度。当输出多于输入时，前向模式自动微分会更高效，但这种情况在机器学习应用中并不常见。通过数组级对偶数 [31, 32]，PyTorch 也可以较容易地扩展为支持前向模式微分。

我们的系统还有一个有趣且不常见的特性：它可以对包含张量原地修改的代码求导，而原地修改正是命令式程序的基本构件之一。为了确保安全，我们为张量实现了版本控制系统，用于跟踪张量的修改，并确保后续始终使用预期的数据。

这里存在一个有趣的权衡。我们本可以使用写时复制等技术来支持任意程序，但从性能角度看，用户通常最好重写代码以避免复制。因此，我们选择不沿着这条路线继续。大多数原地修改都能自动处理；真正复杂的情况会产生用户错误，提醒用户可能需要重新组织程序。这样可以避免引入隐蔽且难以发现的性能陷阱。

## 5 以性能为中心的实现

从 Python 解释器中高效运行深度学习算法是出了名的困难。例如，全局解释器锁（GIL）实际上保证了任意时刻，在多个并发线程中只有一个线程真正运行。基于静态数据流图构造的深度学习框架通过把计算推迟到自定义解释器中，绕开了这个问题。

PyTorch 采用了另一种方式：谨慎优化执行过程的每个方面，同时让用户能够轻松利用额外的优化策略。

### 5.1 高效的 C++ 核心

尽管 PyTorch 与 Python 生态紧密集成，但为了实现高性能，它的大部分代码使用 C++ 编写。这个核心库 `libtorch` 实现了张量数据结构、GPU 和 CPU 运算，以及基础并行原语。它还提供自动微分系统，包括大多数内置函数的梯度公式。

因此，由 PyTorch 核心运算符组合而成的函数，其导数计算可以完全在一个多线程执行器中运行，而不需要持有 Python 的全局解释器锁 [33]。Python 绑定通过 YAML 元数据文件生成。这种设计还有一个有趣的副作用：社区可以快速创建其他语言的绑定，于是出现了 NimTorch [34]、hasktorch [35] 等项目。

这种设计也让我们能够创建一等的 C++ 绑定和模型库，用于 Python 不方便的场景，例如《星际争霸》游戏引擎 [36] 或移动平台。甚至可以把描述 PyTorch 模型的 Python 代码交给 TorchScript 引擎 [37]，在没有 Python 的环境中运行。

### 5.2 分离控制流和数据流

PyTorch 严格分离控制流（即程序分支和循环）与数据流（即张量及其上的操作）。控制流的解析由 Python 和在主机 CPU 上执行的优化 C++ 代码负责，最终在设备上产生一个线性的运算符调用序列。运算符可以在 CPU 或 GPU 上运行。

PyTorch 利用 CUDA stream 机制 [38]，把 CUDA kernel 调用排入 GPU 硬件 FIFO 队列，从而设计成异步地在 GPU 上执行运算符。这使系统能够把 CPU 上 Python 代码的执行与 GPU 上张量运算的执行重叠起来。由于张量运算通常需要相当长的时间，这种方式即使在 Python 这种解释型、开销较高的语言中，也能让 GPU 保持饱和并达到峰值性能。

这个机制对用户几乎是透明的。除非用户实现自己的多流原语，否则所有 CPU-GPU 同步都由库负责。

PyTorch 也可以采用类似机制，在 CPU 上异步执行运算符。然而，跨线程通信和同步的成本会抵消这种优化带来的性能收益。

### 5.3 自定义缓存张量分配器

几乎每个运算符都必须动态分配一个输出张量，用来保存执行结果。因此，优化动态内存分配器的速度至关重要。在 CPU 上，PyTorch 可以依赖经过优化的库 [39–41] 完成这一任务。

但在 GPU 上，`cudaFree` 可能会阻塞调用者，直到所有 GPU 上此前排队的工作全部完成。为避免这个瓶颈，PyTorch 实现了一个自定义分配器：它逐步建立 CUDA 内存缓存，之后的分配直接把这些内存重新分配给后续请求，而不再调用 CUDA API。

渐进式分配对互操作性也很重要，因为如果一开始就占满全部 GPU 内存，用户便无法使用其他支持 GPU 的 Python 包。

为了进一步提高效果，这个分配器针对深度学习的具体内存使用模式进行了调优。例如，它把分配大小向上取整到 512 字节的倍数，以避免碎片问题；同时为每条 CUDA stream（工作队列）维护独立的内存池。

“一条 stream 对应一个内存池”的设计假设简化了实现并提高了分配器性能：由于 CPU 的运行速度领先于 GPU，内存在 GPU 上最后一次使用完成之前，就可能已经在 CPU 上被释放。由于 stream 会串行化执行，如果 CPU 上的释放先于重新分配，那么 GPU 上也会保持相同顺序。因此，只要新的分配与被释放区域使用同一条 stream，分配器就可以立即重新使用 CPU 上已经释放的内存。

但是，如果某次分配最后在一条 stream 上使用，随后又在另一条 stream 上分配，就需要额外同步。

每条 stream 使用独立内存池看起来会受限，因为分配会按 stream 分散，造成碎片。但实践中 PyTorch 几乎从不使用多条 stream。以一种让多个 kernel 协作共享 GPU 的方式编写 CUDA kernel 本来就很难，因为具体调度由硬件控制。在实践中，kernel 编写者通常会采用把多个任务组合在一起的单体 kernel。

数据加载和分布式计算工具是“一条 stream 一个内存池”设计的例外；它们会仔细插入额外同步，以避免与分配器发生不良交互。

虽然这种设计容易受到某些边界情况影响，但在实际代码中几乎不会表现出不期望的行为。大多数用户甚至不知道它的存在。

### 5.4 多进程

由于全局解释器锁（GIL），Python 的默认实现不允许并发线程并行执行。为了缓解这个问题，Python 社区建立了标准的 multiprocessing 模块，包含许多工具，让用户可以轻松创建子进程并实现基本的进程间通信原语。

然而，这些原语的实现使用了与磁盘持久化相同形式的序列化；处理大型数组时效率很低。因此，PyTorch 对 Python 的 multiprocessing 模块进行了扩展，提供 `torch.multiprocessing`，它可以直接替换内置包，并且会自动把发送到其他进程的张量数据放入共享内存，而不是通过通信通道发送。

这种设计显著提高了性能，但也使进程隔离变弱，形成了更接近普通线程程序的编程模型。用户可以轻松实现高度并行的程序，让不同进程操作相互独立的 GPU，之后再使用 all-reduce 风格的原语同步梯度。

这个系统的另一个独特功能是能够透明地处理 CUDA 张量的共享，从而让实现 Hogwild [42] 这类技术变得简单。

### 5.5 引用计数

用户经常会设计模型，使训练过程利用全部可用内存；增加 batch size 是提高速度的常见方法。因此，为了提供良好性能，PyTorch 必须把内存当成稀缺资源并谨慎管理。

具有即时语义的库必须在不知道张量未来如何使用的情况下管理张量内存。垃圾回收是自动完成这一工作的典型方法，因为它具有良好的摊销性能。采用这种方式时，运行时会周期性检查系统状态，枚举正在使用的对象，并释放其余对象。然而，延迟释放会导致程序整体使用更多内存 [43]。考虑到 GPU 内存的稀缺性，这种开销不可接受。

实际上，Torch7 使用 Lua 内置的垃圾回收器，一个常见的反模式是：用户在程序中到处显式触发垃圾回收，希望内存错误能够消失。

PyTorch 采用了不同方案：通过引用计数跟踪每个张量被使用的次数，当计数降为零时立即释放底层内存。需要注意的是，PyTorch 同时跟踪 `libtorch` 库内部的引用，以及用户 Python 代码中的外部引用；后者通过与 Python 自身的引用计数机制集成来实现。

这确保了张量一旦不再需要，内存就能被准确释放。

一个需要注意的限制是：我们只能在以下语言实现中保证所需的性能特性：一类是本身已经使用引用计数的实现（如 CPython、Swift，而不是 PyPy 或许多 Lua 等脚本语言）；另一类是允许用户定义赋值、复制和移动行为的实现（如 C++、Rust）。对于不满足这些条件的语言绑定，需要在 PyTorch 之上实现自己的专用内存管理。

## 6 评估

本节把 PyTorch 与其他几种常用深度学习库的性能进行比较，结果表明，PyTorch 在一系列任务上都具有竞争力。所有实验都在一台工作站上完成，该工作站配有两颗 Intel Xeon E5-2698 v4 CPU 和一块 NVIDIA Quadro GP100 GPU。

### 6.1 异步数据流

我们首先量化 PyTorch 在 GPU 上异步执行数据流的能力。我们使用内置 profiler [44] 对多种基准进行插桩，并记录一次训练步骤中前几个操作的执行时间线。

图 1 展示了 ResNet-50 模型前几个运算符的代表性时间线。负责排队工作的主机 CPU 很快就领先于 GPU 上运算符的执行。这使 PyTorch 几乎可以实现完美的设备利用率。在这个例子中，GPU 执行时间约为 CPU 调度时间的三倍。确切比例取决于主机 CPU 与 GPU 的相对性能、每个张量中的元素数量，以及 GPU 上浮点计算的平均算术复杂度。

**图 1：**ResNet-50 前几个运算符的执行轨迹。顶部一行表示主机 CPU 上控制流的执行，其中灰色区域是由 Python 解释器执行的代码，彩色区域对应主机 CPU 为各种运算符（卷积、批归一化等）排队所做的工作。底部一行显示这些运算符在 GPU 上的对应执行。箭头表示按时间配对的两个事件。

### 6.2 内存管理

我们使用 NVIDIA profiler 跟踪一次 ResNet-50 训练迭代中 CUDA runtime 的执行，以及启动的 CUDA kernel 的执行。如图 2 所示，第一次迭代的行为与后续迭代明显不同。

在第一次迭代中，CUDA 内存管理函数（`cudaMalloc` 和 `cudaFree`）的调用会显著拖慢执行，因为它们会长时间阻塞 CPU 线程，从而降低 GPU 利用率。后续迭代中，这种影响消失了，因为 PyTorch 的缓存内存分配器开始复用之前分配的内存区域。

**图 2：**ResNet-50 在 GPU 上执行过程的标注轨迹。

### 6.3 基准测试

最后，我们通过把 PyTorch 与三个流行的基于计算图的深度学习框架（CNTK、MXNet 和 TensorFlow）、一个运行时定义框架（Chainer），以及一个面向生产的平台（PaddlePaddle）进行比较，来总体了解 PyTorch 在单机即时模式下的性能。附录详细说明了复现实验环境所需的全部步骤。

结果汇总在表 1 中。在所有基准上，PyTorch 的性能都处于最快框架的 17% 以内。我们认为，这一结果来自各工具都把大部分计算卸载给相同版本的 cuDNN 和 cuBLAS 库。

| 框架 | AlexNet | VGG-19 | ResNet-50 | MobileNet | GNMTv2 | NCF |
|---|---:|---:|---:|---:|---:|---:|
| Chainer | 778 ± 15 | 不适用 | 219 ± 1 | 不适用 | 不适用 | 不适用 |
| CNTK | 845 ± 8 | 84 ± 3 | 210 ± 1 | 不适用 | 不适用 | 不适用 |
| MXNet | 1554 ± 22 | 113 ± 1 | 218 ± 2 | 444 ± 2 | 不适用 | 不适用 |
| PaddlePaddle | 933 ± 123 | 112 ± 2 | 192 ± 4 | 557 ± 24 | 不适用 | 不适用 |
| TensorFlow | 1422 ± 27 | 66 ± 2 | 200 ± 1 | 216 ± 15 | 9631 ± 1.3% | 4.8e6 ± 2.9% |
| PyTorch | 1547 ± 316 | 119 ± 1 | 212 ± 2 | 463 ± 17 | 15512 ± 4.8% | 5.4e6 ± 3.4% |

**表 1：**使用 32 位浮点数训练 6 个模型的速度。AlexNet、VGG-19、ResNet-50 和 MobileNet 的吞吐量以每秒图像数表示；GNMTv2 以每秒 token 数表示；NCF 以每秒样本数表示。每个模型的最快速度以粗体显示。原论文表格中的粗体在纯文本转录中无法保留。

### 6.4 采用情况

设计决策对易用性的影响很难直接衡量。作为替代指标，我们尝试统计机器学习社区对 PyTorch 的接受程度：计算自 PyTorch 于 2017 年 1 月首次发布以来，arXiv 论文中各种机器学习工具（包括 Caffe、Chainer、CNTK、Keras、MXNet、PyTorch、TensorFlow 和 Theano）被提及的频率。

图 3 报告了每月提到“PyTorch”的论文数量，占所有提到这些深度学习框架的论文数量的百分比。同一篇论文多次提到某个工具时只计一次；搜索不区分大小写，以涵盖不同拼写形式。

**图 3：**在每个月提到常见深度学习框架的 arXiv 论文中，提到 PyTorch 的论文所占比例。

## 7 结论与未来工作

PyTorch 通过兼顾易用性与谨慎的性能设计，已经成为深度学习研究社区中流行的工具。未来，除了继续支持深度学习领域的最新趋势和进展，我们还计划继续改进 PyTorch 的速度和可扩展性。

其中最重要的工作是 PyTorch JIT：一套允许 PyTorch 程序脱离 Python 解释器执行的工具，并可以在解释器之外进一步优化这些程序。我们还计划通过提供高效的数据并行原语，以及围绕远程过程调用构建的、符合 Python 习惯的模型并行库，改进对分布式计算的支持。

## 8 致谢

我们感谢 PyTorch 社区提供的反馈和贡献，它们极大地影响了 PyTorch 的设计与实现。我们也感谢所有 PyTorch 核心团队成员、贡献者和软件包维护者。原论文在此处列出了完整贡献者名单；为便于与原文和项目记录核对，姓名保留原文拼写：Ailing Zhang、Alex Suhan、Alfredo Mendoza、Alican Bozkurt、Andrew Tulloch、Ansha Yu、Anthony Shoumikhin、Bram Wasti、Brian Vaughan、Christian Puhrsch、David Reiss、David Riazati、Davide Libenzi、Dmytro Dzhulgakov、Dwaraj Rajagopal、Edward Yang、Elias Ellison、Fritz Obermeyer、George Zhang、Hao Lu、Hong Xu、Hung Duong、Igor Fedan、Ilia Cherniavskii、Iurii Zdebskyi、Ivan Kobzarev、James Reed、Jeff Smith、Jerry Chen、Jerry Zhang、Jiakai Liu、Johannes M. Dieterich、Karl Ostmo、Lin Qiao、Martin Yuan、Michael Suo、Mike Ruberry、Mikhail Zolothukhin、Mingzhe Li、Neeraj Pradhan、Nick Korovaiko、Owen Anderson、Pavel Belevich、Peter Johnson、Pritam Damania、Raghuraman Krishnamoorthi、Richard Zou、Roy Li、Rui Zhu、Sebastian Messmer、Shen Li、Simon Wang、Supriya Rao、Tao Xu、Thomas Viehmann、Vincent Quenneville-Belair、Vishwak Srinivasan、Vitaly Fedyunin、Wanchao Liang、Wei Yang、Will Feng、Xiaomeng Yang、Xiaoqiang Zheng、Xintao Chen、Yangqing Jia、Yanli Zhao、Yinghai Lu 和 Zafar Takhirov。

## 参考文献

以下参考文献条目保留原文格式，以便检索原论文和软件项目。

[1] Yangqing “Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor” Darrell. “caffe: Convolutional architecture for fast feature embedding”. “arXiv preprint arXiv:1408.5093”, “2014”.

[2] Frank Seide and Amit Agarwal. Cntk: Microsoft’s open-source deep-learning toolkit. In *Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, KDD ’16, pages 2135–2135, New York, NY, USA, 2016. ACM.

[3] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[4] Theano Development Team. Theano: A Python framework for fast computation of mathematical expressions. arXiv e-prints, abs/1605.02688, May 2016.

[5] Seiya Tokui, Kenta Oono, Shohei Hido, and Justin Clayton. Chainer: a next-generation open source framework for deep learning. In *Proceedings of Workshop on Machine Learning Systems (LearningSys) In The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS)*, 2015.

[6] Ronan Collobert, Samy Bengio, and Johnny Mariéthoz. Torch: a modular machine learning software library. Technical report, Idiap, 2002.

[7] G. Neubig, C. Dyer, Y. Goldberg, et al. DyNet: The Dynamic Neural Network Toolkit. arXiv e-prints, January 2017.

[8] Philip S. Abrams. An APL Machine. PhD thesis, Stanford University, 1970.

[9] The MathWorks, Inc. MATLAB and Statistics Toolbox.

[10] R Core Team. R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria.

[11] Jeff Bezanson, Alan Edelman, Stefan Karpinski, and Viral B. Shah. Julia: A fresh approach to numerical computing. *SIAM Review*, 59(1):65–98, 2017.

[12] Travis Oliphant. *NumPy: A Guide to NumPy*. Trelgol Publishing, 2006.

[13] Gaël Guennebaud, Benoît Jacob, et al. Eigen v3, 2010.

[14] Yann LeCun and Léon Bottou. Lush reference manual. Technical report, 2002.

[15] Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. Automatic differentiation in machine learning: A survey. *Journal of Machine Learning Research*, 18(1):5595–5637, January 2017.

[16] Dougal Maclaurin. *Modeling, Inference and Optimization with Composable Differentiable Procedures*. PhD thesis, Harvard University, April 2016.

[17] Matthew Johnson et al. Jax, 2018.

[18] Mike Innes et al. Flux.jl, 2018.

[19] Eric Jones, Travis Oliphant, Pearu Peterson, et al. SciPy: Open source scientific tools for Python, 2001–.

[20] Wes McKinney. Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference*, pages 51–56, 2010.

[21] Pierre Sermanet, Koray Kavukcuoglu, and Yann LeCun. Eblearn: Open-source energy-based learning in C++. In *2009 21st IEEE International Conference on Tools with Artificial Intelligence*, pages 693–697, 2009.

[22] Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan D. Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer. cuDNN: Efficient primitives for deep learning. CoRR, abs/1410.0759, 2014.

[23] Andrew Lavin. maxDNN: An efficient convolution kernel for deep learning with Maxwell GPUs, January 2015.

[24] Andrew Lavin and Scott Gray. Fast algorithms for convolutional neural networks. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4013–4021, 2016.

[25] Ronan Collobert, Koray Kavukcuoglu, and Clément Farabet. Torch7: A MATLAB-like environment for machine learning. In *NIPS 2011*, 2011.

[26] Richard Gabriel. The rise of worse is better.

[27] Yann LeCun and Corinna Cortes. MNIST handwritten digit database.

[28] Oriol Vinyals, Timo Ewalds, Sergey Bartunov, et al. StarCraft II: A new challenge for reinforcement learning. CoRR, abs/1708.04782, 2017.

[29] DMLC. DLPack: Open in-memory tensor structure.

[30] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in PyTorch. In *NIPS Workshop*, 2017.

[31] Dan Piponi. Automatic differentiation, C++ templates, and photogrammetry. *Journal of Graphics, GPU, & Game Tools*, 9(4):41–55, 2004.

[32] Holger Leuck and Hans-Hellmut Nagel. Automatic differentiation facilitates integration into steering-angle-based road vehicle tracking. In *1999 Conference on Computer Vision and Pattern Recognition*, pages 2360–2365, 1999.

[33] The Python team. The CPython global interpreter lock.

[34] Giovanni Petrantoni and Jörg Wollenschläger. NimTorch.

[35] Austin Huang, Junji Hashimoto, and Sam Stites. Hasktorch.

[36] G. Synnaeve, Z. Lin, J. Gehring, et al. Forward modeling for partial observation strategy games - a StarCraft defogger. In *Advances in Neural Information Processing Systems*, pages 10761–10771, 2018.

[37] The PyTorch team. TorchScript.

[38] Justin Luitjens. CUDA streams. GPU Technology Conference, 2014.

[39] Emery D. Berger, Kathryn S. McKinley, Robert D. Blumofe, and Paul R. Wilson. Hoard: A scalable memory allocator for multithreaded applications. In *ASPLOS IX*, pages 117–128, 2000.

[40] J. Evans. A scalable concurrent malloc(3) implementation for FreeBSD. In *BSDCan*, May 2006.

[41] S. Ghemawat and P. Menage. TCMalloc: Thread-caching malloc.

[42] Benjamin Recht, Christopher Ré, Stephen J. Wright, and Feng Niu. Hogwild: A lock-free approach to parallelizing stochastic gradient descent. In *Advances in Neural Information Processing Systems 24*, pages 693–701, 2011.

[43] Matthew Hertz and Emery D. Berger. Quantifying the performance of garbage collection vs. explicit memory management. In *OOPSLA ’05*, pages 313–326, 2005.

[44] The PyTorch team. PyTorch Autograd Profiler.
