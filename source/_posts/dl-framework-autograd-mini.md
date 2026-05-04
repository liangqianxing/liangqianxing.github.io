---
title: Week 1：DL 框架与 Autograd——从计算图、反向传播到 Mini Autograd 实现
date: 2026-05-05
categories: 技术
tags:
  - 深度学习
  - Autograd
  - PyTorch
  - CMU 10-414
  - Mini Framework
---

如果只用一句话概括 PyTorch / TensorFlow 的本质：**它们是在张量计算之上，自动构建计算图，并用链式法则自动求梯度的系统**。训练神经网络看起来是调用 `loss.backward()` 和 `optimizer.step()`，但底层真正发生的是：前向阶段记录依赖关系，反向阶段沿图逆序传播梯度，同时在显存、计算量和调度开销之间做工程权衡。

这篇文章对应 Week 1 的学习目标，只聚焦 CMU 10-414 中最核心的四块：Computation Graph、Backpropagation、Automatic Differentiation、Memory Optimization。最后我会写一个只支持 `matmul`、`relu`、`softmax`、`cross_entropy` 的 mini autograd，用最少代码把深度学习框架的骨架讲清楚。
<!-- more -->

![Forward and backward graph](/images/posts/mini-autograd/graph.svg)

## 1. 为什么先学 Autograd

深度学习框架表面上提供了很多能力：GPU 张量、神经网络层、优化器、数据加载、分布式训练、模型导出。但如果把这些能力拆到底，核心闭环只有四步：

```text
参数 W 初始化
  -> forward 计算预测值
  -> loss 衡量预测和标签差距
  -> backward 计算每个参数的梯度
  -> optimizer 用梯度更新参数
```

其中最重要的是 `backward`。因为模型训练不是手写每个参数的求导公式，而是让框架自动求导。例如一个两层 MLP：

```python
logits = relu(x @ w1) @ w2
loss = cross_entropy(logits, y)
loss.backward()
```

框架需要知道：

- `logits` 是怎么由 `x`、`w1`、`w2` 算出来的；
- `loss` 对 `logits` 的梯度是多少；
- `relu`、`matmul`、`cross_entropy` 各自的局部导数是什么；
- 多条路径汇合时，梯度如何累加；
- 哪些中间激活要保存，哪些可以反向时重新计算。

所以学 Autograd，本质是在学 PyTorch / TensorFlow 的“训练引擎”。

## 2. Computation Graph：计算图是什么

计算图是一个有向无环图。节点代表数据或操作，边代表依赖关系。

以这段代码为例：

```python
z = x @ w
h = relu(z)
loss = cross_entropy(h, y)
```

可以拆成：

```text
x ----\
       matmul -> z -> relu -> h -> cross_entropy -> loss
w ----/                                      ^
                                            y
```

这里有两类对象：

1. **Tensor / Value 节点**：保存真实数据，例如 `x`、`w`、`z`、`h`、`loss`。
2. **Op / Function 节点**：保存操作规则，例如 `matmul`、`relu`、`cross_entropy`。

动态图框架 PyTorch 通常在 Python 执行前向代码时即时建图。你写 `z = x @ w`，框架马上生成一个新 Tensor，并让这个 Tensor 记住：它来自 `x`、`w`，创建它的操作是 `matmul`。

静态图框架早期 TensorFlow 1.x 则先定义图，再统一编译执行。静态图更利于全局优化，动态图更符合 Python 调试习惯。今天 PyTorch 2.x 通过 `torch.compile` 又把动态图捕获为可优化图，本质上是在易用性和编译优化之间折中。

## 3. Forward Graph：前向图保存什么

前向阶段不只是算数值，还要为反向传播留线索。一个 Tensor 通常至少需要这些字段：

```python
class Tensor:
    data        # 真实数值，例如 numpy.ndarray
    grad        # loss 对当前 Tensor 的梯度
    requires_grad # 是否需要追踪梯度
    parents     # 这个 Tensor 依赖哪些父 Tensor
    op          # 是哪个操作创建了它
    backward_fn # 给定上游梯度，如何算父节点梯度
```

例如：

```python
z = x @ w
```

会创建一个新 Tensor `z`。它的 `parents` 是 `(x, w)`，它的 `backward_fn` 会记录矩阵乘法的导数规则：

```text
z = x @ w
如果上游梯度是 dz，则：
dx = dz @ w.T
dw = x.T @ dz
```

这里的“上游梯度”就是 `loss` 对 `z` 的梯度，通常写作 `dL/dz`。反向函数负责把 `dL/dz` 变成 `dL/dx` 和 `dL/dw`。

## 4. Backpropagation：反向传播的本质

反向传播不是神秘算法，它就是链式法则在计算图上的系统化应用。

假设：

```text
x -> z -> h -> loss
```

那么：

```text
dloss/dx = dloss/dh * dh/dz * dz/dx
```

如果图里有分叉和汇合，例如：

```text
      -> a ->
x           + -> loss
      -> b ->
```

那么 `x` 的梯度需要把所有路径的贡献加起来：

```text
dloss/dx = dloss/da * da/dx + dloss/db * db/dx
```

这就是 Autograd 里 `grad += contribution` 的原因。一个参数可能被多次使用，或者一个中间结果可能影响多个后续节点，梯度天然需要累加。

反向传播的执行顺序必须是拓扑逆序：先算离 loss 最近的节点，再算更早的节点。因为某个节点只有收齐所有下游贡献，才能继续把梯度传给它的父节点。

## 5. Automatic Differentiation：自动微分不是数值微分

常见的求导方法有三种：

| 方法 | 思路 | 问题 |
|---|---|---|
| 手动求导 | 人写每层公式 | 模型一复杂就不可维护 |
| 数值微分 | 用 `(f(x+eps)-f(x))/eps` 近似 | 慢且有数值误差 |
| 自动微分 | 把程序拆成基本算子并套链式法则 | 框架主流方案 |

自动微分不是符号求导。它不会把整个程序化简成一个数学表达式，而是在程序实际执行时记录每一步基本操作，然后对每个基本操作调用已知的局部反向规则。

自动微分主要有两种模式：

1. **Forward-mode AD**：从输入往输出推导导数，适合输入维度小、输出维度大的场景。
2. **Reverse-mode AD**：从输出往输入反传梯度，适合深度学习，因为 loss 通常是一个标量，而参数量巨大。

神经网络训练几乎都用 reverse-mode AD。一次 backward 就能得到所有参数对同一个标量 loss 的梯度。

## 6. 四个核心算子的反向公式

下面是 mini autograd 要支持的四个算子。

### 6.1 MatMul

前向：

```text
C = A @ B
```

形状：

```text
A: [N, D]
B: [D, M]
C: [N, M]
```

反向：

```text
dA = dC @ B.T
dB = A.T @ dC
```

直觉：矩阵乘法把 `A` 的每一行和 `B` 的每一列做内积。上游梯度 `dC` 告诉我们每个输出元素对 loss 的影响，再乘回另一个输入，就得到当前输入的梯度。

### 6.2 ReLU

前向：

```text
relu(x) = max(x, 0)
```

反向：

```text
dx = dout * (x > 0)
```

如果前向时 `x <= 0`，ReLU 输出被截断为 0，局部导数为 0；如果 `x > 0`，ReLU 是恒等映射，局部导数为 1。

### 6.3 Softmax

前向：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

实际实现必须做数值稳定：

```python
shifted = x - max(x)
exp = np.exp(shifted)
prob = exp / exp.sum()
```

如果不减最大值，`exp(1000)` 很容易溢出。

Softmax 的完整 Jacobian 是：

```text
∂s_i/∂x_j = s_i * (1(i=j) - s_j)
```

在代码里我们通常不显式构造 `[C, C]` 的 Jacobian，而是直接写向量-Jacobian 乘积：

```text
dx = s * (dout - sum(dout * s))
```

### 6.4 Cross Entropy

对于分类任务，标签 `y` 是类别 id，预测 `p` 是 softmax 概率：

```text
loss = -log(p_y)
```

如果 batch size 是 `N`：

```text
loss = mean_i -log(p[i, y_i])
```

对 softmax 概率的梯度：

```text
dp[i, y_i] = -1 / p[i, y_i] / N
```

工程里通常会把 `softmax + cross_entropy` 融合成一个更稳定的 `cross_entropy_with_logits`：

```text
dlogits = (softmax(logits) - one_hot(y)) / N
```

这也是 PyTorch 中 `torch.nn.CrossEntropyLoss` 接收 logits 而不是 softmax 后概率的原因。

## 7. Mini Autograd 完整代码

这个实现只依赖 NumPy，代码目标不是功能完整，而是把 autograd 的核心结构讲清楚。

```python
import numpy as np


def ensure_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


class Tensor:
    def __init__(self, data, requires_grad=False, parents=(), op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.parents = tuple(parents)
        self.op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self.op!r})"

    def __matmul__(self, other):
        other = ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="matmul",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="relu",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=axis, keepdims=True)
        out = Tensor(
            probs,
            requires_grad=self.requires_grad,
            parents=(self,),
            op="softmax",
        )

        def _backward():
            if self.requires_grad:
                dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
                self._accumulate_grad(probs * (out.grad - dot))

        out._backward = _backward
        return out

    def cross_entropy(self, target):
        target = np.asarray(target, dtype=np.int64)
        probs = np.clip(self.data, 1e-12, 1.0)
        batch_indices = np.arange(target.shape[0])
        loss_value = -np.log(probs[batch_indices, target]).mean()
        out = Tensor(
            loss_value,
            requires_grad=self.requires_grad,
            parents=(self,),
            op="cross_entropy",
        )

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[batch_indices, target] = -1.0 / probs[batch_indices, target]
                grad /= target.shape[0]
                self._accumulate_grad(out.grad * grad)

        out._backward = _backward
        return out

    def backward(self, grad=None):
        if grad is None:
            if self.data.shape != ():
                raise RuntimeError("grad must be specified for non-scalar tensors")
            grad = np.ones_like(self.data)

        topo = []
        visited = set()

        def build_topo(tensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))
            for parent in tensor.parents:
                build_topo(parent)
            topo.append(tensor)

        build_topo(self)
        self.grad = grad

        for tensor in reversed(topo):
            tensor._backward()

    def zero_grad(self):
        self.grad = None

    def _accumulate_grad(self, grad):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad
```

## 8. 每个结构体和函数细讲

### 8.1 `Tensor.data`

`data` 是真实数值。这里用 `np.ndarray` 保存。真实框架里，Tensor 不只保存数据，还要保存设备、dtype、stride、storage、layout 等信息。例如 PyTorch Tensor 可能在 CPU 或 CUDA 上，可能是 `float32`、`float16`、`bfloat16`，也可能是非连续内存视图。

### 8.2 `Tensor.requires_grad`

`requires_grad` 表示是否需要追踪梯度。输入数据通常不需要梯度，模型参数需要梯度。

```python
x = Tensor([[1, 2]], requires_grad=False)
w = Tensor([[0.1], [0.2]], requires_grad=True)
```

如果一个操作的任意父节点需要梯度，那么输出也需要梯度：

```python
requires_grad=self.requires_grad or other.requires_grad
```

这就是梯度追踪在图上传播的方式。

### 8.3 `Tensor.grad`

`grad` 保存当前 Tensor 的梯度，也就是 `dLoss/dTensor`。注意它不是局部导数，而是最终 loss 对这个 Tensor 的总导数。

为什么初始是 `None` 而不是 0？因为这样可以区分“还没算过梯度”和“梯度确实为 0”。真实训练中每个 step 前都要清空梯度，否则梯度会跨 batch 累加。

```python
w.zero_grad()
loss.backward()
```

PyTorch 也是默认累加梯度，所以训练循环里必须写 `optimizer.zero_grad()`。

### 8.4 `Tensor.parents`

`parents` 保存当前 Tensor 的输入依赖。例如 `z = x @ w`，那么 `z.parents = (x, w)`。反向传播构建拓扑排序时要沿着 `parents` 一直追溯到叶子节点。

叶子节点一般是用户直接创建的 Tensor，例如输入和参数。中间节点是由操作产生的 Tensor。

### 8.5 `Tensor.op`

`op` 只是为了调试展示。真实框架里会有更复杂的 `grad_fn` 或 `Function` 对象，里面保存算子类型、反向规则、上下文缓存等。

例如 PyTorch 里：

```python
z = x @ w
print(z.grad_fn)
```

你会看到类似 `MmBackward` 的对象。

### 8.6 `Tensor._backward`

`_backward` 是 mini autograd 的核心。每个操作在前向时创建输出 Tensor，并给输出 Tensor 塞一个闭包函数。这个闭包知道：

- 当前操作的输入是谁；
- 当前操作的输出是谁；
- 输出的梯度 `out.grad` 如何转成输入的梯度。

例如 matmul：

```python
def _backward():
    if self.requires_grad:
        self._accumulate_grad(out.grad @ other.data.T)
    if other.requires_grad:
        other._accumulate_grad(self.data.T @ out.grad)
```

闭包会捕获 `self`、`other`、`out`。这就是动态图框架非常自然的地方：Python 执行到哪里，反向函数就记录到哪里。

### 8.7 `ensure_tensor`

`ensure_tensor` 负责把普通数字或数组包装成 Tensor。这样以后可以支持：

```python
x @ np_array
```

真实框架中类似逻辑会更复杂，因为要处理 dtype promotion、device 对齐、广播规则等。

### 8.8 `backward`

`backward` 做三件事：

1. 如果当前 Tensor 是标量 loss，则默认上游梯度为 1；
2. 从 loss 出发 DFS 构建拓扑序；
3. 逆拓扑序调用每个 Tensor 的 `_backward`。

关键代码：

```python
for tensor in reversed(topo):
    tensor._backward()
```

为什么要反过来？因为 `topo` 是从叶子到 loss 的顺序，反向传播必须从 loss 回到叶子。

### 8.9 `_accumulate_grad`

梯度必须累加：

```python
self.grad = self.grad + grad
```

如果一个 Tensor 影响 loss 的路径不止一条，每条路径都会贡献一部分梯度。Autograd 的正确性依赖累加，而不是覆盖。

## 9. 跑一个最小训练例子

下面构造一个两层分类模型：

```python
np.random.seed(0)

x = Tensor(np.array([
    [1.0, 2.0, 1.0],
    [2.0, 0.0, 1.0],
    [0.0, 1.0, 2.0],
]), requires_grad=False)

y = np.array([0, 1, 1])

w1 = Tensor(np.random.randn(3, 4) * 0.1, requires_grad=True)
w2 = Tensor(np.random.randn(4, 2) * 0.1, requires_grad=True)

for step in range(50):
    for p in (w1, w2):
        p.zero_grad()

    logits = (x @ w1).relu() @ w2
    probs = logits.softmax(axis=1)
    loss = probs.cross_entropy(y)
    loss.backward()

    lr = 0.5
    w1.data -= lr * w1.grad
    w2.data -= lr * w2.grad

    if step % 10 == 0:
        print(step, loss.data)
```

完整过程是：

```text
x @ w1
  -> relu
  -> @ w2
  -> softmax
  -> cross_entropy
  -> backward
  -> w1.grad / w2.grad
  -> SGD update
```

这个例子虽然小，但已经包含了深度学习训练最重要的机制。

## 10. 梯度流：为什么会消失或爆炸

梯度流指梯度从 loss 往前层传播的过程。每经过一个操作，梯度都会乘上局部导数。

如果很多局部导数小于 1，梯度会越来越小，形成梯度消失；如果很多局部导数大于 1，梯度会越来越大，形成梯度爆炸。

```text
dL/dx = dL/dh_n * dh_n/dh_{n-1} * ... * dh_1/dx
```

这解释了很多网络设计：

- ReLU 比 sigmoid 更常用，因为正区间导数是 1，更利于梯度通过；
- ResNet 用残差连接，让梯度可以沿 identity path 直接传播；
- LayerNorm / BatchNorm 缓解激活分布漂移；
- 合理初始化让前向激活和反向梯度保持稳定尺度；
- 梯度裁剪可以防止 RNN / Transformer 中的梯度爆炸。

理解梯度流之后，很多训练技巧不再是经验魔法，而是为了让链式法则的乘积更稳定。

## 11. Memory Optimization：显存到底花在哪里

训练时显存主要来自：

1. **参数**：模型权重；
2. **梯度**：每个参数对应一份梯度；
3. **优化器状态**：Adam 会保存一阶、二阶动量，通常是参数量的 2 倍；
4. **激活值**：前向中间结果，反向需要用；
5. **临时 buffer**：算子执行过程中的 workspace。

很多人以为显存主要被参数占用，但在大 batch、长序列、深网络里，激活值经常非常可观。因为反向传播需要前向时的中间结果。例如 ReLU backward 要知道前向输入是否大于 0，matmul backward 要知道另一个输入矩阵。

## 12. Activation Checkpoint：用计算换显存

![Activation checkpoint](/images/posts/mini-autograd/checkpoint.svg)

Activation checkpoint 的核心思想：**不要保存所有中间激活，只保存少量 checkpoint；反向传播时，把缺失的中间激活重新算一遍**。

普通训练：

```text
forward: 保存 a1, a2, a3, a4
backward: 直接使用 a1, a2, a3, a4
```

Checkpoint：

```text
forward: 只保存 a0, a4
backward: 从 a0 重新计算 a1, a2, a3，再做局部 backward
```

代价很清楚：

| 方案 | 显存 | 计算 |
|---|---|---|
| 保存所有激活 | 高 | 低 |
| Activation checkpoint | 低 | 高 |

这就是 memory vs compute tradeoff。训练大模型时，如果显存是瓶颈，宁愿多算一点，也要把 batch size、sequence length 或模型规模撑起来。

PyTorch 中常用：

```python
from torch.utils.checkpoint import checkpoint

out = checkpoint(block, x)
```

使用 checkpoint 时要注意：

- 被 checkpoint 的函数最好是纯函数，不要依赖会变化的外部状态；
- dropout 等随机操作需要正确处理 RNG 状态；
- 不是所有层都值得 checkpoint，通常选显存占用大的 block；
- checkpoint 会增加训练时间，不是免费优化。

## 13. PyTorch 和 TensorFlow 本质上做了什么

当你写 PyTorch：

```python
loss = model(x).softmax(dim=-1).log().mean()
loss.backward()
```

框架做的事情和 mini autograd 没有本质区别，只是工程复杂度高很多：

- Tensor 支持 CPU / GPU / 多种 dtype；
- 算子调用高性能 kernel，例如 cuBLAS、cuDNN、Triton；
- Autograd engine 处理拓扑调度、并行反传、线程安全；
- View / inplace / broadcasting 有复杂的梯度语义；
- 编译器会做图捕获、融合、常量折叠、内存规划；
- 分布式训练会插入通信算子，例如 all-reduce；
- 混合精度会处理 loss scaling 和 fp16/bf16 数值稳定。

但心智模型仍然是：

```text
Tensor + Op + Graph + Chain Rule + Scheduler + Memory Planner
```

## 14. Mini Autograd 的限制

这个 mini 版本故意省略了很多真实框架能力：

- 不支持 broadcasting 的梯度反推；
- 不支持 view、reshape、transpose 的复杂共享内存语义；
- 不支持 inplace 修改检测；
- 不支持 GPU；
- 不支持高阶梯度；
- 不支持动态图释放和显存复用；
- `softmax` 和 `cross_entropy` 分开实现，数值稳定性不如融合版。

但它已经完整覆盖了深度学习框架最核心的训练闭环。

## 15. Week 1 学完应该掌握什么

学完这一周，不要求背公式，而是要能解释清楚这些问题：

1. 前向计算时，框架为什么要记录计算图？
2. `loss.backward()` 为什么要按拓扑逆序执行？
3. 多条路径指向同一个 Tensor 时，梯度为什么要累加？
4. `matmul`、`relu`、`softmax`、`cross_entropy` 的反向公式是什么？
5. 为什么深度学习训练主要用 reverse-mode AD？
6. 为什么训练比推理更耗显存？
7. Activation checkpoint 为什么能省显存，代价是什么？
8. PyTorch 的动态图和 TensorFlow 的静态图各有什么取舍？

如果这些问题都能讲顺，基本就理解了深度学习框架的本质。

## 16. 最后总结

Autograd 的核心非常朴素：前向时记录图，反向时套链式法则。复杂的是工程化：如何让它在 GPU 上快、在大模型上省显存、在动态图里易调试、在编译图里可优化。

Mini autograd 的意义不是替代 PyTorch，而是帮我们建立底层直觉。只要理解了 `Tensor` 如何保存 `parents` 和 `_backward`，理解了拓扑逆序和梯度累加，再看 PyTorch、TensorFlow、JAX 的设计，就不会只停留在 API 层，而能真正理解它们为什么这样工作。

补充：完整可运行代码已放在 [/downloads/code/mini_autograd.py](/downloads/code/mini_autograd.py)，可以直接下载运行。

