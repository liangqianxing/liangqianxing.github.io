---
title: 'Stanford CS336 学习记录 01：从 Linear、Embedding 搞清张量形状'
date: '2026-10-20'
description: '从 PyTorch 类、参数与 forward 的关系开始，理解 Linear 为什么保存两行三列的权重、计算时为什么要转置，再区分 Embedding 查表与线性变换。'
series: Stanford CS336
seriesOrder: 1
cover: /images/posts/cs336-linear-embedding-shapes/cs336-linear-embedding-shapes-cover.webp
coverAlt: token 方块、向量表与矩阵变换的 AI 辅助概念插画
categories:
  - 学习笔记
tags:
  - CS336
  - PyTorch
  - Transformer
  - Linear
  - Embedding
draft: false
hidden: false
haloSync: false
haloPublished: false
codeUrl: https://github.com/liangqianxing/cs336-labs/tree/a39eb684710817f7f389626423fcb26ca0f6d629
---

2026 年 10 月 20 日，开始沿着 Stanford CS336 的 Assignment 1 学习。BPE 之前已经接触过，这轮先跳过分词器，从 Transformer 的基础层开始。这里记录的是当时写代码时真正卡住的问题，以及对应测试能确认到的范围。

起步时，我连类的骨架都没完全写顺。`nn.Module`、`nn.Parameter`、`forward` 看起来都与“模型”有关，但分别负责什么并不清楚。与其急着拼一个完整网络，不如先把一个输入 3 维、输出 2 维的 Linear 看明白。

## 类保存什么，forward 计算什么

一个带可训练参数的层，需要同时描述两件事：它拥有什么参数，以及输入来了以后怎样使用这些参数。

`__init__` 在对象创建时准备参数。`forward` 描述一次前向计算。同一个层可以处理很多批输入，权重保存在对象里，不会因为每来一批数据就重新创建。

| 名字 | 在这一层里的工作 |
| --- | --- |
| `nn.Module` | 提供模块、子模块和参数管理机制 |
| `super().__init__()` | 初始化父类的模块管理机制 |
| `nn.Parameter` | 把张量登记为可训练参数 |
| `self.weight` | 保存这个层自己的权重 |
| `forward` | 定义输入怎样变成输出 |

调用层时，通常使用 `layer(x)`。它经过 `nn.Module` 的调用机制，执行 `forward`，也保留 PyTorch 提供的 hooks 等功能。可以把 `forward` 理解为主要的计算内容，但不能把二者的实现机制完全画上等号。

`layer` 也不是特殊关键字，只是给对象起的变量名。前面反复出现“先创建 layer，再把输入传进去”，原因是这个对象需要保存和复用权重。

## 输入 3 维、输出 2 维，权重到底是几行几列

当时被问到权重形状时，我回答了“三行两列”。这个回答对应着一种可以成立的数学存储方向，但本次作业约定权重按 `(out_features, in_features)` 存储，所以应是两行三列。

每一行生成一个输出，每一列对应一个输入特征。用一组便于手算的数：

$$
x=[1,2,3],\qquad
W=\begin{bmatrix}
0.1&0.2&0.3\\
0.4&0.5&0.6
\end{bmatrix}.
$$

第一个输出把输入与第一行逐项相乘再求和：

$$
y_1=1\times0.1+2\times0.2+3\times0.3=1.4.
$$

同样，第二个输出是 $3.2$。最后得到 $[1.4,3.2]$，特征数从 3 变成了 2。

输入按行摆放时，对应的矩阵计算是：

$$
Y=XW^\top,
\qquad
(B,3)\ @\ (3,2)\longrightarrow(B,2).
$$

已经写好的 `linear.py` 中，核心计算只有一行：

```python
return x @ self.weight.T
```

这里的 `@` 做矩阵乘法，`.T` 把二维权重从 `(2, 3)` 转成 `(3, 2)`。转置服务于所选的存储约定；判断方向时，先写出形状，比死记“Linear 一定要转置”可靠。

批量和序列维度也能保留。例如输入是 `(2, 5, 3)`，可以理解为两段序列、每段五个 token、每个 token 三个特征。输出是 `(2, 5, 2)`。这个 Linear 对所有位置应用同一组权重，改变的是最后一维，不会让第一个 token 从第二个 token 那里读取信息。

## 形状、device 和 dtype 是三个独立问题

创建参数时，`torch.empty` 后面既有数字，又有 `device`、`dtype`，最初很容易把它们当成一组含义相近的配置。已有代码中的创建部分是：

```python
torch.empty(out_features, in_features, device=device, dtype=dtype)
```

前两个整数决定形状；`device` 决定张量放在哪里；`dtype` 决定每个元素使用什么数值类型。

| 配置 | 例子 | 回答的问题 |
| --- | --- | --- |
| shape | `(2, 3)` | 有几行、几列 |
| device | `cpu`、`cuda`、`mps` | 数据在哪个计算设备上 |
| dtype | `torch.float32` | 每个数怎样存储和计算 |

这次先在 CPU 上验证前向计算。构造函数接受 `device=None, dtype=None`，表示允许调用者沿用 PyTorch 当前的默认设置。测试适配器则会采用测试权重的设备和类型，避免输入、参数的配置对不上。

## empty 申请空间，初始化决定起点

`empty` 并不是“得到一张全零矩阵”。它分配指定形状的空间，其中的值尚未按模型需要初始化，不能直接拿来当可靠的起始参数。

这份作业的 Linear 使用截断正态初始化，标准差按以下规则选取：

$$
\sigma=\sqrt{\frac{2}{d_{\mathrm{in}}+d_{\mathrm{out}}}}.
$$

均值设为 0，截断边界是 $[-3\sigma,3\sigma]$。输入 3 维、输出 2 维时，$\sigma\approx0.632$，边界大约是 $[-1.897,1.897]$。

这里需要分清标准差和方差；公式外面还有一个平方根。`trunc_normal_` 结尾的下划线则表示操作直接修改已有张量，把抽到的数填入 `self.weight`。初始化只是训练的起点，之后参数仍然需要通过学习调整。

## 测试里的 weights 与 in_features

接上官方 `tests/adapters.py` 时，我又问了一次：`weights` 是什么，`in_features` 又是什么？

适配器拿到的是测试提供的具体数据：`weights` 是指定权重张量，`in_features` 是实际输入张量。适配器创建自己的层，把指定权重加载到层里，再返回这一层对输入的计算结果。

同一个名字在不同函数里可能表示不同东西，尤其是 `in_features`：

| 出现位置 | 本次含义 | 类型 |
| --- | --- | --- |
| `Linear.__init__` | 每个输入有几个特征 | 整数 |
| 官方 `run_linear` | 交给层计算的实际输入 | 张量 |

外部的 `weights` 与内部的 `self.weight` 也不是同一个变量。`load_state_dict` 按名称将测试提供的数值加载到层的参数中。因此这项测试主要核对给定输入、给定权重下的前向计算，不能据此推断随机初始化也已经被完整验证。

## Embedding 是取行，输出多出一个特征维度

Linear 之后写的是 Embedding。它也保存可训练权重，不过输入变成了整数 token ID。

假设词表有 1000 个 token，每个 token 用 64 个数表示，参数表的形状就是 `(1000, 64)`。ID 为 2，就取编号为 2 的行，也就是第三行。

已经写好的 `embedding.py` 用下面这行完成查表：

```python
return self.weight[token_ids]
```

输入 ID `[2, 0]`，就按这个顺序取出第三行和第一行，输出形状为 `(2, 64)`。对于一批序列：

```text
整数 ID：  (batch, seq_len)
权重表：   (vocab_size, embedding_dim)
输出向量：(batch, seq_len, embedding_dim)
```

输入有多少个位置，输出就有多少个向量。Embedding 在这里没有执行 Linear 那样的特征加权求和。它与 Linear 都有可训练参数，具体的前向运算却不同。

本次 Embedding 的初始化使用标准差 1、范围 $[-3,3]$ 的截断正态，不能照抄 Linear 计算标准差的规则。

## 这一步真正遇到的错误

除了数学方向，我还在文件位置和 Python 语法上花了不少时间。最初文件放在仓库根目录，检查时找的是 `cs336_basics/linear.py`，于是出现了“文件明明在，为什么没读到”的误会。模块导入最终依赖的是包目录下的文件。

占位语句也带来过混淆：`NotImplementedError` 曾拼错；写完 `return` 后，后面又留着不会执行的 `raise NotImplementedError`。Embedding 里还写过 `return NotImplementedError`，它会把异常类当成返回结果，并不会报出“尚未实现”的异常。

第一次跑 Embedding 测试时，程序甚至还没有开始比较数值，就在收集测试阶段报了：

```text
SyntaxError: positional argument follows keyword argument
```

原因是适配器中的数据类型参数写成了 `dtype.dtype`，本意是把测试权重的数据类型传给构造函数。这个错误提醒我，先区分“测试没有运行起来”和“运行后结果不对”，排查方向会清楚很多。

## 已验证到哪里

修正上述问题后，历史运行记录里，官方 `test_linear` 与 `test_embedding` 分别得到 `1 passed`。这确认了这两个基础组件在对应官方用例中的结果。

这一阶段还没有训练语言模型，也没有验证完整的 Assignment 1。接下来继续看 [RMSNorm 和 SwiGLU](/posts/cs336-rmsnorm-swiglu/)，重点从矩阵方向转到广播和逐元素门控。

## 发布时的复核

2026 年 10 月 20 日，发布前对[独立代码快照](https://github.com/liangqianxing/cs336-labs/tree/a39eb684710817f7f389626423fcb26ca0f6d629)重新运行了 8 项官方组件测试，结果为 `8 passed`。完整清单和运行环境见[进度记录](https://github.com/liangqianxing/cs336-labs/blob/a39eb684710817f7f389626423fcb26ca0f6d629/docs/progress.md)。

*封面使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 辅助生成，属于概念插画；准确的形状、公式与运算关系以正文为准。*

## 参考资料

- [Stanford CS336 官方课程页面](https://cs336.stanford.edu/)
- [Assignment 1 官方仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)
- [PyTorch：nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch：nn.Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)
- [PyTorch：torch.empty](https://docs.pytorch.org/docs/stable/generated/torch.empty.html)
- [PyTorch：torch.matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
- [PyTorch：nn.Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
