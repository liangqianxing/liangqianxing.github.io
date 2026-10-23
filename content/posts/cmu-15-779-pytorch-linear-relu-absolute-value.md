---
title: 'CMU 15-779 学习记录 04：多输出线性层、ReLU 与绝对值网络'
date: '2026-09-12'
description: '把 nn.Linear 对应到矩阵乘法，弄清多个输出怎样计算，再逐层拆开一个绝对值网络，比较手动设置参数与训练得到的结果。'
series: CMU 15-779
seriesOrder: 4
cover: /images/posts/cmu-15-779-pytorch-linear-relu-absolute-value/cmu-pytorch-absolute-cover-20261015.webp
coverAlt: 双分支经过折角模块后汇入紫色 V 形曲线面板的浅色技术教具插画
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs/tree/ad12568eba934c385d0fcb971cf0288098e28553
attachmentUrl: /materials/cmu-15-779/experiments/pytorch-linear-relu-abs-20261015.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - PyTorch
  - 矩阵乘法
  - ReLU
  - 神经网络
draft: false
hidden: false
haloSync: false
haloPublished: false
---

> 本篇整理的是学习 CMU 15-779 时补充的 PyTorch 预备练习。脚本编号不对应课程讲次，也不是课程官方作业。

接着[前一篇的批量推理与矩阵乘法](/posts/cmu-15-779-pytorch-dataloader-checkpoints-batching/)，我继续把 `model(x)` 拆开看：一层里有多个输出时，每个数是怎么算出来的？两层之间加上 ReLU，又改变了哪一步？

这次最卡住我的地方，是一个手动设置参数的绝对值网络。代码只有两层线性层和一个 ReLU，但看到输出后，我还是问了句“没搞懂怎么算出来的”。后来把输入 `-2` 沿着网络逐步代进去，才把每层的运算对应起来。

本篇对应 `16_matmul.py` 的追加对照，以及 `17_multi_output.py` 到 `19_learn_abs.py`。前两个脚本和训练实验有实际运行记录；逐层计算与跳过 ReLU 的对照在整理时复跑核对。运行设备仍是 Mac 的 CPU。

[查看本篇 GitHub 代码](https://github.com/liangqianxing/cmu-15-779-labs/tree/ad12568eba934c385d0fcb971cf0288098e28553) · [下载本篇代码 ZIP](/materials/cmu-15-779/experiments/pytorch-linear-relu-abs-20261015.zip)

## 把 nn.Linear 和矩阵乘法对上

上一份脚本用手动给定的权重和偏置计算：

```python
x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
weights = torch.tensor([[2., 3.]])
bias = torch.tensor([1.])

prediction = x @ weights.T + bias
```

这里有 3 个样本，每个样本有 2 个特征。计算的形状是：

```text
输入          转置后的权重      输出
[3, 2]    @   [2, 1]       →   [3, 1]
```

例如第一个样本 `[1, 2]`，得到 `1 × 2 + 2 × 3 + 1 = 9`。三个样本的输出依次为 `9、19、29`。

接下来创建 `nn.Linear(2, 1)`，把相同的参数复制进去：

```python
model = torch.nn.Linear(2, 1)

with torch.no_grad():
    model.weight.copy_(weights)
    model.bias.copy_(bias)
    model_prediction = model(x)

print(torch.allclose(prediction, model_prediction))
```

运行结果为 `True`。这次对照验证了当前这组参数和输入下，两种写法得到的结果近似一致。`allclose` 按容差比较浮点数，不是要求每个数的二进制表示完全相同。

期间碰到的错误是把属性写成 `model.weights`。PyTorch 的属性名是单数 `model.weight`，而 `weights` 是我在脚本里定义的变量名，可以这样区分：

```python
model.weight.copy_(weights)
#     属性名          本地变量
```

`copy_()` 将数值复制到已有参数中，末尾的下划线表示原地修改。这里在 `torch.no_grad()` 中手动赋值，没有训练，也没有更新梯度。

## 两个输出，就是两组权重各算一次

`17_multi_output.py` 将模型改为 `nn.Linear(2, 2)`：

```python
model = torch.nn.Linear(2, 2)
weights = torch.tensor([
    [2.0,  3.0],
    [4.0, -1.0],
])
bias = torch.tensor([1.0, 2.0])
```

第一个 `2` 是每个样本的输入特征数，第二个 `2` 是每个样本的输出数。权重的每一行负责一个输出。对输入 `[a, b]`，分别计算：

$$
y_1 = 2a + 3b + 1, \qquad y_2 = 4a - b + 2.
$$

| 输入 | 第一个输出 | 第二个输出 |
| --- | ---: | ---: |
| `[1, 2]` | 9 | 4 |
| `[3, 4]` | 19 | 10 |
| `[5, 6]` | 29 | 16 |

我的终端输出是：

```text
输入形状： torch.Size([3, 2])
权重形状： torch.Size([2, 2])
偏置形状： torch.Size([2])
预测值： tensor([[ 9.,  4.],
        [19., 10.],
        [29., 16.]])
输出形状： torch.Size([3, 2])
```

输出中，每行对应一个样本，每列对应一个输出。偏置的两个数分别加到对应的输出列上。

对于这里的二维批量输入，`nn.Linear(in_features, out_features)` 的权重形状是 `[out_features, in_features]`，所以计算时使用转置：

$$
Y = XW^{\mathsf{T}} + b.
$$

之前用过的 `nn.Linear(1, 8)` 也按这个方式理解：每个样本输入一个数，经过 8 组不同的权重和偏置，得到 8 个数。它并不表示输入有 8 个样本。

## 把绝对值网络逐层代入

`18_relu_steps.py` 使用一个更小的网络：

```text
一个输入 → Linear(1, 2) → ReLU → Linear(2, 1) → 一个输出
```

这一次先手动指定参数，观察前向计算：

```python
fc1 = nn.Linear(1, 2)
relu = nn.ReLU()
fc2 = nn.Linear(2, 1)

with torch.no_grad():
    fc1.weight.copy_(torch.tensor([[1.0], [-1.0]]))
    fc1.bias.zero_()
    fc2.weight.copy_(torch.tensor([[1.0, 1.0]]))
    fc2.bias.zero_()

    hidden = fc1(x)
    activated = relu(hidden)
    prediction = fc2(activated)
```

第一层的两组权重分别是 `1` 和 `-1`，偏置都是 `0`，因此把每个输入 `x` 变成 `[x, -x]`。`.zero_()` 就是把张量中的数值原地设成零。

以 `-2` 为例，三步计算是：

```text
第一层：1 × (-2) = -2，(-1) × (-2) = 2
        得到 [-2, 2]

ReLU：  负数变为 0，非负数保持原值
        得到 [0, 2]

第二层：0 × 1 + 2 × 1 = 2
```

五个输入都按同样的方法计算。整理时复跑的结果与下表一致：

| 输入 | 第一层 `[x, -x]` | ReLU 后 | 第二层相加 |
| ---: | --- | --- | ---: |
| -2 | `[-2, 2]` | `[0, 2]` | 2 |
| -1 | `[-1, 1]` | `[0, 1]` | 1 |
| 0 | `[0, 0]` | `[0, 0]` | 0 |
| 1 | `[1, -1]` | `[1, 0]` | 1 |
| 2 | `[2, -2]` | `[2, 0]` | 2 |

输入形状为 `[5, 1]`，第一层与 ReLU 后都是 `[5, 2]`，最终输出为 `[5, 1]`。ReLU 改变了数值，没有改变张量形状。

这组参数实现的是：

$$
\max(0, x) + \max(0, -x) = |x|.
$$

对于任意实数，一路保留正向部分，另一路保留负向部分的大小，最后相加。

## 跳过 ReLU 会发生什么

整理时还临时把第二层的输入改成第一层的原始输出：

```python
with torch.no_grad():
    prediction_without_relu = fc2(hidden)
```

输出全部为 `0`。原因可以直接代入当前参数：

$$
x + (-x) = 0.
$$

“全部为零”取决于这次手动设置的权重和偏置。一般情况下，两层之间没有激活函数，仍可合并为一个仿射变换。用列向量表示单个样本时：

$$
W_2(W_1x+b_1)+b_2
= (W_2W_1)x + (W_2b_1+b_2).
$$

加上 ReLU 后，中间会根据数值正负改变计算结果，才能用这组参数拼出绝对值的折角。ZIP 中保留了原始脚本的 ReLU 版本；跳过它的这一段是整理时的额外对照。

## 从手动参数走到训练

`19_learn_abs.py` 又换回熟悉的 `1 → 8 → 1` 网络，这次从随机参数开始学习绝对值：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        hidden = self.fc1(x)
        activated = self.relu(hidden)
        return self.fc2(activated)

torch.manual_seed(0)
model = MyModel()
x = torch.linspace(-1.0, 1.0, 21).reshape(-1, 1)
target = x.abs()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
```

`x.abs()` 生成每个输入的绝对值，作为训练答案。每一步用全部 21 个样本计算均方误差，然后反向传播并更新参数，共训练 3000 步。

每隔 500 步，脚本用更新后的模型重新计算一次训练集损失。我的实际输出为：

```text
第 500次：loss=0.00053269
第1000次：loss=0.00013145
第1500次：loss=0.00010063
第2000次：loss=0.00008567
第2500次：loss=0.00007076
第3000次：loss=0.00005678
```

三个检查输入在训练前后的预测是：

| 输入 | 目标绝对值 | 训练前 | 训练后 |
| ---: | ---: | ---: | ---: |
| -0.8 | 0.8 | -0.2028 | 0.8022 |
| 0.0 | 0.0 | 0.0089 | 0.0227 |
| 0.8 | 0.8 | 0.2678 | 0.8001 |

整体训练损失下降了，但 `0` 这个点的预测从约 `0.0089` 变成了 `0.0227`，单点误差反而变大。优化器降低的是全体训练样本的平均损失，并不保证每个点都同时变好。

这三个检查点也包含在 21 个训练点中，不能用它们来说明独立测试集上的效果。当前结果能说明这份网络对训练数据的拟合已有改善，尚未验证区间外表现。

手动设置的 `1 → 2 → 1` 网络能按公式精确构造绝对值；训练得到的 `1 → 8 → 1` 网络仍有拟合误差。训练并不要求找到和手工构造相同的参数。

## 代码与复现

本篇 ZIP 包含以下文件，和代码仓库中的本次发布版本对应：

| 文件 | 内容 |
| --- | --- |
| `16_matmul.py` | 手写矩阵乘法与 `nn.Linear` 的对照 |
| `17_multi_output.py` | 两个输入特征、两个输出 |
| `18_relu_steps.py` | 手动设参数，逐层计算绝对值 |
| `19_learn_abs.py` | 从随机参数训练绝对值网络 |
| `requirements.txt`、`README.md` | 依赖版本、运行方式与结果说明 |

创建环境后，可以依次运行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 16_matmul.py
.venv/bin/python 17_multi_output.py
.venv/bin/python 18_relu_steps.py
.venv/bin/python 19_learn_abs.py
```

前三个脚本使用手动指定的参数，不需要训练或加载模型文件。最后一个脚本设置了随机种子，在本次 CPU 环境复跑时与记录一致；不同运行环境的浮点末位可能存在差异。

[GitHub 代码（本篇版本）](https://github.com/liangqianxing/cmu-15-779-labs/tree/ad12568eba934c385d0fcb971cf0288098e28553) · [下载本篇代码 ZIP](/materials/cmu-15-779/experiments/pytorch-linear-relu-abs-20261015.zip)
