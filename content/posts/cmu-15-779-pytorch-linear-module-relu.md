---
title: 'CMU 15-779 学习记录 02：nn.Linear、张量形状与 ReLU'
date: '2026-10-15'
description: '从带偏置的线性回归开始，弄清 nn.Linear 和 nn.Module 的分工，再用平方函数观察线性模型的限制、ReLU 的作用和测试点的选择。'
series: CMU 15-779
seriesOrder: 2
cover: /images/posts/cmu-15-779-pytorch-linear-module-relu/cmu-pytorch-relu-cover-20261015.webp
coverAlt: 输入经过两层神经网络和 ReLU 折角模块，输出曲线拟合结果的技术插画
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs
attachmentUrl: /materials/cmu-15-779/experiments/pytorch-models-20261015.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - PyTorch
  - 神经网络
  - ReLU
draft: false
hidden: false
haloSync: false
haloPublished: false
---

[上一篇](/posts/cmu-15-779-pytorch-autograd-linear-regression/)里，我用一个参数拟合了 $y=2x$。2026 年 10 月 15 日继续练习，先加上偏置，再把手写参数换成 `nn.Linear`，最后尝试拟合平方函数。

这一段对应代码仓库里的 `04_linear_bias.py` 到 `10_more_data.py`。环境仍是 Python 3.11、PyTorch 2.14.1，使用 CPU。

[查看 GitHub 代码](https://github.com/liangqianxing/cmu-15-779-labs) · [下载本篇代码 ZIP](/materials/cmu-15-779/experiments/pytorch-models-20261015.zip)

## 加一个偏置，学出 y = 2x + 1

这次输入还是 `[1, 2, 3, 4]`，目标改为 `[3, 5, 7, 9]`。模型也增加一个参数：

$$
\hat y = wx+b
$$

`bias` 就是偏置 $b$。$w$ 控制直线的斜率，$b$ 控制整体上下移动。只用 $wx$ 时，直线必须经过原点；有了 $b$，就可以拟合截距为 1 的这组数据。

`04_linear_bias.py` 仍然手动声明两个张量：

```python
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.05)

prediction = w * x + b
loss = ((prediction - target) ** 2).mean()
```

我在这里又问了一遍：有 `optimizer.step()`，为什么还需要 `loss.backward()`？两者的工作不同。`backward()` 计算梯度，写入 `w.grad`、`b.grad`；`step()` 读取这些梯度并更新参数。训练循环依然要依次清除旧梯度、计算损失、反向传播，再更新参数。

最初只训练 100 次，打印出的参数约为 `w=2.0216`、`b=0.9366`。增加到 500 次后，我得到：

```text
第500次：w=2.0001,b=0.9998,loss=0.0000
新输入： tensor([5., 6.])
预测值： tensor([11.0001, 13.0002])
```

训练次数影响了这个例子的收敛程度。打印出来的 `0.0000` 仍是四位小数的显示结果。

## model(x) 在计算什么

接下来把手写的 `w`、`b` 换成：

```python
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
prediction = model(x)
```

我当时问，`model(x)` 是不是会自动识别公式。这里的计算形式由 `nn.Linear` 决定，仍然是输入乘权重再加偏置。训练负责调整参数，输入数据不会让这一层自动变成别的函数。

`Linear(1, 1)` 表示每个样本有 1 个输入特征、输出 1 个数。参数在 `model.weight` 和 `model.bias` 中，`model.parameters()` 把需要训练的参数交给优化器。

这一版把输入也改成二维：

```python
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
target = torch.tensor([[3.0], [5.0], [7.0], [9.0]])
```

这里的形状是 `[4, 1]`，即 4 个样本，每个样本 1 个特征。我的一次输出为 `w=1.9999`、`b=1.0003`，对 `[[5], [6]]` 的预测接近 `[[11], [13]]`。这一层使用随机初始化，所以没有固定种子的几次运行，最后几位小数可能不同。

## 两个输入特征，权重为什么是 [1, 2]

`06_two_inputs.py` 使用下面的训练数据：

```python
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
target = torch.tensor([[1.0], [3.0], [4.0], [6.0]])
model = torch.nn.Linear(2, 1)
```

它要学的关系是：

$$
\hat y = w_1x_1+w_2x_2+b
\quad\longrightarrow\quad
3x_1+2x_2+1
$$

训练后，两个权重接近 `[3, 2]`，偏置接近 1。输入 `[[2, 3], [6, 7]]` 时，我得到的预测约为 `[[12.9990], [32.9969]]`，与目标 13、33 接近。

| 张量 | 本次形状 | 含义 |
| --- | --- | --- |
| `x` | `[4, 2]` | 4 个样本，每个样本有 2 个特征 |
| `model.weight` | `[1, 2]` | 1 个输出，对应 2 个输入权重 |
| `model.bias` | `[1]` | 1 个输出偏置 |
| `model(x)` | `[4, 1]` | 每个样本得到 1 个预测 |

打印参数时还遇到一个小错误：`model.weight.tolist()` 返回列表，后面不能直接加 `:.4f`。这次直接打印列表；单个数才用 `.item()` 取值后格式化。

## 用 nn.Module 包住自己的模型

`07_custom_model.py` 将同一层放进自定义类：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)
```

`nn.Module` 是 PyTorch 模型的基类。把层赋给 `self.linear` 后，它会作为子模块登记，里面的参数也能通过 `model.parameters()` 找到。`forward()` 定义输入经过哪些运算，使用时仍然调用 `model(x)`。

`super().__init__()` 初始化父类。在这个普通实例方法中，它与 `super(MyModel, self).__init__()` 的作用相同。

包了一层以后，我还按旧写法访问 `model.weight`，于是得到：

```text
AttributeError: 'MyModel' object has no attribute 'weight'
```

参数现在属于内部的 `linear` 层，应写成 `model.linear.weight` 和 `model.linear.bias`。后来把网络改成 `fc1`、`fc2` 后，旧的 `model.linear` 也失效了。模型结构变了，检查参数的代码也要跟着改。

查看全部参数时，可以用：

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

## 平方函数为什么被拟合成了 0.5

`08_square.py` 把目标换成 $y=x^2$，训练点是 `[-1, -0.5, 0, 0.5, 1]`，但模型还是单个 `Linear(1, 1)`。

第一次预测时，我把三个输入写成了 `[[-1, 0, 1]]`。它的形状是 `[1, 3]`，表示一个有 3 个特征的样本，与这一层要求的 1 个特征不符，因此报错：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 1x1)
```

改成 `[[-1], [0], [1]]` 后，程序能运行了，但三个预测都接近 0.5。参数也显示权重接近 0，偏置接近 0.5。

这个结果与模型的限制有关。单个线性层只能给出直线，训练数据却沿平方曲线分布。对这五个对称样本，最小二乘意义下的最佳直线是水平线：

$$
w=0,\qquad
b=\frac{1+0.25+0+0.25+1}{5}=0.5
$$

继续训练这条直线，也无法让它弯成抛物线。

## 加入 ReLU，再检查训练点之外的预测

`09_square_relu.py` 使用两层线性层，中间插入 ReLU：

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
```

ReLU 计算 $\max(0,z)$。如果中间没有非线性激活，两层这样的仿射变换合起来仍是一个仿射变换。加入 ReLU 后，这个小网络可以用分段线性的函数近似曲线。

| 参数 | 形状 | 元素数 |
| --- | --- | ---: |
| `fc1.weight` | `[8, 1]` | 8 |
| `fc1.bias` | `[8]` | 8 |
| `fc2.weight` | `[1, 8]` | 8 |
| `fc2.bias` | `[1]` | 1 |

总共 25 个参数。从这一步开始，脚本在创建模型前设置了 `torch.manual_seed(0)`，方便复跑时比较结果。训练仍用 SGD，学习率为 0.05，更新 3000 次。

训练损失打印为 `0.00000000` 后，我又输入了 `[-0.75, 0.25, 0.75]`：

| 输入 | 真实平方值 | 预测值 |
| ---: | ---: | ---: |
| -0.75 | 0.5625 | 0.6250 |
| 0.25 | 0.0625 | 0.0845 |
| 0.75 | 0.5625 | 0.6250 |

这三个点的均方误差为 `0.0027660925406962633`。五个训练点拟合得很准，点与点之间仍有误差。例如 0.5 和 1 对应的目标是 0.25 和 1，连接它们的直线在 0.75 处给出 0.625，而真正的平方值是 0.5625。

## 增加数据时，也要重新检查测试点

`10_more_data.py` 保留网络和优化器，将训练点改为：

```python
x = torch.linspace(-1.0, 1.0, 25).reshape(-1, 1)
target = x ** 2
```

`linspace` 生成等间距数值；`reshape(-1, 1)` 将其整理成每个样本一个特征。相同三个比较点上的预测变成约 `0.5892`、`0.0675`、`0.5972`，均方误差降到 `0.0006467862986028194`。

整理代码时需要纠正这里的实验解释：这 25 个等间距训练点已经包含了 `-0.75`、`0.25` 和 `0.75`。脚本虽然仍把变量叫作 `test_x`，这一次得到的数值实际是在训练样本上重新计算的，不能与前面的误差直接组成独立测试集上的泛化对比。

后面的保存模型实验改用 21 个训练点，并用 `[-0.73, 0.23, 0.77]` 检查预测。这些点没有出现在那组训练数据中，具体结果放在[下一篇记录](/posts/cmu-15-779-pytorch-dataloader-checkpoints-batching/)里。

## 本篇代码

| 脚本 | 对应练习 |
| --- | --- |
| `04_linear_bias.py` | 手写权重与偏置，拟合 $y=2x+1$ |
| `05_nn_linear.py` | 用 `nn.Linear(1, 1)` 替换手写参数 |
| `06_two_inputs.py` | 两个输入特征与参数形状 |
| `07_custom_model.py` | 用 `nn.Module` 定义自己的模型 |
| `08_square.py` | 单个线性层拟合平方函数的限制 |
| `09_square_relu.py` | 使用 ReLU，检查未参与训练的三个点 |
| `10_more_data.py` | 增加到 25 个训练点，核对比较点是否重叠 |

[代码仓库](https://github.com/liangqianxing/cmu-15-779-labs)和[本篇 ZIP](/materials/cmu-15-779/experiments/pytorch-models-20261015.zip)包含脚本与依赖说明。建立环境后，可先运行 `.venv/bin/python 04_linear_bias.py`，再按编号逐个运行。
