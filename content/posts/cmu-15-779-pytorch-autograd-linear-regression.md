---
title: 'CMU 15-779 学习记录 01：PyTorch 自动求导与线性回归'
date: '2026-10-13'
description: '从自动求导和手动梯度下降开始，比较学习率，再用 SGD 训练线性回归。记录 w * 4 写错后停在 1.25 的原因和修正结果。'
series: CMU 15-779
seriesOrder: 1
cover: /images/posts/cmu-15-779-pytorch-autograd-linear-regression/cmu-pytorch-autograd-cover-20261015.webp
coverAlt: 梯度曲面上的小球沿路径下降，前侧面板展示散点与拟合直线的技术插画
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs
attachmentUrl: /materials/cmu-15-779/experiments/pytorch-basics-20261013.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - PyTorch
  - 自动求导
  - 线性回归
draft: false
hidden: false
haloSync: false
haloPublished: false
---

开始学习 CMU 15-779 时，我会 Python，但还没用过 PyTorch。这次先在 Mac 上用 CPU 做基础练习：计算导数、手动更新参数，再用 SGD 从一组数据里学出线性关系。

这篇记录整理的是 2026 年 10 月 13 日的练习。我把运行结果和中间写错的地方一起留下，方便以后回看。

实验使用 Python 3.11.15、PyTorch 2.14.1，张量采用默认的 `float32`，运行设备是 CPU。三个脚本和依赖文件可以在 [GitHub 代码仓库](https://github.com/liangqianxing/cmu-15-779-labs)查看，也可以下载[实验代码 ZIP](/materials/cmu-15-779/experiments/pytorch-basics-20261013.zip)。参考阅读是 [PyTorch 中文译稿](/learning/cmu-15-779/materials/pytorch-paper-zh/)的第 4.3 节「自动微分」。

## 自动求导算的是哪个导数

最开始的例子很短：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * x

print("y =", y.item())
print("求导前 =", x.grad)

y.backward()

print("求导后 =", x.grad.item())
```

这里 $y=x^2$，导数是 $2x$。在 $x=2$ 时，函数值和导数都等于 4，但它们的含义不同。`requires_grad=True` 让 PyTorch 记录后续运算的求导关系；调用 `backward()` 后，关于 `x` 的梯度存入 `x.grad`。`.item()` 用来取出单元素张量中的数值。

我接着把计算改成了：

```python
z = y * y
z.backward()
print(x.grad)
```

这一段接在前面创建 `x`、`y` 的代码后面，替换原来的 `y.backward()`。此时求的是 $z$ 关于 $x$ 的导数：

$$
y=x^2,\qquad z=y^2=x^4
$$

$$
\frac{dz}{dx}
=\frac{dz}{dy}\frac{dy}{dx}
=2y\cdot 2x
=4x^3
$$

所以在 $x=2$ 时，`x.grad` 对应 32。我原来的注释写成了「输出 y 的梯度函数」，应当改为「z 关于 x 的导数，在 x=2 处的值」。`x.grad` 保存的是这个位置的梯度数值。

当时我还问：把 `x` 声明为 2 后，是不是只能在这个位置求导？可以换位置，但需要重新创建输入并重新计算表达式。例如，令 $x=3$ 后重新计算 $z=x^4$，这个位置的导数就是 $4\times 3^3=108$。旧的前向计算对应旧输入，不能直接用它代表新位置。

参数名称也容易写错：正确写法是 `requires_grad`，带一个 `s`；`require_grad` 和 `request_grad` 都不能作为 `torch.tensor()` 的这个参数。

## 用梯度把参数推向目标

下一步是让参数 $w$ 从 0 开始接近 3。损失函数定义为：

$$
L(w)=(w-3)^2,\qquad \frac{dL}{dw}=2(w-3)
$$

先计算损失和梯度，再沿梯度的反方向更新：

```python
import torch

w = torch.tensor(0.0, requires_grad=True)
learn_rate = 0.1

loss = (w - 3) ** 2
loss.backward()

with torch.no_grad():
    w -= learn_rate * w.grad

new_loss = (w - 3) ** 2
print(f"Old loss: {loss.item()}, New loss: {new_loss.item()}")
```

我运行后的输出是：

```text
Old loss: 9.0, New loss: 5.760000228881836
```

这一步可以手算核对。初始梯度是 $-6$，学习率为 0.1，因此：

$$
w_{\mathrm{new}}=0-0.1\times(-6)=0.6
$$

更新后的损失是 $(0.6-3)^2=5.76$，与输出一致。打印出来的长小数来自浮点数表示。

### 两种学习率，比较相同的 10 次更新

我把单次更新写成循环，每轮都重新计算损失和梯度。求导前用 `w.grad = None` 清除上一轮的梯度，再分别用 0.1 和 0.01 运行 10 次。

| 学习率 | 第 1 次更新后的 $w$ | 第 10 次更新后的 $w$ | 第 10 次更新后的损失 |
| --- | ---: | ---: | ---: |
| 0.1 | 0.6000 | 2.6779 | 0.1038 |
| 0.01 | 0.0600 | 0.5488 | 6.0085 |

在这个函数、相同初值和更新次数下，0.1 的损失下降更快。这次结果说明了两个学习率在这个实验中的差别；学习率过大仍可能造成震荡或发散。

循环中清除梯度很关键，因为 `backward()` 会累加梯度。第一轮梯度是 $-6$，更新到 $w=0.6$ 后，第二轮梯度应是 $-4.8$。如果不清除，保存的梯度会变成 $-10.8$，就不再只是当前位置这一轮的梯度。

压缩包里的 `01_autograd.py` 是连续更新版本，保留了最后一次实验使用的 `learn_rate = 0.01`。把它改为 0.1，就可以复现另一组结果。

## 换成 SGD 后，每一步负责什么

接下来用 PyTorch 的优化器替代手动更新：

```python
optimizer = torch.optim.SGD([w], lr=0.1)
```

`[w]` 告诉优化器需要更新哪些参数。这次使用的普通 SGD 没有配置动量和权重衰减，更新规则是：

$$
w_{t+1}=w_t-\eta\frac{dL}{dw}(w_t)
$$

对应的训练循环是：

```python
for step in range(10):
    optimizer.zero_grad()
    loss = (w - 3) ** 2
    loss.backward()
    optimizer.step()
```

| 调用 | 在这个循环里的作用 |
| --- | --- |
| `optimizer.zero_grad()` | 清除优化器管理的参数上一次保存的梯度 |
| `loss.backward()` | 计算梯度，写入参数的 `.grad` |
| `optimizer.step()` | 使用梯度和优化器配置更新参数 |

我当时问 `optimizer.step()` 是怎样动态调整的。这个例子里的学习率始终是 0.1，每轮变化的是梯度。$w$ 越接近 3，梯度的绝对值越小，更新距离也就越小。

手动更新可以写成：

```python
with torch.no_grad():
    w -= learn_rate * w.grad
```

这里是 `w.grad`，不能写成 `w.grad()`。它是张量属性，不是函数。

我还把 `lr=0.1` 误写成了 `lr-0.1`，运行时报 `NameError`。前者是在给优化器传参数，后者会尝试用一个名为 `lr` 的变量做减法。

### `zero_grad()` 和 `no_grad()` 容易混淆

在线性回归的代码里，每隔 10 轮有一个 `with torch.no_grad():`，我一度以为梯度也是每隔 10 轮才清除。

实际上，`optimizer.zero_grad()` 放在 `for` 循环内、`if` 条件外，每轮都执行。每隔 10 轮执行的是更新后的损失检查和打印。

`torch.no_grad()` 暂时停止记录缩进代码块内普通运算的求导关系，计算本身仍会进行，结果也可以保存。它不会清除已有梯度。离开这个代码块后，会恢复进入前的梯度记录模式。

因此，下面的检查不影响前面的训练更新，也不需要为打印结果建立新的求导关系：

```python
if (step + 1) % 10 == 0:
    with torch.no_grad():
        new_loss = ((w * x - target) ** 2).mean()
```

## 用四个样本训练线性回归

标量实验之后，我开始用一组输入和目标训练模型：

```text
x      = [1, 2, 3, 4]
target = [2, 4, 6, 8]
```

模型只有一个需要学习的参数 $w$，预测是 $\hat y_i=wx_i$。`w * x` 会把同一个参数乘到四个输入上，得到四个预测值。`x` 和 `target` 是给定的数据，这次需要求导和更新的是 `w`。

损失采用均方误差：

$$
L(w)=\frac{1}{4}\sum_{i=1}^{4}(wx_i-y_i)^2
$$

对应代码是 `((prediction - target) ** 2).mean()`：先计算每个样本的误差，再平方并求平均。

最终运行通过的代码如下，与附件中的 `03_linear.py` 使用相同的计算式和训练配置：

```python
import torch

w = torch.tensor(0.0, requires_grad=True)
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([2.0, 4.0, 6.0, 8.0])
optimizer = torch.optim.SGD([w], lr=0.01)

for step in range(100):
    optimizer.zero_grad()
    prediction = w * x
    loss = ((prediction - target) ** 2).mean()
    loss.backward()
    optimizer.step()

    if (step + 1) % 10 == 0:
        with torch.no_grad():
            new_loss = ((w * x - target) ** 2).mean()

        print(
            f"第{step + 1:3d}次："
            f"w={w.item():.4f},"
            f"loss={new_loss.item():.4f}"
        )

with torch.no_grad():
    test_x = torch.tensor([5.0, 6.0])
    print("新输入：", test_x)
    print("预测值：", w * test_x)
```

我实际运行后得到：

```text
第 10次：w=1.6063,loss=1.1628
第 20次：w=1.9225,loss=0.0451
第 30次：w=1.9847,loss=0.0017
第 40次：w=1.9970,loss=0.0001
第 50次：w=1.9994,loss=0.0000
第 60次：w=1.9999,loss=0.0000
第 70次：w=2.0000,loss=0.0000
第 80次：w=2.0000,loss=0.0000
第 90次：w=2.0000,loss=0.0000
第100次：w=2.0000,loss=0.0000
新输入： tensor([5., 6.])
预测值： tensor([10.0000, 12.0000])
```

$w$ 逐渐接近 2，在训练时没出现过的输入 `[5, 6]` 上，预测接近 `[10, 12]`。

输出中的 `:3d` 表示整数至少占 3 个字符，不够时补空格；`:.4f` 表示保留小数点后 4 位。显示为 `loss=0.0000` 是格式化后的结果，不能据此认定误差已经严格为零。

## 写成 `w * 4` 后，为什么停在 1.25

这个实验中，我把训练时的预测误写成了：

```python
prediction = w * 4
```

这样得到的是一个标量，和四个目标计算误差时会发生广播。模型给所有样本提供同一个预测值 $4w$，完全没有使用输入 `x`。

四个目标是 `[2, 4, 6, 8]`。平方误差最小的统一预测值是它们的平均值 5，因此：

$$
4w=\frac{2+4+6+8}{4}=5
\quad\Rightarrow\quad w=1.25
$$

所以参数稳定在 1.25 有具体原因：代码训练出来的是统一预测值为 5 的模型。

我后来只把检查损失的公式改成了 `w * x`，训练部分仍是 `w * 4`。这时打印的损失变了，但参数还是停在 1.25，因为参数更新依然依据传给 `backward()` 的那个训练损失。

下面是这几次运行的最终输出对比，数值按当时的打印精度记录：

| 训练时的预测 | 检查损失时的预测 | $w$ | 打印的损失 | 对 `[5, 6]` 的预测 |
| --- | --- | ---: | ---: | --- |
| `w * 4` | `w * 4` | 1.2500 | 5.0000 | `[6.2500, 7.5000]` |
| `w * 4` | `w * x` | 1.2500 | 4.2188 | `[6.2500, 7.5000]` |
| `w * x` | `w * x` | 2.0000 | 0.0000 | `[10.0000, 12.0000]` |

把训练中的那一行也改成 `prediction = w * x` 后，才得到前面列出的正确输出。这次调试让我记住：训练、检查损失和最终预测必须使用同一套模型计算式。

## 实验代码和复现方式

[GitHub 代码仓库](https://github.com/liangqianxing/cmu-15-779-labs)持续保存源代码；也可以[下载实验代码压缩包](/materials/cmu-15-779/experiments/pytorch-basics-20261013.zip)，解压后包含：

| 文件 | 内容 |
| --- | --- |
| `01_autograd.py` | 手动梯度下降，连续更新 10 次，当前学习率为 0.01 |
| `02_SGD.py` | 使用 SGD 更新标量参数，学习率为 0.1 |
| `03_linear.py` | 修正后的线性回归，训练 100 次并预测新输入 |
| `requirements.txt` | 实验环境的依赖版本 |
| `README.md` | 环境建立、运行命令和结果核对说明 |

进入解压后的目录，在 Python 3.11 环境中运行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 01_autograd.py
.venv/bin/python 02_SGD.py
.venv/bin/python 03_linear.py
```

附件保留了这次练习结束时的三个脚本。初始的 $x^2$、$x^4$ 求导示例和单次参数更新代码，则保留在这篇记录的正文中。
