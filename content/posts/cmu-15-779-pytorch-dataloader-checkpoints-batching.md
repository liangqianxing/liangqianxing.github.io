---
title: 'CMU 15-779 学习记录 03：DataLoader、模型保存与批量推理'
date: '2026-10-15'
description: '用 DataLoader 分批训练平方函数，保存并加载模型参数，再比较 1000 个样本与 1 个样本的推理耗时，最后把批量计算对应到矩阵乘法。'
series: CMU 15-779
seriesOrder: 3
cover: /images/posts/cmu-15-779-pytorch-dataloader-checkpoints-batching/cmu-pytorch-batching-cover-20261015.webp
coverAlt: 样本卡片按批次进入计算模块，旁边配有参数存储盒和输出矩阵面板的技术插画
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs
attachmentUrl: /materials/cmu-15-779/experiments/pytorch-batching-20261015.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - PyTorch
  - DataLoader
  - 批量推理
  - 模型保存
draft: false
hidden: false
haloSync: false
haloPublished: false
---

接着[前一篇的平方函数实验](/posts/cmu-15-779-pytorch-linear-module-relu/)，我在 2026 年 10 月 15 日继续练习分批训练、保存参数和独立预测。最后做了一次计时：同一个小网络，逐个处理 1000 个样本与一次处理整批样本，耗时相差很大；把样本数改为 1 后，差距又明显缩小。

本篇对应 `11_batches.py` 到 `16_matmul.py`。网络仍是 `Linear(1, 8) → ReLU → Linear(8, 1)`，运行设备是 Mac 的 CPU。

[查看 GitHub 代码](https://github.com/liangqianxing/cmu-15-779-labs) · [下载本篇代码 ZIP](/materials/cmu-15-779/experiments/pytorch-batching-20261015.zip)

## DataLoader 怎样把样本分成批次

`11_batches.py` 先只观察数据，不训练模型：

```python
from torch.utils.data import DataLoader, TensorDataset

x = torch.linspace(-1.0, 1.0, 21).reshape(-1, 1)
target = x ** 2
dataset = TensorDataset(x, target)
loader = DataLoader(dataset, batch_size=5, shuffle=False)

for batch_index, (batch_x, batch_target) in enumerate(loader, start=1):
    print(batch_index, batch_x.shape, batch_target.shape)
```

`TensorDataset` 按位置将输入与目标配成一对，`DataLoader` 再从中取出批次。这里有 21 个样本，每批最多 5 个，得到的样本数依次是 `5、5、5、5、1`。最后一个批次只有 1 个样本，输入和目标的形状都变成 `[1, 1]`。

`shuffle=False` 表示按原顺序读取，便于这次观察。它不会自动把样本顺序打乱。

## 一个 epoch 会更新几次参数

`12_mini_batch.py` 把熟悉的训练步骤放进批次循环：

```python
for epoch in range(600):
    for batch_x, batch_target in loader:
        optimizer.zero_grad()
        prediction = model(batch_x)
        loss = ((prediction - batch_target) ** 2).mean()
        loss.backward()
        optimizer.step()
```

一个 batch 是一次用来计算损失和梯度的样本集合，一个 epoch 是把整个训练集遍历一次。

这份脚本使用 25 个训练点，`batch_size=5`，每个 epoch 有 5 批，也就更新 5 次参数。600 个 epoch 总共更新 3000 次。

| 脚本 | 样本数 | 每批样本数 | 每个 epoch 的批次数 |
| --- | ---: | ---: | ---: |
| `11_batches.py` | 21 | 最多 5 | 5，最后一批为 1 |
| `12_mini_batch.py` | 25 | 5 | 5 |
| `13_save_load.py` | 21 | 最多 5 | 5，最后一批为 1 |

每批计算的梯度都来自这一批数据，所以分批更新与用全部数据一次更新，会沿着不同的参数轨迹前进。即使更新次数相同，也不能当作完全相同的训练过程。

打印每个 epoch 的损失时，代码重新对整个训练集计算：

```python
with torch.no_grad():
    new_loss = ((model(x) - target) ** 2).mean()
```

这样得到的是全体训练样本的均方误差。若最后一批只有 1 个样本，直接平均各批的平均损失，会给最后一个样本过高的权重；重新计算全量损失可以避开这个问题。

`12_mini_batch.py` 最后在 `[-0.75, 0.25, 0.75]` 上打印的误差是 `0.0006145233637653291`。这三个点同样包含在该脚本的 25 个训练点里，含义与前一篇说明的一样，属于训练样本上的检查。

## 保存参数，再恢复同一个模型

`13_save_load.py` 使用 21 个等间距训练点、每批最多 5 个样本，训练 600 个 epoch。检查输入改成了 `[-0.73, 0.23, 0.77]`，这三个点没有出现在训练集中。

我的运行记录为：

```text
第 100次：loss=0.00350851
第 200次：loss=0.00156298
第 300次：loss=0.00089876
第 400次：loss=0.00065406
第 500次：loss=0.00056592
第 600次：loss=0.00051049
```

| 输入 | 目标平方值 | 预测值 |
| ---: | ---: | ---: |
| -0.73 | 0.5329 | 0.5622 |
| 0.23 | 0.0529 | 0.0643 |
| 0.77 | 0.5929 | 0.6319 |

这三个检查点的均方误差为 `0.0008369663264602423`。它们位于训练区间内，可以用来观察点间的预测；仅凭这三个点，还不能说明模型在整个区间或区间外都准确。

参数保存和恢复的核心代码是：

```python
torch.save(model.state_dict(), 'square_model.pth')

loaded_model = MyModel()
parameters = torch.load('square_model.pth', weights_only=True)
loaded_model.load_state_dict(parameters)
loaded_model.eval()
```

`state_dict()` 保存参数及持久化缓冲区的数值。这份模型里对应的是 `fc1.weight`、`fc1.bias`、`fc2.weight`、`fc2.bias`。恢复时先用 `MyModel()` 创建同样的结构，再将这些数值装回去。

比较保存前后的结果时，我一度看到两组不同的数字。检查代码后发现，后面又把输入改回了 `[-0.75, 0.25, 0.75]`。两组输出用的输入不同，不能直接比较。最后改成复用同一个 `test_x`：

```python
with torch.no_grad():
    print('比较使用的输入：', test_x)
    print('原模型预测：', model(test_x))
    print('加载后预测：', loaded_model(test_x))
```

整理时复跑确认，同一输入上的保存前后预测一致。

## 独立预测脚本只需要模型结构和参数

`14_predict.py` 没有优化器，也没有训练循环。它创建结构、加载参数，然后计算新输入的输出。独立运行后，三个预测仍是约 `0.5622`、`0.0643`、`0.6319`，误差与保存前相同。

这里碰到的拼写错误是：

```python
torch.load('square_model.pth', weight_only=True)
```

正确参数名是 `weights_only`，带一个 `s`。误写后，程序报了 `Unpickler.__init__() got an unexpected keyword argument 'weight_only'`，改正后就能加载。

`model.eval()` 和 `torch.no_grad()` 的用途也要分清。`eval()` 切换模型的训练/评估模式，会影响 Dropout、BatchNorm 等依赖模式的层；`no_grad()` 暂停记录普通运算的求导关系。这次网络只有线性层和 ReLU，切换评估模式不会改变这些层的数值计算，但预测时仍按这个习惯写。

## 1000 个样本：逐个预测与整批预测

`15_batch_benchmark.py` 加载同一份模型，比较两种处理方式：

```python
x = torch.linspace(-1.0, 1.0, 1000).reshape(-1, 1)

def predict_one_by_one():
    outputs = []
    for i in range(x.shape[0]):
        outputs.append(model(x[i:i+1]))
    return torch.cat(outputs, dim=0)

def predict_batch():
    return model(x)
```

逐个版本调用模型 1000 次，每次输入形状是 `[1, 1]`；整批版本只调用一次，输入形状是 `[1000, 1]`。切片使用 `x[i:i+1]`，可以保留批次这一维。

计时前先检查两组输出近似一致，再各预热 3 次。正式计时将同一种方式重复运行 5 次，取每次执行的平均毫秒数：

```python
def measure_ms(function):
    start = perf_counter()
    for _ in range(5):
        function()
    return (perf_counter() - start) / 5 * 1000
```

整个预测和计时过程放在 `torch.no_grad()` 中。下面保留的是我当时终端里的输出，计时会随机器状态和运行次数波动：

| 样本数量 | 输出近似一致 | 逐个预测 | 整批预测 | 当次加速比 |
| ---: | --- | ---: | ---: | ---: |
| 1000 | `True` | 13.0410 ms | 0.0139 ms | 939.32 倍 |
| 1 | `True` | 0.0120 ms | 0.0083 ms | 1.44 倍 |

加速比使用未四舍五入的计时值计算，因此用表格里的四位小数直接相除，可能与最后一列略有不同。

这个模型很小。逐个版本反复执行 Python 循环、切片、模型调用，最后还要拼接输出；整批版本把这些样本集中交给底层张量运算，减少了重复调用的开销。939.32 倍描述的是这次 CPU、小模型和 1000 个样本条件下，两段完整预测流程的耗时比。

把样本数改为 1 后，两种方式都只调用模型一次，差距明显缩小。剩余差别包含切片、列表、拼接和计时波动。这两次结果不能外推成所有模型、所有设备上固定的批量加速倍数。

当前上传的脚本保留最后一次实验使用的 **1 个样本**。要复现 1000 个样本的版本，将 `torch.linspace(-1.0, 1.0, 1)` 的最后一个参数改成 `1000` 即可。

## 用矩阵乘法理解批量输入

接着运行的 `16_matmul.py` 手动指定了权重和偏置，不进行训练：

```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
weights = torch.tensor([[2.0, 3.0]])
bias = torch.tensor([1.0])

prediction = x @ weights.T + bias
```

`@` 表示矩阵乘法，`.T` 将这里的二维权重从 `[1, 2]` 转为 `[2, 1]`。形状关系是：

$$
\underbrace{X}_{3\times2}
\underbrace{W^T}_{2\times1}
+\underbrace{b}_{1}
=\underbrace{\hat Y}_{3\times1}
$$

偏置沿着样本维度广播。每一行都计算 $2x_1+3x_2+1$，一次得到三个样本的预测。我运行后得到：

```text
输入形状： torch.Size([3, 2])
转置后的权重形状： torch.Size([2, 1])
预测值： tensor([[ 9.],
        [19.],
        [29.]])
输出形状： torch.Size([3, 1])
```

`nn.Linear(2, 1)` 对二维输入做的也是 $XW^T+b$。这次手动给定的参数让我能直接核对每个结果，也把前面 `model(x)` 接收整批样本的写法对应到了矩阵运算。

## 代码与运行顺序

| 文件 | 内容 |
| --- | --- |
| `11_batches.py` | 观察 21 个样本如何分成 5 批 |
| `12_mini_batch.py` | 对 25 个样本进行分批训练 |
| `13_save_load.py` | 对 21 个样本训练，保存并恢复参数 |
| `14_predict.py` | 在独立脚本中加载参数并预测 |
| `15_batch_benchmark.py` | 比较逐个与整批预测，当前默认 1 个样本 |
| `16_matmul.py` | 手动使用矩阵乘法计算 3 个样本的输出 |
| `square_model.pth` | 本次 21 点训练实验保存的 25 个参数 |

[GitHub 仓库](https://github.com/liangqianxing/cmu-15-779-labs)和[本篇 ZIP](/materials/cmu-15-779/experiments/pytorch-batching-20261015.zip)都包含这份小模型的参数文件。建立 Python 环境后，在代码所在目录运行：

```bash
.venv/bin/python 11_batches.py
.venv/bin/python 12_mini_batch.py
.venv/bin/python 13_save_load.py
.venv/bin/python 14_predict.py
.venv/bin/python 15_batch_benchmark.py
.venv/bin/python 16_matmul.py
```

`13_save_load.py` 会在当前目录重新生成 `square_model.pth`。如果只想检查已经保存的结果，可以直接运行 `14_predict.py`；计算矩阵乘法的 `16_matmul.py` 不依赖参数文件。
