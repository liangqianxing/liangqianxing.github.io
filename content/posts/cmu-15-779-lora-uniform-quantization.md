---
title: 'CMU 15-779 学习记录 12：LoRA 首步梯度、权重合并与均匀 4-bit 打包'
date: '2026-10-06'
description: '用一次 CPU SGD 更新检查 LoRA 初始化、冻结基座与合并等价，再分别记录均匀 4-bit 存储字节数和量化误差。'
series: CMU 15-779
seriesOrder: 12
cover: /images/posts/cmu-15-779-lora-uniform-quantization/cmu-lora-cover-20261019.webp
coverAlt: 浅色手绘的大计算板、两块细长适配板和成对打包的小方块
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs/tree/c4851e0cedadf13af7c421867e83c8ff57244fa0
attachmentUrl: /materials/cmu-15-779/experiments/cpu-parallel-serving-lora-moe-20261019.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - LoRA
  - Quantization
draft: false
hidden: false
haloSync: false
haloPublished: false
---

> 第 17 讲的个人配套实验，35 号脚本实际执行一次 CPU LoRA 更新，并实现均匀 4-bit 打包。实验检查机制与数值误差，不评估微调任务效果，也没有实现完整 QLoRA。

这次把两个问题分开记录：低秩分支是否能合并进原权重，低比特保存又会引入多少误差。源码与原始结果保存在实验快照中；整理时复跑，两部分结果均与原报告一致。

[查看代码版本](https://github.com/liangqianxing/cmu-15-779-labs/tree/c4851e0cedadf13af7c421867e83c8ff57244fa0) · [下载 31 至 37 号代码与结果 ZIP](/materials/cmu-15-779/experiments/cpu-parallel-serving-lora-moe-20261019.zip)

## 192 个基座参数与 84 个可训练参数

脚本沿用 PyTorch 的 `W[out,in]` 约定，冻结 `W[12,16]`，训练 `A[3,16]` 和 `B[12,3]`，rank 为 3、alpha 为 6，缩放系数 s 为 2：

$$
Y=XW^T+s(XA^T)B^T,\qquad s=\frac{\alpha}{r}.
$$

基座参数量为 `12×16=192`，适配器为 `3×16+12×3=84`。W 以 buffer 保存，优化器只更新 A、B。降低可训练参数量不等于基座计算和激活都消失。

A 使用随机小值，B 初始化为零。初始低秩分支输出为零，与基座输出的最大误差为 0。对 7 个样本计算 MSE 并反向后，梯度范数为：

| 参数 | 首步梯度范数 |
| --- | ---: |
| A | 0 |
| B | 0.1857457504 |

由于 A 的梯度路径乘上 B，零 B 会使 A 的首步梯度为零；B 仍可通过非零 A 获得梯度。若 A、B 都置零，这两个因子的更新会遇到零梯度问题。

## 合并检查保持同一个表达式

脚本执行一次学习率 `0.1` 的 SGD，确认基座 W 保持原值。然后构造：

$$
W_{\mathrm{merged}}=W+sBA.
$$

低秩分支输出与 `X @ W_merged.T` 的最大绝对误差为 `1.78e-15`。单元测试另使用非零 A、B，核对两种表达式对输入及适配器参数的梯度。

这个对照使用同一组高精度权重，差异来自浮点计算顺序。实验只验证一个线性模块和一步更新，没有训练真实语言模型，也没有任务精度结论。

## 两个 4-bit 码打包进一个字节

量化实验使用 16 个在 `[-1,1]` 上均匀分布的码点，每 64 个权重共享一个 absmax 尺度，尺度保存为 FP32。两个码打包到一个 uint8：

```python
packed = codes[::2] | (codes[1::2] << 4)
# 解包
low = packed & 15
high = packed >> 4
```

192 个权重恰好分成 3 块，code 占 96B，3 个 FP32 尺度占 12B，合计 108B。原始 float64 权重数据为 1536B；这些数字只比较张量数据，不包含 Python 对象和其他运行内存。非整除情况还需计入末块填充，现有测试覆盖了这类边界。

| 原始报告的数值对照 | 最大绝对误差 |
| --- | ---: |
| 反量化权重与原始权重 | 0.1754975613 |
| 当前输入的线性输出 | 0.8775516177 |

这次量化会改变权重，输出误差不能套用 LoRA 合并的等价容限。运行时如果同时保留打包张量和反量化后的权重，总驻留内存也会超过 108B。

均匀码本没有实现 QLoRA 的 NF4 非均匀码点，也未实现 double quantization、paged optimizer 或低比特计算 kernel。这里的 LoRA 更新和量化对照是两项独立检查，不能合称已经完成 QLoRA 训练。

## 复现与课程材料

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 35_lora_quantization.py --output results/35-rerun.json
.venv/bin/python -m unittest discover -s tests -v
```

共享 ZIP 包含 31 至 37 号冻结脚本及 22 项测试，其中 3 项直接检查 LoRA 与量化。保留完整共享测试文件所需模块后，可独立运行；完整发布仓库的 52 项测试也已通过。

可对照原课[微调课件](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/17-LLM-finetuning.pdf)、[LoRA](https://arxiv.org/abs/2106.09685)和 [QLoRA](https://arxiv.org/abs/2305.14314)。真实 GPU 微调按学习安排跳过，论文阅读确认与课程项目仍单独验收。
