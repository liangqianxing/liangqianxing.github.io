---
title: 'CMU 15-779 学习记录 13：MoE 路由、稀疏分发与容量溢出'
date: '2026-10-14'
description: '用 CPU top-2 MoE 检查稀疏分发的输出与梯度，统计 expert-token 行数，并保留容量丢弃会改变输出的反例。'
series: CMU 15-779
seriesOrder: 13
cover: /images/posts/cmu-15-779-moe-routing-capacity/cmu-moe-cover-20261019.webp
coverAlt: 浅色手绘分拣器将小珠分配到四个专家工作屋，队列呈现负载不均
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs/tree/c4851e0cedadf13af7c421867e83c8ff57244fa0
attachmentUrl: /materials/cmu-15-779/experiments/cpu-parallel-serving-lora-moe-20261019.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - MoE
  - Routing
draft: false
hidden: false
haloSync: false
haloPublished: false
---

> 第 18 讲的个人配套实验。36 号脚本串行运行 CPU 专家，检查 top-k 路由后的分发与合并；没有训练 router，也没有执行 GPU grouped GEMM 或跨设备 all-to-all。

[上一节](/posts/cmu-15-779-lora-uniform-quantization/)记录低秩适配，这次转向每个 token 只激活部分专家的计算。源码与原始报告保存在实验快照中，整理时从独立副本复跑，结果一致。

[查看代码版本](https://github.com/liangqianxing/cmu-15-779-labs/tree/c4851e0cedadf13af7c421867e83c8ff57244fa0) · [下载 31 至 37 号代码与结果 ZIP](/materials/cmu-15-779/experiments/cpu-parallel-serving-lora-moe-20261019.zip)

## 先固定 gate 的含义

输入为 13 个 token，每个 8 维，配置 4 个专家，top-2 路由。每个专家执行 `ReLU(x @ W_up) @ W_down`，中间维度为 12。

脚本先选 router logits 最大的两个专家，再只在这两个 logits 内做 softmax：

```python
selected, indices = torch.topk(logits, top_k, dim=-1)
gates = torch.softmax(selected, dim=-1)
```

对 token i，若所选专家集合为 S，输出为：

$$
y_i=\sum_{e\in S}g_{i,e}E_e(x_i).
$$

其他 MoE 架构可能采用不同的 gate 归一化约定。输出对照必须使用相同约定，否则改变的可能是模型本身。

## 稀疏分发怎样恢复原 token 顺序

代码按专家收集 token，只执行被选中的行，最后通过 `index_add_` 将结果加回原 token 位置：

```python
token_ids, slots = torch.where(indices == expert)
expert_output = expert_forward(x[token_ids], up[expert], down[expert])
output.index_add_(0, token_ids,
                  expert_output * gates[token_ids, slots, None])
```

密集参考先让全部专家处理全部 token，再选取同样的 top-2 输出并加权。默认运行的最大绝对误差为 `3.55e-15`。

| 计数 | expert-token 行数 |
| --- | ---: |
| 全部 4 个专家计算后再选择 | 52 |
| 仅执行 top-2 分配 | 26 |

默认路由给四个专家的分配数为 `[7,8,2,9]`。测试还在 top-k 为 1、2、3 时，比较稀疏与密集表达式对输入、router logits、专家权重的梯度。梯度检查针对同一次 top-k 选择，没有把离散的专家索引选择变成可微操作。

行数减半只反映执行的专家工作量。这里没有计时，分组、搬移和小矩阵调用也有成本，因此不能据此声称吞吐翻倍。

## 容量限制会丢掉什么

反例让全部 token 都倾向同样的两个专家：每行 logits 为 `[9,8,-1,-2]`，需求变成 `[13,13,0,0]`。按 `ceil(13×2/4)` 给每个专家容量 7，脚本保留 token 顺序最早的分配，丢弃其余分配，并保留原 gate，不重新归一化。

| 倾斜路由 | 每专家处理的分配数 | 丢弃分配数 |
| --- | --- | ---: |
| 不限容量 | `[13,13,0,0]` | 0 |
| 每专家容量为 7 | `[7,7,0,0]` | 12 |

容量限制后的输出与不限容量时，最大绝对差为 `11.3812313563`。这里丢弃的是 12 个 expert-token 分配；具体例子中，后 6 个 token 的两个分配都被丢弃。这个策略显式改变了输出，不能当作保持语义的等价优化。

增加空闲专家也不会自动让 router 改选它们。负载均衡损失、路由约束或无丢弃执行需要单独设计和评估，本脚本没有训练这些机制。

## 专家计算之外还有系统成本

不同专家收到的 token 数不同，会形成不同大小的矩阵任务。真实实现需要考虑分桶或排序、grouped GEMM、dispatch/combine 融合等执行问题。expert parallel 把专家放到不同设备后，还要发送输入 token 并回传结果，通常涉及 all-to-all；热点专家和网络都可能限制吞吐。

当前 CPU 实验只串行执行专家，没有测量这些 kernel 或通信成本。总参数量、每 token 激活的参数量、实际延迟需要分别观察，不能互相代替。

## 复现与课程位置

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 36_moe_dispatch.py --output results/36-rerun.json
.venv/bin/python -m unittest discover -s tests -v
```

共享 ZIP 包含 31 至 37 号脚本、原始结果及 22 项测试，其中 2 项直接检查 MoE 的输出/梯度和容量反例。完整发布仓库的 52 项测试也在冻结副本中通过。

对应原课[第 18 讲 MoE 课件](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/18-mixture-of-experts.pdf)。本篇完成的是这项 CPU 配套实验；真实 GPU 测量按学习安排跳过，课程项目、阅读理解和课程完成状态仍独立核验。
