---
title: 'CMU 15-779 学习记录 08：分块 Attention、在线 Softmax 与 KV cache'
date: '2026-10-16'
description: '在 CPU 上对照完整 Attention、在线 Softmax 分块实现与 PyTorch SDPA，检查因果掩码、浮点误差和分数块大小，理解 FlashAttention 的一部分算法基础。'
series: CMU 15-779
seriesOrder: 8
cover: /images/posts/cmu-15-779-attention-online-softmax/cmu-online-attention-cover-20261016.webp
coverAlt: 浅紫色卡片分批进入绿色计算托盘，旁边三个状态容器保存累积量的浅色手绘插画
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs/tree/9261bea6aedf31cdccbbab2541ca693fc97b3743
attachmentUrl: /materials/cmu-15-779/experiments/attention-online-softmax-kv-cache-20261019.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - Attention
  - Softmax
  - FlashAttention
  - PyTorch
draft: false
hidden: false
haloSync: false
haloPublished: false
---

> 第 5 讲 Transformers & Attention 的个人配套实验。代码只运行 CPU、单头、前向计算，包含在线归一化和 KV cache；没有实现 FlashAttention GPU kernel，也没有速度或内存峰值测量。

[上一节](/posts/cmu-15-779-matmul-tiling-tree-reduction/)通过分块共用矩阵输入。这次接到 Attention：能不能一块一块读取分数，同时得到与完整 Softmax 一致的输出？

`28_attention_online_softmax.py` 已保存完整计算、分块计算和 PyTorch SDPA MATH 的 CPU 对照结果。原始记录来自 2026 年 10 月 16 日；整理时在冻结副本中复跑了默认配置和因果配置。2026 年 10 月 19 日补入 29 号 KV cache 实验，并在独立冻结副本中复验。

[查看代码版本](https://github.com/liangqianxing/cmu-15-779-labs/tree/9261bea6aedf31cdccbbab2541ca693fc97b3743) · [下载代码与结果 ZIP](/materials/cmu-15-779/experiments/attention-online-softmax-kv-cache-20261019.zip)

## 完整 Attention 会生成什么

本次只看一个 attention head，输入形状为 `Q[N,d]`、`K[N,d]`、`V[N,dv]`：

$$
S=\frac{QK^\mathsf{T}}{\sqrt d},\qquad P=\operatorname{softmax}(S),\qquad O=PV.
$$

Softmax 沿每一行的 key 方向归一化。完整实现先得到 `N×N` 分数矩阵，再计算权重和输出。因果模式先把未来位置的分数设为负无穷，确保当前位置看不到后面的 K、V。

默认配置为 `N=64, d=16, dv=8`，使用 float64，输出形状为 `[64,8]`。SDPA 参考明确指定 `SDPBackend.MATH`，输入位于 CPU，不能把这个对照称为调用了 GPU FlashAttention。

## 为什么不能分别做 Softmax 再拼起来

一行中的所有分数共享同一个归一化分母。若每个小块各自做 Softmax，它们各自的权重和都等于 1；把这些权重拼接，无法得到整行的权重。

分块实现需要保存每一行已经看过的最大分数 m、指数和 l，以及未归一化的输出向量 u。遇到新分数块 s 和对应值向量块 V 后，更新为：

$$
\begin{aligned}
m'&=\max(m,\max_j s_j),\\
\alpha&=\exp(m-m'),\\
p_j&=\exp(s_j-m'),\\
l'&=\alpha l+\sum_j p_j,\\
u'&=\alpha u+\sum_j p_jV_j.
\end{aligned}
$$

所有块处理完，再计算 `O = u / l`。初始最大值 m 取负无穷，l 和 u 均置零。

这里容易漏掉的是旧状态修正：新块如果出现更大的分数，旧的 l 和 u 必须一起乘上 `exp(m-m')`，才能在同一个最大值基准下相加。实际代码对应：

```python
next_max = torch.maximum(row_max, scores.max(dim=-1, keepdim=True).values)
correction = torch.exp(row_max - next_max)
block_weights = torch.exp(scores - next_max)
denominator = correction * denominator + block_weights.sum(dim=-1, keepdim=True)
v_tile = v[k_start:k_start + key_block]
numerator = correction * numerator + block_weights @ v_tile
```

这个分块递推对应 [FlashAttention 论文](https://papers.nips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)讨论的在线归一化思想。本次实现用普通 PyTorch CPU 操作验证公式，不包含论文中的 GPU 内存调度或反向重计算。

## 对照结果与因果检查

两份原始报告得到以下最大绝对误差，参考均为 CPU SDPA MATH：

| 配置 | 完整实现误差 | 分块实现误差 |
| --- | ---: | ---: |
| N=64，float64，非因果，块 16×16 | 0 | 4.44×10⁻¹⁶ |
| N=17，float32，因果，块 8×5 | 3.58×10⁻⁷ | 2.98×10⁻⁷ |

两份报告中的默认例子、六组边界配置均通过数值容差检查，完整权重的每行和为 1。float64 使用 `rtol=atol=1e-12`；float32 使用 `2e-5`。计算顺序不同会带来浮点舍入差异，因此这里比较容差内一致。

测试还覆盖了这些容易出错的位置：新块改变运行最大值后的旧状态修正；非整除分块；float32/float64；非连续输入；零分数对应均值；较大有限分数的稳定性。因果测试修改后半段的 K、V，前半段输出保持不变，验证未来输入不会影响前缀。

当前实现按 key 从 0 开始遍历，所以每一行在第一个块中至少能看见 key 0；后续完全被因果掩码遮住的块贡献为零。这不等于已经支持任意 padding mask 或整行全被遮蔽的情况。

## 分数块变小，不等于测出了总内存下降

固定 query/key 块为 `16×16`，默认 float64 扫描结果为：

| N | 完整分数矩阵元素数 | 最大单块分数元素数 | 实际计算的分数位置数 |
| ---: | ---: | ---: | ---: |
| 16 | 256 | 256 | 256 |
| 64 | 4096 | 256 | 4096 |
| 128 | 16384 | 256 | 16384 |

N=64 时，单张完整分数矩阵是 32768 字节，最大单块分数是 2048 字节。这里记录的是单个分数张量的尺寸；Q、K、V、输出、权重块和运行状态仍然占内存，不能将这些数字当作程序的峰值内存。

这个模型仍计算所有 `N²` 个 query/key 位置。因果模式也先计算块内分数，再应用掩码，没有跳过未来块来减少计算量。N 不超过块尺寸时，最大单块也可能就是整张分数矩阵。

本次结果说明分块递推可以避免在较长序列配置中保存完整分数矩阵；它没有测量 GPU IO、运行速度或总内存峰值，也没有把 Attention 的算术复杂度降成线性。

## KV cache 省下的是哪一部分重复工作

Prefill 一次处理整段 prompt；decode 每步只有一个新 query，却仍要读取合法历史 K/V。`29_kv_cache.py` 对固定输入表示、固定投影权重的一层 Attention 做了两条路径：每个前缀重算全部 Q/K/V 和完整因果 Attention，再取最后一行；或者 prompt 做一次 prefill，之后只投影新增位置，把 K/V 追加到缓存。

默认 prompt 为 8 个位置，再加入 4 个位置。长度 8、9、10、11、12 的最后一行输出均通过 float64 容差检查：

| 观察项 | 前缀重算 | KV cache |
| --- | ---: | ---: |
| 每项投影行数 | 50 | 12 |
| 分数对数 | 510 | 106 |
| 最大绝对误差 | 基准 | 2.22×10⁻¹⁶ |

缓存路径的分数对为 `8² + 9 + 10 + 11 + 12 = 106`，其中 prefill 计算完整分数矩阵，decode 每次只计算新 query 的一行。原始报告和本次复跑的数值、计数一致；另有无 decode、单位置和较长前缀等边界配置。

这些计数依赖本实验选定的完整前缀基线。若基线保存了其他中间状态或只计算最后一行 Attention，工作量会不同。表中的比值不代表通用加速倍数，也没有测量 FLOPs、耗时或峰值内存。

### 单 query 不能照搬左上三角掩码

当前 query 位于最后一个位置，而缓存只含过去和当前位置的 key。因此这一行中的每个 key 都合法：

```python
query, key, value = project(x[position:position + 1], weights)
cached_k = torch.cat((cached_k, key), dim=0)
cached_v = torch.cat((cached_v, value), dim=0)
output = torch.softmax(query @ cached_k.T / math.sqrt(query.shape[1]), dim=-1) @ cached_v
```

如果对 `1×prefix_length` 分数套用左上对齐的三角掩码，会只留下第 0 个 key。测试用零分数和 V 值 `[1,2,6]` 检查这一点：正确输出应为 3。另一项测试修改未来输入，较早位置的因果输出保持不变。

### 缓存会随上下文增长

本例 K、V 宽度均为 8，使用 float64，到第 12 个位置时保存的 K/V 数据为：

$$
12\times(8+8)\times8=1536\ \text{bytes}.
$$

这是最终缓存数据量，没有包含追加时的临时副本或整个程序内存。多层、多请求和多个 KV 头还需分别计入相应维度；GQA/MQA 中的 KV 头数也可能少于 query 头数。

脚本用 `torch.cat` 便于逐步核对，增长时会复制旧内容。它不是优化后的推理服务，也不生成文字。KV cache 避免重投影历史位置，当前 query 的注意力仍要读取历史 K/V，开销不会变成与上下文长度无关。

## 复现与范围

更新后的 ZIP 包含 28/29 号脚本、三份原始 JSON、依赖和 18 项归约/Attention/KV cache 测试。共享测试还加载归约模块，因此附上 `27_tree_reduction.py`。完整发布仓库连同矩阵和编译实验共 30 项测试，本次冻结副本中全部通过；ZIP 自带的测试为其中 18 项。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 28_attention_online_softmax.py \
  --output results/attention-rerun.json
.venv/bin/python 28_attention_online_softmax.py \
  --n 17 --d 8 --value-dim 4 --query-block 8 --key-block 5 \
  --causal --dtype float32 --output results/causal-rerun.json
.venv/bin/python 29_kv_cache.py --output results/kv-cache-rerun.json
.venv/bin/python -m unittest discover -s tests -v
```

本次验证只包含固定输入的一层单头前向，不包含训练、反向传播、dropout、padding mask、多头模型或完整 Transformer。真实 GPU 实验按学习安排跳过，不能把 CPU 对照写成 GPU prefill/decode 性能结果。

材料可对照[第 5 讲原课幻灯片](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/05-transformers-attention.pdf)、[Attention Is All You Need](https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)和站内[Transformer 中文学习资料](/learning/cmu-15-779/materials/transformer-paper-zh/)。这些是实验的阅读入口，不表示整篇论文评述或整门课程已经完成。

[本篇 GitHub 代码](https://github.com/liangqianxing/cmu-15-779-labs/tree/9261bea6aedf31cdccbbab2541ca693fc97b3743) · [下载代码与结果 ZIP](/materials/cmu-15-779/experiments/attention-online-softmax-kv-cache-20261019.zip)
