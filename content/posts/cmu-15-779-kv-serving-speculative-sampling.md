---
title: 'CMU 15-779 学习记录 11：分页 KV、连续批处理与推测采样'
date: '2026-10-19'
description: '在 CPU 上验证 KV 页共享和写时复制，用单位成本模型检查连续批处理，再用条件概率小模型核对推测采样的接受与补采样。'
series: CMU 15-779
seriesOrder: 11
cover: /images/posts/cmu-15-779-kv-serving-speculative-sampling/cmu-serving-cover-20261019.webp
coverAlt: 浅色手绘的共享卡片抽屉与不断补入小珠子的请求托盘
codeUrl: https://github.com/liangqianxing/cmu-15-779-labs/tree/c4851e0cedadf13af7c421867e83c8ff57244fa0
attachmentUrl: /materials/cmu-15-779/experiments/cpu-parallel-serving-lora-moe-20261019.zip
categories:
  - 技术
tags:
  - CMU 15-779
  - LLM Serving
  - KV Cache
  - Speculative Decoding
draft: false
hidden: false
haloSync: false
haloPublished: false
---

> 第 15、16 讲的个人配套实验。33 号实现 CPU 分页 KV 存储，37 号模拟请求准入，34 号运行小概率模型的推测采样。它们分别验证存储、调度和分布问题，没有测量真实 LLM 服务吞吐或 GPU 延迟。

[记录 08](/posts/cmu-15-779-attention-online-softmax/)已经对照过单个请求的 KV cache。这次继续检查：多个分支怎样共享前缀，短请求结束后怎样补入新请求，以及草稿 token 怎样在保持目标分布的条件下接受或拒绝。代码与原始结果冻结于 2026 年 10 月 19 日，整理时复跑结果一致。

[查看代码版本](https://github.com/liangqianxing/cmu-15-779-labs/tree/c4851e0cedadf13af7c421867e83c8ff57244fa0) · [下载 31 至 37 号代码与结果 ZIP](/materials/cmu-15-779/experiments/cpu-parallel-serving-lora-moe-20261019.zip)

## 共享前缀遇到部分页追加

33 号用 K/V 物理页池、每序列逻辑页表、引用计数和空闲页列表管理缓存。`fork` 复制页表并增加引用计数；`release` 只有在某页引用归零时才把它交还空闲列表。

块长为 4 时，一个 6-token 前缀占 2 页。父序列与两个分支刚建立时，共用这 2 页；每个分支各追加 1 token，会写入仍被共享的第二页，因此各自申请新页并复制已有的 2 个有效 token 槽。

| 时刻 | 活跃物理页 | 活跃页容量字节 | 累计复制的 token 槽 |
| --- | ---: | ---: | ---: |
| 父序列与两分支共享前缀 | 2 | 896 | 0 |
| 两分支各追加一个 token | 4 | 1792 | 4 |
| 全部释放 | 0 | 0 | 4 |

每 token 包含 8 个 K 元素与 6 个 V 元素，float64 合计 112B。页池一次预分配 16 页，即 7168B；释放逻辑页后，这个张量大小仍为 7168B。表中的活跃页字节不能直接解释成进程 RSS 或分配器峰值。

两个分支的 Attention 与各自连续存储的参考完全一致，报告中的最大误差为 0，父前缀也保持原值。测试在多种块长下随机执行 fork、指定前缀长度、追加和释放，并检查新页申请及写时复制遇到 OOM 时不破坏原状态。

实现中的 `gather` 仍会额外物化连续 K/V，再计算 Attention。它没有实现直接遍历页表的 GPU kernel、radix tree 或跨请求语义匹配。可共享的 KV 身份还取决于模型、位置和输入上下文，不能只看两个文本片段是否相同。

## 连续批处理怎样填补空位

37 号固定 batch 容量为 2，四个请求同时到达，生成长度依次为 `[1,4,1,4]`。每个 decode 迭代固定耗费一个模拟单位，活跃请求每轮各输出一个 token。

固定 cohort 要等整组结束才接纳下一组；连续策略在每轮开始时把空位交给等待请求。请求从 0 编号，执行轨迹为：

| 模拟轮次 | 固定 cohort | 连续批处理 |
| --- | --- | --- |
| 0 | 0、1 | 0、1 |
| 1 | 1 | 1、2 |
| 2 | 1 | 1、3 |
| 3 | 1 | 1、3 |
| 4 | 2、3 | 3 |
| 5 | 3 | 3 |
| 6 | 3 | 已结束 |
| 7 | 3 | 已结束 |

总时长分别为 8 与 6 个模拟单位；各请求的首 token 时间分别为 `[1,1,5,5]` 和 `[1,1,2,3]`。两种策略都输出总计 10 个 token，测试还检查请求不能早于到达时间运行。

这个结果来自“每轮成本与 batch、上下文长度无关”的假设。真实服务还要计入 prefill、decode 成本差异和 KV 容量，并联合检查吞吐、首 token 延迟、每 token 延迟与公平性。这里的 8/6 不能写成 GPU 加速比。

## 推测采样为什么需要修正拒绝分支

34 号的 target 与 draft 都是依赖前一 token 的小概率模型，每轮先提议 4 个 token，再逐个判断。对同一前缀下的目标分布 p、草稿分布 q，草稿 x 的接受概率为：

$$
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
$$

首次拒绝后，从下列残差分布补采一个 token，并丢弃后续草稿：

$$
r(x)=\frac{\max(p(x)-q(x),0)}{\sum_y\max(p(y)-q(y),0)}.
$$

接受分支已贡献 `min(p,q)` 的概率质量，拒绝分支补足 `(p-q)+`，二者恢复 p。拒绝后直接从 p 再采会重复计算已经贡献过的质量。若全部草稿接受，则再从对应前缀的目标分布追加一个 token。

默认例子的目标首 token 分布为 `[0.65,0.25,0.10]`。脚本枚举单步接受/拒绝质量，精确恢复这一分布；固定种子运行 20000 轮后，首 token 频率为 `[0.65255,0.24975,0.09770]`。平均每轮接受 3.3182 个草稿，输出 4.3182 个 token。

频率只统计每轮第一个 token，没有检验完整序列分布。确定性测试另覆盖支持不相交、p=q、总是拒绝和额外目标 token。q 为零的位置不会由 q 提议，因此测试不把这些位置直接代入除法。

脚本逐位置调用 CPU 概率函数，没有 Transformer 的并行验证、KV 回滚或 SpecInfer 的 token tree。平均输出 token 数也不构成速度证据：真实收益还取决于草稿成本、验证成本和接受率。

## 复现与课程材料

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 33_paged_kv_cache.py --output results/33-rerun.json
.venv/bin/python 37_continuous_batching.py --output results/37-rerun.json
.venv/bin/python 34_speculative_sampling.py --output results/34-rerun.json
.venv/bin/python -m unittest discover -s tests -v
```

共享 ZIP 包含 31 至 37 号脚本、原始结果和 22 项测试，其中 9 项直接检查本篇的缓存、采样与调度。共享测试文件还导入 LoRA、MoE 模块，附件保留了这些依赖。完整发布仓库的 52 项测试也已通过。

原课的[服务一](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/15-LLM-serving-part1.pdf)、[服务二](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/16-LLM-serving-part2.pdf)可与 [PagedAttention](https://arxiv.org/abs/2309.06180)、[SGLang](https://arxiv.org/abs/2312.07104)、[推测解码论文](https://proceedings.mlr.press/v202/leviathan23a.html)和 [SpecInfer](https://arxiv.org/abs/2305.09781)对照。实验完成不表示这些论文已由学习者完成阅读确认；继续推进中的课程项目也不属于本篇冻结范围。
