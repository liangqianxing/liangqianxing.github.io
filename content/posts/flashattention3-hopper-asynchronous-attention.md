---
title: "FlashAttention-3 精读：用异步流水与 FP8 加速 Hopper Attention"
date: 2026-07-21 09:15:00
description: 从 Hopper 的 TMA、WGMMA 和 FP8 出发，拆解 FlashAttention-3 如何用 warp specialization、GEMM-Softmax 流水与低误差量化提升精确注意力吞吐。
series: 三大会论文精读
seriesOrder: 5
categories:
  - AI
tags:
  - AI Systems
  - FlashAttention
  - GPU Kernel
  - Hopper
  - FP8
  - NeurIPS
---

FlashAttention-3 研究的不是一种新的注意力近似，也没有改变 Transformer 的输出。它问的是一个更底层的问题：**当 H100 已经把矩阵乘、显存搬运和低精度计算做成相对独立的硬件单元时，Attention kernel 该怎样重新排程，才能让这些单元尽量同时工作？**

论文的答案包含三层：用 warp specialization 分离搬运与计算，用跨迭代流水把 Softmax 藏在异步矩阵乘之后，再用分块量化和 incoherent processing 控制 FP8 误差。NeurIPS 2024 最终版报告 BF16 前向最高 840 TFLOPs/s，约为 H100 理论峰值的 85%；FP8 最高达到 1.3 PFLOPs/s。

先划清结论边界：这些数字来自单张 H100 上的 Attention kernel 微基准，不是完整 LLM 的端到端训练或推理吞吐。论文也明确把 LLM inference 优化和大规模低精度训练效果留作后续工作。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** |
| 作者 | Jay Shah、Ganesh Bikshandi、Ying Zhang、Vijay Thakkar、Pradeep Ramani、Tri Dao |
| 会议 | NeurIPS 2024，Main Conference Track |
| 专题子方向 | AI Systems：GPU kernel、异步流水与低精度计算 |
| 正式论文 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html) |
| 作者版本与许可 | [arXiv:2407.08608v2](https://arxiv.org/abs/2407.08608)，CC BY 4.0 |
| 作者解读 | [Tri Dao: FlashAttention-3](https://tridao.me/blog/2024/flash3/) |
| 代码 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)，BSD-3-Clause |

**选择理由**：它是硬件感知算法设计的代表案例，既有清楚的流水图和可复核的消融，也把性能收益与数值误差放在同一篇论文中讨论。它适合工程读者理解“减少 HBM 访问”之后，Attention 优化为什么进一步转向异步执行、寄存器压力和数据布局。

## 问题背景：只减少 HBM 访问还不够

单头注意力写成：

$$
S=\alpha QK^\top,\qquad P=\operatorname{softmax}(S),\qquad O=PV,
$$

其中 $Q,K,V\in\mathbb{R}^{N\times d}$，通常 $\alpha=1/\sqrt d$。标准实现若把 $N\times N$ 的 $S$ 和 $P$ 写回 HBM，会产生大量慢速显存读写。FlashAttention 用 tiling 和在线 Softmax 在片上处理分块，不再物化完整 $S$、$P$；FlashAttention-2 又改善了序列维度并行和任务划分。

但在 H100 上，FlashAttention-2 仍只达到约 35% 的峰值利用率，而优化过的 GEMM 可以达到 80%-85%。原因不只是 kernel 尚未换用新指令，更在于它的算法结构仍接近同步执行：加载、$QK^\top$、Softmax、$PV$ 之间有显式依赖，硬件单元会轮流等待。

Hopper 提供了三个关键条件：

1. **TMA**（Tensor Memory Accelerator）异步搬运 GMEM 与 SMEM 之间的数据；
2. **WGMMA** 让一个 warpgroup 异步驱动 Tensor Core，并可直接读取 shared memory；
3. **FP8 Tensor Core** 的理论矩阵乘吞吐约为 FP16/BF16 的两倍。

另一个容易忽略的瓶颈是 Softmax。论文按 H100 SXM5 计算，FP16 矩阵乘峰值为 989 TFLOPs/s，而指数等特殊函数只有约 3.9 TFLOPs/s。对 head dimension 128，矩阵乘 FLOPs 虽然比指数运算多 512 倍，但指数吞吐低 256 倍，因此 Softmax 仍可能占到矩阵乘约一半的周期。优化目标遂变成：**让 Tensor Core 做 GEMM 时，其他执行单元同时完成 Softmax。**

## 核心贡献

论文贡献可以归纳为三点：

1. **生产者-消费者异步化**：用独立 warps 发出 TMA 加载，consumer warpgroups 负责 WGMMA 与 Softmax，并用循环 SMEM buffer 解耦搬运和计算；
2. **两级 GEMM-Softmax 流水**：跨 block 迭代打破“GEMM0 -> Softmax -> GEMM1”的完全串行链，让下一块的 $QK^\top$ 与当前块的 Softmax、$PV$ 重叠；
3. **可用的 FP8 Attention**：解决 FP8 WGMMA 的数据布局约束，并通过分块量化和随机 Hadamard 变换减轻离群值带来的量化误差。

**论文结论**：这些设计使 FlashAttention-3 在 H100 上比 FlashAttention-2 的 BF16 前向快 1.5-2.0 倍、反向快 1.5-1.75 倍，同时把 FP8 基线的数值误差降低 2.6 倍。

**我的判断**：论文最重要的不是某个峰值数字，而是把 kernel 优化从“减少数据移动”推进到“显式调度异构执行单元”。这种思路会随硬件演进继续存在，但具体指令、tile 大小和流水深度高度依赖架构与编译器。

## 方法总览：CTA 内的三层流水

FlashAttention-3 仍按 batch、head 和 query block 并行。一个 CTA 负责 $Q_i$ 并遍历 $K_j,V_j$：producer warpgroup 用 TMA 把数据装入多级循环 SMEM buffer；consumer warpgroups 等待相应 stage 就绪，在寄存器和 SMEM 上完成两次矩阵乘及在线 Softmax。

### 在线 Softmax 保持精确结果

对第 $j$ 个 score block：

$$
S_i^{(j)}=\alpha Q_iK_j^\top,
$$

令当前行最大值和归一化分母为 $m_i$、$\ell_i$，更新为：

$$
m_i^{\text{new}}=\max\left(m_i^{\text{old}},\operatorname{rowmax}(S_i^{(j)})\right),
$$

$$
\widetilde P_i^{(j)}=\exp\left(S_i^{(j)}-m_i^{\text{new}}\right),
$$

$$
\ell_i^{\text{new}}
=e^{m_i^{\text{old}}-m_i^{\text{new}}}\ell_i^{\text{old}}
+\operatorname{rowsum}(\widetilde P_i^{(j)}).
$$

输出累加器也按相同系数重标定，再加上 $\widetilde P_i^{(j)}V_j$。遍历完所有 key/value blocks 后除以 $\ell_i$，得到与标准 Softmax Attention 相同的数学结果。这里没有稀疏化或低秩近似；BF16/FP16 路径的差异来自浮点执行顺序，而不是注意力定义被替换。

## 关键模块一：Warp Specialization 与 Pingpong

![两个 consumer warpgroups 的 Pingpong 排程](/images/posts/flashattention3-hopper-asynchronous-attention/pingpong-scheduling.png)

*图源：Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608), Figure 1, NeurIPS 2024；取自作者 CC BY 4.0 arXiv v2 源码。原图用于论文解读。*

Producer 与 consumer 分工后，TMA 加载不会占用负责矩阵乘的 warps。Hopper 的 `setmaxnreg` 还能把 producer 不需要的寄存器额度转给 consumer，允许后者保存更大的 tile 和中间状态。

两个 consumer warpgroups 再采用 pingpong 排程：warpgroup 1 发出 GEMM 时，warpgroup 2 做 Softmax；随后交换角色。论文说明示意图比真实调度理想化，但在 arXiv v2 的 FP16、head dimension 128、sequence length 8192 设置中，吞吐可由约 570 提升到 620-640 TFLOPs/s。

## 关键模块二：跨迭代的两级流水

![WGMMA 与 Softmax 的两级流水](/images/posts/flashattention3-hopper-asynchronous-attention/two-stage-pipeline.png)

*图源：Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608), Figure 2, NeurIPS 2024；取自作者 CC BY 4.0 arXiv v2 源码。原图用于论文解读。*

单个 warpgroup 内仍有依赖：当前 block 的 Softmax 必须等 $Q_iK_j^\top$，而 $P_jV_j$ 又必须等 Softmax。论文用跨迭代流水绕开这条串行链：

1. 先异步发出下一块的 $Q_iK_j^\top$，不立即等待；
2. 再发出上一块的 $\widetilde P_{j-1}V_{j-1}$；
3. 只等待下一块 score 就绪，执行它的在线 Softmax；
4. 等待上一块输出矩阵乘完成，重标定输出累加器；
5. 交换 `current` 与 `next` buffer，进入下一轮。

SASS 附录显示编译器确实把部分 `MUFU.EX2`、FP32 到 FP16 转换和第一组 HGMMA 指令交错排布。但这不是免费收益：两级流水必须额外保存一个 $S_{\text{next}}$，增加寄存器压力。论文还尝试三级流水，结果反而更差，因为编译器没有按预期重叠第二个 WGMMA，额外中间状态又迫使 kernel 选择更小 tile。

## 关键模块三：FP8 布局与误差控制

Hopper 的 FP8 WGMMA 只接受 k-major 输入，而 Attention 中 $V$ 通常在 head dimension 连续。FlashAttention-3 没有为推理额外启动一次全局 transpose kernel，而是在 kernel 内用 LDSM/STSM 把 $V$ tile 从 SMEM 搬到寄存器再写回，同时完成转置。FP32 accumulator 和第二次 FP8 WGMMA operand 的寄存器布局也不一致，论文再用 byte permute 重排寄存器条目，并让 $V$ 的行置换与之匹配。

布局正确只解决“能算”，还要控制离群值引起的 FP8 误差：

### 分块量化

把 $Q,K,V$ 切成与 kernel tile 对齐的 blocks，各自选择 scale，而不是让整个 tensor 共用一个 scale。Attention 本来就按 block 计算，因此 score block 可直接吸收对应缩放系数。作者指出，量化可以与 rotary embedding 之类的前序内存带宽受限算子融合。

### Incoherent Processing

量化前对 $Q,K$ 同乘随机正交矩阵 $M$：

$$
(QM)(KM)^\top=QMM^\top K^\top=QK^\top.
$$

因此精确算术下 Attention score 不变，但原本集中在少数维度的离群值被“摊开”。实现采用随机符号对角矩阵与 Hadamard 矩阵的乘积，把计算从一般正交变换的 $O(d^2)$ 降到 $O(d\log d)$，并可与 rotary embedding 融合。

## 训练与推理流程

这篇论文没有训练新模型，“训练流程”实际是 Attention kernel 的前向与反向：

### 前向

1. 按 query tile 分派 CTA；
2. producer 用 TMA 预取 $Q_i,K_j,V_j$ 到循环 SMEM buffer；
3. consumers 用 WGMMA 计算 score blocks，并在线更新行最大值、归一化项与输出；
4. 两个 warpgroups 做 pingpong，不同迭代在单个 warpgroup 内再做两级流水；
5. 写回 $O_i$ 和反向所需的 log-sum-exp 向量 $L_i$。

### 反向

反向先计算 $D=\operatorname{rowsum}(dO\circ O)$，再重算局部 $P$，累积 $dK,dV$ 并生成局部 $dQ$。由于多个 threadblocks 会写同一 $dQ$ 区域，论文增加专门的 `dQ-writer` warp，通过 semaphore 与原子累加处理争用，让 consumer warps 尽快继续矩阵乘。

### 推理

代码可作为精确 Attention primitive，也兼容 MQA/GQA 的索引方式，但论文没有报告完整 LLM prefill、decode 或端到端吞吐。作者明确承认 LLM inference 仍需专门优化；因此不能把 kernel 的 1.5-2.0 倍直接换算成模型服务同等加速。

## 实验设置

NeurIPS 最终版使用单张 NVIDIA H100 80GB SXM5（700W），固定 1830 MHz，软件环境为 CUDA 12.3、cuDNN 9.5.0.50、CUTLASS 3.6、FlashAttention 2.6.3、Triton 3.1、PyTorch 2.5.0。每个 benchmark 重复 10 次取平均。

序列长度覆盖 512 到 16K，总 token 数固定为 16K；hidden dimension 为 2048，head dimension 为 64、128 或 256。前向 FLOPs 按下式估算：

$$
4\times \text{seqlen}^2\times \text{head dimension}\times \text{number of heads}.
$$

因果掩码近似除以 2；反向按前向的 2.5 倍计算，因为前向有 2 次矩阵乘，反向连同重计算共有 5 次。对比项包括 PyTorch standard attention、FlashAttention-2、使用 Hopper 指令的 Triton 实现，以及闭源的 cuDNN 实现。

## 主要结果

### BF16/FP16 前向与反向

NeurIPS 最终版报告：

- BF16 前向比 FlashAttention-2 快 1.5-2.0 倍，最高 840 TFLOPs/s、约 85% 峰值利用率；
- BF16 反向比 FlashAttention-2 快 1.5-1.75 倍；
- 相对会物化中间矩阵的 standard attention，部分设置达到 3-16 倍；
- 在 1K 及以上的中长序列上，论文实现通常超过当时针对 H100 优化的 cuDNN。

![FP16 非因果前向吞吐随序列长度变化](/images/posts/flashattention3-hopper-asynchronous-attention/figure5c-fp16-forward.png)

*图源：Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608), Figure 5(c), NeurIPS 2024；从作者 CC BY 4.0 arXiv v2 源码中的矢量子图直接栅格化，坐标轴、图例和数据标签均保留。原图用于论文解读。该 v2 图表早于 NeurIPS 最终版性能更新。*

这张 arXiv v2 图展示 head dimension 128、无因果掩码的较早结果：16K 时 FlashAttention-3 为 648 TFLOPs/s，FlashAttention-2 为 370，cuDNN 为 595。它支持“长序列下超过旧实现”的趋势，但不应拿 648 与正式版 840 的峰值直接比较，因为后者来自更新后的最终实现和完整设置。

### FP8 前向

最终版 FP8 峰值为 1.3 PFLOPs/s。FP8 并非在所有点都领先 cuDNN：短序列和因果掩码下可能落后，论文将原因之一归于 FP8 kernel 缺少 persistent kernel 与 load balancing；BF16 kernel 则已经包含这些设计。

![FP8 非因果前向吞吐随序列长度变化](/images/posts/flashattention3-hopper-asynchronous-attention/figure7a-fp8-forward.png)

*图源：Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608), Figure 7(a), NeurIPS 2024；从作者 CC BY 4.0 arXiv v2 源码中的矢量子图直接栅格化，坐标轴、图例和数据标签均保留。原图用于论文解读。该 v2 图表早于 NeurIPS 最终版性能更新。*

在这张 arXiv v2 的 head dimension 256、无因果掩码图中，FlashAttention-3 从 512 的 510 TFLOPs/s 增至 16K 的 1171 TFLOPs/s；cuDNN 在 512 到 4K 更快，8K 后 FlashAttention-3 才反超。这比只看峰值更能说明 kernel 启动、并行度和负载均衡对短序列很重要。

## 消融分析

论文在非因果 FP16、`batch=4, seqlen=8448, nheads=16, hdim=128` 上做消融：

| 配置 | 时间 | 吞吐 |
| --- | ---: | ---: |
| 完整 FlashAttention-3 | 3.538 ms | 661 TFLOPs/s |
| 无 GEMM-Softmax 流水，保留 warp specialization | 4.021 ms | 582 TFLOPs/s |
| 保留 GEMM-Softmax 流水，无 warp specialization | 4.105 ms | 570 TFLOPs/s |

完整方案相对两个删减版本分别提高约 13.6% 和 16.0%。两种机制都有独立贡献，但表中没有“二者都关闭”的第四组，所以不能从这张表单独推导严格的加法分解。

数值误差实验用 FP64 作为参考，并人为构造离群值：每个 $Q,K,V$ 元素先采样 $\mathcal N(0,1)$，再以 0.1% 概率叠加 $\mathcal N(0,100)$。结果为：

| 方法 | RMSE |
| --- | ---: |
| Baseline FP16 | $3.2\times10^{-4}$ |
| FlashAttention-2 FP16 | $1.9\times10^{-4}$ |
| FlashAttention-3 FP16 | $1.9\times10^{-4}$ |
| Baseline FP8 | $2.4\times10^{-2}$ |
| FlashAttention-3 FP8 | $9.1\times10^{-3}$ |
| FP8 去掉 block quantization | $9.3\times10^{-3}$ |
| FP8 去掉 incoherent processing | $2.4\times10^{-2}$ |

在这组合成数据上，主要误差改善来自 incoherent processing；去掉分块量化只从 $9.1\times10^{-3}$ 变为 $9.3\times10^{-3}$。这不等于分块量化普遍无用，只说明该实验的特定离群分布更突出地检验了随机正交变换。

## 失败案例与局限

论文没有给自然语言或视觉任务的样本级失败案例，因为研究对象是 kernel；它给出的失败模式主要发生在系统设计层：

1. **三级流水失败**：编译器没有按预期重叠第二个 WGMMA，寄存器占用又限制 tile，性能低于两级流水；
2. **FP8 短序列不稳定占优**：缺少 persistent kernel 和 load balancing，短序列或 causal 设置可能慢于 cuDNN；
3. **硬件范围窄**：实验只覆盖 H100，论文对其他具备异步执行与低精度能力的加速器仅提出预期，没有实测；
4. **不是端到端评测**：没有训练模型、语言建模质量、TTFT、TPOT、吞吐或成本数据；
5. **FP8 证据有限**：误差验证基于合成离群分布，尚未回答大规模低精度训练是否稳定；
6. **比较依赖版本**：cuDNN 为闭源实现，且最终版固定在 2024 年 10 月的软件栈，不能视为长期不变的系统排名。

作者在结论中明确承认的两项后续工作是 LLM inference 优化，以及理解低精度 Attention 在大规模训练中的影响。论文不支持“换成 FlashAttention-3 就能让任意 LLM 端到端快两倍”这一说法。

## 可复现资源

- [NeurIPS 最终版 PDF 与 BibTeX](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html)：用于核对最终作者、收录、环境与更新后的性能数字；
- [arXiv v2 全文与源码](https://arxiv.org/abs/2407.08608)：CC BY 4.0，本文 4 张配图均来自这份作者源码；
- [FlashAttention 官方代码库](https://github.com/Dao-AILab/flash-attention)：BSD-3-Clause，Hopper 实现在 `hopper/`，包含 kernel 与 benchmark 脚本；
- [作者技术解读](https://tridao.me/blog/2024/flash3/)：解释 TMA、WGMMA、pingpong、两级流水和 FP8 误差控制；
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)：论文实现使用的 WGMMA、TMA 等底层抽象。

复现应固定论文记录的软件版本和 H100 时钟，并区分 arXiv v2 与 NeurIPS 最终版。当前代码库仍在演进，直接使用 `main` 得到的结果不一定等于 2024 年论文快照；比较时还应同时记录 kernel 形状、causal mask、dtype、head dimension、总 token 数与 warm-up/计时方法。

## 个人判断：这是算法与硬件共同定义的 Attention

FlashAttention-3 展示了一个很有代表性的系统规律：当 GEMM 足够快时，过去被当作“边角开销”的 Softmax、数据布局、同步指令和寄存器分配会成为主导因素。继续优化不能只数 FLOPs，而要同时考虑每种硬件单元的吞吐、依赖图和编译器实际生成的指令顺序。

对工程团队，我认为有三条直接启示：

1. 不要用 kernel 峰值替代端到端收益，应在目标模型上分别测 prefill、decode、训练前向和反向；
2. 低精度优化必须同时报告速度与误差，并用真实激活分布补充合成离群实验；
3. kernel 参数和流水深度应由目标 GPU、形状分布与编译器版本共同决定，不能把 H100 上的最优排程机械迁移。

因此，FlashAttention-3 更像一份 Hopper 时代的 Attention kernel 设计教材，而不是一个跨硬件、跨任务的统一结论。它证明了显式异步排程与 FP8 可以继续释放精确 Attention 的潜力，也同样清楚地暴露了收益对硬件、输入形状和实现版本的依赖。

## 参考资料

1. Shah et al. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html). NeurIPS 2024.
2. Shah et al. [FlashAttention-3, arXiv:2407.08608](https://arxiv.org/abs/2407.08608). CC BY 4.0.
3. Dao et al. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html). NeurIPS 2022.
4. Dao. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). 2023.
5. Chee et al. [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304). NeurIPS 2023.
6. NVIDIA. [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/). WGMMA、TMA 与寄存器控制参考。
