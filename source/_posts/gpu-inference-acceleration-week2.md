---
title: Week 2：GPU 与推理加速——从 Kernel、算子融合到 LLM Serving
date: 2026-05-05
categories: 技术
tags:
  - 深度学习
  - GPU
  - 推理加速
  - LLM
  - CMU 10-414
  - vLLM
  - TensorRT
---

Week 1 我们从 Autograd 理解了深度学习框架的训练本质：Tensor、计算图、反向传播和内存优化。Week 2 要切到更贴近论文和系统落地的部分：**GPU 与推理加速**。

如果说训练框架的核心问题是“如何自动求梯度”，那么推理系统的核心问题就是：**如何把有限的 GPU 算力、显存带宽和请求调度能力，用在最多的 token 上**。尤其是 LLM 推理，慢往往不是因为“计算公式复杂”，而是因为 memory bound、KV cache、batch scheduling、streaming generation 共同决定了端到端延迟。
<!-- more -->

![LLM inference pipeline](/images/posts/gpu-inference/llm-inference.svg)

## 1. Week 2 学什么

这一周只看 CMU 10-414 中和系统性能最相关的模块：

- GPU architecture basics：理解 GPU 为什么适合大规模并行；
- Kernel launch overhead：理解小算子为什么慢；
- Operator fusion：理解为什么要融合算子；
- Batching / throughput vs latency：理解服务系统的核心取舍；
- Distributed training overview：只看概念，理解数据并行、张量并行、流水并行的基本动机。

学完后你应该能解释：

- 为什么 LLM 推理经常是 memory bound；
- KV cache 为什么既加速 attention，又吃掉大量显存；
- batch scheduling 为什么决定 serving 系统吞吐；
- streaming generation 为什么提升体感速度但不减少总计算；
- vLLM、TensorRT、diffusion acceleration 分别在优化什么。

## 2. GPU 架构基础：GPU 快在哪里

CPU 擅长复杂控制流、低延迟任务和通用逻辑；GPU 擅长把同一个操作并行应用到大量数据上。深度学习里的矩阵乘法、卷积、attention 都天然符合这种模式。

一个非常简化的 GPU 层级可以理解为：

```text
GPU
  -> 多个 SM（Streaming Multiprocessor）
      -> 多个 warp
          -> 多个 thread
  -> HBM 显存
  -> L2 cache / shared memory / register
```

几个关键词必须理解。

### 2.1 Thread

Thread 是最细粒度的执行单元。一个 kernel 会启动大量 thread，每个 thread 处理一小块数据。

例如向量加法：

```cuda
c[i] = a[i] + b[i]
```

每个 thread 可以负责一个 `i`。如果向量有一百万个元素，就可以启动大量线程并行处理。

### 2.2 Warp

NVIDIA GPU 中，一个 warp 通常是 32 个线程。Warp 内线程执行同一条指令，但处理不同数据，这叫 SIMT：Single Instruction, Multiple Threads。

如果 warp 内线程走不同分支，例如一半执行 `if`，一半执行 `else`，就会产生 warp divergence，实际执行效率下降。

### 2.3 SM

SM 可以理解为 GPU 的计算核心。每个 SM 管理多个 warp，并在 warp 之间切换，用大量并发隐藏内存访问延迟。

GPU 高吞吐的关键不是单个线程很快，而是同时跑很多线程。

### 2.4 Register / Shared Memory / HBM

不同内存层级速度差别很大：

| 层级 | 特点 | 用法 |
|---|---|---|
| Register | 最快，线程私有 | 保存局部变量 |
| Shared Memory | 很快，block 内共享 | tile 复用、局部缓存 |
| L2 Cache | GPU 全局缓存 | 缓解重复访问 |
| HBM | 容量大但慢于片上存储 | 存参数、激活、KV cache |

推理优化的一个核心目标就是：**尽量把数据复用发生在 register/shared memory/cache，而不是频繁往返 HBM**。

## 3. Roofline：为什么要区分 compute bound 和 memory bound

一个算子慢，可能有两种原因：

1. **Compute bound**：计算单元忙不过来，瓶颈是 FLOPs；
2. **Memory bound**：计算单元在等数据，瓶颈是显存带宽。

判断关键是 arithmetic intensity：

```text
arithmetic intensity = FLOPs / bytes moved
```

如果每读 1 byte 数据能做很多计算，例如大矩阵乘法，通常更接近 compute bound。如果每读很多数据只做很少计算，例如 LayerNorm、Elementwise Add、Decode 阶段的小 batch GEMV，就容易 memory bound。

LLM 推理尤其在 decode 阶段经常 memory bound。因为每生成一个 token，都要读一遍大量模型权重，但 batch 可能很小，每个权重参与的计算复用不够。

## 4. Kernel 是什么

Kernel 是运行在 GPU 上的函数。Python 代码本身不在 GPU 上执行，它只是通过 CUDA runtime / driver 让 GPU 启动 kernel。

例如 PyTorch：

```python
y = torch.relu(x)
```

背后通常会触发一个 GPU kernel：对 `x` 的每个元素并行执行 `max(x, 0)`。

一次 kernel launch 包含固定开销：

```text
Python 调用
  -> C++ dispatcher
  -> CUDA runtime / driver
  -> GPU 排队执行 kernel
  -> kernel 真正运行
```

如果 kernel 本身工作量很大，例如大矩阵乘，启动开销可以忽略。如果 kernel 很小，例如几个 elementwise 操作，启动开销和显存读写可能比计算本身更贵。

## 5. Kernel Launch Overhead：为什么小算子很慢

假设你写：

```python
y = gelu(x + bias)
z = dropout(y)
```

如果没有融合，可能触发多个 kernel：

```text
add kernel
  -> 写中间结果到 HBM
gelu kernel
  -> 从 HBM 读中间结果，再写回 HBM
dropout kernel
  -> 再读再写
```

问题有两个：

1. 每个 kernel launch 都有固定调度成本；
2. 中间结果反复读写 HBM，浪费带宽。

这就是为什么深度学习编译器和推理引擎都非常重视 operator fusion。

## 6. Operator Fusion：算子融合

![Operator fusion](/images/posts/gpu-inference/fusion.svg)

算子融合就是把多个算子合成一个 kernel。例如：

```text
未融合：MatMul -> BiasAdd -> GELU -> Dropout
融合后：FusedMatMulBiasGELUDropout
```

融合的收益：

- 减少 kernel launch 次数；
- 减少中间激活写回 HBM；
- 增加寄存器和 shared memory 内的数据复用；
- 给编译器更多优化空间。

但融合也有代价：

- kernel 更复杂，开发和调试难度更高；
- 动态 shape、复杂控制流会降低融合机会；
- 过度融合可能导致 register pressure 过大，反而降低 occupancy。

TensorRT、XLA、TVM、TorchInductor、Triton 都在不同层面做 fusion。你看到推理框架性能大幅提升，很多时候不是数学变了，而是执行计划变了。

## 7. Batching：吞吐与延迟的永恒取舍

推理服务面对的不是一个固定输入，而是持续到来的请求流。Batching 的作用是把多个请求合在一起，提高 GPU 利用率。

```text
request A: prompt length 20
request B: prompt length 300
request C: prompt length 80
```

如果一个一个跑，GPU 可能吃不满。如果合成 batch，矩阵乘规模变大，权重复用更好，吞吐上升。

但 batch 太大也会增加延迟：

- 请求要在队列里等待凑 batch；
- 长 prompt 可能拖慢短 prompt；
- decode 阶段每个请求生成长度不同，会产生调度碎片；
- 显存中的 KV cache 随 batch 和序列长度增长。

所以 serving 系统要在两个指标间取舍：

| 指标 | 含义 | 偏好 |
|---|---|---|
| Throughput | 单位时间处理多少 token/request | 大 batch、有队列等待 |
| Latency | 单个请求多久返回 | 小 batch、少等待 |

在线聊天更看重低延迟，离线批处理更看重高吞吐。

## 8. LLM 推理流程：Prefill 与 Decode

LLM 生成可以分成两个阶段。

### 8.1 Prefill

Prefill 处理输入 prompt，一次性计算 prompt 中所有 token 的 hidden states，并建立 KV cache。

特点：

- 输入长度可能很长；
- attention 可以并行处理整个 prompt；
- 更像大矩阵乘，GPU 利用率通常较高；
- 决定 time to first token 的重要部分。

### 8.2 Decode

Decode 每次生成一个新 token，并把这个 token 的 K/V 追加到 KV cache。

特点：

- 每轮只新增一个 token；
- 必须自回归，不能把未来 token 并行算出来；
- batch 小时很容易 memory bound；
- 端到端生成时间主要由 decode token 数决定。

完整流程：

```text
prompt tokens
  -> prefill
  -> first token
  -> decode step 1
  -> decode step 2
  -> ...
  -> EOS / max_tokens
```

这解释了为什么输入很长会影响首 token 延迟，而输出很长会影响总生成时间。

## 9. 为什么 LLM 推理慢

### 9.1 Memory Bound

LLM 参数巨大。每生成一个 token，Transformer 的每一层都要访问大量权重。如果 batch 很小，这些权重读出来后只服务少量 token，复用不足。

以 decode 为例，很多操作更接近矩阵向量乘或小 batch 矩阵乘：

```text
hidden: [batch, hidden_dim]
weight: [hidden_dim, 4 * hidden_dim]
```

当 batch 小时，读 weight 的成本很高，计算单元可能等数据。这就是 memory bound。

优化方向包括：

- 增大 continuous batching，提高权重复用；
- 量化权重，减少 bytes moved；
- 使用更高效 kernel，减少额外内存访问；
- KV cache 分页管理，减少显存碎片；
- speculative decoding，减少大模型调用步数。

### 9.2 KV Cache

Transformer attention 每个 token 都需要看之前的 token。如果每一步都重新计算历史 token 的 K/V，成本会爆炸。

KV cache 保存每层历史 token 的 Key 和 Value：

```text
layer 1: K_cache, V_cache
layer 2: K_cache, V_cache
...
layer L: K_cache, V_cache
```

生成第 t 个 token 时，只需要计算新 token 的 K/V，然后和历史 K/V 做 attention。

KV cache 的好处：

- 避免重复计算历史 token；
- 让自回归 decode 可用。

KV cache 的问题：

- 显存占用随 batch、层数、hidden size、序列长度线性增长；
- 请求长度不同会造成碎片；
- cache 读写本身也会产生带宽压力；
- 长上下文推理时 KV cache 可能比权重更难管理。

一个粗略估算：

```text
KV cache bytes ≈ batch_size * seq_len * num_layers * 2(K,V) * hidden_size * bytes_per_element
```

如果用了多头注意力，还要考虑 head 数、head dim、GQA/MQA 等结构。GQA 和 MQA 的重要收益之一就是减少 KV cache 规模。

### 9.3 Batch Scheduling

LLM 请求是动态的：有人 prompt 长，有人 prompt 短；有人生成 20 token，有人生成 2000 token。传统静态 batch 很容易低效。

难点包括：

- 新请求什么时候插入正在 decode 的 batch；
- 已完成请求如何从 batch 中移除；
- 不同长度序列如何管理 KV cache；
- 如何避免短请求被长请求拖死；
- 如何在吞吐和 P99 延迟之间平衡。

这就是 vLLM、TGI、TensorRT-LLM 等 serving 系统的核心竞争点之一。

### 9.4 Streaming Generation

Streaming generation 是边生成边返回：

```text
用户看到：你 -> 好 -> ， -> 我 -> 来 -> 帮 -> 你
```

它的优点是降低用户体感延迟，特别是 time to first token 很重要。

但 streaming 不会减少总计算量。模型还是要一个 token 一个 token 地自回归生成。它优化的是交互体验，不是数学成本。

## 10. vLLM 在优化什么

vLLM 最出名的是 PagedAttention。它的动机来自一个非常工程的问题：KV cache 很大，而且不同请求长度不同，如果用连续大块显存保存，很容易碎片化和浪费。

PagedAttention 借鉴操作系统虚拟内存的思想，把 KV cache 切成 block/page：

```text
logical sequence blocks -> physical KV cache blocks
```

好处：

- 不要求每个请求的 KV cache 在显存中连续；
- 可以更灵活地增长、释放、复用 KV block；
- 支持更高效的 continuous batching；
- 降低显存碎片，提高可服务 batch 数。

vLLM 的核心不是“让模型少算公式”，而是让 serving 系统能把更多请求稳定塞进 GPU，并减少 KV cache 管理浪费。最终表现为吞吐提升、并发提升、显存利用率提升。

## 11. TensorRT / TensorRT-LLM 在优化什么

TensorRT 是 NVIDIA 的高性能推理优化引擎。它更像一个面向部署的编译器和 runtime。

常见优化包括：

- 图优化：删除无用节点、常量折叠、算子融合；
- kernel auto-tuning：为具体 shape 选择最快 kernel；
- 精度优化：FP16、BF16、INT8、FP8；
- 内存规划：复用 buffer，减少峰值显存；
- 插件 kernel：为 attention、layernorm、gemm 等提供特化实现；
- 多 GPU 推理：张量并行、流水并行等。

TensorRT-LLM 则进一步针对 LLM：

- fused attention；
- paged KV cache；
- inflight batching；
- quantization；
- tensor parallel；
- speculative decoding 支持。

如果 vLLM 的关键词是 serving scheduler + KV cache 管理，那么 TensorRT 的关键词是 graph optimization + kernel optimization + deployment runtime。

## 12. Diffusion Acceleration 在优化什么

Diffusion 模型和 LLM 都是生成模型，但瓶颈不完全一样。

Diffusion 生成图片通常需要多步 denoising：

```text
noise x_T
  -> denoise step T
  -> denoise step T-1
  -> ...
  -> image x_0
```

慢的原因：

- U-Net / DiT 要重复执行很多步；
- 每一步都有大量卷积或 attention；
- 高分辨率图片带来巨大的激活和计算量；
- classifier-free guidance 可能让一次 step 跑两次网络。

加速方向：

- 减少采样步数：DDIM、DPM-Solver、LCM、Turbo 类模型；
- 蒸馏：把多步模型蒸馏成少步模型；
- 算子优化：fused attention、xFormers、FlashAttention；
- 量化：FP16、INT8、FP8；
- 编译优化：TensorRT、TorchInductor、ONNX Runtime；
- 缓存复用：在视频或交互编辑里复用部分特征。

所以你看 diffusion acceleration 时，要先判断它是在减少 step 数，还是在加速每一步 kernel，还是在减少内存访问。

## 13. Distributed Training Overview：只看概念

虽然这一周重点是推理，但分布式训练的概念也要知道，因为很多推理并行策略来自训练。

### 13.1 Data Parallel

每张 GPU 放一份完整模型，处理不同 batch，反向后同步梯度。

```text
GPU0: model + batch0
GPU1: model + batch1
all-reduce gradients
```

优点简单，缺点是每张卡都要放完整模型。

### 13.2 Tensor Parallel

把单层的大矩阵切到多张 GPU 上。例如一个线性层的 weight 按列或按行切分。

```text
W = [W0, W1, W2, W3]
```

适合单卡放不下或单层计算太大的模型，但需要 GPU 间通信。

### 13.3 Pipeline Parallel

把不同层放到不同 GPU 上：

```text
GPU0: layer 0-9
GPU1: layer 10-19
GPU2: layer 20-29
```

适合层数很深的模型，但会有 pipeline bubble，需要 micro-batch 填流水线。

### 13.4 推理中的并行

LLM 推理也会用 tensor parallel 和 pipeline parallel，尤其是模型单卡放不下时。但推理还有 serving 特有问题：batch 动态变化、KV cache 分布、跨卡通信延迟、首 token 延迟等。

## 14. 一个性能分析心法

遇到任何推理加速论文或系统，先问五个问题：

1. 它减少了 FLOPs，还是减少了 bytes moved？
2. 它优化 prefill，还是优化 decode？
3. 它优化单请求 latency，还是多请求 throughput？
4. 它改变模型数学结果，还是只改变执行方式？
5. 它的代价是什么：精度、显存、工程复杂度、兼容性，还是调度公平性？

例如：

| 技术 | 主要优化 | 代价 |
|---|---|---|
| Quantization | 减少权重/激活 bytes | 可能掉精度 |
| FlashAttention | 减少 attention HBM 读写 | kernel 更复杂 |
| Operator Fusion | 减少 launch 和中间写回 | 编译/调试复杂 |
| Continuous Batching | 提高吞吐和权重复用 | 调度复杂，可能影响延迟 |
| PagedAttention | 降低 KV cache 碎片 | cache 管理更复杂 |
| Speculative Decoding | 减少大模型 decode 步数 | 需要小模型/草稿模型 |
| Distillation | 减少模型或采样步数 | 训练成本和质量风险 |

## 15. Week 2 学完应该能讲清楚

这一周不要求你会手写 CUDA kernel，但要能建立系统直觉：

- GPU 的并行来自大量线程、warp、SM，而不是单线程快；
- kernel launch 有固定成本，小算子多会拖慢端到端；
- operator fusion 的本质是减少 launch 和 HBM 往返；
- compute bound 和 memory bound 要用 arithmetic intensity 区分；
- LLM prefill 和 decode 的性能特征完全不同；
- KV cache 是 LLM 自回归推理的关键状态，也是显存管理难点；
- batch scheduling 是 serving 系统吞吐与延迟的核心；
- streaming generation 提升体感速度，但不减少总计算；
- vLLM 更偏 serving 和 KV cache 管理，TensorRT 更偏编译和 kernel 优化；
- diffusion acceleration 要区分减少采样步数和加速单步网络。

## 16. 最后总结

GPU 推理加速不是单一技术，而是一组围绕硬件瓶颈展开的系统工程：算子层面要减少 kernel launch 和显存访问，图层面要做 fusion 和内存规划，服务层面要做 batching 和 KV cache 管理，模型层面要做量化、蒸馏或结构改造。

理解这些之后，再看 vLLM、TensorRT、FlashAttention、diffusion turbo、speculative decoding，你会更容易判断它们到底在优化哪一层、为什么有效、代价是什么。这也是读推理系统论文时最重要的底层框架。
