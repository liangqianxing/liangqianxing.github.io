---
title: 从 PyTorch 到 CUDA Kernel：融合 RMSNorm 在 RTX 4090D 上加速 4.27 倍
date: 2026-07-22 18:16:50
description: 从零实现 PyTorch CUDA 扩展，拆解 RMSNorm 的访存瓶颈、warp 归约、向量化加载、算子融合与冷热缓存 benchmark。
series: AI Infra 实战
seriesOrder: 1
categories:
  - 技术
tags:
  - CUDA
  - AI Infra
  - GPU
  - PyTorch
  - LLM 推理
  - Kernel 优化
hidden: true
haloPublished: true
---

为了准备 AI Infra 岗位，我做了一个尽量小而完整的 CUDA 项目：不追求堆很多算子，而是选一个真实的 LLM 推理路径，把实现、正确性、benchmark 和性能解释全部做扎实。

项目实现了两个 PyTorch CUDA 算子：

```python
rms_norm(x, weight)
fused_add_rms_norm(x, residual, weight)
```

最终在 RTX 4090D 的 `256 × 4096 FP16` workload 上，融合算子相对 `torch.compile(..., fullgraph=True)` 获得 **4.27 倍 hot-cache 加速**和 **3.11 倍 cold-cache 加速**。代码、测试和原始 CSV 都已公开：

- [GitHub：fast-llm-kernels](https://github.com/liangqianxing/fast-llm-kernels)
- [完整 benchmark 结果](https://github.com/liangqianxing/fast-llm-kernels/tree/main/results)

这篇文章不只展示结果，更重要的是解释这些数字为什么成立，以及它们不能说明什么。

![Fused RMSNorm kernel data flow](/images/posts/cuda-fused-rmsnorm/kernel-pipeline.svg)

*图：本文原创绘制。一个 CUDA block 负责一个 token row，先完成 residual add 与平方和归约，再进行归一化和缩放。*

## 0. 阅读前的前置知识

如果已经熟悉混合精度和 CUDA 编程，可以直接跳到下一节。第一次接触 GPU kernel，则建议先建立下面这些概念之间的关系。

### 0.1 Token、hidden state 和 hidden size

模型不会直接处理文字，而是先把文字切分成 token。每个 token 在 Transformer 内部对应一个浮点数向量，称为 hidden state：

```text
一个 token -> [0.12, -0.53, 0.87, ...]
```

向量中元素的数量就是 hidden size。例如输入 shape 为 `[256, 4096]`，表示有 256 个 token，每个 token 用 4096 个浮点数表示。RMSNorm 会分别处理这 256 行，每一行之间没有数据依赖。

### 0.2 FP32、FP16 和 BF16

FP 表示 floating point，即浮点数。后面的数字表示一个数占用的 bit 数：

| 格式 | 每个元素 | 指数位 | 尾数位 | 主要特点 |
|---|---:|---:|---:|---|
| FP32 | 4B | 8 | 23 | 精度和范围高，但显存与带宽开销大 |
| FP16 | 2B | 5 | 10 | 精度较高，表示范围比 FP32 小 |
| BF16 | 2B | 8 | 7 | 范围接近 FP32，小数精度低于 FP16 |

大模型推理通常使用 FP16 或 BF16，以减少显存占用和数据搬运。但 RMSNorm 需要累加几千个平方值，直接用低精度累加容易产生明显误差。因此本项目采用 mixed precision：

```text
FP16/BF16 输入
    -> 转成 FP32 计算平方和
    -> 得到归一化系数
    -> 转回 FP16/BF16 输出
```

这就是文中 **FP32 accumulation** 的含义。它不一定表示输入输出都是 FP32，而是指关键的 reduction 使用 FP32。

### 0.3 RMSNorm、LayerNorm 和 residual

Normalization 的目标是控制 hidden state 的数值尺度，避免深层网络中的激活越来越大或越来越小。

LayerNorm 会先减去均值，再除以标准差：

```text
(x - mean(x)) / sqrt(variance(x) + eps)
```

RMSNorm 不计算和减去均值，只根据平方均值缩放：

```text
x / sqrt(mean(x²) + eps)
```

因此 RMSNorm 的计算更简单，被 Llama、Qwen、Mistral 等模型广泛使用。

Residual 是残差连接。Transformer 不直接丢弃上一层的输入，而是把新结果加回原信息：

```python
residual_out = x + residual
output = rms_norm(residual_out, weight)
```

这样既能保留旧信息，也有利于深层网络训练。本文优化的正是这条 residual-add + RMSNorm 路径。

### 0.4 Kernel、thread、warp、block 和 SM

CUDA kernel 是运行在 GPU 上的函数。CPU 负责发起 kernel launch，GPU 再让大量线程执行它。

NVIDIA GPU 的执行层级可以简化为：

```text
GPU
  -> 多个 SM
      -> 多个 block
          -> 多个 warp
              -> 32 个 thread
```

- **Thread**：最小执行单元，每个线程处理一部分元素；
- **Warp**：GPU 的基本调度单位，NVIDIA GPU 上通常包含 32 个线程；
- **Block**：一组可以同步、共享 shared memory 的线程；
- **SM**：真正执行 warp 和 block 的硬件单元。

本项目采用一个 block 处理一个 token row，每个 block 启动 256 个线程，也就是 8 个 warp。

### 0.5 Register、shared memory、L2 和显存

GPU 存储层级的速度与容量差异很大：

| 层级 | 速度 | 可见范围 | 本项目中的用途 |
|---|---|---|---|
| Register | 最快 | 单个线程 | 保存局部平方和 |
| Shared memory | 很快 | 一个 block | 保存 8 个 warp 的归约结果 |
| L2 cache | 较快 | 整个 GPU | 缓存重复读取的输入和 weight |
| VRAM / GDDR / HBM | 较慢、容量大 | 整个 GPU | 存放 tensor 数据 |

GPU kernel 优化经常不是减少数学运算，而是减少数据在这些存储层级之间的搬运。

### 0.6 Reduction、warp shuffle 和同步

Reduction 是把一组数据合并成一个值，例如：

```text
[1, 2, 3, 4] -> sum -> 10
```

RMSNorm 需要将一行中所有 `x[i]²` 合并成平方和。每个线程先计算自己的局部结果，再通过 warp shuffle 在 warp 内交换 register 数据。8 个 warp 的结果写入 shared memory，最后合并成整个 block 的平方和。

`__syncthreads()` 是 block 内同步屏障：所有线程到达这里后才能继续，避免某些线程还没写完 shared memory，其他线程就提前读取。

### 0.7 Memory-bound、fusion 和向量化

如果算子主要受计算单元吞吐限制，称为 compute-bound；如果大量时间花在等待数据，称为 memory-bound。

RMSNorm 每读取一个元素只进行少量平方、加法和乘法，因此通常是 memory-bound。优化重点是减少显存访问和 kernel launch，而不是使用更多计算单元。

Kernel fusion 是把多个操作合进一次 kernel launch。Pack-4 向量化则让每个线程一次读取四个连续元素，减少访存指令。二者分别优化“启动与中间结果”和“单次访存效率”。

### 0.8 Latency、speedup、hot cache 和 cold cache

- **Latency**：一次算子执行需要多少时间，本文使用微秒（us）；
- **Speedup**：baseline latency 除以 CUDA latency，例如 `40 / 10 = 4x`；
- **Hot cache**：重复使用同一组输入，数据很可能已经位于 L2；
- **Cold cache**：计时前主动驱逐 L2，让输入更多地从显存读取；
- **`torch.compile`**：PyTorch 的图编译功能，会尝试融合操作并生成优化 kernel，是比 eager PyTorch 更强的 baseline。

真实推理一般位于 hot 和 cold 两种极端之间，因此后文会同时报告两组数据，而不是只选择最好看的结果。

## 1. 为什么选择 RMSNorm

RMSNorm 广泛用于 Llama、Mistral、Qwen 等 decoder-only 模型。对一行 hidden state，它计算：

$$
\operatorname{RMSNorm}(x_i) = x_i \cdot \frac{1}{\sqrt{\frac{1}{H}\sum_{j=1}^{H}x_j^2 + \epsilon}} \cdot w_i
$$

其中 $H$ 是 hidden size，$w$ 是可学习缩放参数。

这个算子的 FLOPs 很少，却需要读取整行输入、计算 reduction，再写出结果。它的 arithmetic intensity 很低，通常不是算力不够，而是受以下因素限制：

- global memory traffic；
- kernel launch overhead；
- reduction 的同步开销；
- 小 batch 下不足的并行度。

因此 RMSNorm 很适合用来练习 AI Infra 中真正重要的能力：判断瓶颈、设计 fusion 边界、处理混合精度，并建立可信的性能实验。

## 2. 为什么融合 residual add

Transformer 的 pre-norm 路径通常包含：

```python
residual_out = x + residual
output = rms_norm(residual_out, weight)
```

如果分开执行，至少需要一次 add kernel 和一次 normalization 路径。中间的 `residual_out` 需要写回显存，再被后续 kernel 读取。

本项目将 residual add、平方和归约、归一化与 weight scaling 放进一次 kernel launch：

```text
x + residual
    -> FP32 square accumulation
    -> block reduction
    -> rsqrt(mean + eps)
    -> normalize * weight
```

融合后仍然返回 `residual_out`，因为下一个 decoder block 还需要这条 residual stream。这个细节很重要：kernel fusion 不能只追求理论上的最少写入，还必须保持模型调用方需要的语义。

## 3. Kernel 映射：一行一个 block

把输入展平为 `[tokens, hidden_size]` 后，每一行的 RMSNorm 彼此独立，因此使用：

```text
1 CUDA block <-> 1 token row
256 threads   <-> 并行遍历 hidden dimension
```

每个线程处理若干元素并累积局部平方和。典型 hidden size 是 4096 或 8192，256 个线程能让每个线程处理多个连续元素，同时保持足够的并发。

### 3.1 Pack-4 向量化访存

当 hidden size 能被 4 整除时，kernel 把连续四个元素包装成一个对齐 pack：

```cpp
template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  T value[N];
};
```

这能减少 load/store 指令数量并提高 memory transaction 利用率。对于 255 这类非对齐维度，代码会自动切换到标量 fallback，而不是假设所有模型 shape 都是整齐的。

### 3.2 FP32 累加

即使输入是 FP16 或 BF16，平方和也使用 FP32：

```cpp
float square_sum = 0.0f;
square_sum += value * value;
```

hidden size 往往有几千维，直接用低精度累加会放大舍入误差，甚至产生溢出。输出仍然转换回原始 dtype，因此这是数值稳定性与吞吐之间的常见折中。

fused path 还有一个容易忽略的语义：`x + residual` 必须先舍入到目标 dtype，再用这个实际写入 residual stream 的值计算平方和。否则 BF16 下归一化使用的值和返回的 `residual_out` 会不一致。

## 4. Warp shuffle reduction

每个线程得到局部平方和后，需要在 block 内合并。

第一阶段在每个 warp 内使用：

```cpp
value += __shfl_down_sync(0xffffffff, value, offset);
```

32 个线程的数据直接通过寄存器交换完成，不需要把每一步都写入 shared memory。

第二阶段只把 8 个 warp 的结果写入 shared memory，再由第一个 warp 完成最终归约。256-thread kernel 因此只使用 **32B shared memory**。

得到整行平方和后计算 `inv_rms`，然后执行第二遍读取并写出归一化结果。为什么不把整行输入一直放在寄存器中？因为保存 4096 或 8192 个值会显著增加 register pressure，降低 occupancy。这里选择第二遍读取，是经过资源约束后的工程取舍。

## 5. PyTorch 扩展边界

Kernel 快不等于扩展可用。项目还处理了：

- CUDA、contiguous、shape 和 dtype 校验；
- FP32、FP16、BF16 类型 dispatch；
- `CUDAGuard` 多卡设备保护；
- 当前 PyTorch CUDA stream；
- `C10_CUDA_KERNEL_LAUNCH_CHECK` 异步启动错误。

这使它可以像普通 PyTorch 算子一样调用，而不是一个只能独立运行的 `.cu` 示例。

## 6. Hot cache 和 cold cache

只测一组重复输入，GPU 很可能从 L2 cache 读取数据，得到非常漂亮的结果，但这不能完全代表显存压力较大的真实 workload。

因此 benchmark 同时报告两种模式：

| 模式 | 测量方式 | 主要观察内容 |
|---|---|---|
| Hot cache | 连续使用同一组 tensor | 最低延迟、launch 与计算效率 |
| Cold cache | 每次计时前覆盖并同步 256MB GPU buffer | 更接近 DRAM-bound 的工作集 |

Cold cache 也不等于“完全没有缓存”。同一 kernel 内第一次和第二次读取之间仍可能发生 L2 复用。因此本文给出的 GB/s 是逻辑 tensor traffic 除以延迟，不是 DRAM 硬件计数器。

计时方法包括：

- CUDA Event；
- 50 次 warmup；
- 50 次测量；
- 5 轮重复并取中位数；
- eager PyTorch 与 `torch.compile(fullgraph=True)` 两种 baseline；
- FP16 与 BF16；
- 1、32、256、1024 tokens 以及 4096/8192 hidden size。

## 7. 实测结果

测试环境：RTX 4090D 24GB、CUDA 12.8、PyTorch 2.8.0、`sm_89`。下表使用更强的 `torch.compile` baseline，shape 为 `256 × 4096`。

| dtype | cache | operation | PyTorch | CUDA | speedup |
|---|---|---|---:|---:|---:|
| FP16 | hot | RMSNorm | 35.122 us | 6.861 us | 5.12x |
| FP16 | hot | add + RMSNorm | 39.670 us | 9.298 us | 4.27x |
| FP16 | cold | RMSNorm | 57.344 us | 16.384 us | 3.50x |
| FP16 | cold | add + RMSNorm | 63.648 us | 20.480 us | 3.11x |
| BF16 | hot | RMSNorm | 36.415 us | 6.984 us | 5.21x |
| BF16 | hot | add + RMSNorm | 40.572 us | 9.298 us | 4.36x |
| BF16 | cold | RMSNorm | 62.720 us | 16.384 us | 3.83x |
| BF16 | cold | add + RMSNorm | 69.648 us | 20.592 us | 3.38x |

![FP16 benchmark comparison](/images/posts/cuda-fused-rmsnorm/fp16-benchmark.svg)

*图：本文基于公开 CSV 原创绘制。横轴为单次延迟，越短越好；baseline 为 `torch.compile(fullgraph=True)`。*

结果显示两件事：

1. 小而专用的 CUDA kernel 即使面对 `torch.compile`，仍能显著减少通用路径和 reduction 的开销；
2. cold-cache 的加速比低于 hot-cache，说明这个算子确实受到内存层级影响，只报告 hot-cache 会高估收益。

## 8. 正确性和资源占用

项目在真实 GPU 上通过 23 项测试，覆盖：

- FP32、FP16、BF16；
- 255 维 scalar fallback；
- 256、4096、8192 维 vectorized path；
- 二维和三维输入；
- fused residual output；
- PyTorch `F.rms_norm` 参考结果。

`cuobjdump --dump-resource-usage` 给出的 vectorized fused kernel 资源如下：

| dtype | registers/thread | shared | local | stack |
|---|---:|---:|---:|---:|
| FP16 | 30 | 32B | 0B | 0B |
| BF16 | 26 | 32B | 0B | 0B |
| FP32 | 29 | 32B | 0B | 0B |

没有 local memory 和 stack 使用，说明当前版本没有发生寄存器溢出。

我也尝试使用 Nsight Compute 获取 DRAM throughput、L2 hit rate 和 warp stall reason，但 AutoDL 容器禁止访问 GPU performance counters，返回 `ERR_NVGPUCTRPERM`。这里选择明确记录限制，而不是用逻辑带宽冒充硬件计数器。

## 9. 为什么这个项目还不能叫“生产级”

当前版本有清晰的边界：

- 只实现 forward，适合推理，没有 backward；
- block size 固定为 256，没有按 GPU 架构和 hidden size autotune；
- 只测了单算子，没有接入完整 decoder block；
- 只有 RTX 4090D 数据，缺少 A100/H100 等数据中心 GPU 对比；
- 还没有加入 Apex 和独立 Triton baseline。

下一阶段最值得做的不是继续堆零散 kernel，而是加入 block-size dispatch、Triton 对照实现，并测量融合算子对端到端 time per output token 的影响。

## 10. 总结

这个项目让我把“算子融合可以减少访存”这句话落到了可验证的工程细节上：

- 先从模型调用语义确定 fusion 边界；
- 用 roofline 思维判断 RMSNorm 是 memory-bound；
- 用 pack-4 和 warp shuffle 实现具体优化；
- 用 FP32 accumulation 和 dtype 舍入保证数值正确；
- 用 hot/cold cache 与强 baseline 设计可信实验；
- 最后如实报告硬件、原始数据和测量限制。

对 AI Infra 项目而言，真正有价值的不是一个最大的加速数字，而是能解释每一项优化为什么成立、在哪些条件下成立，以及下一步该验证什么。
