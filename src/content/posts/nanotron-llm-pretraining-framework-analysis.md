---
title: Nanotron 项目详解：Hugging Face 的大模型预训练框架怎么做分布式训练
date: 2026-05-10 12:10:00
tags:
  - LLM
  - 大模型训练
  - 分布式训练
  - Nanotron
  - Hugging Face
  - AI Infra
  - PyTorch
categories:
  - AI
---

这篇文章选一个真正和“大模型训练”强相关的开源项目来讲：Hugging Face 的 **Nanotron**。

项目地址：<https://github.com/huggingface/nanotron>

Nanotron 是 Hugging Face 开源的大模型预训练框架，定位是 **Pretraining transformer models made easy**。它不是一个简单的训练脚本，而是一个面向大规模 Transformer 预训练的训练基础设施项目，核心能力包括 3D 并行（DP + TP + PP）、MoE Expert Parallelism、Pipeline Parallel schedule、ZeRO-1、FP32 梯度累积、参数 tying/sharding、大模型 checkpoint、CUDA event 性能计时等。

如果你想理解“大模型训练项目到底在做什么”，Nanotron 是一个非常适合拆解的项目。



## 1. 项目一句话介绍

Nanotron 可以概括为：

> Nanotron 是 Hugging Face 开源的 Transformer 大模型预训练框架，围绕分布式并行、训练配置、数据加载、模型切分、优化器、checkpoint 和性能监控构建了一套可扩展的大模型训练基础设施。

如果放到面试里，可以这样介绍：

> Nanotron 不是普通的单卡训练脚本，而是一个面向 LLM pretraining 的分布式训练框架。它通过数据并行、张量并行和流水线并行组成 3D parallelism，把模型参数、计算和 batch 拆到多张 GPU 甚至多节点上；通过 ZeRO-1 降低优化器状态内存；通过 pipeline schedule 提升多 stage 训练效率；通过 YAML 配置统一管理模型、数据、优化器、并行策略、checkpoint 和日志，从而让大模型预训练流程工程化、可复现、可扩展。

## 2. 为什么选择 Nanotron

大模型训练项目有很多，例如 Megatron-LM、DeepSpeed、Colossal-AI、LLaMA-Factory、Axolotl 等。这里选择 Nanotron 的原因是：

1. 它来自 Hugging Face，生态链接很强。
2. 它专注于 pretraining，而不只是 SFT/LoRA 微调。
3. 它显式支持 DP、TP、PP 的 3D 并行。
4. 它的训练入口、配置和文档比较清晰。
5. 它适合用来学习“大模型训练基础设施”而不是只会调 trainer。

如果你的目标是理解“从零训练一个 Transformer 大模型需要哪些系统能力”，Nanotron 比很多只做微调的项目更合适。

## 3. 大模型训练为什么复杂

普通 PyTorch 训练大概是：

```text
model -> dataloader -> forward -> loss -> backward -> optimizer.step
```

但大模型训练会遇到完全不同的问题：

1. 模型参数太大，一张 GPU 放不下。
2. 激活值太大，batch 稍大就 OOM。
3. 优化器状态通常是参数量的数倍。
4. 多 GPU 通信开销很高。
5. 多节点训练容易出现 straggler 和网络瓶颈。
6. checkpoint 很大，保存和恢复都很慢。
7. 数据吞吐必须跟得上 GPU 消耗。
8. 训练中断后必须能恢复。
9. loss spike、梯度溢出、nan 都要监控。
10. 性能不能只看能不能跑，还要看 MFU、吞吐和尾部延迟。

Nanotron 的价值就是把这些复杂问题拆成一个个可配置的训练模块。

## 4. Nanotron 的整体训练链路

一个 Nanotron 训练任务大致是：

```text
读取 YAML 配置
  -> 初始化 torch.distributed
  -> 构建并行拓扑 DP / TP / PP
  -> 构建 tokenizer 和 dataloader
  -> 构建 Transformer 模型
  -> 按 TP / PP 切分模型
  -> 构建 optimizer / scheduler
  -> 加载 checkpoint 或从头训练
  -> 训练循环 forward / backward / step
  -> 定期保存 checkpoint
  -> 记录 loss、吞吐、显存、计时指标
```

相比一个普通训练脚本，它多出来的核心是：

- 分布式进程组管理。
- 模型并行切分。
- Pipeline 调度。
- 优化器状态分片。
- 大规模 checkpoint 管理。
- 性能监控和 benchmark。

## 5. Quick Start 怎么跑

Nanotron README 给出的最小训练命令是：

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```

这个命令里有几个关键点：

- `torchrun`：PyTorch 官方分布式启动器。
- `--nproc_per_node=8`：单节点启动 8 个进程，通常对应 8 张 GPU。
- `run_train.py`：Nanotron 的训练入口。
- `--config-file`：训练配置文件。
- `CUDA_DEVICE_MAX_CONNECTIONS=1`：对某些分布式通信和 kernel 调度有影响，常用于提升特定并行模式下的稳定性或性能。

这说明 Nanotron 的训练方式不是在 Python 里手动 `for gpu in gpus`，而是使用标准分布式多进程模型。

## 6. 配置驱动训练

Nanotron 使用 YAML 配置训练参数。一个大模型训练任务通常包括：

```text
model:
  architecture
  hidden_size
  num_layers
  num_attention_heads
  sequence_length

tokens:
  tokenizer
  vocab_size

data:
  dataset
  num_loading_workers
  seed

parallelism:
  dp
  tp
  pp

optimizer:
  learning_rate
  weight_decay
  betas
  grad_clip

scheduler:
  warmup_steps
  total_steps

checkpoints:
  checkpoint_interval
  save_dir

logging:
  wandb
  log_interval
```

配置驱动的好处：

1. 实验可复现。
2. 多组实验只需要改配置。
3. 训练脚本不用写死参数。
4. 多人协作时更容易审查训练设置。
5. 方便在 Slurm 多节点环境里提交任务。

## 7. 3D Parallelism：DP + TP + PP

Nanotron 的核心能力之一是 3D 并行，也就是：

```text
Data Parallelism + Tensor Parallelism + Pipeline Parallelism
```

这是大模型训练里最重要的概念之一。

## 8. DP：数据并行

数据并行是最容易理解的并行方式。

每张 GPU 放一份完整模型，但喂不同 batch：

```text
GPU0: model + batch0
GPU1: model + batch1
GPU2: model + batch2
GPU3: model + batch3
```

每张卡独立 forward/backward，然后通过 AllReduce 同步梯度。

优点：

- 实现简单。
- 扩展 batch size 直接。
- 适合模型能放进单卡的情况。

缺点：

- 每张卡都要保存完整模型参数。
- 模型太大时单卡放不下。
- 优化器状态也会重复保存。

所以大模型不能只靠 DP。

## 9. TP：张量并行

张量并行是把单层内部的矩阵计算拆到多张 GPU 上。

以 Transformer 里的线性层为例：

```text
Y = XW
```

如果 W 太大，可以按列或按行切分：

```text
W = [W0, W1, W2, W3]
GPU0 计算 XW0
GPU1 计算 XW1
GPU2 计算 XW2
GPU3 计算 XW3
最后再通信合并
```

TP 的优点：

- 单层参数可以拆到多张卡。
- 适合超大 hidden size 和 attention/MLP 层。
- 降低单卡参数和激活压力。

缺点：

- 层内通信频繁。
- 对 GPU 间互联要求高，例如 NVLink / NVSwitch。
- 跨节点 TP 通常性能较差。

因此 TP 通常放在单节点内做。

## 10. PP：流水线并行

流水线并行是按层切分模型。

例如一个 32 层 Transformer，可以分成 4 个 stage：

```text
GPU0: layers 0-7
GPU1: layers 8-15
GPU2: layers 16-23
GPU3: layers 24-31
```

数据从 stage 0 流到 stage 3，反向传播再从 stage 3 回到 stage 0。

PP 的优点：

- 适合层数很多的大模型。
- 每张卡只保存一部分层。
- 可以跨节点扩展。

缺点：

- 有 pipeline bubble，也就是部分 GPU 在等待。
- 需要 micro-batch 切分提高流水线利用率。
- 调度复杂，debug 难度高。

## 11. 3D 并行怎么组合

假设有 64 张 GPU，可以组合为：

```text
dp = 4
tp = 4
pp = 4
总 GPU = dp * tp * pp = 64
```

含义是：

- 每个数据并行副本有 `tp * pp = 16` 张 GPU。
- 模型在一个副本内部被 TP 和 PP 切开。
- 一共有 4 个这样的副本处理不同数据。
- 不同 DP 副本之间同步梯度。

Nanotron 的配置里就需要明确这些并行维度。

## 12. Global Batch Size 怎么算

Nanotron 文档里提到，global batch size 通常是：

```text
micro_batch_size * batch_accumulation_per_replica * dp
```

这里每个变量的含义是：

- `micro_batch_size`：每次 forward 的小 batch。
- `batch_accumulation_per_replica`：梯度累积步数。
- `dp`：数据并行副本数量。

为什么需要梯度累积？因为显存有限，不能一次塞很大的 batch。可以多次 forward/backward 累积梯度，再执行一次 optimizer step。

例如：

```text
micro_batch_size = 2
batch_accumulation_per_replica = 8
dp = 16
global_batch_size = 2 * 8 * 16 = 256
```

## 13. Pipeline Schedule：AFAB 和 1F1B

Nanotron 支持 Pipeline Parallel 的 schedule，例如 AFAB 和 1F1B。

### 13.1 AFAB

AFAB 可以理解为：

```text
All Forward, All Backward
```

先把所有 micro-batch 的 forward 做完，再做 backward。

优点：简单。

缺点：激活保存多，显存压力大，pipeline bubble 可能明显。

### 13.2 1F1B

1F1B 是：

```text
One Forward, One Backward
```

流水线填满后，每个 stage 交替做一个 forward 和一个 backward。

优点：

- 显存更友好。
- Pipeline 利用率更高。
- 大模型训练更常用。

缺点：实现复杂，调度和通信更难 debug。

## 14. ZeRO-1：优化器状态分片

大模型训练里，优化器状态非常占显存。

以 Adam 为例，每个参数除了权重本身，还要维护：

```text
param
Gradient
Momentum m
Variance v
```

如果用混合精度，还可能有 FP32 master weight。

ZeRO 的核心思想是把冗余状态切分到不同 DP rank 上。

ZeRO-1 主要分片 optimizer states：

```text
DP rank0 保存一部分 optimizer state
DP rank1 保存一部分 optimizer state
DP rank2 保存一部分 optimizer state
...
```

这样可以减少每张 GPU 上重复保存的优化器状态。

Nanotron 当前支持 ZeRO-1，roadmap 中有 ZeRO-3 / FSDP。

## 15. FP32 梯度累积

Nanotron 支持 FP32 gradient accumulation。

这点对训练稳定性很重要。大模型训练通常会用 BF16/FP16 做 forward/backward，但梯度累积如果精度太低，可能带来数值误差。

FP32 累积的含义是：

```text
每个 micro-batch 得到梯度
  -> 转成 / 保持 FP32 累积
  -> 累积若干步
  -> optimizer step
```

这样可以提升训练稳定性，尤其是在大 batch、长序列和深层网络下。

## 16. Parameter Tying 和 Sharding

Transformer 语言模型中常见参数 tying：

```text
input embedding weight == output lm_head weight
```

这样可以减少参数量，也可能提升泛化。

但在 TP/PP/DP 并行下，参数 tying 会复杂很多：

- tied 参数可能在不同 pipeline stage。
- 参数可能被 tensor parallel 切分。
- checkpoint 保存和加载要知道它们共享同一份权重。
- 梯度同步不能重复或漏掉。

Nanotron 支持 parameter tying/sharding，说明它不是只处理单卡模型，而是考虑了大模型并行训练中的真实边界情况。

## 17. Checkpoint：大模型训练的生命线

大模型训练动辄跑几天、几周甚至几个月，checkpoint 是生命线。

Checkpoint 至少要保存：

```text
模型参数
优化器状态
scheduler 状态
训练步数
随机数状态
并行切分信息
tokenizer / config
```

在 3D 并行下，checkpoint 更复杂：

- 每个 rank 只保存自己负责的参数 shard。
- 恢复时必须匹配 TP/PP/DP 拓扑。
- 如果并行配置变化，还需要 reshard。
- 保存太频繁会拖慢训练。
- 保存太少则失败恢复成本高。

Nanotron 配置中可以设置 `checkpoint_interval` 控制保存频率。

## 18. 数据加载：吞吐不能拖后腿

大模型训练时 GPU 很贵，最怕 GPU 等数据。

一个训练数据链路通常是：

```text
原始文本数据
  -> 清洗、去重、过滤
  -> tokenizer 编码
  -> packed sequence
  -> dataloader 多进程加载
  -> batch 送入 GPU
```

Nanotron 支持 Hugging Face datasets，也支持自定义 dataloader。

文档里提到，如果要使用自定义数据加载器，可以把 dataset 配置设为 null，然后自己实现 dataloader。

这在真实训练中很常见，因为公司内部数据格式、数据清洗逻辑和采样策略往往是定制的。

## 19. 自定义 Dataloader 为什么重要

大模型训练的数据不是简单的图片分类数据集，而是海量 token 流。

自定义 dataloader 可能要处理：

- 多数据源混合采样。
- 数据权重配比。
- 文档边界。
- sequence packing。
- tokenizer 版本。
- curriculum learning。
- 数据去重标记。
- 断点恢复时的数据位置。

如果数据加载不可控，训练结果就不可控。

## 20. 性能监控：不能只看 loss

Nanotron 支持 CUDA event-based timing，用于更准确地测量 GPU 性能。

大模型训练要关注的指标包括：

```text
loss
learning rate
grad norm
tokens/sec
samples/sec
GPU memory
MFU
step time
communication time
pipeline bubble
checkpoint time
```

其中 MFU 是 Model FLOPS Utilization，表示模型实际使用了理论峰值算力的多少。

高质量训练系统不只要 loss 降，还要知道训练资源利用率怎么样。

## 21. MFU 为什么重要

如果 100 张 H100 只跑出很低的 MFU，就意味着大量钱花在空等通信、数据加载或 pipeline bubble 上。

MFU 低可能来自：

- batch 太小。
- TP 通信过重。
- PP bubble 太大。
- dataloader 跟不上。
- checkpoint 阻塞训练。
- kernel 没有融合。
- 序列长度和模型大小不匹配硬件。

Nanotron README 中也强调了 benchmark 和 Ultrascale Playbook，说明它很关注训练效率。

## 22. 和 Megatron-LM / DeepSpeed 的关系

Nanotron 的很多思想和 Megatron-LM、DeepSpeed 类似。

| 项目 | 重点 | 适合学习什么 |
|---|---|---|
| Megatron-LM | NVIDIA 大规模 Transformer 训练 | TP/PP、GPT 预训练、高性能 kernel |
| DeepSpeed | 分布式训练优化库 | ZeRO、offload、训练系统优化 |
| Nanotron | Hugging Face 预训练框架 | 配置化 LLM pretraining、3D 并行、训练工程化 |
| LLaMA-Factory | 微调框架 | SFT、LoRA、DPO、数据格式适配 |
| Axolotl | 微调配置框架 | 指令微调、多模型配置 |

所以 Nanotron 更适合作为“大模型预训练系统”的学习案例，而不是“怎么快速微调一个模型”的工具。

## 23. 训练一个 tiny Llama 的流程

Nanotron Quick Start 是训练 tiny Llama。

完整理解可以拆成：

```text
1. 安装 Python / PyTorch / Nanotron
2. 安装 datasets、transformers、wandb、flash-attn 等依赖
3. 登录 Hugging Face 和 W&B
4. 准备 config_tiny_llama.yaml
5. torchrun 启动 8 GPU 训练
6. 训练中保存 checkpoint
7. 使用 run_generate.py 从 checkpoint 生成文本
```

生成命令类似：

```bash
torchrun --nproc_per_node=1 run_generate.py --ckpt-path checkpoints/{checkpoint_number}/ --tp 1 --pp 1
```

这里 `--tp` 和 `--pp` 也可以用于生成阶段的并行设置。

## 24. 多节点训练

单节点训练只需要 `torchrun --nproc_per_node=8`，多节点训练通常还需要：

```text
nnodes
node_rank
master_addr
master_port
```

以及集群调度系统，例如 Slurm。

多节点训练难点包括：

- 节点间网络延迟。
- NCCL 配置。
- master 节点发现。
- rank 映射。
- 故障恢复。
- 日志聚合。
- checkpoint 存储共享。

Nanotron 提供 multi-node training 文档，说明它不是只考虑单机多卡。

## 25. 大模型训练中的常见故障

### 25.1 OOM

原因可能是：

- micro batch 太大。
- sequence length 太长。
- TP/PP 配置不合理。
- 激活 checkpoint 没开。
- optimizer state 太大。

解决方式：

- 降低 micro batch。
- 增加 TP/PP。
- 使用梯度累积保持 global batch。
- 使用 checkpointing。
- 使用 ZeRO。

### 25.2 Loss Spike

原因可能是：

- 学习率过高。
- warmup 不足。
- 数据异常。
- 梯度裁剪缺失。
- 混合精度溢出。

解决方式：

- 调低学习率。
- 增加 warmup。
- 检查数据。
- 开启 grad clip。
- 监控 grad norm。

### 25.3 GPU 利用率低

原因可能是：

- dataloader 慢。
- 通信开销大。
- pipeline bubble。
- batch 太小。
- checkpoint 阻塞。

解决方式：

- 增加 num workers。
- 优化并行配置。
- 增加 micro batch 数。
- 使用更合适的 schedule。
- 异步或降低 checkpoint 频率。

## 26. Nanotron 的项目难点

### 26.1 并行拓扑管理

DP、TP、PP 同时存在时，每个 rank 属于多个进程组。框架必须知道：

```text
当前 rank 属于哪个 DP group
当前 rank 属于哪个 TP group
当前 rank 属于哪个 PP stage
哪些 rank 需要通信
哪些参数需要同步
```

这比普通 DDP 复杂得多。

### 26.2 Pipeline 调度

Pipeline parallel 要处理 micro-batch 的 forward/backward 顺序、激活保存、通信、bubble 和梯度累积。schedule 写错会导致死锁、梯度错误或性能很差。

### 26.3 参数切分和 checkpoint

模型参数可能被 TP 切分，也可能按 PP 分布在不同 stage。保存和恢复 checkpoint 时必须知道每个 shard 的位置和含义。

### 26.4 数据吞吐和训练稳定性

训练系统不仅要能跑，还要稳定跑很多 step。数据加载、随机种子、checkpoint、日志、异常恢复都必须工程化。

## 27. 如果要读源码，建议顺序

读 Nanotron 不建议一开始就钻进所有并行细节，可以按这个顺序：

1. `README.md`：理解项目定位和 quick start。
2. `examples/config_tiny_llama.yaml`：理解配置项。
3. `run_train.py`：看训练入口。
4. 配置解析相关代码：看 YAML 怎么变成训练对象。
5. distributed / parallel context：看 DP/TP/PP group 怎么建。
6. model 构建：看 Llama / Transformer 怎么定义。
7. pipeline schedule：看 forward/backward 怎么调度。
8. optimizer：看 ZeRO-1 和梯度累积。
9. checkpoint：看保存和恢复。
10. dataloader：看数据如何进入训练。

## 28. 面试 1 分钟讲法

如果面试官让你讲 Nanotron，可以这样说：

> Nanotron 是 Hugging Face 开源的大模型预训练框架，主要用于 Transformer / LLM pretraining。它的核心是把大模型训练工程化：通过 YAML 配置描述模型、数据、优化器、并行策略和 checkpoint；通过 torchrun 启动多进程分布式训练；通过 DP、TP、PP 组成 3D 并行，把 batch、张量和模型层分别拆到不同 GPU 上；通过 ZeRO-1 分片优化器状态降低显存；通过 AFAB/1F1B pipeline schedule 提升流水线训练效率；同时支持 FP32 梯度累积、参数 tying/sharding、大模型 checkpoint 和 CUDA event 计时。它适合学习从单机训练到多节点 LLM 预训练所需的训练基础设施。

## 29. 面试高频问答

### Q1：Nanotron 和普通 Trainer 有什么区别？

普通 Trainer 更偏单机或简单 DDP 训练，Nanotron 面向大模型预训练，重点是 3D 并行、pipeline schedule、optimizer state sharding、checkpoint sharding 和性能监控。

### Q2：为什么大模型训练不能只用数据并行？

因为数据并行要求每张 GPU 都保存完整模型和优化器状态。模型太大时单卡放不下，即使放得下，优化器状态和激活也会占用大量显存。

### Q3：TP 和 PP 的区别是什么？

TP 是把单层内部的矩阵计算切到多张 GPU；PP 是把模型的不同层切到不同 GPU。TP 更依赖高速互联，PP 更适合按层扩展超深模型。

### Q4：global batch size 怎么算？

通常是 `micro_batch_size * batch_accumulation_per_replica * dp`。如果显存不够，就减小 micro batch，用梯度累积保持 global batch。

### Q5：1F1B 为什么比 AFAB 更常用？

1F1B 在流水线填满后交替执行 forward 和 backward，可以减少激活保存压力，并提升 pipeline 利用率。AFAB 简单但显存压力更大。

### Q6：ZeRO-1 节省什么？

ZeRO-1 主要分片优化器状态，例如 Adam 的 momentum 和 variance，减少数据并行副本间重复保存 optimizer states 的显存开销。

### Q7：为什么 checkpoint 在大模型训练里复杂？

因为参数可能被 TP/PP 切分，不同 rank 保存不同 shard；恢复时必须匹配并行拓扑，还要恢复 optimizer、scheduler、step 和随机状态。

### Q8：为什么要 FP32 梯度累积？

混合精度训练中，FP32 累积能减少数值误差，提高训练稳定性，尤其是在长序列、大 batch 和深层网络中。

### Q9：MFU 是什么？

MFU 是 Model FLOPS Utilization，表示模型训练实际利用了理论峰值算力的比例。它能反映训练系统是否高效利用 GPU。

### Q10：Nanotron 适合微调吗？

Nanotron 更偏预训练基础设施。如果目标是 LoRA/SFT/DPO，LLaMA-Factory 或 Axolotl 这类微调框架更直接；如果想学大规模 pretraining，Nanotron 更合适。

## 30. 简历写法建议

如果你想把 Nanotron 源码学习写进简历，可以这样写：

> 深入学习 Hugging Face Nanotron 大模型预训练框架，梳理其基于 torchrun 的分布式训练入口、YAML 配置系统、DP/TP/PP 3D 并行拓扑、Pipeline Parallel schedule、ZeRO-1 优化器状态分片、FP32 梯度累积、参数 tying/sharding、checkpoint 保存恢复和 CUDA event 性能计时机制；理解 LLM pretraining 中显存、通信、吞吐、checkpoint 与训练稳定性的系统性权衡。

如果是做项目复现，可以写：

> 基于 Nanotron 复现 tiny Llama 预训练流程，完成训练配置构建、数据加载、8 卡 torchrun 启动、checkpoint 保存与生成测试，分析 DP/TP/PP 并行配置对 global batch size、显存占用和训练吞吐的影响。

## 31. 总结

Nanotron 是一个非常适合学习大模型训练基础设施的项目。它把 LLM 预训练中最关键的系统问题都摆在了台面上：模型太大怎么办、显存不够怎么办、多卡怎么切、多节点怎么跑、pipeline 怎么调度、优化器状态怎么省、checkpoint 怎么存、数据怎么喂、性能怎么量化。

如果只是想“把模型跑起来”，可能用一个 Trainer 就够了；但如果想理解“大模型训练工程到底怎么做”，Nanotron 这种项目更值得认真读。
