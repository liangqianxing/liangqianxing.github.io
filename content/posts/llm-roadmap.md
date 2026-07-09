---
title: LLM 技术路线：从 Transformer 到训练、推理与应用
date: 2026-07-09 11:00:00
description: 一份系统化 LLM 学习路线，覆盖 Transformer、预训练、SFT、RLHF、RAG、Agent、推理优化、评测和工程项目。
categories:
  - 技术
tags:
  - LLM
  - 大模型训练
  - 推理优化
  - AI Infra
  - RAG
  - Agent
  - 面试
---

这篇文章整理一条 **LLM 技术路线**。它更偏模型与算法视角：从 Transformer 开始，到预训练、微调、对齐、推理优化、RAG、Agent、评测和工程落地。

如果把 LLM 学习拆成一句话：

> LLM 路线要同时理解“模型为什么能工作”和“系统如何把它跑起来”。

只会调 API，面试会被问穿；只会推公式，不会落地，也很难做项目。比较好的路线是：先理解模型结构，再理解训练范式，然后补推理、评测和应用系统。

---

## 0. LLM 路线总览

可以按八层来学：

```text
数学与深度学习基础
  -> Transformer
  -> Tokenizer / Data
  -> Pretraining
  -> SFT / PEFT
  -> Alignment
  -> Inference
  -> RAG / Agent / Evaluation
```

每一层都对应一个核心问题：

| 层级 | 核心问题 |
|---|---|
| 数学基础 | 为什么梯度下降能训练模型 |
| 深度学习 | MLP、Attention、Normalization 如何工作 |
| Transformer | LLM 的基本架构是什么 |
| 数据 | 语料、tokenizer、数据配比如何影响模型 |
| 预训练 | next token prediction 如何学到能力 |
| 微调 | SFT / LoRA 如何改变模型行为 |
| 对齐 | RLHF / DPO 如何让模型更符合偏好 |
| 推理 | KV Cache、batching、quantization 如何提速 |
| 应用 | RAG、Agent、评测如何让模型解决真实问题 |

---

## 1. 第一阶段：数学和深度学习基础

不需要把所有数学都学成研究生水平，但至少要能看懂 LLM 论文和代码。

### 必备数学

- 线性代数：矩阵乘法、向量空间、特征值、范数；
- 概率统计：条件概率、交叉熵、KL 散度、采样；
- 微积分：导数、链式法则、梯度；
- 优化：SGD、Adam、学习率、weight decay；
- 信息论：entropy、perplexity、bits per token。

### 必备深度学习

- MLP；
- embedding；
- softmax；
- cross entropy；
- backpropagation；
- dropout；
- layer normalization；
- residual connection；
- optimizer；
- overfitting / underfitting。

### 建议项目

先不要急着上大模型。可以从零写一个小的 autograd 或 MLP：

```text
Tensor
  -> forward graph
  -> backward
  -> Linear
  -> ReLU
  -> CrossEntropy
  -> SGD / Adam
```

写过一遍之后，再看 Transformer 会轻松很多。

---

## 2. 第二阶段：Transformer

Transformer 是 LLM 的骨架。必须能从输入 token 一路讲到输出 logits。

### 主线结构

```text
Token IDs
  -> Token Embedding
  -> Positional Encoding / RoPE
  -> Transformer Blocks
      -> LayerNorm
      -> Multi-Head Self Attention
      -> Residual
      -> LayerNorm
      -> FFN / MLP
      -> Residual
  -> LM Head
  -> Logits
  -> Softmax
  -> Next Token
```

### 必须理解的问题

| 模块 | 问题 |
|---|---|
| Self-Attention | Q/K/V 是什么，为什么要除以 sqrt(d) |
| Multi-Head | 多头为什么有用，不同 head 是否学不同关系 |
| Causal Mask | 为什么 decoder-only 只能看左边 |
| Position | 绝对位置、RoPE、ALiBi 的差异 |
| LayerNorm | 为什么稳定训练 |
| FFN | 为什么中间维度通常更大 |
| Residual | 为什么深层网络离不开残差 |
| LM Head | hidden state 如何映射到词表概率 |

### 推荐项目

写一个 **Mini GPT**：

- 支持 tokenizer；
- 支持 causal self-attention；
- 支持训练一个小语料；
- 支持 generate；
- 打印 loss 和 perplexity；
- 保存 checkpoint。

这个项目是 LLM 路线的分水岭。你能不能讲清 LLM，本质上看你能不能讲清这个项目。

---

## 3. 第三阶段：Tokenizer 与数据

很多人学 LLM 只看模型结构，但真实训练里数据极其重要。

### Tokenizer 要懂什么

- BPE；
- WordPiece；
- SentencePiece；
- byte-level BPE；
- special tokens；
- 中文 tokenization 问题；
- vocabulary size；
- token 压缩率。

### 数据要懂什么

- 数据清洗；
- 去重；
- 质量过滤；
- 语言配比；
- domain mixture；
- contamination；
- packing；
- sequence length；
- train / eval split。

### 关键认知

LLM 不是“架构一换就变强”。很多能力来自：

- 语料规模；
- 语料质量；
- 数据配比；
- 训练稳定性；
- 指令数据；
- 人类偏好数据；
- 评测反馈。

所以准备 LLM 面试时，不能只背 Transformer，也要能讲数据工程。

---

## 4. 第四阶段：预训练

预训练的核心目标通常是 next token prediction：

```text
给定 x1, x2, ..., xt
预测 x(t+1)
```

看起来简单，但大模型的能力就是从这个目标里涌现出来的。

### 要掌握的概念

- causal language modeling；
- loss curve；
- perplexity；
- batch size；
- gradient accumulation；
- learning rate schedule；
- warmup；
- cosine decay；
- checkpoint；
- mixed precision；
- distributed training；
- data parallel；
- tensor parallel；
- pipeline parallel；
- ZeRO；
- gradient checkpointing。

### 训练问题排查

| 现象 | 可能原因 |
|---|---|
| loss 不下降 | 学习率、数据、mask、label shift、初始化 |
| loss 爆炸 | 学习率过高、梯度裁剪缺失、数值精度问题 |
| eval 很差 | 数据污染、过拟合、验证集不合理 |
| 训练很慢 | batch 太小、I/O 瓶颈、通信瓶颈 |
| 显存不够 | activation、optimizer state、sequence length |

### 推荐项目

训练一个小模型：

- 数据用 TinyStories / WikiText / 自己的中文语料；
- 模型规模可以很小；
- 跑通 tokenizer、dataset、dataloader、training loop；
- 记录 loss；
- 支持 checkpoint 和 resume。

目的不是训练出强模型，而是理解训练系统。

---

## 5. 第五阶段：SFT 与 PEFT

预训练模型会补全文本，但不一定会听指令。所以需要 SFT。

### SFT

SFT 的训练数据一般是：

```text
Instruction
Input
Response
```

模型学习的是在给定指令下输出期望回答。

要理解：

- prompt template；
- chat template；
- assistant mask；
- loss 只算 assistant 部分；
- 多轮对话格式；
- 数据质量比数量更重要；
- instruction diversity。

### PEFT

全量微调成本高，所以常用参数高效微调：

- LoRA；
- QLoRA；
- Adapter；
- Prefix tuning；
- Prompt tuning。

LoRA 的核心思想：

> 不直接更新原始大矩阵，而是学习一个低秩增量矩阵。

可以表达为：

```text
W' = W + BA
```

其中 W 冻结，只训练 A 和 B。

### 推荐项目

做一个领域微调：

- 准备 500-2000 条高质量 instruction 数据；
- 用 LoRA 微调一个小模型；
- 对比 base model 和 fine-tuned model；
- 加一个简单评测集；
- 记录错误样例。

---

## 6. 第六阶段：对齐

SFT 让模型会回答，对齐让模型更符合人类偏好。

### 典型路线

```text
Pretrain
  -> SFT
  -> Preference Data
  -> Reward Model / DPO
  -> Aligned Model
```

### RLHF

RLHF 通常包含：

1. 收集人类偏好数据；
2. 训练 reward model；
3. 用 PPO 等方法优化策略模型；
4. 约束模型不要偏离原始能力太远。

### DPO

DPO 更简单，直接使用 chosen / rejected 偏好对优化模型，不需要显式训练 reward model。

面试中可以这样说：

> RLHF 把偏好学习拆成 reward model 和强化学习优化，DPO 则把偏好对直接转成监督式目标，工程上更简单、稳定性更好。

### 要理解的风险

- reward hacking；
- over-optimization；
- helpfulness / harmlessness trade-off；
- refusal 过度；
- benchmark overfitting；
- 数据偏差。

---

## 7. 第七阶段：推理优化

推理是 LLM 工程落地的核心。训练可以离线慢慢跑，推理必须在线服务用户。

### 推理链路

```text
Prompt Tokens
  -> Prefill
  -> KV Cache
  -> Decode one token
  -> Append token
  -> Repeat
```

### 必须掌握

| 技术 | 作用 |
|---|---|
| KV Cache | 避免重复计算历史 token |
| Continuous Batching | 动态合并不同请求，提高吞吐 |
| PagedAttention | 更高效管理 KV Cache 内存 |
| Quantization | INT8 / INT4 降低显存和带宽 |
| Speculative Decoding | 小模型草稿，大模型验证 |
| FlashAttention | 优化 attention 计算和显存访问 |
| Tensor Parallel | 多卡切分矩阵计算 |
| Prefix Cache | 复用相同 system prompt / prefix |

### 关键指标

- TTFT：time to first token；
- TPOT：time per output token；
- throughput：tokens/s；
- latency p50 / p95 / p99；
- GPU utilization；
- KV cache hit rate；
- batch size；
- cost per 1k tokens。

### 推荐项目

做一个推理服务压测：

- 用 vLLM / TensorRT-LLM / llama.cpp 任选；
- 比较不同 batch、max_tokens、并发数；
- 记录 TTFT、tokens/s、显存；
- 尝试量化；
- 尝试 prefix cache。

这个项目很适合写进 LLM Infra 简历。

---

## 8. 第八阶段：RAG、Agent 与应用

LLM 本身只是模型，真实产品通常要接外部知识和工具。

### RAG 要解决

- 知识更新；
- 私有数据；
- 引用来源；
- 减少幻觉；
- 降低长上下文成本。

### Agent 要解决

- 多步任务；
- 工具调用；
- 计划执行；
- 文件操作；
- 浏览器操作；
- 代码执行；
- 状态恢复。

### 应用工程的关键

- prompt version；
- output schema；
- tool schema；
- eval dataset；
- trace；
- fallback；
- human-in-the-loop；
- safety confirmation；
- cost budget。

LLM 路线到这里会和 AI Infra 路线汇合。

---

## 9. 第九阶段：评测

不会评测，就无法迭代。

### 模型评测

- MMLU；
- GSM8K；
- HumanEval；
- BBH；
- C-Eval；
- CMMLU；
- MT-Bench；
- Arena；
- domain benchmark。

### 应用评测

- answer correctness；
- faithfulness；
- citation accuracy；
- tool success rate；
- format success rate；
- refusal accuracy；
- latency；
- cost；
- user feedback。

### 评测项目

做一个 **LLM Eval Harness**：

- 支持 JSONL 测试集；
- 支持多模型对比；
- 支持规则评分；
- 支持 LLM-as-judge；
- 保存 bad case；
- 输出 HTML/Markdown 报告。

这个项目能体现你不是只会调用模型，而是能持续改进模型系统。

---

## 10. 推荐学习顺序

### 第 1 个月：Transformer 基础

- 学 attention；
- 写 mini GPT；
- 跑通训练和生成；
- 理解 tokenizer。

### 第 2 个月：训练与微调

- 跑一个小规模预训练；
- 做 SFT；
- 做 LoRA；
- 对比微调前后效果。

### 第 3 个月：推理优化

- 学 KV Cache；
- 学 vLLM；
- 做压测；
- 理解量化和 batching。

### 第 4 个月：RAG / Agent / Eval

- 做 hybrid RAG；
- 做 Agent tool calling；
- 做 eval harness；
- 整理成完整项目。

---

## 11. 面试准备重点

LLM 面试经常问这些问题：

1. Transformer 为什么要用 self-attention？
2. Q/K/V 分别是什么？
3. causal mask 为什么必要？
4. RoPE 怎么理解？
5. Pretrain 和 SFT 的区别？
6. LoRA 为什么参数少？
7. RLHF 和 DPO 区别？
8. KV Cache 为什么能加速？
9. vLLM 的 PagedAttention 解决什么问题？
10. RAG 为什么会召回错误？
11. Agent 为什么容易失控？
12. 如何评测一个 LLM 应用？

回答时不要只背概念，最好结合项目说。

例如 KV Cache：

> 自回归解码每次只生成一个 token，如果每一步都重新计算全部历史 token，复杂度和延迟会很高。KV Cache 会保存历史 token 在每层 attention 中的 key/value，新 token 只需要和已有 K/V 做 attention，从而避免重复计算。它提升 decode 阶段速度，但会带来显存占用问题，所以 vLLM 这类系统会重点优化 KV Cache 管理。

---

## 12. 项目组合建议

如果要包装成 LLM 方向简历，可以准备三个项目：

### 项目一：Mini GPT

证明你懂模型结构和训练：

- tokenizer；
- transformer block；
- causal attention；
- training loop；
- checkpoint；
- text generation。

### 项目二：LLM Fine-tuning

证明你懂微调和数据：

- instruction dataset；
- chat template；
- LoRA / QLoRA；
- eval set；
- before / after 对比。

### 项目三：LLM Serving + RAG

证明你懂工程落地：

- vLLM serving；
- streaming API；
- hybrid RAG；
- rerank；
- eval；
- trace；
- latency benchmark。

这三个项目合起来，基本覆盖“模型理解 + 训练微调 + 工程落地”。

---

## 13. 最后总结

LLM 路线可以总结成三句话：

1. **从 Transformer 入手**，弄懂模型如何从 token 变成 next-token distribution。
2. **从训练和微调深入**，理解数据、loss、SFT、LoRA、对齐如何改变模型能力。
3. **从推理和应用落地**，掌握 KV Cache、batching、RAG、Agent、评测和观测。

真正有竞争力的 LLM 工程能力，不是“知道很多名词”，而是能把下面这条链路讲清楚、做出来、测明白：

```text
数据 -> 模型 -> 训练 -> 对齐 -> 推理 -> 检索 -> 工具 -> 评测 -> 观测 -> 迭代
```

这也是从 LLM 学习者走向 LLM 工程师的关键路径。
