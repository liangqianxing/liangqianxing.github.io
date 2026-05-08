---
title: 基模训练八股：从数据、架构到预训练、对齐与工程面试
date: 2026-05-08 00:00:00
tags:
  - LLM
  - 基础模型
  - 预训练
  - AI Infra
  - 面试
categories:
  - AI
mathjax: true
---

这篇文章整理一份“基模训练八股”，目标不是写成论文综述，而是面向面试、工程实践和系统复盘：如果有人问“一个大语言模型从零训练出来需要哪些环节”，你应该能从数据、Tokenizer、模型架构、预训练目标、并行训练、稳定性、后训练、评测、部署和成本几个层面完整回答。

所谓基模（Foundation Model），一般指在大规模通用数据上训练出来、具备较强迁移能力的基础模型。它可以是语言模型、多模态模型、视觉模型、语音模型，也可以是统一建模文本、图像、音频、视频和动作的模型。本文主要以大语言模型（LLM）为主线，但许多原则同样适用于多模态基模。

<!-- more -->

## 1. 基模训练整体流程

训练一个基模通常可以拆成几层：

1. 数据层：收集、清洗、去重、过滤、配比、采样。
2. 表示层：设计 Tokenizer、词表、特殊 token、序列格式。
3. 模型层：选择 Transformer 架构、参数规模、上下文长度、位置编码、归一化方式。
4. 预训练层：定义 next token prediction 等目标，在大规模语料上训练。
5. 系统层：分布式并行、混合精度、通信优化、checkpoint、容错。
6. 稳定性层：loss spike、梯度爆炸、数值溢出、数据异常、学习率策略。
7. 后训练层：SFT、偏好优化、RLHF、DPO、工具调用、多轮对话。
8. 评测层：通用能力、推理能力、代码能力、安全性、长上下文、幻觉。
9. 部署层：推理加速、KV Cache、量化、并发调度、成本优化。
10. 迭代层：根据评测和线上反馈反推数据、训练和对齐策略。

面试时可以先给一个总览：

> 基模训练不是“把数据扔进 Transformer”这么简单，而是数据工程、模型设计、训练系统、稳定性治理、后训练对齐和评测闭环共同构成的大工程。

## 2. 什么是 Foundation Model

Foundation Model 的核心特征有三个。

第一，预训练数据规模大。它不是只在某个任务数据集上训练，而是在网页、书籍、论文、代码、问答、多语言、多模态内容等大规模数据上训练。

第二，任务泛化能力强。训练时可能只是预测下一个 token，但模型会涌现出翻译、摘要、代码、推理、问答、规划等能力。

第三，可以通过后训练适配。预训练模型通常不是最终产品，还需要 SFT、偏好对齐、工具增强、安全对齐和领域微调。

一个简单区分是：

- 预训练模型：学世界知识、语言规律、代码模式和通用表示。
- 指令模型：学会按人的指令回答。
- 对话模型：学会多轮上下文、角色一致性和安全边界。
- Agent 模型：学会工具调用、规划、执行和反思。

## 3. 数据是基模训练的第一性问题

在基模训练里，数据质量通常比模型结构小改动更重要。很多能力差异并不来自“模型层数多一层”，而来自数据覆盖、清洗质量、去重策略、课程配比和后训练数据。

### 3.1 数据来源

常见预训练数据包括：

- Web 文本：Common Crawl、新闻、论坛、博客、百科、问答社区。
- 书籍与论文：长文本、高质量知识密度、结构化表达。
- 代码数据：GitHub、开源仓库、文档、issue、代码解释。
- 数学与推理：教材、题库、证明、竞赛题、解题过程。
- 多语言数据：中文、英文、小语种、混合语料。
- 对话数据：问答、客服、多轮聊天、合成指令。
- 领域数据：金融、医疗、法律、教育、科研等专业语料。

多模态模型还会加入：

- 图文对数据。
- 视频字幕与帧序列。
- OCR 数据。
- 音频转写数据。
- 机器人轨迹和动作数据。

### 3.2 数据清洗

数据清洗的目标不是让文本“看起来干净”，而是减少模型学习到低质量模式。

常见清洗步骤：

- HTML 解析：去除导航栏、广告、脚本、样式、页脚。
- 语言识别：过滤语言不明或混杂严重的文本。
- 编码修复：处理乱码、非法字符、重复 Unicode。
- 质量打分：用规则或模型判断文本是否有价值。
- 长度过滤：去掉过短、过长、重复模板文本。
- 敏感过滤：处理隐私、违法、有害内容。
- 格式归一：统一空格、换行、标点和特殊符号。
- 代码清洗：过滤生成文件、压缩文件、依赖目录、重复 vendor 代码。

面试回答时可以强调：清洗策略不能只靠规则，通常会结合启发式规则、质量分类器、困惑度过滤、embedding 聚类和人工抽检。

### 3.3 数据去重

去重非常关键，因为重复数据会导致：

- 训练效率降低。
- 模型记忆增强。
- 评测污染风险变高。
- 生成内容更容易复读。
- 某些数据源权重被意外放大。

常见去重粒度：

- 文档级去重：完全相同或高度相似文档。
- 段落级去重：重复段落、模板文本。
- 句子级去重：常见垃圾句、免责声明。
- n-gram 去重：检测局部重复。
- MinHash / SimHash：近似去重。
- embedding 去重：语义层面的近似重复。

一个经典八股点：训练集和评测集之间也要去重，尤其是 benchmark contamination。否则模型可能不是“会做题”，而是“见过题”。

### 3.4 数据配比

数据配比决定模型能力画像。

如果代码占比高，模型代码能力会增强，但自然语言可能受影响。如果数学推理数据太少，模型会缺乏 chain-of-thought 风格的推理模式。如果中文数据比例低，中文表达和知识覆盖会弱。

常见配比考虑：

- 通用文本 vs 专业文本。
- 中文 vs 英文 vs 多语言。
- 代码 vs 自然语言。
- 高质量小数据 vs 低质量大数据。
- 长文本 vs 短文本。
- 新数据 vs 旧数据。
- 人类数据 vs 合成数据。

配比不是一次性决定的，通常需要根据中间 checkpoint 的评测结果迭代调整。

## 4. Tokenizer 八股

Tokenizer 决定文本如何变成 token 序列。对 LLM 来说，它不是边角料，而是影响训练效率、多语言能力、代码能力和上下文利用率的关键组件。

### 4.1 常见 Tokenizer

常见方案包括：

- BPE：Byte Pair Encoding，GPT 系列常见。
- WordPiece：BERT 系列常见。
- SentencePiece：常用于多语言模型，可不依赖空格。
- Unigram LM：SentencePiece 支持的一类概率 tokenizer。
- Byte-level BPE：对任意字节鲁棒，避免 OOV。

现代 LLM 通常倾向使用 byte-level 或 byte fallback，保证任何字符都能编码。

### 4.2 词表大小怎么选

词表太小：

- 序列变长。
- 训练和推理成本增加。
- 长上下文中有效信息减少。

词表太大：

- embedding 和 LM head 参数增加。
- 低频 token 学不好。
- 多语言下词表浪费。

常见 LLM 词表规模从几万到十几万不等。中文模型通常需要考虑中文字符、词组、英文、代码符号和多语言字符的平衡。

### 4.3 Tokenizer 对中文的影响

中文没有天然空格，如果 tokenizer 对中文不友好，会导致一个词被切成很多 token，直接影响：

- 中文上下文长度。
- 中文推理效率。
- 中文生成流畅度。
- 训练成本。

所以中文数据足够多时，通常希望 tokenizer 学到常见中文词、短语和标点组合。

### 4.4 特殊 token

常见特殊 token：

- BOS：序列开始。
- EOS：序列结束。
- PAD：padding。
- UNK：未知 token，现代 byte-level 模型可弱化。
- MASK：BERT 类模型使用。
- system / user / assistant：对话格式。
- tool call / tool result：工具调用格式。
- image / audio / video placeholder：多模态占位。

特殊 token 的设计会影响后训练和推理格式，一旦模型大规模训练后再改 tokenizer 成本很高。

## 5. 模型架构八股

现代 LLM 主体通常是 decoder-only Transformer。

### 5.1 为什么 GPT 类模型用 decoder-only

BERT 是 encoder-only，适合理解任务；T5 是 encoder-decoder，适合 seq2seq；GPT 是 decoder-only，适合自回归生成。

基模尤其是对话模型常用 decoder-only，原因包括：

- next token prediction 目标简单。
- 预训练和生成过程一致。
- 扩展到大规模更直接。
- KV Cache 推理友好。
- 适合统一建模文本、代码和对话。

### 5.2 Transformer Block 组成

一个典型 decoder block 包括：

1. RMSNorm 或 LayerNorm。
2. Self-Attention。
3. 残差连接。
4. RMSNorm 或 LayerNorm。
5. MLP / FFN。
6. 残差连接。

常见结构是 Pre-Norm：

$$
x = x + \text{Attention}(\text{Norm}(x))
$$

$$
x = x + \text{MLP}(\text{Norm}(x))
$$

Pre-Norm 比 Post-Norm 更适合深层网络稳定训练。

### 5.3 Attention

标准自注意力：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

在 decoder-only 中需要 causal mask，保证当前位置只能看见过去 token，不能看未来。

### 5.4 MHA、MQA、GQA

- MHA：Multi-Head Attention，每个 head 有自己的 Q/K/V。
- MQA：Multi-Query Attention，多个 query head 共享一组 K/V。
- GQA：Grouped-Query Attention，多个 query head 分组共享 K/V。

为什么 MQA/GQA 重要？

推理时 KV Cache 占用很大。共享 K/V 可以显著减少 KV Cache 大小，提高长上下文和高并发推理效率。很多现代大模型采用 GQA，在质量和推理效率之间折中。

### 5.5 MLP 结构

传统 Transformer 用 FFN：

$$
\text{FFN}(x)=W_2 \sigma(W_1x)
$$

现代 LLM 常用 SwiGLU / GEGLU：

$$
\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3)W_2
$$

SwiGLU 通常比普通 GELU FFN 效果更好。

### 5.6 位置编码

常见位置编码：

- 绝对位置编码：早期 Transformer 使用。
- 相对位置编码：建模相对距离。
- RoPE：Rotary Position Embedding，现代 LLM 常用。
- ALiBi：用 attention bias 表示距离衰减。

RoPE 的优势是相对位置性质好，外推和实现都比较友好。长上下文扩展时常见技巧包括 RoPE scaling、NTK scaling、YaRN、位置插值等。

### 5.7 Norm 选择

LayerNorm 对均值和方差归一化，RMSNorm 只使用均方根，计算更简单。很多现代 LLM 使用 RMSNorm：

$$
\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g
$$

它通常训练稳定且效率较高。

## 6. 参数规模与 Scaling Law

Scaling Law 的核心观点是：在一定范围内，模型效果随参数量、数据量和计算量按幂律改善。

常见三要素：

- 参数量 N。
- 数据 token 数 D。
- 训练计算量 C。

大致关系是：

$$
C \approx 6ND
$$

这里的 6 是 Transformer dense LM 训练中一次 forward + backward 的粗略 FLOPs 系数。

### 6.1 Chinchilla 结论

Chinchilla 提出：很多早期大模型是“参数太多、数据太少”，在固定计算预算下，应该用更小模型训练更多 token。

经验上，计算最优训练会让参数量和 token 数一起增长，而不是盲目堆参数。

面试中可以这样回答：

> 如果算力固定，不能只问模型做多大，还要问训练多少 token。欠训练的大模型可能不如充分训练的小模型。

### 6.2 参数量怎么分配

模型超参包括：

- 层数 depth。
- hidden size。
- attention heads。
- intermediate size。
- vocab size。
- context length。

常见趋势：

- 更深模型表达层级更强，但训练稳定性和通信成本更高。
- 更宽模型并行友好，但参数和激活成本大。
- MLP 参数通常占大头。
- vocab embedding 在超大模型中占比不一定最大，但大词表会明显增加小模型成本。

## 7. 预训练目标

LLM 最常见目标是自回归 next token prediction。

给定序列：

$$
x_1, x_2, \ldots, x_T
$$

模型最大化：

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t \mid x_{<t})
$$

训练时最小化交叉熵：

$$
\mathcal{L}=-\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})
$$

这就是为什么 LLM 看起来只是“预测下一个词”，却能学到大量能力：因为要预测下一个 token，模型必须压缩语言、知识、逻辑、代码、世界状态和上下文模式。

### 7.1 Causal LM vs Masked LM

Causal LM：

- 从左到右预测。
- 适合生成。
- GPT 类模型常用。

Masked LM：

- 随机 mask token，让模型恢复。
- 双向上下文。
- BERT 类模型常用。
- 更适合理解和表示学习。

### 7.2 Prefix LM 和 Span Corruption

有些模型使用 Prefix LM 或 span corruption，以增强条件生成或文本恢复能力。但主流对话 LLM 仍以 causal LM 为主。

## 8. 训练数据组织

训练不是简单按文档逐个喂给模型。

### 8.1 Packing

为了提高训练效率，会把多个短文档拼接到一个固定长度序列中，这叫 packing。

注意点：

- 文档之间要加 EOS。
- loss mask 要正确。
- 对话数据要 mask 用户输入还是全量训练，需要按目标决定。
- packing 不当可能导致模型跨文档错误建模。

### 8.2 Sequence Length Curriculum

长上下文训练成本高，因为 attention 是 $O(n^2)$。常见做法：

- 先用较短上下文训练主体能力。
- 后期扩展到长上下文。
- 用高质量长文档做 continued pretraining。
- 结合位置编码外推策略。

### 8.3 数据采样

不是所有数据按原始比例采样。常见采样策略：

- temperature sampling：提高低资源语言或小数据源采样概率。
- domain upsampling：提高代码、数学、高质量数据比例。
- curriculum learning：从简单到复杂。
- dynamic sampling：根据阶段调整数据配比。

## 9. 优化器与学习率

### 9.1 AdamW

大模型训练常用 AdamW：

- Adam 使用一阶矩和二阶矩自适应更新。
- AdamW 将 weight decay 与梯度更新解耦。

常见参数：

- $\beta_1=0.9$
- $\beta_2=0.95$ 或 $0.999$
- weight decay 约 $0.1$
- gradient clipping 约 $1.0$

具体值取决于模型规模和训练 recipe。

### 9.2 学习率策略

常见 schedule：

1. warmup：开始阶段逐渐增大学习率，避免训练不稳定。
2. cosine decay：后期平滑降低学习率。
3. constant + decay：中间保持一段平台。
4. cooldown：训练末期降低学习率做收敛。

学习率过大容易 loss spike 或发散；过小则训练效率低。

### 9.3 Batch Size

大模型通常使用很大的 global batch size，以提高吞吐和训练稳定性。

但 batch size 不是越大越好：

- 太小：梯度噪声大，吞吐低。
- 太大：泛化可能变差，学习率需要调整。
- 受限于显存和并行策略。

通常会通过 gradient accumulation 扩大等效 batch。

## 10. 混合精度与数值稳定

### 10.1 FP16、BF16、FP32

- FP32：精度高，成本大。
- FP16：速度快、省显存，但动态范围小，容易 overflow。
- BF16：动态范围接近 FP32，精度低于 FP16，但训练更稳。

现代大模型训练常用 BF16。某些关键状态如 optimizer state、master weights、归约累加可能仍使用 FP32。

### 10.2 Loss Scaling

FP16 训练中常用 loss scaling 防止梯度 underflow。BF16 因动态范围更大，对 loss scaling 依赖较少。

### 10.3 常见数值问题

- softmax overflow。
- attention score 过大。
- 梯度爆炸。
- NaN / Inf。
- LayerNorm / RMSNorm epsilon 不合适。
- 数据中异常超长 token 或脏样本。

工程上要有 NaN 检测、梯度范数监控、参数范数监控、loss spike 回滚和 checkpoint 恢复机制。

## 11. 分布式训练并行策略

单卡无法训练大模型，需要多维并行。

### 11.1 Data Parallel

Data Parallel 每张卡放一份完整模型，不同卡处理不同 batch，最后同步梯度。

优点：简单。
缺点：模型太大时单卡放不下。

### 11.2 Tensor Parallel

Tensor Parallel 把单层矩阵乘法切到多张卡上。例如把线性层的 weight 按列或按行切分。

优点：解决单层太大的问题。
缺点：层内需要频繁通信，对带宽敏感。

### 11.3 Pipeline Parallel

Pipeline Parallel 把模型不同层切到不同设备上，micro-batch 流水执行。

优点：解决层数太多放不下的问题。
缺点：pipeline bubble、调度复杂、激活传输开销。

### 11.4 ZeRO / FSDP

ZeRO 把优化器状态、梯度、参数分片到不同 GPU 上。

- ZeRO-1：切 optimizer state。
- ZeRO-2：切 optimizer state + gradients。
- ZeRO-3：切 optimizer state + gradients + parameters。

PyTorch FSDP 类似 ZeRO-3 思路，按需 all-gather 参数，用完释放。

### 11.5 Sequence Parallel 和 Context Parallel

长上下文训练时，序列维度本身也可能太长，需要把 sequence 维度切分。

- Sequence Parallel：常用于配合 Tensor Parallel，降低激活内存。
- Context Parallel：把长上下文切到多卡，解决超长序列 attention 的显存和计算问题。

### 11.6 Expert Parallel

MoE 模型中，不同 expert 分布在不同设备上，token 通过 router 分发到 expert。

优点：在相同激活参数下增加总参数量。
缺点：通信复杂、负载均衡难、训练稳定性更难。

## 12. 内存构成与优化

训练显存主要由几部分组成：

- 参数。
- 梯度。
- 优化器状态。
- 激活值。
- 临时 buffer。
- 通信 buffer。

Adam 优化器通常很吃显存，因为需要一阶矩和二阶矩。如果用 FP32 optimizer state，显存会进一步增加。

常见优化：

- activation checkpointing：重算激活换显存。
- ZeRO / FSDP：切分状态。
- mixed precision：降低参数和激活精度。
- FlashAttention：降低 attention 显存和提高速度。
- gradient accumulation：小 micro-batch 累积成大 batch。
- offload：把部分状态放 CPU/NVMe，但速度会受影响。

## 13. FlashAttention 八股

标准 attention 的瓶颈不只是 FLOPs，还有 HBM 读写。

FlashAttention 的核心思想：

- 不显式存完整 $n \times n$ attention matrix。
- 分块计算 softmax。
- 利用 online softmax 保持数值稳定。
- 减少 HBM 访问，提高速度并降低显存。

它把 attention 从“显存 IO 密集”优化为更适合 GPU SRAM tile 的计算方式。长序列训练和推理中非常重要。

## 14. Checkpoint 与容错

大规模训练动辄几周几个月，硬件故障是常态。

Checkpoint 需要保存：

- model weights。
- optimizer state。
- LR scheduler state。
- random seed / RNG state。
- dataloader state。
- consumed tokens。
- parallel states。

如果只保存模型权重，恢复后训练轨迹可能不一致。

常见策略：

- 定期保存。
- 保存最近 N 个。
- 关键里程碑永久保存。
- 异步 checkpoint 降低阻塞。
- 分布式 checkpoint 避免单点瓶颈。

## 15. 训练稳定性问题

### 15.1 Loss Spike

Loss spike 是大模型训练常见问题。可能原因：

- 学习率过大。
- 数据 batch 异常。
- 梯度爆炸。
- 混合精度 overflow。
- attention logits 异常。
- 分布式通信错误。
- optimizer state 损坏。

处理方式：

- 降低学习率。
- gradient clipping。
- 跳过异常 batch。
- 回滚 checkpoint。
- 检查数据样本。
- 检查 NaN/Inf。
- 调整初始化和 norm。

### 15.2 训练发散

发散通常体现为 loss 快速升高、梯度范数爆炸、参数出现 NaN。

排查顺序：

1. 数据是否异常。
2. loss mask 是否正确。
3. label shift 是否正确。
4. 学习率和 warmup 是否合理。
5. 混合精度是否 overflow。
6. attention mask 是否正确。
7. 分布式梯度同步是否正确。
8. checkpoint 恢复是否完整。

### 15.3 Loss 不降

可能原因：

- 数据和 label 错位。
- 模型没看到有效上下文。
- 学习率太小或太大。
- optimizer 参数错误。
- tokenizer 和数据不匹配。
- loss mask 全错。
- 梯度没有回传。

面试时如果被问“训练 loss 不降怎么办”，最好从数据、目标、模型、优化器、分布式五个层面排查。

## 16. MoE 基模

MoE（Mixture of Experts）通过稀疏激活增加模型总参数量。

一个 MoE 层通常包括：

- router / gate：决定 token 去哪些 expert。
- experts：多个 FFN 专家。
- top-k routing：每个 token 选择 k 个 expert。
- load balancing loss：避免所有 token 挤到少数 expert。

优点：

- 总参数量大。
- 每 token 激活参数较少。
- 训练和推理 FLOPs 相对可控。

缺点：

- all-to-all 通信复杂。
- expert 负载不均衡。
- router 容易不稳定。
- 小 batch 或低并发推理效率可能不好。
- 部署系统更复杂。

面试回答重点：MoE 不是免费午餐，它用通信复杂度和系统复杂度换取参数规模扩展。

## 17. 长上下文训练

长上下文能力不是把 RoPE scale 一下就完事。

核心挑战：

- attention 计算和显存随长度平方增长。
- 长文档数据质量难保证。
- 模型容易只利用局部上下文。
- 位置外推可能退化。
- 评测容易被“针在 haystack”误导。

常见方法：

- RoPE scaling / YaRN / NTK scaling。
- 长上下文 continued pretraining。
- 高质量长文档数据。
- FlashAttention / ring attention。
- context parallel。
- sliding window attention。
- sparse attention。
- retrieval augmentation。

长上下文评测不仅要测 needle retrieval，还要测多跳推理、长文档总结、跨段引用、指令保持和抗干扰能力。

## 18. 代码模型训练

代码能力强通常来自几个因素：

- 高质量代码语料。
- 多语言代码覆盖。
- README、文档、issue、commit message。
- 单元测试和执行反馈。
- SFT 中加入代码问答和修复任务。
- RL / rejection sampling 优化可执行正确性。

代码数据清洗要注意：

- 过滤 vendored dependencies。
- 过滤 minified 文件。
- 保留文件路径和项目结构信息。
- 处理 license。
- 去重 fork 仓库。
- 过滤二进制和生成文件。

代码模型评测不能只看 pass@1，还要看多文件修改、依赖理解、测试修复、长上下文 repo 理解和工具调用能力。

## 19. 数学与推理训练

数学能力通常不是单靠预训练自然涌现到很强，需要：

- 高质量数学语料。
- 分步解题数据。
- 合成推理数据。
- verifier / reward model。
- rejection sampling。
- self-consistency。
- process supervision。

但也要注意：过多低质量 chain-of-thought 会让模型学会“看起来很会推理，但实际乱编”。

推理训练的关键不只是最终答案，而是过程是否可靠、可检验、可泛化。

## 20. 后训练：从基模到助手

预训练模型只会续写，不一定会听指令。后训练让它变成可用助手。

### 20.1 SFT

SFT（Supervised Fine-Tuning）用指令-回答数据训练模型。

数据类型：

- 单轮问答。
- 多轮对话。
- 写作、总结、翻译。
- 代码生成和解释。
- 数学解题。
- 工具调用。
- 安全拒答。
- 角色扮演。

SFT 关键是质量，不是越多越好。少量高质量数据往往比大量模板化数据更有效。

### 20.2 RLHF

RLHF 通常包括：

1. 训练 SFT 模型。
2. 收集人类偏好比较数据。
3. 训练 reward model。
4. 用 PPO 等 RL 方法优化策略。

目标是让模型输出更符合人类偏好。

缺点：

- 工程复杂。
- reward hacking。
- 训练不稳定。
- 标注成本高。

### 20.3 DPO

DPO（Direct Preference Optimization）直接使用偏好对优化模型，不显式训练 reward model，也不需要 PPO 那样复杂的 RL loop。

直观理解：给定 chosen 和 rejected，提升 chosen 概率，降低 rejected 概率，同时约束模型不要偏离 reference model 太远。

DPO 工程上更简单，因此很常用。

### 20.4 RLAIF

RLAIF 使用 AI feedback 替代或辅助人类反馈。优点是规模大、成本低；缺点是可能继承 judge model 的偏见和盲点。

### 20.5 对齐税

后训练可能提高人类偏好和安全性，但也可能损伤某些能力，这叫 alignment tax。

例如：

- 回答变啰嗦。
- 拒答过度。
- 创造性下降。
- 代码能力下降。
- 数学推理变保守。

所以后训练需要持续评测基础能力，不能只看聊天体验。

## 21. 安全训练

安全训练包括：

- 拒绝违法有害请求。
- 隐私保护。
- 防止泄露系统提示词。
- 防止生成恶意代码。
- 降低偏见和歧视。
- 幻觉控制。
- 医疗、法律、金融等高风险领域免责声明。

常用手段：

- 安全 SFT 数据。
- 偏好数据。
- red teaming。
- policy model / moderation model。
- refusal style 优化。
- adversarial prompt 训练。

安全训练的难点是平衡 helpfulness 和 harmlessness。过度安全会导致模型什么都不答；过度开放则风险高。

## 22. 评测体系

基模评测要分层。

### 22.1 预训练中间评测

训练过程中需要定期评估：

- validation loss。
- 不同数据域 loss。
- benchmark 子集。
- 语言能力。
- 代码能力。
- 数学能力。
- 长上下文能力。

Validation loss 下降不代表所有能力都提升，尤其是后期需要看能力评测。

### 22.2 通用 Benchmark

常见方向：

- 知识问答。
- 阅读理解。
- 常识推理。
- 数学推理。
- 代码生成。
- 多语言能力。
- 指令跟随。
- 安全性。

评测要注意污染、prompt 格式、采样参数、是否使用 CoT、是否使用工具、是否 few-shot。

### 22.3 人类评测

自动评测无法完全覆盖用户体验。人类评测关注：

- 是否有帮助。
- 是否事实正确。
- 是否遵循指令。
- 是否表达清晰。
- 是否安全。
- 是否符合风格要求。

### 22.4 LLM-as-a-Judge

用强模型当裁判可以提高评测效率，但有偏差：

- 偏好长回答。
- 偏好格式好看的回答。
- 对事实错误不敏感。
- 对自身风格有偏好。
- 容易被 prompt hack。

所以最好结合人工抽检和客观评测。

## 23. Benchmark 污染

Benchmark contamination 是基模训练绕不开的问题。

污染来源：

- 训练数据包含 benchmark 原题。
- 网页中有题解。
- 合成数据引用 benchmark。
- 后训练数据混入评测集。
- 模型蒸馏数据来自已见过题的模型。

解决方法：

- 训练集与 benchmark 去重。
- 使用时间切分的新数据集。
- 构造私有评测集。
- 动态生成题目。
- 评测过程记录 prompt 和输出。
- 对可疑高分做人工分析。

## 24. 推理部署八股

基模训练后还要能高效服务。

### 24.1 Prefill 和 Decode

LLM 推理分两段：

- Prefill：处理 prompt，计算初始 KV Cache，可并行度高。
- Decode：逐 token 生成，每步依赖上一步，延迟敏感。

Prefill 更像大矩阵计算，decode 更容易受内存带宽和 KV Cache 影响。

### 24.2 KV Cache

自回归生成时，过去 token 的 K/V 可以缓存，避免每步重复计算。

KV Cache 大小与以下因素相关：

- batch size。
- sequence length。
- layers。
- KV heads。
- head dimension。
- precision。

GQA/MQA 能显著降低 KV Cache。

### 24.3 Continuous Batching

请求长度和生成长度不同，如果静态 batching，吞吐会很差。Continuous batching 会动态加入和移除请求，提高 GPU 利用率。

### 24.4 PagedAttention

PagedAttention 借鉴操作系统分页思想管理 KV Cache，减少碎片，提高大规模并发服务效率。

### 24.5 量化

常见量化：

- FP16 / BF16。
- INT8 weight-only。
- INT4 weight-only。
- GPTQ / AWQ。
- SmoothQuant。
- KV Cache quantization。

量化可以降低显存和带宽，但可能损伤质量，尤其是数学、代码和长上下文能力。

## 25. 成本估算

训练成本可以粗略由 token 数、参数量和硬件效率估算。

训练 FLOPs：

$$
\text{FLOPs} \approx 6ND
$$

如果模型参数 $N=7B$，训练 token 数 $D=1T$：

$$
6 \times 7 \times 10^9 \times 10^{12}=4.2\times10^{22}\ \text{FLOPs}
$$

实际训练时间取决于：

- GPU 数量。
- GPU 理论 FLOPs。
- MFU（Model FLOPs Utilization）。
- 通信效率。
- checkpoint 和数据加载开销。
- 故障重启。

MFU 是衡量训练系统效率的重要指标。高 MFU 说明硬件利用较好；低 MFU 可能是通信、数据加载、kernel、pipeline bubble 或小 batch 导致。

## 26. 数据合成与蒸馏

现代基模训练越来越依赖合成数据。

合成数据用途：

- 指令数据扩充。
- 数学解题过程。
- 代码单测生成。
- 多轮对话。
- 工具调用轨迹。
- 安全红队数据。
- 小语种增强。

蒸馏方式：

- response distillation：学习强模型回答。
- logit distillation：学习 soft targets。
- reasoning distillation：学习推理轨迹。
- preference distillation：学习偏好排序。

风险：

- 模型风格同质化。
- 错误被放大。
- 数据多样性下降。
- judge model 偏见传递。
- 合成痕迹过重。

所以合成数据需要过滤、打分、多样化采样和人工抽检。

## 27. 多模态基模训练

多模态模型通常要解决“不同模态如何对齐”的问题。

常见路线：

1. 图像编码器 + LLM：用视觉 encoder 抽特征，再投影到 LLM token space。
2. 统一 tokenization：把图像、音频、视频离散化成 token。
3. Diffusion / Flow + LLM：LLM 负责语义规划，生成模型负责图像或视频生成。
4. Any-to-any 模型：输入输出都支持多模态。

训练阶段通常包括：

- 视觉-语言对齐预训练。
- 图文指令微调。
- 多轮视觉问答。
- OCR 和文档理解。
- 视频时序理解。
- 多模态偏好对齐。

难点：

- 多模态数据质量。
- 图文错配。
- 视觉幻觉。
- 高分辨率成本。
- 视频序列太长。
- 多模态安全问题。

## 28. 常见面试题速答

### 28.1 为什么 LLM 用 next token prediction 能学会推理

因为预测下一个 token 不是简单统计词频。为了在大量文本、代码和推理数据中降低 loss，模型必须学习语法、语义、事实知识、上下文依赖、代码执行模式和推理路径。推理能力是大规模数据、模型容量和训练目标共同作用下的涌现结果。

### 28.2 为什么数据去重重要

去重可以提升训练效率、降低记忆风险、减少评测污染，并避免重复数据让模型过拟合某些模板或来源。

### 28.3 为什么训练用 BF16

BF16 动态范围接近 FP32，比 FP16 更不容易 overflow，同时显存和计算效率接近低精度训练，因此大模型训练更稳定。

### 28.4 为什么要 warmup

训练初期参数和 optimizer state 还不稳定，直接使用大学习率容易导致梯度爆炸或 loss spike。warmup 让学习率逐步升高，提升稳定性。

### 28.5 ZeRO-3 和 Tensor Parallel 区别

ZeRO-3/FSDP 是切参数、梯度和优化器状态，主要解决状态显存问题；Tensor Parallel 是切单层矩阵计算，主要解决单层模型太大和计算并行问题。实际训练中常组合使用。

### 28.6 为什么 GQA 能加速推理

GQA 减少 K/V head 数量，降低 KV Cache 大小和内存带宽压力，因此 decode 阶段更高效，同时质量通常比 MQA 更稳。

### 28.7 FlashAttention 为什么快

它不显式写出完整 attention matrix，而是分块在 SRAM 中计算，并用 online softmax 保持数值稳定，减少 HBM 读写，因此更快更省显存。

### 28.8 SFT 和 RLHF 区别

SFT 学习示范答案，让模型会按指令回答；RLHF 使用偏好信号进一步优化回答质量、安全性和人类偏好。SFT 是模仿，RLHF 是偏好优化。

### 28.9 DPO 为什么比 PPO 简单

DPO 直接用偏好对优化语言模型，不需要单独训练 reward model，也不需要复杂 RL rollout 和 PPO 更新，因此工程上更简单稳定。

### 28.10 为什么 validation loss 低不一定用户体验好

Validation loss 衡量平均 token 预测能力，但用户体验还包括指令遵循、事实性、推理可靠性、安全性、格式、风格和多轮一致性。这些需要专门评测和后训练。

## 29. 基模训练排障清单

如果训练出问题，可以按这个 checklist 排查。

数据侧：

- 是否有乱码、空样本、重复样本。
- tokenizer 是否和数据匹配。
- packing 是否跨文档污染。
- loss mask 是否正确。
- 数据配比是否异常。

模型侧：

- attention mask 是否正确。
- label shift 是否正确。
- position ids 是否正确。
- norm epsilon 是否合理。
- 初始化是否正确。

优化侧：

- 学习率是否过大。
- warmup 是否太短。
- gradient clipping 是否开启。
- AdamW 参数是否合理。
- weight decay 是否错误作用在 norm/bias 上。

系统侧：

- 梯度同步是否正确。
- mixed precision 是否 overflow。
- checkpoint 是否完整。
- 随机种子和 dataloader 状态是否恢复。
- 通信是否有 silent error。

评测侧：

- benchmark 是否污染。
- prompt 格式是否一致。
- decoding 参数是否一致。
- 是否区分 base model 和 chat model。

## 30. 一个面试中的完整回答模板

如果面试官问：“你讲一下大模型基模训练流程。”

可以这样回答：

> 我会从数据、模型、训练系统、稳定性和后训练五个层面讲。首先数据是基模训练的核心，需要收集多源语料，做清洗、质量过滤、去重、隐私和安全处理，并按能力目标设计数据配比。然后设计 tokenizer 和 decoder-only Transformer 架构，包括 RoPE、RMSNorm、SwiGLU、GQA 等常见组件。预训练目标通常是 causal LM 的 next token prediction，用 AdamW、warmup + cosine decay、BF16 混合精度训练。系统上会结合 data parallel、tensor parallel、pipeline parallel、ZeRO/FSDP、activation checkpointing 和 FlashAttention 来解决显存与吞吐问题。训练中要监控 loss、梯度范数、NaN、MFU、数据域 loss，并通过 checkpoint 容错。预训练完成后，还要做 SFT、DPO/RLHF、安全对齐和工具调用训练，最后通过通用 benchmark、人类评测、红队测试和线上反馈迭代。

这个回答的优点是结构完整，而且能自然展开到任何细节。

## 31. 总结

基模训练可以理解成一个由“数据规模、模型容量、计算系统、训练稳定性和后训练对齐”共同决定的复杂工程。

真正重要的不是背某一个公式，而是理解每个环节为什么存在：

- 数据决定模型学什么。
- Tokenizer 决定信息如何进入模型。
- 架构决定容量和效率。
- 预训练目标决定基础能力形成方式。
- 分布式系统决定能不能训得动。
- 稳定性治理决定能不能训完。
- 后训练决定能不能变成好用助手。
- 评测闭环决定能不能持续进化。

如果把基模训练比作造一座城市，模型架构只是建筑结构，数据是土地和人口，训练系统是电网和交通，后训练是公共服务，评测是城市治理。只有这些系统共同工作，才可能训练出真正可用的大模型。
