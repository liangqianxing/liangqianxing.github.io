---
title: 讲透 mini-llm-engine：从显存碎片到六大推理优化
date: 2026-06-30 14:00:00
tags:
  - LLM
  - AI Infra
  - vLLM
  - PagedAttention
  - KV Cache
  - 推理优化
  - 投机解码
  - 实习求职
categories:
  - AI
---

上一篇写了 [mini-llm-engine 的实现过程](/posts/mini-llm-engine-from-scratch)，有人问我能不能把原理再讲清楚一点。这篇就专门讲"为什么"——为什么要这么设计，每个优化到底解决了什么问题。

项目地址：[github.com/liangqianxing/mini-llm-engine](https://github.com/liangqianxing/mini-llm-engine)

---

## 一、背景：LLM 推理的两个核心瓶颈

跑一个 LLM（比如 GPT-2、LLaMA）时，显存里有一个叫 **KV Cache** 的东西。

Transformer 每一层都有 Attention，计算时需要存当前 token 之前所有 token 的 Key 和 Value，这就是 KV Cache。它让 decode 每步只算当前 token，而不是重算整个序列。

### 瓶颈一：显存碎片（利用率只有 17%）

传统框架一开始就给每个请求预留 `max_seq_len` 个 token 的空间：

```
请求进来，分配 128 token 的显存槽：
[prompt:20tok | output:15tok | ......空的...... ]
                                ↑ 93 个 token 的空间浪费了
```

实际生成 35 个 token，却占了 128 个 token 的显存，利用率只有 27%。不同长度的请求导致显存碎片严重，多余内存无法被其他请求复用。

### 瓶颈二：GPU 空转（静态批处理）

```
同时跑 A、B、C、D 四个请求，等最长的 A 跑完才能开始下一批：

Step 1~10：[A][B][C][D]  都在跑
Step 11   ：[A][B][ ][ ]  C、D 完了，两个槽空着等
Step 30   ：[A][ ][ ][ ]  只剩 A，75% 算力浪费
Step 40   ：A 完成，才能开始下一批
```

短序列完成后，GPU 槽位空着等最长的序列，算力白白浪费。

**这两个问题，就是整个项目要解决的。**

---

## 二、项目架构一览

七个核心模块，从下往上：

```
┌──────────────────────────────────────────────────┐
│                   LLMEngine                       │  ← 对外 API
├──────────────────────────────────────────────────┤
│   Scheduler       │   KVCacheManager             │  ← 调度 + 内存
├──────────────────────────────────────────────────┤
│   BlockAllocator  │   PrefixCache │ SwapManager  │  ← 底层机制
├──────────────────────────────────────────────────┤
│   Sequence / Block                               │  ← 基础数据结构
└──────────────────────────────────────────────────┘
```

另有：`ModelRunner`（推理后端）、`MetricsCollector`（性能统计）、`SpeculativeDecoder`（投机解码）。

---

## 三、六个优化，逐一拆解

### 3.1 Paged KV Cache —— 解决显存碎片

**类比 OS 虚拟内存。** 操作系统不会给每个进程分一块连续的物理内存，而是分成固定大小的"页"按需分配。这里完全照搬：

- `PhysicalBlock`：一个物理块，存 16 个 token 的 KV Cache，相当于一页
- `BlockAllocator`：空闲链表，O(1) 分配和回收
- `block_table`：每个序列的"页表"，记录逻辑块 → 物理块的映射

```python
@dataclass
class PhysicalBlock:
    block_id: int        # GPU 显存中的索引
    ref_count: int = 0   # 引用计数，>1 时触发 CoW
    content_hash: int = None  # 内容哈希，用于 Prefix Cache

class BlockAllocator:
    def allocate(self) -> PhysicalBlock:
        return self._free_blocks.popleft()   # O(1)
    
    def free(self, block: PhysicalBlock) -> None:
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free_blocks.append(block)  # O(1)
```

序列产生第 17 个 token 时，第一个 block 满了，立刻申请新块：

```python
if seq.needs_new_block():
    new_block = allocator.allocate()
    seq.block_table[new_idx] = new_block
```

序列结束时，所有物理块立刻回到空闲链表。**内存利用率从 17% 提升到 45–65%**。

---

### 3.2 Continuous Batching —— 解决 GPU 空转

核心思想：**每一步都重新调度，而不是等一批跑完再换。**

调度器每步做三件事：

```
Step N 结束后：
1. 检查 running 里有没有完成的序列 → 释放它的 block
2. 检查 waiting 队列里有没有新请求 → 立刻拉进来 prefill
3. 剩余的继续 decode
```

```
Step N:   [A:decode] [B:decode] [C:decode]
Step N+1: C 完成 → 立刻拉入 D
          [A:decode] [B:decode] [D:prefill]
Step N+2: [A:decode] [B:decode] [D:decode]
```

核心代码：

```python
def schedule(self):
    # 先给运行中的序列分配 decode slot
    for seq in self.running:
        self.kv_cache.append_slot(seq)
        output.decode_seqs.append(seq)
    
    # 再从等待队列拉入新请求
    while self.waiting:
        if not self.kv_cache.can_allocate(next_seq):
            break  # 内存不够就停
        self.kv_cache.allocate(next_seq)
        output.prefill_chunks.append(next_seq)
```

**实测：吞吐量 2.4 倍提升**（50 请求，2ms/step 模拟延迟）。

---

### 3.3 Chunked Prefill —— 长 prompt 不阻塞短请求

**问题**：一个 512 token 的长 prompt，prefill 要跑整整一步，这期间所有 decode 请求都被阻塞。

**解法**（来自 [Sarathi-Serve, OSDI 2024](https://arxiv.org/abs/2308.16369)）：把长 prompt 切成 chunk，和 decode 穿插着跑：

```
没有 Chunked Prefill：
│──── prefill(512 tok) ────│decode│decode│decode│...

有 Chunked Prefill（chunk_size=64）：
│chunk(64)│decode│chunk(64)│decode│...│
```

实现：给序列加了状态 `PREFILLING` 和进度字段 `num_prefilled_tokens`：

```python
class SequenceStatus(Enum):
    WAITING    = auto()
    PREFILLING = auto()  # 长 prompt 分批处理中
    RUNNING    = auto()
    SWAPPED    = auto()  # KV cache 已卸载到 CPU
    FINISHED   = auto()
```

调度器每步只给这个序列分配 `chunk_size` 个 token，下步继续接着来：

```python
start, end = seq.get_next_prefill_range(chunk_size=64)
# 本步处理 token[start:end]，下步继续 token[end:end+64]
```

效果：短请求的 TTFT（首 token 延迟）不再受长 prompt 拖累。

---

### 3.4 Prefix Caching —— 共享 system prompt 的 KV Cache

**场景**：你的 API 服务里所有请求都有同一个 system prompt（比如"你是一个有帮助的助手..."）。每次都重新计算这 256 个 token 的 KV Cache 是纯粹的浪费。

**解法**：把满块的内容哈希存到全局缓存，第二个请求直接引用同一个物理块：

```python
class PrefixCache:
    _cache: Dict[int, PhysicalBlock]  # hash(token_ids) → PhysicalBlock
    
    def lookup(self, content_hash):
        block = self._cache.get(content_hash)
        if block:
            block.ref_count += 1  # 共享了，引用计数 +1
        return block
```

共享了怎么保证安全写入？**Copy-on-Write**：写入前检查 `ref_count > 1`，是的话先 fork 出一个新块再写：

```python
def cow_if_needed(self, block):
    if block.ref_count <= 1:
        return block, False   # 独占，直接写
    new_block = allocator.allocate()   # fork 一份
    block.ref_count -= 1               # 旧块减引用
    return new_block, True
```

这和 Linux 的 fork-on-write 是完全一样的思路。

**实测**（30 个请求，48 token 共享前缀 + 16 token 独有后缀）：
- 缓存命中率：71.3%（最后一个不满的块必然 miss，这是设计上的取舍）
- 大量省去重复 prefill 计算，显存节省 ~75%

> **为什么只缓存"满块"？** 不满的最后一块内容还会变化（decode 阶段还会往里追加 token），哈希不稳定，缓存了也没意义。只有满块内容固定，哈希才可靠。

---

### 3.5 CPU Swap —— 显存不够时的降级策略

内存不够必须抢占某个请求时，有两种选择：

- **Recompute**：丢掉它的 KV Cache，等下次轮到它重新 prefill（简单，但长序列代价大）
- **Swap**：把 KV Cache 搬到 CPU RAM，等内存够了再搬回来（复杂，但避免重算）

```python
class SwapManager:
    def swap_out(self, seq):
        # 1. 记录 logical_block → cpu_slot 的映射
        # 2. 释放 GPU 物理块（显存立刻还给其他请求）
        seq.status = SequenceStatus.SWAPPED
    
    def swap_in(self, seq):
        # 1. 重新分配 GPU 物理块
        # 2. 恢复 block_table 映射
        seq.status = SequenceStatus.RUNNING
```

**权衡**：PCIe 带宽（GPU↔CPU ~16 GB/s）远低于 GPU 显存带宽（~2 TB/s），swap 延迟约是 GPU 内部操作的 100 倍。但对长序列（512+ tokens）来说，swap 延迟 << 重新 prefill 延迟，值得做。

这与 vLLM 的 `preemption_mode="swap"` 完全对应。

---

### 3.6 Speculative Decoding —— 让 GPU 每步多生几个 token

标准 decode 每步只生 1 个 token，GPU 大部分时间在等显存读写（memory-bound，计算利用率低）。

**投机解码**（[Leviathan et al., ICML 2023](https://arxiv.org/abs/2211.17192)）：

```
Step 1 (Draft)  ：小模型快速生成 K=4 个候选 token [t₁, t₂, t₃, t₄]
Step 2 (Verify) ：大模型一次并行验证所有 4 个（等价于 1 次 prefill，并行快）
Step 3 (Accept) ：t₁✓ t₂✓ t₃✗ → 接受 [t₁, t₂]，用大模型的纠正 token
```

接受率 α 下，期望每步接受 token 数：

```
E[accepted] = (1 - α^(K+1)) / (1 - α)
```

当 α=0.7、K=4 时，期望 **3.3 个 token/步**，而不是 1 个，理论加速 **3.3×**。

```python
def step(self, seqs):
    # Phase 1: Draft（K 步，小模型快）
    draft_tokens = {}
    for _ in range(self.K):
        new_draft = self.draft_runner.step(decode_seqs=seqs)
        for seq in seqs:
            draft_tokens[seq.seq_id].append(new_draft[seq.seq_id])
    
    # Phase 2: Verify（1 步，大模型并行）
    verify_tokens = self.target_runner.step(decode_seqs=seqs)
    
    # Phase 3: Accept/Reject（从左到右截断）
    for seq in seqs:
        accepted = []
        for draft_tok in draft_tokens[seq.seq_id]:
            if random() < acceptance_rate:
                accepted.append(draft_tok)        # draft 被接受
            else:
                accepted.append(verify_tokens[seq.seq_id])  # 用 target 纠正
                break
        else:
            accepted.append(verify_tokens[seq.seq_id])      # bonus token
```

实践中，draft model 选同家族的小模型（如 Llama-3.2-1B 配 Llama-3-70B），接受率通常在 0.6–0.85 之间。

---

## 四、三个关键设计决策

**为什么用 deque 做空闲链表？**

`popleft()` 和 `append()` 都是 O(1)，比数组（O(n) 查找空闲）或 bitmap（需要扫描）更快。实际上 vLLM 的 block allocator 也是类似的空闲列表。

**ModelRunner 为什么抽象成接口？**

`MockModelRunner`（纯 Python，无 GPU）和 `GPT2ModelRunner`（真实推理）实现同一个 `step()` 接口，调度器完全不知道背后用的什么模型。这样可以单独测试调度逻辑，不用 GPU 也能跑 benchmark，也方便 CI。

**Prefix Cache 为什么只缓存满块？**

不满的最后一块在 decode 阶段还会往里追加 token，内容还没固定，哈希不稳定。只有满块内容不再变化，缓存才有意义。这是一个设计取舍：牺牲最后一个 block 的命中率，换取实现的简洁性。SGLang 的 RadixAttention 用 Radix Tree 能做到更精细的最长前缀匹配，但实现复杂很多。

---

## 五、整体数据流（一张图）

```
用户提交 "Hello world"
  │
  ▼
LLMEngine.add_request()
  │ 创建 Sequence（WAITING），加入 scheduler.waiting
  │
  ▼ ── 每步循环 ──────────────────────────────────────
Scheduler.schedule()
  ├── 为 RUNNING 序列 append_slot（可能触发 CoW）
  ├── 为 PREFILLING 序列推进 chunk 进度
  ├── 尝试 swap_in SWAPPED 序列
  └── 从 waiting 拉入新请求（分配物理块）
  │
  ▼ SchedulerOutput
  │
ModelRunner.step(prefill_seqs, decode_seqs)
  │ 返回 Dict[seq_id → new_token_id]
  │
  ▼
Scheduler.on_step_done()
  │ 追加 token，检查 EOS / max_tokens
  │
  ▼
MetricsCollector.record_step()
  │ 记录 KV util、队列深度、TTFT
  │ ──────────────────────────────────────────────────
  │
  ▼ 序列完成时
RequestOutput(latency, ttft, output_token_ids)
```

---

## 六、这个项目能聊什么

做完这个项目，下面这些问题我能讲清楚：

**Q：vLLM 为什么比 HuggingFace 快 20 倍？**  
A：两个核心：Continuous Batching 消除 GPU 空转，PagedAttention 消除显存碎片。GPU 利用率从 30% 提升到 90%+，显存释放后能跑更多并发请求，两者相乘放大了差距。

**Q：Prefix Cache 的 CoW 怎么实现的？**  
A：满块计算 `hash(tuple(token_ids))` 作为 key 存入全局字典。命中时 `ref_count+1` 直接共享物理块，不用重新分配。写入前检查 `ref_count > 1`，如果是共享块就先 fork 一个新块再写，旧块 `ref_count-1`。完全照搬 Linux fork 的思路。

**Q：显存不够时怎么处理？**  
A：三个策略按优先级：① 先看能不能 swap out 到 CPU RAM（避免重算，有带宽代价）；② 不行就 recompute（丢 KV Cache，下次重新 prefill，长序列代价高）；③ 实在不行就让请求继续排队。vLLM 通过 `preemption_mode` 参数让用户选择。

**Q：Chunked Prefill 和 Continuous Batching 什么关系？**  
A：CB 解决"谁来跑"（批次动态变化，完成了立刻换人）；Chunked Prefill 解决"怎么跑"（长 prefill 拆碎，不独占一整步）。二者正交，可以同时开启，也是 vLLM 的默认配置。

**Q：Speculative Decoding 什么时候不适用？**  
A：draft 和 target 分布差异太大时（接受率 α < 0.5），每步 K 次 draft 的时间开销超过多拿 token 的收益，不如直接跑 target。实践中要选同家族模型做 draft，比如 Llama-3.2-1B 配 Llama-3-70B，接受率能到 0.7+ 才合算。

---

代码在这里，可以直接跑：

```bash
git clone https://github.com/liangqianxing/mini-llm-engine
cd mini-llm-engine
pip install pytest
pytest tests/ -v                               # 67 tests, all pass
python examples/basic_usage.py                 # 完整 demo
python run_all_benchmarks.py --fast --no-plot  # 5 个 benchmark
```
