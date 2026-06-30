---
title: 从零实现 LLM 推理引擎：深挖 vLLM 的六大核心优化
date: 2026-06-30 10:00:00
tags:
  - LLM
  - AI Infra
  - vLLM
  - PagedAttention
  - 推理优化
  - 投机解码
  - 实习求职
categories:
  - AI
---

> 本文对应 GitHub 项目：[mini-llm-engine](https://github.com/liangqianxing/mini-llm-engine)  
> 代码量：~3500 行 Python · 67 个单元测试 · 全部通过 ✅

---

## 一、为什么要自己实现？

用 vLLM 很容易，一行 `from vllm import LLM` 就能跑起来。但面试时被问到"vLLM 为什么比 HuggingFace 快 20 倍"，能说清楚的人并不多。

我花了两周时间，从零实现了 vLLM 的核心调度和内存管理逻辑，把六个关键优化全部用 Python 写了一遍，并且每个都配了 benchmark。

这篇文章记录整个过程——不只是"是什么"，而是**"为什么这么设计"**。

---

## 二、传统推理有什么问题？

先理解痛点，才能理解优化的价值。

### 问题一：显存碎片（内存利用率只有 17%）

传统服务框架在请求开始时就预留 `max_seq_len` 个 token 的 KV cache：

```
请求 A（实际生成 20 tokens）：
[prompt KV | generated KV | pad | pad | pad | pad | ...]
 ← 20 tokens 实际使用 →  ← 108 tokens 浪费 →
```

如果 `max_seq_len=128`，实际生成 20 个 token，显存利用率只有 15%。不同长度的请求导致显存碎片严重，多余内存无法被其他请求复用。

**实测数据（200 个请求，max_tokens=128，eos_prob=5%）：**

| 策略 | 利用率 | 碎片率 | 最大并发 |
|------|--------|--------|---------|
| 静态预分配 | 17–28% | 72–83% | ~8 条 |
| Paged KV Cache | 45–65% | 35–55% | ~38 条 |

### 问题二：静态批处理的 GPU 空闲

```
Batch [A, B, C, D]（batch_size=4）：

Step 1  : A B C D  ← 全部 decode
Step 10 : A B _ _  ← C、D 已完成，但 A、B 还在跑，两个槽位空着浪费！
Step 30 : A _ _ _  ← 只剩 A，75% 的算力在等待
Step 40 : 完成，才能开始下一批
```

短序列完成后，GPU 槽位空着等最长的序列，算力白白浪费。

---

## 三、六大优化的实现原理

### 3.1 Paged KV Cache（虚拟内存分页）

**论文**：[PagedAttention, SOSP 2023](https://arxiv.org/abs/2309.06180)

核心思路类比 OS 的虚拟内存：

```
传统：  [连续大块预分配，max_seq_len 对齐]

Paged：[block₀][block₁][block₂]...  ← 每块 16 个 token
       ↑ 用完一块才分配下一块，结束立即回收
```

**关键数据结构：**

```python
@dataclass
class PhysicalBlock:
    block_id: int        # GPU 显存中的索引
    ref_count: int = 0   # 引用计数（>1 时触发 CoW）
    content_hash: int = None  # 块内容哈希（用于 Prefix Cache）

class BlockAllocator:
    # O(1) 分配/回收：deque 作为空闲链表
    _free_blocks: Deque[PhysicalBlock]
    
    def allocate(self) -> PhysicalBlock:
        return self._free_blocks.popleft()   # O(1)
    
    def free(self, block: PhysicalBlock) -> None:
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free_blocks.append(block)  # O(1)
```

每个序列维护一张"页表"（`block_table`），把逻辑块编号映射到物理块：

```python
class Sequence:
    block_table: Dict[int, PhysicalBlock]  # logical_idx → PhysicalBlock
    logical_blocks: List[LogicalTokenBlock]
    
    def needs_new_block(self) -> bool:
        return not self.logical_blocks or self.logical_blocks[-1].is_full
```

**生命周期**：请求开始时按需分配块 → decode 时若当前块满则申请新块 → 请求完成时批量回收所有块。

---

### 3.2 Continuous Batching（连续批处理）

**论文**：[Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)

每步结束后立即检查完成的序列，有空槽就从等待队列拉入新请求：

```
Step N:   [A:decode] [B:decode] [C:decode]
Step N+1: C 完成 → 立即拉入 D
          [A:decode] [B:decode] [D:prefill]
Step N+2: [A:decode] [B:decode] [D:decode]
```

**调度器核心逻辑（简化版）：**

```python
def schedule(self) -> SchedulerOutput:
    # 1. 为 running 序列分配 decode slot（内存不够就抢占）
    for seq_group in self.running:
        for seq in seq_group.get_seqs(RUNNING):
            if not self.kv_cache.can_append_slot(seq):
                self._preempt(seq_group, output)  # OOM → 抢占
                continue
            self.kv_cache.append_slot(seq)
            output.decode_seqs.append(seq)
    
    # 2. 从 waiting 拉入新请求（prefill）
    while self.waiting:
        sg = self.waiting[0]
        if not self.kv_cache.can_allocate(sg.seqs[0]):
            break  # 显存不够，停止
        self.kv_cache.allocate(sg.seqs[0])
        output.prefill_chunks.append(...)
        self.running.append(sg)
```

**实测吞吐量提升（50 个请求，max_tokens=64，2ms/step 模拟延迟）：**

```
Naive Batching    ：152 tok/s
Continuous Batch  ：360 tok/s   🚀 2.4x
```

---

### 3.3 Chunked Prefill（分块预填充）

**论文**：[Sarathi-Serve, OSDI 2024](https://arxiv.org/abs/2308.16369)

**问题**：512 token 的长 prompt prefill 占据整个步骤，decode 序列被阻塞：

```
Without：│──── prefill(512 tok) ────│decode│decode│...
   With：│chunk(32)│decode│chunk(32)│decode│...│
```

**实现**：序列新增 `PREFILLING` 状态和 `num_prefilled_tokens` 进度字段：

```python
class SequenceStatus(Enum):
    WAITING    = auto()
    PREFILLING = auto()  # ← 新增：长 prompt 分批处理中
    RUNNING    = auto()
    SWAPPED    = auto()  # ← 新增：KV cache 已卸载到 CPU
    FINISHED   = auto()

class Sequence:
    num_prefilled_tokens: int = 0   # 已处理的 prompt token 数
    
    def get_next_prefill_range(self, chunk_size: int):
        start = self.num_prefilled_tokens
        end = min(start + chunk_size, self.prompt_len)
        return start, end
```

效果：短请求的 TTFT 不再受长 prompt 拖累。

---

### 3.4 Prefix Caching（前缀缓存 / Copy-on-Write）

**论文**：[SGLang RadixAttention](https://arxiv.org/abs/2312.07104)

当多个请求共享相同 system prompt，KV cache 只需计算一次：

```
Req 1：[system prompt KV 256 tok] [user query 1 KV]  ← 全部计算
Req 2：[system prompt KV 256 tok] [user query 2 KV]  ← 前 256 tok 直接复用！
       ^────── ref_count=2，CoW 保护 ──────^
```

**实现**：以满块的 `content_hash` 为 key，命中则 `ref_count+1` 直接引用：

```python
class PrefixCache:
    _cache: Dict[int, PhysicalBlock]  # hash → PhysicalBlock
    
    def lookup(self, content_hash: int) -> Optional[PhysicalBlock]:
        block = self._cache.get(content_hash)
        if block:
            block.ref_count += 1   # 共享引用
        return block
    
    def cow_if_needed(self, block: PhysicalBlock):
        if block.ref_count <= 1:
            return block, False  # 独占，安全写入
        # 触发 CoW：分配新块，旧块 ref_count-1
        new_block = self.allocator.allocate()
        block.ref_count -= 1
        return new_block, True
```

**实测（30 个请求，48 token 共享前缀 + 16 token 独有后缀）：**

```
缓存命中率：71.3%
显存节省：~75% 的前缀 KV cache 块被共享，无需重复分配
```

---

### 3.5 CPU Swap（显存卸载）

当 GPU 显存不足，传统做法是丢弃被抢占序列的 KV cache，等轮到它时重新 prefill（代价高）。**CPU Swap** 将 KV cache 临时卸载到 CPU RAM：

```
GPU OOM：swap_out(seq) → GPU 块释放，映射记录在 CPU 字典
         swap_in(seq)  → 分配新 GPU 块，从 CPU 字典恢复映射
                       ← 避免昂贵的重新 prefill ←
```

**权衡**：PCIe 带宽（GPU↔CPU ~16 GB/s）远低于 GPU 显存带宽（~2 TB/s），swap 延迟约为 GPU 内部操作的 100 倍。但对长序列（512+ tokens）来说，swap 延迟 << 重新 prefill 延迟。

这与 vLLM 的 `preemption_mode="swap"` 完全对应（另一模式是 `"recompute"`）。

---

### 3.6 Speculative Decoding（投机解码）

**论文**：[Leviathan et al., ICML 2023](https://arxiv.org/abs/2211.17192)

标准解码每步只生成 1 个 token（memory-bound，GPU 计算利用率低）。投机解码利用并行验证加速：

```
Step 1 (Draft)  ：小模型连续生成 K=4 个候选 token [t₁, t₂, t₃, t₄]
Step 2 (Verify) ：大模型并行验证所有 K 个（等价于 1 次 prefill）
Step 3 (Accept) ：t₁✓ t₂✓ t₃✗ → 接受 [t₁, t₂]，用大模型的纠正 token
```

期望每步接受的 token 数：`E[accepted] = (1 - α^(K+1)) / (1 - α)`

当 `α=0.7, K=4`：期望 ≈ **3.3 tokens/步**（vs 标准的 1 token/步）

实际部署中，draft model 通常比 target model 小 10-20 倍（如 GPT-2 small vs GPT-2 large），验证步骤的成本接近 1 个标准 decode step，但产出 3.3 个 token，理论加速 **3.3x**。

---

## 四、系统整体数据流

```
用户请求
  │
  ▼
LLMEngine.add_request()
  │ 创建 Sequence（状态 WAITING），加入 Scheduler.waiting 队列
  ▼
Scheduler.schedule()  ── 每步调用 ──▶
  │
  ├── 1. 为 RUNNING 序列 append_slot（可能触发 CoW）
  ├── 2. 为 PREFILLING 序列推进 chunk 进度
  ├── 3. 尝试 swap_in SWAPPED 序列
  └── 4. 从 waiting 拉入新请求（分配物理块）
  │
  ▼ SchedulerOutput（prefill_chunks + decode_seqs）
  │
ModelRunner.step()  →  Dict[seq_id → new_token_id]
  │
  ▼
Scheduler.on_step_done()  →  追加 token，检查完成条件
  │
MetricsCollector  →  p50/p95/p99 latency、TTFT、KV utilization
  │
  ▼
RequestOutput（latency, ttft, output_text）
```

---

## 五、工程亮点

### 测试驱动，67 个单元测试

```bash
tests/test_block_allocator.py   # 9 tests：OOM、CoW、批量释放
tests/test_kv_cache.py          # 10 tests：分配、append_slot、CoW
tests/test_scheduler.py         # 8 tests：连续批处理核心逻辑
tests/test_new_features.py      # 40 tests：6 个新功能的完整测试
```

### 干净的接口分离

`ModelRunner` 是可替换的后端接口，调度器完全不感知后端差异：

```python
class BaseModelRunner(ABC):
    @abstractmethod
    def step(self, prefill_seqs, decode_seqs) -> Dict[int, int]: ...
```

`MockModelRunner`（无 GPU，用于 CI）和 `GPT2ModelRunner`（真实推理）实现同一接口。

### GitHub Actions CI

矩阵测试（Python 3.9/3.10/3.11）+ benchmark smoke test，每次 push 自动验证。

---

## 六、面试中的高频考点

通过实现这个项目，能清楚回答以下问题：

**Q：vLLM 为什么比 HuggingFace 快？**  
→ Continuous Batching（避免 GPU 空闲）+ PagedAttention（消除显存碎片），GPU 利用率从 ~30% 提升到 ~90%+。

**Q：Prefix Cache 的 CoW 怎么实现的？**  
→ 每个满块计算 `content_hash = hash(tuple(token_ids))`，命中时 `ref_count+1` 直接引用；写入前检查 `ref_count > 1` 则分配新块、旧块 `ref_count-1`。

**Q：显存不够时怎么处理？**  
→ 三种策略：① Swap out 到 CPU（保留 KV cache）；② Recompute（丢弃，适合短序列）；③ 请求排队（等内存释放）。

**Q：Chunked Prefill 和 Continuous Batching 的关系？**  
→ CB 解决"谁来跑"的问题（批次动态变化）；Chunked Prefill 解决"prefill 步骤阻塞 decode"的问题（把长 prefill 拆碎，穿插到 decode 步骤之间）。二者配合使用。

**Q：Speculative Decoding 什么时候失效？**  
→ draft 和 target 分布差异大时（α 低），每步 draft 时间 > 节省的 token 时间。通常 α < 0.5 时不如标准解码。实践中选择同家族小模型（如 Llama-3.2-1B 作为 Llama-3-70B 的 draft）。

---

## 七、与 vLLM 的对比

| 功能 | 本项目 | vLLM |
|------|--------|------|
| Paged KV Cache | ✅ Python 模拟 | ✅ CUDA kernel（PagedAttention） |
| Continuous Batching | ✅ | ✅ |
| Chunked Prefill | ✅ | ✅（Sarathi-Serve 集成） |
| Prefix Caching | ✅ 哈希平铺 | ✅ Radix Tree（最长前缀匹配） |
| CPU Swap | ✅ 模拟 | ✅ 真实 CUDA memcpy |
| Speculative Decoding | ✅ | ✅（Eagle、Medusa 等变体） |
| 多 GPU | ❌ | ✅（Tensor + Pipeline Parallelism） |

本项目的价值在于**让调度策略层可读、可测试、可解释**。

---

## 八、总结

推荐路线（如果你也想做）：

1. **先读论文摘要**：PagedAttention（SOSP'23）、Orca（OSDI'22）、Sarathi（OSDI'24）——各 10 分钟，理解直觉
2. **先实现 BlockAllocator**：最纯粹的数据结构，没有依赖，容易测试
3. **再实现 Scheduler**：连续批处理是核心，其他功能都是在它上面叠加
4. **最后加高级功能**：Prefix Cache → CPU Swap → Speculative Decoding

代码在 GitHub：**[liangqianxing/mini-llm-engine](https://github.com/liangqianxing/mini-llm-engine)**

```bash
git clone https://github.com/liangqianxing/mini-llm-engine
cd mini-llm-engine
pip install pytest
pytest tests/ -v                              # 67 tests, all pass
python run_all_benchmarks.py --fast --no-plot # 5 个 benchmark
```
