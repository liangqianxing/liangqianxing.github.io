---
title: "高并发 LLM 推理网关架构实践：两级精确/语义缓存、50ms 有界动态批处理与尾部延迟治理"
date: '2026-10-19'
description: "深度拆解生产级大模型推理网关 LLM-Gateway 的核心系统设计：基于 SHA-256 与 Redis 向量余弦相似度的两级精确/语义缓存、50ms 有界等待动态批处理调度器（Dynamic Batching）、令牌桶多租户限流与 Prometheus 全链路可观测性，将命中请求 P99 延迟从 5,307 ms 压降至 17 ms。"
cover: /images/posts/llm-gateway-semantic-cache-dynamic-batching/llm-gateway-cover-20261019.webp
coverAlt: 高并发请求流依次穿过精确哈希滤网、向量球面语义缓存层与 50ms 动态批处理时间窗拼装器的等距系统架构插画
categories:
  - AI Infra
tags:
  - AI Infra
  - LLM 推理
  - 分布式系统
  - Redis
  - FastAPI
  - 后端
mathjax: true
hidden: false
haloPublished: true
---

在将大语言模型接入高并发业务系统时，如果让上游微服务直接裸连底层推理引擎（vLLM / SGLang 或外部模型 API），系统很快会暴露出三个典型的生产痛点：

1. **高频重复与近义查询消耗巨量算力**：大量用户请求在语义上高度等价（例如“如何重置登录密码？”与“忘记密码怎么找回？”），每次都穿透到底层跑完整的 Prefill + Decode，不仅浪费昂贵的 GPU 算力，还拉高了整体排队水位；
2. **碎片化小请求导致 GPU 批次利用率低下**：突发流量往往以单条请求（Batch Size = 1）零散到达，缺乏网关层的微批聚合（Micro-batching），底层算子长期处于 Memory-Bound 状态；
3. **长尾请求缺乏多租户隔离与熔断观测**：少数超长 Prompt 请求占满并发槽位，导致普通短请求的 P99 尾部延迟飙升至数十秒。

为此，我设计并实现了开源推理服务网关 **[LLM-Gateway](https://github.com/LiangQianXing/llm-gateway)**（配套压测产物保存在 `experiments/gateway-bench/results.json`）。本文系统拆解其四大核心组件——**两级精确/语义缓存（Two-Tier Exact & Semantic Cache）**、**50 ms 有界等待动态批处理调度器（Bounded-Wait Dynamic Batcher）**、**令牌桶限流（Token-Bucket Rate Limiter）**与 **Prometheus 指标体系**，并给出多组并发压测下的 P50 / P95 / P99 实测数据。

---

## 一、整体架构：四层流水线网关拓扑

LLM-Gateway 位于业务调用方与底层推理集群之间，所有请求按以下异步非阻塞流水线依次流转：

```text
客户端并发请求 (HTTP / SSE)
   │
   ▼
[1. 令牌桶限流与配额网关] ──► 超出租户 QPS / Token 水位时快速返回 HTTP 429
   │
   ▼
[2. L1 精确哈希缓存 (Redis String)] ──► SHA-256(model + normalized_messages + temp) 命中 (~2 ms)
   │ Miss
   ▼
[3. L2 向量语义缓存 (Embedding + Cosine ≥ τ)] ──► 语义等价查询命中 (~15 ms)
   │ Miss
   ▼
[4. 50ms 有界动态批处理调度器] ──► 聚合时间窗内同模型请求为 Micro-batch ──► 底层推理引擎
   │
   └──► 异步回填 L1 & L2 缓存 + 暴露 Prometheus 直方图指标 (/metrics)
```

---

## 二、核心模块设计与源码实现

### 1. L1 精确哈希 + L2 向量余弦相似度两级缓存

为什么不能只做精确字符串匹配，或者只做向量检索？
- **L1 精确缓存**基于规范化后的请求签名计算确定性 SHA-256 哈希，时间复杂度为 $O(1)$，不需要运行任何 Embedding 模型，可将完全相同请求的响应压缩到 **1～3 ms**；
- **L2 语义缓存**针对 L1 未命中的自然语言查询，将其编码为单位长度稠密向量 $\mathbf{e}_q \in \mathbb{R}^d$（$\|\mathbf{e}_q\|_2 = 1$），在 Redis 语义索引中检索满足余弦相似度阈值 $\tau$ 的历史问答对：

$$
\text{sim}(\mathbf{e}_q, \mathbf{e}_i) = \mathbf{e}_q^\top \mathbf{e}_i \ge \tau \quad (\text{默认 }\tau = 0.92)
$$

需要注意的是，**并非所有请求都允许进入语义缓存**：当请求包含实时工具调用结果（`role == "tool"`）、高随机性采样温度（`temperature > 0.3`）或显式指定 `cache_control: no-cache` 时，网关会自动旁路 L2 语义层，防止跨上下文串话。

### 2. 50 ms 有界等待动态批处理调度器（Bounded-Wait Dynamic Batcher）

对于穿透两级缓存的真实推理请求，网关通过 `asyncio.Queue` 与 `asyncio.Future` 实现了一个无锁异步动态批处理器。其核心停止条件由**最大批次容量 $B_{\max}$** 与**最大等待时间窗 $T_{\max}$（默认 50 ms）**共同约束：

```python
import asyncio
import time
from dataclasses import dataclass

@dataclass
class PendingRequest:
    payload: dict
    future: asyncio.Future
    enqueue_ts: float

class DynamicBatchScheduler:
    def __init__(self, backend_client, max_batch_size: int = 16, max_wait_ms: float = 50.0):
        self.backend = backend_client
        self.max_batch_size = max_batch_size
        self.max_wait_sec = max_wait_ms / 1000.0
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue()

    async def submit(self, payload: dict) -> dict:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self.queue.put(PendingRequest(payload=payload, future=fut, enqueue_ts=time.perf_counter()))
        return await fut

    async def worker_loop(self) -> None:
        while True:
            # 1. 阻塞等待批次的首个请求到达
            first_req = await self.queue.get()
            batch = [first_req]
            deadline = time.perf_counter() + self.max_wait_sec

            # 2. 在 50ms 有界时间窗内尽可能凑满 max_batch_size
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break

            # 3. 批量下发到底层推理后端并按序唤醒各请求的 Future
            asyncio.create_task(self._dispatch_batch(batch))

    async def _dispatch_batch(self, batch: list[PendingRequest]) -> None:
        try:
            responses = await self.backend.generate_batch([item.payload for item in batch])
            for item, resp in zip(batch, responses):
                if not item.future.done():
                    item.future.set_result(resp)
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
```

这种“首包触发计时 + 容量/超时双阈值截断”的设计保证了：**在低峰期（单请求稀疏到达），请求最多额外等待 50 ms（相比数秒的 LLM 推理耗时仅占 $<2\%$）；而在高峰期（并发涌入），队列可在几毫秒内凑满 $B_{\max}$ 立即发车**。

---

## 三、受控压测与尾部延迟（P50 / P95 / P99）对比

我们在真实 LLM 推理负载下对网关进行了三组受控压测（完整原始数据保存在 `experiments/gateway-bench/results.json`），分别考察**精确缓存热命中**、**混合流量下的语义缓存**以及**高并发突发流量**：

### 1. 场景 A：L1 精确缓存冷启动 vs. 热命中（50 请求，并发度 $C=5$）

| 压测阶段 | 缓存命中率 | 吞吐量 (RPS) | 平均延迟 (ms) | P50 延迟 (ms) | P95 延迟 (ms) | P99 延迟 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **冷缓存穿透（Cold Pass）** | 0.0% | 1.38 req/s | 3,522.6 ms | 3,372.6 ms | 4,768.2 ms | **5,307.0 ms** |
| **热缓存命中（Warm Exact Hit）** | 100.0% | **316.58 req/s** | 14.5 ms | 15.6 ms | 16.9 ms | **17.0 ms** |

在热点问题重复到达时，L1 精确缓存将 P99 尾部延迟从 **5,307.0 ms 直接压降至 17.0 ms（降低 99.7%）**，系统吞吐量从 `1.38 RPS` 跃升至 **`316.58 RPS`（提升 229×）**。

### 2. 场景 B：混合流量下的两级精确 + 语义缓存（60 请求，并发度 $C=5$）

在包含 20 条首次唯一查询、20 条精确重复查询与 20 条句式改写近义查询的混合流量中：

| 指标项 | 无缓存基线（Direct LLM） | 开启两级缓存（L1 Exact + L2 Semantic） | 核心收益对比 |
| --- | ---: | ---: | --- |
| **总耗时（Wall Time）** | 84.52 s | **30.52 s** | 端到端总耗时缩短 **63.9%** |
| **整体缓存拦截率** | 0.0% | **65.0%**（39 / 60 命中） | 直接节省约三分之二的底层 GPU 推理调用 |
| **未命中请求 P50 延迟** | 3,372.6 ms | 3,332.9 ms | 缓存未命中路径几乎零额外损耗 |
| **语义缓存命中延迟（L2 Hit）** | — | **15.1 ms（P99 $\le 37.0\text{ ms}$）** | 在本地完成向量编码与余弦匹配，免除秒级生成等待 |

### 3. 场景 C：高并发压力测试（100 请求，并发度 $C=10, 20, 50$）

在 65% 稳态缓存命中率的混合生产流量模拟下，逐步拉高客户端并发度：

| 客户端并发度 ($C$) | 成功率 | 网关总吞吐 (RPS) | 整体 P50 延迟 | 缓存命中路径 P99 延迟 | 穿透推理路径 P99 延迟 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| $C = 10$ | 100.0% | 4.86 req/s | 1.9 ms | 18.5 ms | 7,280.7 ms |
| $C = 20$ | 100.0% | 4.86 req/s | 1.6 ms | 24.3 ms | 14,193.3 ms |
| $C = 50$ | 100.0% | **5.55 req/s** | **2.2 ms** | **37.0 ms** | 21,963.1 ms |

可以看到，即使在 $C=50$ 的突发洪峰下（底层 LLM 因并发排队导致穿透路径 P99 达到 `21,963.1 ms`），网关层的异步事件循环依然未被阻塞，**全部命中缓存的请求 P50 保持在 `2.2 ms`、P99 稳定在 `37.0 ms` 以内**，实现了快慢路径的彻底解耦。

---

## 四、工程设计总结

1. **快慢路径解耦是治理 LLM 尾部延迟的第一要务**：通过前置的 L1 精确哈希与 L2 向量语义缓存拦截 60%+ 的重复与同义长尾流量，能防止底层 GPU 队列被低价值重复计算打满；
2. **语义缓存必须受控启用**：仅对确定性、无状态问答开启高阈值（$\tau \ge 0.92$）余弦匹配，对带实时 Tool Observation 或高采样温度的请求严格旁路；
3. **有界时间窗（50 ms）动态批处理兼顾了延迟与吞吐**：用微小的 50 ms 聚合窗口将零散并发请求打包下发，配合 Prometheus `/metrics` 暴露的缓存命中率与队列水位直方图，构成了生产级推理网关的标准底座。
