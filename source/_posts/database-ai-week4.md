---
title: Week 4：数据库速成——从 Storage、Index、Query Optimization 到 Vector DB 与 RAG
date: 2026-05-05 12:00:00
categories: 技术
tags:
  - 数据库
  - CMU 15-445
  - Vector DB
  - RAG
  - LLM Memory
  - Query Optimization
  - Caching
---

Week 1 我们理解了 Autograd，Week 2 理解了 GPU 推理加速，Week 3 理解了分布式系统。Week 4 要补的是数据库，但目标不是成为数据库内核工程师，而是学会用数据库视角理解 AI 系统里的 **Vector DB、RAG pipeline、LLM memory 和 retrieval latency**。

只看 CMU 15-445 中和 AI 最相关的四块：Storage & Buffer Pool、Index（B+ Tree / LSM）、Query Optimization、Caching。你会发现，很多所谓“AI memory 系统”的核心问题，本质仍然是数据库老问题：数据怎么存、索引怎么建、查询怎么优化、缓存怎么命中。
<!-- more -->

![Storage buffer cache](/images/posts/database-ai/storage-buffer-cache.svg)

## 1. 为什么 AI 系统要学数据库

一个 RAG 系统看起来像是 LLM 应用：

```text
用户问题 -> embedding -> 向量检索 -> rerank -> 拼 prompt -> LLM 回答
```

但如果拆到底，它其实是一条数据库查询链路：

```text
query
  -> query embedding
  -> index lookup
  -> metadata filter
  -> top-k candidate
  -> rerank
  -> fetch document chunks
  -> cache result
```

这条链路里每一步都像数据库：

- 文档 chunk 是 record；
- embedding 是向量字段；
- metadata 是标量字段；
- vector index 是特殊索引；
- top-k retrieval 是近似查询；
- rerank 是二阶段排序；
- prompt context 是查询结果；
- memory 是不断更新的知识表；
- cache 命中率决定尾延迟。

所以理解数据库不是为了写 SQL，而是为了知道 retrieval 为什么慢、Vector DB 为什么这么设计、LLM memory 为什么难做。

## 2. 数据库视角下的 AI 数据

AI 应用里常见数据可以分成几类：

| 数据 | 数据库类比 | 例子 |
|---|---|---|
| 原始文档 | heap file / object storage | PDF、网页、Markdown |
| 文本块 | record / row | chunk text |
| embedding | vector column | 768/1024/1536 维向量 |
| metadata | scalar columns | source、time、author、tag |
| 索引 | access path | B+Tree、HNSW、IVF、LSM |
| 对话记忆 | mutable state | user profile、session memory |
| 检索缓存 | result cache | query -> top-k chunks |
| embedding 缓存 | computed feature cache | text -> vector |

当你说“让 LLM 有长期记忆”，工程上往往意味着：设计一张或多张表，存储内容、时间、来源、embedding、重要性分数、过期策略，然后提供低延迟检索和更新。

## 3. Storage：数据到底怎么放

数据库首先要解决 storage。数据不是抽象地“存在库里”，而是以 page、file、segment、SSTable、object 等形式落在磁盘或对象存储中。

传统数据库常见单位是 page，例如 4KB、8KB、16KB。Buffer pool 以 page 为单位把磁盘数据加载到内存。

AI 系统里也有类似问题：

- 文档原文放对象存储还是数据库；
- chunk text 和 embedding 放一起还是分开；
- metadata 和 vector index 是否共存；
- 大 embedding 是否压缩；
- 冷数据是否下沉到便宜存储；
- 热门 chunk 是否常驻内存。

如果数据布局不合理，即使模型很强，检索也会慢。

## 4. Row Store 与 Column Store

数据库常见两种布局：row store 和 column store。

Row store 把一行的数据放在一起：

```text
[id, text, embedding, source, timestamp]
[id, text, embedding, source, timestamp]
```

适合按主键取完整记录，例如取某个 chunk 的 text、metadata 和向量。

Column store 把同一列连续存放：

```text
id column
text column
embedding column
source column
timestamp column
```

适合分析查询和扫描某些列，例如只扫描 timestamp 或 source。

Vector DB 里经常混合使用：向量索引用特殊结构保存，metadata 用标量索引保存，chunk 原文可能放在独立 doc store 中。查询时先通过向量索引拿到 candidate id，再回表读取文本和 metadata。

## 5. Buffer Pool：为什么缓存 page 很重要

Buffer pool 是数据库在内存中管理磁盘 page 的组件。它解决的问题是：内存放不下所有数据，但频繁访问磁盘太慢，所以要把热 page 缓存在内存里。

基本流程：

```text
query needs page P
  -> check buffer pool
  -> hit: return memory page
  -> miss: read page from disk
  -> maybe evict another page
```

几个关键概念：

- page table：记录 page id 到 frame 的映射；
- frame：内存中的 page slot；
- pin count：防止正在使用的 page 被淘汰；
- dirty bit：page 是否被修改过，淘汰前是否要写回；
- replacement policy：淘汰谁，例如 LRU、Clock、LRU-K。

对应到 RAG：

- 热门文档 chunk 应该更容易留在 cache；
- 热门 query 的 top-k 结果可以缓存；
- embedding 计算结果可以缓存；
- rerank 结果可以缓存；
- 长尾冷数据可以接受更高延迟。

Buffer pool 思想告诉我们：性能不是只靠索引，缓存命中率同样关键。

## 6. Index：索引是为了减少扫描

没有索引时，查询只能全表扫描：

```text
for row in table:
    if row.source == "paper":
        return row
```

数据小的时候没问题，数据大了就不可接受。索引的作用是提供 access path，让查询直接跳到可能相关的数据。

AI 系统里至少有两类查询：

1. 标量查询：`source = paper`、`timestamp > 2025`、`user_id = 123`；
2. 向量查询：找 embedding 距离 query embedding 最近的 top-k。

因此 Vector DB 通常需要同时支持：

- 标量索引：B+ Tree、Hash、Bitmap；
- 向量索引：HNSW、IVF、PQ、DiskANN；
- 混合查询：vector search + metadata filter。

## 7. B+ Tree：最经典的范围查询索引

B+ Tree 是数据库里最常见的索引结构之一。它的特点是：

- 多叉树，高度低；
- 内部节点只存 key 和指针；
- 叶子节点存 key 和 record pointer；
- 叶子节点之间有链表，方便范围扫描；
- 适合磁盘和 page 访问模型。

结构大概是：

```text
          [30 | 60]
        /    |     \
   [1..29] [30..59] [60..99]
```

为什么不用普通二叉树？因为磁盘 I/O 很贵，B+ Tree 一个节点可以放很多 key，对应一个 page。这样树高度很低，几次 page read 就能找到数据。

B+ Tree 适合：

- 主键查询；
- 范围查询；
- 排序扫描；
- metadata filter，例如 timestamp、doc_id、user_id。

在 RAG 中，B+ Tree 可以用来做标量过滤：

```text
先过滤 user_id = 123 and timestamp > 2025
再在过滤后的文档集合里做向量检索
```

或者反过来：

```text
先 ANN 召回 top-1000
再用 metadata filter 过滤
```

哪个更好，就是 query optimization 的问题。

## 8. LSM Tree：写优化索引

LSM Tree 常见于 RocksDB、LevelDB、Cassandra 等系统。它适合写多读也多的场景。

核心思想：先写内存，再批量刷盘，磁盘上以有序文件保存。

```text
write -> WAL
      -> MemTable
      -> flush to SSTable
      -> compaction
```

关键组件：

- WAL：write-ahead log，防止内存数据丢失；
- MemTable：内存中的有序结构；
- SSTable：磁盘上的不可变有序文件；
- Compaction：合并多个 SSTable，清理旧版本和删除标记；
- Bloom Filter：快速判断某个 key 是否可能存在。

LSM 的优点是写入吞吐高，因为随机写变成顺序写。缺点是读可能需要查多个层级，compaction 也会带来后台开销。

对应 AI memory：如果你的记忆系统频繁写入新事实、新对话、新工具结果，LSM 思想就很重要。很多向量库或嵌入式存储底层会用 LSM KV 来保存 metadata 或对象。

## 9. 向量索引：为什么不是简单 B+ Tree

Embedding 是高维向量，例如 768 维或 1536 维。我们关心的是相似度：

```text
cosine similarity(query_vector, doc_vector)
```

高维空间里，B+ Tree 这类一维有序索引不适合直接做 nearest neighbor。向量检索通常用 ANN：Approximate Nearest Neighbor。

ANN 的核心取舍是：牺牲一点召回精度，换取大幅查询速度提升。

常见向量索引：

| 索引 | 思想 | 特点 |
|---|---|---|
| HNSW | 小世界图，沿图贪心搜索 | 召回高、内存占用大 |
| IVF | 聚类分桶，先找近的 bucket | 速度快，依赖聚类质量 |
| PQ | 向量量化压缩 | 省内存，可能损失精度 |
| DiskANN | 面向磁盘/SSD 的图索引 | 大规模低成本存储 |

Vector DB 的核心不是“存向量”，而是在延迟、召回率、内存占用、更新成本之间做取舍。

## 10. RAG Pipeline：数据库视角

![RAG vector database pipeline](/images/posts/database-ai/rag-vector-db.svg)

一个典型 RAG pipeline：

```text
离线：
documents
  -> parse
  -> chunk
  -> embedding
  -> build vector index
  -> store metadata and text

在线：
query
  -> query embedding
  -> ANN search
  -> metadata filter
  -> rerank
  -> fetch top-k chunks
  -> build prompt
  -> LLM answer
```

从数据库视角看：

- parse/chunk 是 ETL；
- embedding 是特征生成；
- vector index 是 access method；
- metadata filter 是 predicate；
- rerank 是二阶段 query processing；
- top-k 是排序和截断；
- prompt assembly 是结果 materialization；
- query cache 是 result cache。

所以 RAG 优化不能只调 prompt。很多时候更该问：chunk 是否合理、索引参数是否合理、metadata filter 顺序是否合理、cache 是否命中、rerank 是否太慢。

## 11. Query Optimization：为什么执行顺序很重要

同一个查询有多种执行计划。数据库优化器要选择成本最低的计划。

比如用户问：

```text
只在 2024 年后的论文里，找和 diffusion acceleration 相关的段落
```

可能有两种计划：

### Plan A：先向量检索，再过滤

```text
ANN search top-1000
  -> filter year >= 2024
  -> rerank top-50
```

如果过滤条件不严格，这个计划很好。

### Plan B：先 metadata filter，再向量检索

```text
filter year >= 2024
  -> ANN search within filtered subset
  -> rerank top-50
```

如果过滤条件很严格，这个计划可能更好。

优化器要估算：

- 过滤条件选择率；
- ANN search 成本；
- rerank 成本；
- 回表读取成本；
- cache 命中率；
- top-k 大小；
- 网络开销。

这就是为什么一些 Vector DB 提供 hybrid search 和 filter pushdown。它们本质上是在做查询优化。

## 12. Cost Model：数据库如何估算成本

数据库优化器会用 cost model 比较执行计划。简化看：

```text
cost = I/O cost + CPU cost + network cost + memory cost
```

RAG 里可以类似估算：

```text
retrieval latency = embedding latency
                  + ANN search latency
                  + metadata filter latency
                  + fetch chunk latency
                  + rerank latency
                  + network latency
```

如果 rerank 模型很大，rerank 可能成为瓶颈。如果 chunk 存在远程对象存储，fetch chunk 可能成为瓶颈。如果 query embedding 没缓存，embedding API 延迟也可能主导整体体验。

优化前一定要测量，不要凭感觉。

## 13. Caching：缓存什么最划算

AI 系统里缓存非常重要。常见缓存层：

| 缓存 | Key | Value | 适用场景 |
|---|---|---|---|
| Embedding cache | text hash | vector | 重复文本、重复 query |
| Retrieval cache | query hash | top-k ids | 热门问题 |
| Chunk cache | chunk id | text | 热门文档 |
| Rerank cache | query + candidate ids | rerank scores | 重复检索结果 |
| Prompt cache | prefix tokens | KV cache / token states | 长系统提示、固定上下文 |
| Answer cache | normalized query | final answer | FAQ 类场景 |

缓存要考虑四个问题：

1. 命中率高不高；
2. value 计算成本贵不贵；
3. value 会不会过期；
4. 缓存错误会不会影响正确性。

比如 embedding cache 通常很安全，因为同一文本的 embedding 稳定。answer cache 风险更高，因为答案可能依赖时间、权限、上下文。

## 14. Cache Invalidation：缓存失效比缓存更难

缓存最大的问题是失效。文档更新后，旧的 embedding、旧的 retrieval result、旧的 answer 都可能过期。

常见策略：

- TTL：过一段时间自动失效；
- version：文档版本变化后 cache key 变化；
- explicit invalidation：更新文档时主动删缓存；
- write-through：写入时同步更新缓存；
- lazy refresh：读到旧缓存时异步刷新；
- namespace：按用户、项目、知识库隔离缓存。

RAG 系统尤其要注意权限。如果 retrieval cache 没有把 user_id、tenant_id、permission version 放进 cache key，可能出现越权召回。

## 15. LLM Memory：长期记忆是数据库问题

LLM memory 常被包装得很神秘，但工程上通常是：

```text
memory item = {
  id,
  user_id,
  content,
  embedding,
  importance,
  timestamp,
  source,
  access_count,
  expires_at
}
```

写入 memory：

```text
conversation/tool result
  -> extract facts
  -> score importance
  -> deduplicate
  -> embed
  -> upsert memory store
```

读取 memory：

```text
current query
  -> embed query
  -> retrieve relevant memories
  -> filter by user/session/permission
  -> rerank by relevance + recency + importance
  -> inject into prompt
```

这就是一个带向量索引、标量过滤、更新策略、缓存策略的数据库系统。

LLM memory 难点：

- 什么时候写入；
- 写入什么粒度；
- 如何去重和合并；
- 如何处理过期记忆；
- 如何避免错误记忆污染；
- 如何控制检索延迟；
- 如何保证用户隔离和隐私。

## 16. Retrieval Latency：慢在哪里

一次 RAG 检索可能慢在很多地方：

```text
query normalization
  -> embedding API
  -> vector DB network call
  -> ANN index search
  -> metadata filtering
  -> fetch full chunk text
  -> rerank model
  -> prompt assembly
```

优化要先定位瓶颈。

### 16.1 Embedding 慢

优化：embedding cache、批量 embedding、本地 embedding 模型、更小模型、异步预计算。

### 16.2 ANN search 慢

优化：调 HNSW ef_search、IVF nprobe、减少搜索范围、增加内存、使用量化、冷热分层。

### 16.3 Metadata filter 慢

优化：建立标量索引、filter pushdown、先过滤再向量搜、权限 bitmap。

### 16.4 Fetch chunk 慢

优化：chunk cache、把 hot text 放近、减少回表、列裁剪、压缩。

### 16.5 Rerank 慢

优化：减少候选数、batch rerank、小模型 rerank、只对高价值 query rerank、缓存 rerank 分数。

## 17. B+ Tree、LSM、Vector Index 怎么选

一个 AI memory / RAG 系统里通常不是只用一种索引。

| 查询需求 | 合适结构 |
|---|---|
| 按 id 查 chunk | Hash / B+ Tree |
| 按时间范围查 | B+ Tree |
| 按用户过滤 | Hash / B+ Tree / Bitmap |
| 高频写入 memory | LSM KV |
| 向量相似度 top-k | HNSW / IVF / PQ / DiskANN |
| 大规模冷数据 | DiskANN / object store + metadata index |
| 混合检索 | Vector index + scalar index + optimizer |

真正的系统通常是组合拳：LSM 存 metadata，object store 存原文，HNSW 存热向量，DiskANN 存冷向量，Redis 缓存热门结果。

## 18. Vector DB 不是魔法

Vector DB 一般包括：

- storage layer：保存向量、metadata、文档引用；
- index layer：HNSW、IVF、PQ 等；
- query layer：top-k、filter、hybrid search；
- update layer：insert、delete、compaction、rebuild；
- cache layer：热向量、热结果、热文档；
- distributed layer：sharding、replication、load balancing；
- consistency layer：写入可见性、快照、版本控制。

所以评价一个 Vector DB，要看：

- recall-latency 曲线；
- 写入和删除成本；
- metadata filter 能力；
- 是否支持多租户隔离；
- 索引重建成本；
- 热更新是否影响查询；
- 内存占用；
- tail latency；
- 运维复杂度。

## 19. RAG 优化 Checklist

做 RAG 系统时，可以按数据库思路检查：

1. Chunk 粒度是否适合查询；
2. Embedding 模型是否和领域匹配；
3. 向量索引参数是否测过 recall-latency；
4. Metadata filter 是否有索引；
5. Filter 是先做还是后做；
6. Top-k 候选数是否合理；
7. Rerank 是否成为瓶颈；
8. Chunk text 是否频繁回表；
9. Embedding / retrieval / rerank 是否有缓存；
10. Cache key 是否包含用户权限和版本；
11. 文档更新后索引和缓存如何失效；
12. P95/P99 latency 慢在哪里；
13. 召回错误是 chunk 问题、embedding 问题还是 index 问题；
14. 是否需要冷热分层；
15. 是否需要分片和副本。

## 20. Week 4 学完应该掌握什么

这一周不要求你能实现数据库内核，但要能讲清楚：

- Storage 为什么决定数据读取路径；
- Buffer pool 为什么能显著影响性能；
- B+ Tree 为什么适合范围查询和 metadata filter；
- LSM Tree 为什么适合高写入场景；
- 向量索引为什么要做 ANN，而不是简单排序全量向量；
- Query optimizer 为什么要选择执行计划；
- RAG pipeline 为什么是一条数据库查询链路；
- Caching 应该缓存 embedding、retrieval、chunk、rerank 还是 answer；
- LLM memory 为什么本质是带向量检索的数据库状态；
- Retrieval latency 应该如何拆解和定位。

## 21. 最后总结

数据库不是 AI 系统的外围组件，而是 RAG、Vector DB、LLM memory 的底层骨架。Storage 决定数据怎么放，Buffer Pool 决定热数据怎么留在内存，Index 决定如何避免全量扫描，Query Optimization 决定执行顺序，Caching 决定尾延迟。

当你用这套视角看 RAG，就不会只停留在“换 embedding 模型”或“调 prompt”。你会开始系统性地问：数据布局对吗，索引选型对吗，filter 顺序对吗，缓存 key 对吗，P99 慢在哪里。这才是把 AI 应用做成可靠系统的关键。

