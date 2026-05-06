---
title: 阿里 Agent Infra 工程师面试准备：AI Infra、稳定性工程与项目包装
date: 2026-05-06 10:00:00
categories: 技术
tags:
  - 面试
  - Agent
  - AI Infra
  - 稳定性工程
  - RAG
  - LLM
  - 阿里巴巴
---

这篇文章整理的是阿里巴巴控股集团「Agent Infra 工程师」岗位的面试准备路线。岗位归属是平台技术-稳定性工程，所以准备重点不是“会调用大模型 API”，而是：**能把 Agent / LLM 应用做成稳定、可观测、可扩展、可恢复的基础设施**。

如果只用一句话概括这个岗位：它需要你理解 Agent 系统，也需要你理解 Infra。也就是说，既要知道 Planner、Tool Calling、Memory、RAG、Workflow 怎么设计，又要知道超时、重试、幂等、限流、熔断、降级、监控、Trace、SLO 怎么落地。
<!-- more -->

## 1. 岗位信息拆解

投递信息大概是：

```text
公司：阿里巴巴控股集团
岗位：Agent Infra 工程师
招聘类型：阿里巴巴 2027 届实习生
部门：阿里巴巴控股集团-平台技术-稳定性工程
状态：面试中
```

这个岗位名称里有两个关键词：

1. **Agent Infra**：Agent 运行时、任务编排、工具调用、记忆系统、RAG、评测、权限、沙箱、观测平台。
2. **稳定性工程**：高可用、故障恢复、容量治理、限流熔断、降级、监控告警、SLO、成本控制。

所以面试时不能只讲“我会写 prompt”或“我接过 OpenAI API”。更好的表达是：

> 我关注的是如何把 Agent 从 demo 做成平台能力：任务可编排、状态可恢复、工具调用可审计、检索链路可观测、失败可降级、成本可控制。

## 2. 岗位画像

### 2.1 Agent Infra 做什么

Agent Infra 可以理解成 Agent 应用背后的基础设施层。典型能力包括：

- Agent runtime：负责运行 Agent loop；
- Tool calling：统一工具注册、参数校验、权限控制、超时重试；
- Workflow engine：把复杂任务拆成可恢复的步骤；
- Memory system：短期记忆、长期记忆、用户画像、任务状态；
- RAG pipeline：embedding、向量检索、metadata filter、rerank、context packing；
- Evaluation：自动评测、人工反馈、回归测试；
- Observability：日志、指标、链路追踪、错误分析；
- Sandbox：代码执行、浏览器操作、文件操作的隔离；
- Cost control：token 成本、模型路由、缓存、限额。

### 2.2 稳定性工程做什么

稳定性工程关注系统在真实生产环境里能不能扛住不确定性：

- 外部模型 API 抖动怎么办；
- 工具调用超时怎么办；
- Agent 死循环怎么办；
- RAG 检索变慢怎么办；
- worker crash 后任务如何恢复；
- prompt 或 tool 输出异常如何兜底；
- 高峰期如何限流；
- 依赖服务故障时如何降级；
- 如何定义 SLO；
- 如何快速定位线上问题。

一句话：Agent Infra 偏“怎么搭平台”，稳定性工程偏“平台坏了怎么办，以及如何不让它坏”。

## 3. 准备方向总览

这个岗位可以按六块准备：

| 模块 | 必须掌握的问题 |
|---|---|
| Agent 系统设计 | Planner、Executor、Tool、Memory、Evaluator 怎么拆 |
| RAG / Memory | chunk、embedding、ANN、rerank、cache、权限隔离 |
| LLM Serving | prefill、decode、KV cache、batching、streaming、vLLM |
| 稳定性工程 | timeout、retry、idempotency、限流、熔断、降级、SLO |
| 分布式系统 | task queue、worker crash、checkpoint、Raft 概念、workflow |
| 工程基础 | Python/Go/Java、Redis、MySQL、MQ、Docker、K8s、Linux |

你博客里 Week 1–4 的学习路线其实很贴这个岗位：

- Week 1 Autograd：理解 DL 框架底层；
- Week 2 GPU 推理加速：理解 LLM serving；
- Week 3 分布式系统：理解 Ray、workflow、多 Agent；
- Week 4 数据库：理解 Vector DB、RAG、LLM memory。

面试时可以把这条学习路线包装成：我不是只会用模型，而是在系统性补 AI Infra 的底层能力。

## 4. Agent 系统设计

### 4.1 一个 Agent 平台怎么拆

可以把一个 Agent 平台拆成：

```text
User Request
  -> API Gateway
  -> Planner
  -> Task Scheduler
  -> Agent Runtime
  -> Tool Executor
  -> Memory / RAG
  -> Evaluator
  -> Response Aggregator
```

每个组件的职责：

| 组件 | 职责 |
|---|---|
| API Gateway | 鉴权、限流、请求入口、租户隔离 |
| Planner | 把用户目标拆成步骤 |
| Scheduler | 调度任务到 worker，管理并发 |
| Agent Runtime | 执行 Agent loop，管理上下文 |
| Tool Executor | 工具调用、参数校验、沙箱、超时 |
| Memory / RAG | 检索相关知识和历史状态 |
| Evaluator | 质量评估、安全检查、终止判断 |
| Aggregator | 汇总结果，生成最终回答 |

### 4.2 Agent loop 怎么设计

一个最小 Agent loop：

```text
observe user request
  -> plan next action
  -> call tool or LLM
  -> observe result
  -> update memory/state
  -> decide continue or stop
```

工程上要加很多约束：

- 最大步数限制，防止死循环；
- 最大 token 限制，控制成本；
- tool timeout，防止阻塞；
- tool allowlist，控制权限；
- sensitive action confirmation，防止危险操作；
- trace id，串起完整链路；
- checkpoint，支持失败恢复。

### 4.3 Tool Calling 怎么做稳定

Tool calling 是 Agent Infra 面试高频点。

核心问题：LLM 生成的工具调用不一定可靠。它可能参数错、调用不存在的工具、重复调用、有副作用。

稳定设计：

- 工具注册中心：每个工具有 schema、权限、超时、重试策略；
- 参数校验：用 JSON Schema / Pydantic 校验；
- 权限控制：不同用户、租户、Agent 可用工具不同；
- 超时控制：每个工具必须有 timeout；
- 幂等设计：可重试工具必须有 request id；
- 审计日志：记录谁在何时调用了什么工具；
- 沙箱隔离：代码执行、文件操作、浏览器操作不能裸跑；
- 结果截断：防止工具返回过大内容撑爆上下文。

## 5. RAG 与 LLM Memory

### 5.1 RAG pipeline

典型 RAG：

```text
Offline:
document -> parse -> chunk -> embedding -> vector index

Online:
query -> query embedding -> ANN search -> metadata filter -> rerank -> context packing -> LLM
```

面试要能讲清楚每一步的作用和瓶颈。

### 5.2 RAG 慢怎么排查

一次 retrieval latency 可以拆成：

```text
query rewrite latency
+ embedding latency
+ vector DB network latency
+ ANN search latency
+ metadata filter latency
+ fetch chunk latency
+ rerank latency
+ prompt assembly latency
```

优化方向：

- embedding cache；
- retrieval result cache；
- metadata filter pushdown；
- 调整 HNSW `ef_search` / IVF `nprobe`；
- 减少 rerank candidate 数；
- chunk text 缓存；
- 热点知识库常驻内存；
- 异步预取和并行检索。

### 5.3 Memory 怎么设计

LLM memory 本质是数据库问题。一个 memory item 可以这样设计：

```json
{
  "id": "mem_001",
  "user_id": "u123",
  "content": "用户更喜欢简洁直接的回答",
  "embedding": [0.01, -0.02, 0.03],
  "importance": 0.82,
  "timestamp": "2026-05-06T10:00:00+08:00",
  "source": "conversation",
  "expires_at": null
}
```

读取 memory：

```text
current query
  -> embed
  -> retrieve relevant memories
  -> filter by user/session/permission
  -> rerank by relevance + recency + importance
  -> inject into prompt
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

难点：错误记忆污染、过期策略、去重合并、权限隔离、延迟控制。

## 6. LLM Serving 与推理系统

### 6.1 Prefill 和 Decode

LLM 生成分两段：

- **Prefill**：处理 prompt，建立 KV cache；
- **Decode**：一次生成一个 token，逐步追加 KV cache。

Prefill 通常更像大矩阵乘，GPU 利用率较高；decode 每步只生成一个 token，batch 小时更容易 memory bound。

### 6.2 为什么 LLM 推理慢

主要原因：

- 模型参数大，每个 token 都要读大量权重；
- decode 自回归，不能并行生成未来 token；
- KV cache 随 batch、序列长度、层数增长；
- 小 batch 时 GPU 利用率低；
- serving 系统要处理动态请求和不同输出长度。

### 6.3 vLLM 为什么快

vLLM 的核心是 PagedAttention 和 continuous batching。

PagedAttention 借鉴操作系统分页思想，把 KV cache 切成 block/page，避免连续大块显存分配导致的碎片。

好处：

- KV cache 管理更灵活；
- 支持更高并发；
- 降低显存浪费；
- continuous batching 更容易做。

### 6.4 Streaming Generation

Streaming 能降低用户体感延迟，但不减少总计算量。

面试可以这样说：

> Streaming 优化的是交互体验，尤其是 time to first token；但自回归生成的总 token 数不变，所以总计算成本没有本质减少。

## 7. 稳定性工程重点

### 7.1 Timeout

所有外部调用都要有 timeout：

- LLM API timeout；
- embedding timeout；
- vector DB timeout；
- tool call timeout；
- browser / code execution timeout。

没有 timeout，系统会被慢请求拖死。

### 7.2 Retry

Retry 必须谨慎。重试不是免费午餐。

风险：

- 重复扣费；
- 重复写入；
- 重复发送消息；
- 重复执行危险工具。

解决：

- request id；
- 幂等 key；
- 去重表；
- 状态机；
- 对副作用操作禁用自动重试；
- 重试使用指数退避。

### 7.3 Idempotency

幂等性是稳定性工程核心。

```text
同一个请求执行一次和执行多次，最终结果一致。
```

比如：

```text
create_task(request_id=abc)
```

如果请求超时，客户端重试，服务端发现 `request_id=abc` 已处理，就直接返回原结果，而不是创建两个任务。

### 7.4 限流、熔断、降级

- **限流**：保护系统不被打爆；
- **熔断**：依赖服务持续失败时暂时停止调用；
- **降级**：核心路径保住，非核心能力关闭。

Agent 平台降级例子：

- rerank 服务挂了，退化为 vector top-k；
- 高级模型限流，切到便宜模型；
- browser tool 失败，退化为 search snippet；
- memory 检索超时，跳过长期记忆；
- evaluator 超时，返回带风险提示的结果。

### 7.5 SLO 和核心指标

Agent 平台可以定义：

| 指标 | 含义 |
|---|---|
| Availability | 服务可用性 |
| P95 / P99 latency | 尾延迟 |
| Time to first token | 首 token 延迟 |
| Task success rate | 任务成功率 |
| Tool success rate | 工具调用成功率 |
| Retrieval hit rate | 检索命中率 |
| Cache hit rate | 缓存命中率 |
| Cost per task | 单任务成本 |
| Token usage | token 消耗 |
| Error budget | 错误预算 |

## 8. 分布式系统与 Workflow

Agent Infra 本质上是分布式系统。

### 8.1 任务状态机

一个 workflow step 可以有状态：

```text
PENDING -> RUNNING -> SUCCEEDED
                 -> FAILED
                 -> TIMEOUT
                 -> CANCELED
```

状态必须持久化，这样 worker crash 后可以恢复。

### 8.2 Worker 挂了怎么办

方案：

- heartbeat 检测 worker 是否活着；
- lease 超时后任务重新入队；
- step 输出持久化；
- 幂等 step 可以重试；
- 非幂等 step 需要人工确认或补偿。

### 8.3 Checkpoint

长任务不能失败后从头开始。

例如 DeepScientist：

```text
已完成：搜索论文、解析 PDF、抽取引用
失败：生成 related work
恢复：从 related work 这一步继续
```

这就是 checkpoint 的价值。

### 8.4 多 Agent 协作

多 Agent 系统可以看成：

```text
coordinator
  -> planner
  -> researcher workers
  -> coder worker
  -> reviewer worker
  -> aggregator
```

分布式问题：

- 多 worker 并发；
- 共享状态冲突；
- slow worker；
- worker 输出不稳定；
- 结果聚合；
- 失败恢复。

## 9. 项目包装：DeepScientist 怎么讲

如果你有 DeepScientist 项目，可以这样包装。

### 9.1 一句话介绍

> DeepScientist 是一个面向科研任务的 multi-agent workflow 系统，本质上是一个 mini distributed system。它把复杂科研问题拆成检索、阅读、总结、写作、评估等可恢复步骤，并通过 RAG 和 memory 管理中间知识。

### 9.2 架构

```text
User Query
  -> Planner
  -> Search Workers
  -> PDF / Web Parser Workers
  -> Memory / Vector Store
  -> Draft Writer
  -> Critic / Evaluator
  -> Final Aggregator
```

### 9.3 Infra 点

- Planner 拆任务；
- Worker 执行工具调用；
- RAG 检索文献；
- Memory 保存中间证据；
- Workflow 记录 step 状态；
- Evaluator 检查输出质量；
- Aggregator 汇总最终报告。

### 9.4 稳定性点

- tool call timeout；
- search / parser 失败重试；
- LLM 调用失败 fallback；
- 中间结果 checkpoint；
- trace 记录每一步耗时；
- 缓存搜索和 embedding 结果；
- 对长文本做截断和压缩；
- 对外部 API 做限流。

### 9.5 面试表达

可以这样说：

> 我一开始把它当成 LLM 应用做，但后来发现真正难点在 Infra：多步骤任务状态管理、外部工具不稳定、RAG latency、token 成本、失败恢复和可观测性。因此我把它抽象成 workflow + worker + memory + evaluator 的架构。

这个表达比“我做了一个调用大模型写报告的项目”强很多。

## 10. 高频系统设计题

### 10.1 设计一个企业内部 Agent 平台

要点：

- 多租户鉴权；
- tool registry；
- workflow runtime；
- memory / RAG；
- sandbox；
- observability；
- evaluation；
- model router；
- cost control。

### 10.2 Agent 调工具超时怎么办

回答结构：

1. 每个工具配置 timeout；
2. 超时后根据工具类型决定是否重试；
3. 幂等工具可 retry；
4. 有副作用工具不自动 retry；
5. 记录 tool call trace；
6. 返回可解释错误或降级结果；
7. 必要时人工确认。

### 10.3 RAG 检索慢怎么排查

按链路拆：

```text
embedding -> vector search -> metadata filter -> fetch chunk -> rerank -> context packing
```

逐段看耗时和 P99。

### 10.4 Agent 死循环怎么办

- 最大 step 数；
- 最大 token budget；
- evaluator 判断是否继续；
- 重复 action 检测；
- planner 重新规划；
- 人工接管；
- trace 分析循环原因。

### 10.5 如何做 Agent 可观测性

记录：

- request id；
- user id / tenant id；
- model name；
- prompt hash；
- token usage；
- tool call input/output；
- retrieval top-k；
- rerank scores；
- step latency；
- error stack；
- final status。

## 11. 高频八股题清单

### 11.1 AI Infra

- prefill 和 decode 区别是什么？
- KV cache 是什么，为什么占显存？
- vLLM 的 PagedAttention 解决什么问题？
- streaming generation 优化了什么？
- RAG 和 fine-tuning 的区别？
- rerank 为什么能提升效果？
- embedding cache 怎么设计？
- prompt cache 和 KV cache 有什么区别？

### 11.2 稳定性

- timeout 和 retry 怎么设计？
- 什么是幂等？
- 如何防止重复执行副作用操作？
- 熔断和限流区别是什么？
- 什么是 SLO / SLA / error budget？
- 如何排查 P99 latency？
- 服务雪崩怎么处理？
- 灰度发布怎么做？

### 11.3 分布式

- MapReduce 的核心思想是什么？
- Raft 为什么需要 majority？
- leader 挂了怎么办？
- task queue 如何保证任务不丢？
- worker crash 后任务如何恢复？
- checkpoint 和 snapshot 区别？
- eventual consistency 和 strong consistency 区别？

### 11.4 数据库/RAG

- B+ Tree 和 LSM Tree 区别？
- Vector DB 为什么需要 ANN？
- HNSW 的基本思想？
- metadata filter 先做还是后做？
- cache invalidation 怎么处理？
- LLM memory 为什么是数据库问题？

## 12. 7 天冲刺计划

### Day 1：Agent Infra 架构

目标：能画出 Agent 平台架构图。

准备：

- Agent runtime；
- tool registry；
- workflow；
- memory；
- evaluator；
- sandbox；
- observability。

输出：一张“企业 Agent 平台系统设计图”。

### Day 2：RAG / Vector DB / Memory

目标：能讲清 retrieval latency。

准备：

- chunk；
- embedding；
- ANN；
- metadata filter；
- rerank；
- cache；
- memory schema。

输出：一套 RAG 优化 checklist。

### Day 3：LLM Serving

目标：能解释 vLLM 为什么快。

准备：

- prefill/decode；
- KV cache；
- batching；
- streaming；
- memory bound；
- PagedAttention。

输出：一页 LLM serving 笔记。

### Day 4：稳定性工程

目标：能回答线上故障治理。

准备：

- timeout；
- retry；
- idempotency；
- rate limit；
- circuit breaker；
- fallback；
- SLO；
- monitoring。

输出：一个“工具调用超时”的故障处理方案。

### Day 5：分布式系统

目标：能把 Agent 讲成 distributed system。

准备：

- task queue；
- worker heartbeat；
- checkpoint；
- Raft 概念；
- workflow engine；
- idempotent task。

输出：DeepScientist = mini distributed system 的讲稿。

### Day 6：项目包装

目标：把 DeepScientist 包装成 Agent Infra 项目。

准备：

- 项目背景；
- 架构图；
- 技术难点；
- 稳定性设计；
- 性能优化；
- 可观测性；
- 反思改进。

输出：3 分钟项目介绍 + 10 分钟项目深挖。

### Day 7：模拟面试

目标：熟练输出。

准备：

- 3 个系统设计题；
- 20 个八股题；
- 1 个项目深挖；
- 1 个线上故障排查题。

输出：录音复盘，修改表达。

## 13. 面试自我介绍模板

可以这样说：

> 面试官您好，我是古恩豪。我主要关注 AI Infra 和 Agent 系统工程，最近系统学习了深度学习框架、GPU 推理加速、分布式系统和数据库/RAG。项目上我做过一个 DeepScientist，多 Agent 科研工作流系统，里面涉及任务拆解、工具调用、RAG 检索、memory 管理、失败重试和中间状态持久化。我对如何把 Agent 从 demo 做成稳定、可观测、可恢复的平台能力比较感兴趣，也希望在 Agent Infra 和稳定性工程方向深入实践。

## 14. 面试反问问题

可以反问：

1. 团队现在的 Agent Infra 更偏 runtime、tool platform，还是 serving / evaluation？
2. 稳定性工程在 Agent 场景里最关注哪些指标？
3. 目前 Agent 平台最大的技术挑战是 tool reliability、latency、cost，还是 observability？
4. 实习生进去后会参与平台建设、稳定性治理，还是具体业务 Agent 落地？
5. 团队技术栈主要是 Java、Go、Python，还是混合？

这些问题能体现你真的理解岗位，而不是泛泛求职。

## 15. 最后总结

Agent Infra 工程师不是单纯的大模型应用开发，而是 AI 时代的平台工程岗位。核心能力是把 Agent 应用做成可靠系统：可编排、可恢复、可观测、可扩展、可降级、可控成本。

准备时要始终围绕四个关键词：

```text
Agent Runtime
RAG / Memory
LLM Serving
Reliability Engineering
```

面试表达上，不要只说“我会用 LangChain”或“我会调 API”。要尽量说：我理解 Agent 系统背后的任务调度、状态管理、工具调用稳定性、检索延迟、推理服务瓶颈和可观测性建设。

这才是 Agent Infra 岗位最想听到的能力。

补充：打印版 PDF 已放在 [/downloads/pdf/alibaba-agent-infra-interview.pdf](/downloads/pdf/alibaba-agent-infra-interview.pdf)，可以直接下载打印。

