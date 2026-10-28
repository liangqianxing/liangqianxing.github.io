---
title: AI Infra 工程路线：从后端基础到 Agent 平台
date: 2026-07-09 10:00:00
description: 一份面向工程同学的 AI Infra 学习路线，覆盖后端基础、RAG、Agent Runtime、模型服务、评测观测、稳定性工程和项目落地。
categories:
  - 技术
tags:
  - AI Infra
  - Agent
  - RAG
  - 后端架构
  - 稳定性工程
  - 面试
hidden: true
haloPublished: true
---

这篇文章整理一条面向工程同学的 **AI Infra 路线**。它不是“会调大模型 API”的路线，而是从后端基础出发，逐步走到 RAG、Agent Runtime、模型服务、评测系统、观测平台和稳定性工程。

如果只用一句话概括：

> AI Infra 的核心不是把 LLM 接进项目，而是把模型能力做成稳定、可扩展、可评测、可观测、可恢复的平台能力。

适合的人：

- 有后端、数据、平台、算法工程基础，想转向 AI Infra；
- 做过 RAG / Agent demo，但想把它做成系统；
- 准备 Agent Infra、LLM Infra、AI 平台、模型应用工程相关面试；
- 想知道从项目到岗位能力之间还差哪些模块。

---

## 0. 先建立岗位画像

AI Infra 不是一个单点岗位，而是一组工程能力的交集：

| 方向 | 典型问题 |
|---|---|
| RAG Infra | 文档如何切分、索引、召回、重排、拼上下文、评估效果 |
| Agent Infra | Agent loop 如何运行，工具如何注册，状态如何恢复 |
| Model Serving | 模型如何部署、路由、限流、批处理、缓存、降成本 |
| Evaluation | 如何离线评测、线上回归、构造测试集、判断答案质量 |
| Observability | 如何看每次请求的 trace、token、延迟、失败原因 |
| Reliability | 超时、重试、熔断、降级、幂等、SLO、故障恢复 |
| Platform | 如何把能力封装成 SDK、API、控制台和多租户平台 |

真正的 AI Infra 工程师要能回答两个问题：

1. **能力怎么做出来**：检索、推理、工具调用、记忆、评测、观测。
2. **系统坏了怎么办**：超时、失败、幻觉、成本失控、链路不稳定、线上回归。

---

## 1. 第一阶段：后端与系统基础

AI Infra 最底层还是后端系统。没有后端基础，RAG 和 Agent 都很容易停留在 demo。

### 必须掌握

- HTTP / SSE / WebSocket / gRPC；
- MySQL / PostgreSQL 的索引、事务、隔离级别、慢查询；
- Redis 的缓存、分布式锁、限流、队列、热点 key；
- Kafka / RabbitMQ 的消息模型、消费位点、重试、死信队列；
- Docker / Compose / 基础部署；
- 日志、指标、Trace、告警；
- 并发模型、连接池、线程池、异步 I/O。

### 推荐项目

做一个“AI 文档问答后端”：

```text
用户上传文档
  -> 后端解析文件
  -> 切分 chunk
  -> embedding
  -> 写入向量库 / PostgreSQL
  -> 查询时召回相关 chunk
  -> 拼接 prompt
  -> SSE 流式返回答案
```

这个项目不难，但能把 AI Infra 的第一层链路串起来。

### 面试表达

不要只说：

> 我做过一个 RAG 项目。

更好的表达：

> 我做过文档问答系统，链路包括文件上传、异步解析、chunk 切分、embedding、向量检索、上下文拼接和 SSE 流式返回。后端侧重点是任务异步化、索引更新一致性、检索延迟控制和失败重试。

---

## 2. 第二阶段：RAG Infra

RAG 是 AI Infra 最常见的入口。很多人以为 RAG 就是“向量库 + top-k”，但真实系统复杂得多。

### 学习主线

```text
Document Parsing
  -> Chunking
  -> Embedding
  -> Vector Index
  -> Sparse Retrieval
  -> Hybrid Retrieval
  -> Rerank
  -> Context Packing
  -> Generation
  -> Evaluation
```

### 关键问题

| 模块 | 要理解的问题 |
|---|---|
| 文档解析 | PDF、HTML、Markdown、图片、表格如何抽取结构 |
| Chunking | 按长度切、按标题切、语义切分，各有什么问题 |
| Embedding | 向量维度、归一化、批处理、缓存、版本管理 |
| 向量库 | HNSW、IVF、metadata filter、索引构建与更新 |
| BM25 | 为什么稀疏检索仍然重要，如何处理精确关键词 |
| Hybrid | dense + sparse 如何融合，RRF 为什么常用 |
| Rerank | cross-encoder / LLM rerank 如何提升精度 |
| Context | 如何压缩、去重、排序、控制 token 预算 |
| Eval | 如何构造 query、ground truth、bad case 和回归集 |

### 推荐实现顺序

1. 先写一个纯文本 RAG。
2. 加 BM25，做 hybrid retrieval。
3. 加 reranker，对比 Recall@k / MRR@k。
4. 加 metadata filter，例如文档类型、时间、作者、权限。
5. 加评测脚本，不再靠肉眼判断效果。
6. 加 trace，把每次召回的 chunk、分数、prompt、token 都记录下来。

### 能做出来的项目

可以做一个 **RAG Evaluation Lab**：

- 支持导入文档；
- 支持 dense / BM25 / hybrid / rerank 多策略；
- 每次 query 展示召回结果；
- 支持 Recall@k、MRR、人工标注；
- 支持 bad case 保存和回归测试。

这个项目比普通“文档问答 demo”更能体现 AI Infra 味道。

---

## 3. 第三阶段：Agent Runtime

Agent Infra 的难点不是“让模型调用工具”，而是让它可控、可恢复、可审计。

### Agent Runtime 要解决什么

一个 Agent 请求通常不是一次 LLM call，而是一个循环：

```text
User Task
  -> Planner
  -> LLM decides action
  -> Tool call
  -> Tool result
  -> Update state
  -> Continue / Stop
```

里面每一步都会出问题：

- 模型输出了非法参数；
- 工具调用超时；
- 工具结果太长；
- Agent 死循环；
- 中途 worker crash；
- 用户取消任务；
- 多个工具有权限边界；
- 任务执行到一半需要恢复。

### 必须掌握的模块

| 模块 | 关键设计 |
|---|---|
| Tool Registry | 工具描述、参数 schema、权限、超时、重试 |
| State Machine | pending / running / failed / paused / completed |
| Workflow | DAG、步骤依赖、失败恢复、人工确认 |
| Memory | 会话历史、任务状态、长期记忆、用户画像 |
| Sandbox | 文件、浏览器、代码执行、网络访问隔离 |
| Guardrails | 参数校验、输出约束、敏感操作确认 |
| Trace | 每一步输入、输出、耗时、token、错误 |

### 推荐项目

做一个 **Agent Task Runner**：

- 用户输入一个任务；
- Agent 可以调用搜索、文件读写、代码执行等工具；
- 每一步都落库；
- 支持暂停、恢复、失败重试；
- 支持工具调用 trace；
- 支持最大轮数和成本预算。

如果能把这个项目做好，面试 Agent Infra 会非常有说服力。

---

## 4. 第四阶段：Model Serving 与成本优化

很多 AI 应用上线后最先遇到的问题不是效果，而是延迟和成本。

### 要理解的能力

- 模型 API 网关；
- 多模型路由；
- prompt cache；
- embedding cache；
- 语义缓存；
- 批处理；
- 限流与配额；
- token 成本统计；
- fallback 模型；
- 流式输出；
- 超时取消；
- GPU serving 基础。

### 典型设计

```text
Client
  -> AI Gateway
  -> Auth / Quota
  -> Model Router
  -> Cache
  -> Provider Adapter
  -> LLM / Embedding / Reranker
  -> Metrics / Trace / Billing
```

AI Gateway 的价值是把所有模型调用统一管理：

- 哪个业务用了多少 token；
- 哪个模型延迟高；
- 哪些 prompt 命中缓存；
- 哪些请求失败；
- 哪个租户超过配额；
- 出故障时切到哪个备用模型。

---

## 5. 第五阶段：评测与观测

AI Infra 和传统后端最大的区别之一是：输出质量不稳定。

传统接口通常可以写断言：

```text
输入 A -> 输出 B
```

但 LLM 系统更像：

```text
输入 A -> 输出可能对、可能错、可能格式错、可能引用错
```

所以评测和观测必须前置。

### 离线评测

需要准备：

- query 集合；
- ground truth；
- 期望引用文档；
- 答案评分规则；
- bad case 集；
- 回归测试脚本。

指标包括：

- Recall@k；
- MRR / NDCG；
- faithfulness；
- answer relevance；
- citation accuracy；
- tool success rate；
- latency p50 / p95 / p99；
- token cost。

### 在线观测

每次请求至少记录：

- request_id / user_id / tenant_id；
- prompt 版本；
- model 版本；
- retrieved chunks；
- rerank 分数；
- tool calls；
- token usage；
- latency；
- error type；
- fallback 是否触发；
- 用户反馈。

这就是为什么 AI Infra 岗位经常和平台工程、稳定性工程、观测系统绑在一起。

---

## 6. 第六阶段：稳定性工程

Agent / RAG 系统上线后，稳定性问题会比普通 API 更多。

### 常见故障

| 故障 | 处理方式 |
|---|---|
| 模型 API 超时 | 超时控制、重试、fallback、异步任务 |
| 检索为空 | query rewrite、扩大召回、降级提示 |
| 工具调用失败 | 参数校验、重试、人工确认、替代工具 |
| Agent 死循环 | 最大步数、状态检测、停止条件 |
| token 超限 | context compression、摘要、裁剪 |
| 成本飙升 | quota、cache、模型路由、预算中断 |
| 结果幻觉 | citation、faithfulness check、拒答策略 |
| 向量索引延迟 | 异步索引、版本切换、增量更新 |

### 面试可讲的 SLO

可以这样定义：

- RAG 查询 p95 延迟 < 3s；
- Agent 单步工具调用成功率 > 99%；
- 检索服务可用性 > 99.9%；
- 模型调用失败 fallback 成功率 > 95%；
- 单请求 token 成本不超过预算；
- 核心评测集回归不下降。

---

## 7. 推荐学习路径

### 1-2 周：补后端链路

- FastAPI / Go / Node.js 任选一个；
- PostgreSQL + Redis；
- Docker Compose；
- SSE 流式返回；
- 基础日志和错误处理。

### 3-4 周：做 RAG

- 文档解析；
- chunking；
- embedding；
- dense retrieval；
- BM25；
- hybrid retrieval；
- rerank；
- RAG eval。

### 5-6 周：做 Agent Runtime

- tool registry；
- tool schema；
- state machine；
- task persistence；
- retry / timeout；
- trace。

### 7-8 周：做平台化

- AI Gateway；
- model router；
- token usage；
- prompt version；
- quota；
- dashboard；
- offline evaluation。

---

## 8. 简历项目包装

项目不要写成：

> 基于 LangChain 实现文档问答系统。

建议写成：

> 设计并实现一个面向企业知识库的 RAG 平台，支持文档解析、语义切分、dense/BM25 hybrid retrieval、rerank、上下文压缩、SSE 流式问答和离线评测。系统记录完整 query trace，包括召回 chunk、分数、prompt、token、延迟和错误类型，用于 bad case 分析和回归测试。

Agent 项目可以写成：

> 设计并实现 Agent Runtime，支持工具注册、JSON schema 参数校验、工具调用超时重试、任务状态持久化、执行 trace、最大步数限制和失败恢复。通过状态机管理 Agent 执行生命周期，避免死循环和不可恢复任务。

这类表达比“接入了大模型”更接近岗位要求。

---

## 9. 最后总结

AI Infra 的学习路线可以压缩成一句话：

> 先把后端系统做稳，再把 RAG 做准，把 Agent 做可控，把模型调用做便宜，把评测观测做完整，最后用稳定性工程把它变成平台。

不要急着追每一个新框架。真正能拉开差距的是：

- 你是否理解完整链路；
- 你是否能定位 bad case；
- 你是否能解释延迟和成本；
- 你是否能让任务失败后恢复；
- 你是否能把 demo 抽象成平台能力。

这就是 AI Infra 和普通 LLM 应用开发之间的分水岭。
