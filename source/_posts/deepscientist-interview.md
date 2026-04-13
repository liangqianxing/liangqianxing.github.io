---
title: DeepScientist 项目面经：AI 科研平台的 Agent 设计与工程实践
date: 2026-04-13
categories: 技术
tags:
  - 面试
  - LLM
  - Agent
  - FastAPI
  - SSE
---

在西湖大学 NLP 实验室（张岳教授组）实习期间，我负责 DeepScientist 平台的 AI Copilot 模块开发，包括多轮对话链路、工具调用、SSE 流式输出等核心功能。平台用户 5000+，这篇文章整理面试中关于这个项目的常见问题和回答思路。

<!-- more -->

## 如何介绍这个项目

**背景说明**

DeepScientist 是一个面向科研人员的 AI 辅助平台，核心是帮助研究者完成文献检索、论文分析、实验设计等科研任务。我在其中负责 AI Copilot 模块，也就是平台的智能对话核心。

科研场景和普通问答有本质区别：用户的问题往往需要跨多个步骤才能回答，比如"帮我找最近三年关于 diffusion model 加速的论文，总结一下主流方法"——这不是一次检索能搞定的，需要 Agent 拆解任务、调用工具、整合结果。

**内容介绍**

整个 AI Copilot 模块分三层：

- **对话管理层**：多轮上下文维护、会话隔离、历史摘要压缩
- **Agent 执行层**：基于 ReAct 模式的工具调用链路，支持文献检索、论文解析、代码执行等工具
- **输出层**：SSE 流式输出，前端实时渲染，支持 Markdown、代码块、引用卡片

**要点**

三个核心技术点：ReAct 工具调用、SSE 流式输出、多轮对话上下文管理。

**价值**

- 用户从"手动检索+人工整理"变成"一句话描述需求，Agent 自动完成"
- 平台 DAU 提升，用户平均会话时长从 3 分钟增加到 12 分钟
- 复杂科研任务的完成率从 40% 提升到 75%

---

## 热门面试题

### 简单介绍一下这个项目

DeepScientist 是西湖大学 NLP 实验室开发的科研 AI 平台，我负责其中 AI Copilot 模块的核心链路开发。

科研场景的特点是任务复杂、需要多步推理。比如用户问"帮我分析这篇论文的方法论并和最新工作对比"，这需要先解析 PDF、再检索相关论文、再做对比分析，三步缺一不可。

我基于 ReAct 模式设计了工具调用链路，让 Agent 能够拆解任务、按步骤调用工具（文献检索、PDF 解析、代码执行等），最终整合结果输出。输出层用 SSE 实现流式推送，用户能实时看到 Agent 的思考和输出过程，体验接近 ChatGPT。

平台目前用户 5000+，复杂任务完成率从 40% 提升到 75%。

---

### 你在项目里是什么角色

AI Copilot 模块从 0 到 1 由我负责，包括：

- 后端：FastAPI + 异步架构，对话管理、工具注册、Agent 执行循环
- 流式输出：SSE 链路设计，前后端协议定义
- 工具开发：文献检索工具、PDF 解析工具、代码执行沙箱接口
- Prompt 工程：系统 Prompt 设计、工具描述优化、输出格式约束

---

### 你提到了 ReAct，详细说说

ReAct 是 Reasoning + Acting 的结合，核心是让大模型像人一样"先想再做"。

传统问答模式：用户问 → 模型直接答。遇到需要外部信息的问题就只能靠训练数据，容易幻觉。

ReAct 模式：用户问 → 模型思考（需要什么信息）→ 调用工具 → 观察结果 → 继续思考 → 最终回答。

举个 DeepScientist 里的例子：

```
用户：帮我找 2024 年关于 LoRA 微调效率的论文，总结主要贡献

第1轮：
  思考：需要检索 2024 年 LoRA 相关论文
  行动：调用 search_papers(query="LoRA fine-tuning efficiency 2024", limit=10)
  观察：返回 8 篇论文，含标题、摘要、引用数

第2轮：
  思考：有了论文列表，需要逐篇分析主要贡献
  行动：调用 parse_abstract(paper_ids=[...])
  观察：返回结构化的方法摘要

第3轮：
  思考：信息足够，可以整合输出
  输出：综合分析报告
```

ReAct 的关键优势是**动态决策**——Agent 不是按固定流程走，而是根据每步的结果决定下一步做什么。这对科研场景非常重要，因为用户的问题往往没有固定的解法路径。

---

### SSE 是什么，为什么用 SSE 而不是 WebSocket

SSE（Server-Sent Events）是一种服务器向客户端单向推送数据的协议，基于 HTTP 长连接。

**为什么选 SSE 而不是 WebSocket：**

| 维度 | SSE | WebSocket |
|------|-----|-----------|
| 方向 | 单向（服务器→客户端） | 双向 |
| 协议 | HTTP | WS |
| 断线重连 | 浏览器原生支持 | 需要手动实现 |
| 实现复杂度 | 低 | 高 |
| 适用场景 | 流式输出、通知推送 | 实时聊天、游戏 |

AI 对话的输出是单向的——服务器流式推送 token，客户端只需要接收渲染。SSE 完全够用，而且实现更简单，不需要维护 WebSocket 连接状态。

**FastAPI 里的实现：**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def stream_generator(session_id: str, message: str):
    async for chunk in agent.astream(session_id, message):
        # SSE 格式：data: <内容>\n\n
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        stream_generator(request.session_id, request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，否则流式会失效
        }
    )
```

前端用 `EventSource` 接收：

```javascript
const es = new EventSource('/chat/stream');
es.onmessage = (e) => {
    if (e.data === '[DONE]') { es.close(); return; }
    appendToken(e.data);  // 实时渲染
};
```

**踩过的坑：** Nginx 默认会缓冲响应，导致流式输出变成"一次性输出"。需要在响应头加 `X-Accel-Buffering: no` 或在 Nginx 配置里关闭 `proxy_buffering`。

---

### 多轮对话的上下文是怎么管理的

多轮对话有两个核心问题：**上下文窗口有限**和**会话隔离**。

**会话隔离：**

每个用户会话有独立的 `session_id`，对应独立的消息历史。用 Redis 存储，key 是 `session:{session_id}:messages`，TTL 24 小时。

**上下文窗口管理（三层策略）：**

```
第1层：滑动窗口
  保留最近 N 轮完整对话（我设的是 8 轮）
  直接放入 prompt，保证短期连贯性

第2层：摘要压缩
  超过 8 轮的历史，用小模型（haiku）压缩成摘要
  摘要保留：用户核心需求、已解决的问题、重要参数
  10 轮对话 → 3-4 句摘要

第3层：向量检索（可选）
  所有历史向量化存入 pgvector
  用户提到"之前那篇论文"时，语义检索历史
```

**token 预算控制：**

```python
def build_context(session_id: str, new_message: str) -> list[dict]:
    messages = get_recent_messages(session_id, n=8)
    summary = get_session_summary(session_id)
    
    # 估算 token 数
    total_tokens = count_tokens(messages) + count_tokens(summary)
    
    if total_tokens > TOKEN_BUDGET * 0.8:
        # 触发压缩：把最早的几轮压缩进摘要
        messages, summary = compress_oldest(messages, summary)
    
    context = []
    if summary:
        context.append({"role": "system", "content": f"对话摘要：{summary}"})
    context.extend(messages)
    context.append({"role": "user", "content": new_message})
    return context
```

---

### 工具调用怎么设计的，如何保证大模型不调用错误的工具

**工具注册：**

每个工具是一个 Python 函数 + 描述 schema，注册到工具注册表：

```python
@tool(
    name="search_papers",
    description="搜索学术论文。当用户需要查找特定主题的论文时使用。",
    parameters={
        "query": {"type": "string", "description": "搜索关键词，建议用英文"},
        "year_from": {"type": "integer", "description": "起始年份，如 2023"},
        "limit": {"type": "integer", "description": "返回数量，默认 10，最大 20"},
    }
)
async def search_papers(query: str, year_from: int = 2020, limit: int = 10):
    ...
```

**三层保障防止错误调用：**

1. **精确的工具描述**：每个工具的 `description` 写清楚"什么时候用"、"不适合什么场景"，参数写清楚格式约束和枚举值
2. **参数校验**：工具执行前用 Pydantic 校验参数，不合法直接返回错误信息给模型，让它重新调用
3. **重试机制**：调用失败时，把错误信息加入上下文，让 Agent 自动修正参数重试，最多 3 次

---

### 项目的难点是什么

**最难的是流式输出和工具调用的结合。**

普通流式输出很简单，但 ReAct 模式下，Agent 在输出最终答案之前要先调用工具——而工具调用是同步的（需要等结果），这期间前端没有任何输出，用户会以为卡住了。

解决方案是**分阶段流式**：

```
阶段1：流式输出 Agent 的"思考过程"（Thought）
  → 前端实时显示"正在分析您的问题..."

阶段2：工具调用（同步等待）
  → 前端显示"正在检索论文..."（进度提示）

阶段3：流式输出最终答案
  → 前端实时渲染结果
```

SSE 事件类型区分：

```python
# 思考过程
yield f"event: thinking\ndata: {thought}\n\n"

# 工具调用状态
yield f"event: tool_call\ndata: {json.dumps({'tool': name, 'status': 'running'})}\n\n"

# 工具结果
yield f"event: tool_result\ndata: {json.dumps({'tool': name, 'status': 'done'})}\n\n"

# 最终答案 token
yield f"event: answer\ndata: {token}\n\n"
```

前端根据事件类型渲染不同 UI，用户全程有反馈，不会感觉卡顿。

---

### 如何避免大模型幻觉

科研场景对准确性要求很高，幻觉是最大的风险。

**四层防护：**

1. **Prompt 约束**：系统 Prompt 明确要求"只基于工具返回的信息回答，不允许使用训练数据中的论文细节"
2. **来源标注**：要求模型在回答中标注每个论点的来源（哪篇论文、哪个工具返回的）
3. **相似度阈值**：RAG 召回时设置 0.75 的相似度阈值，低于阈值直接告知用户"未找到相关文献"
4. **用户反馈**：界面有 👎 按钮，点击后记录 session_id 和消息 id，我会人工 trace 日志分析原因

---

### 如果重新设计，你会改什么

1. **引入评测基准**：目前靠用户反馈，不够系统化。应该建立一套科研问答的 benchmark，定期跑评测
2. **工具结果缓存**：同一个检索请求在短时间内可能被多次调用，加 Redis 缓存能显著降低延迟和 API 成本
3. **Agent 可观测性**：每次工具调用的耗时、token 消耗、召回相似度都应该有 trace，方便性能分析
4. **多模态支持**：科研场景经常需要分析图表，接入多模态模型让 Agent 能理解论文里的图

---

## 面试节奏建议

介绍项目时按这个顺序：**背景痛点 → 解决方案 → 技术亮点 → 业务价值**，每段控制在 30 秒左右，主动在技术亮点处停顿，给面试官提问的机会。

面试官最感兴趣的点通常是：SSE 流式实现细节、ReAct 工具调用链路、多轮对话上下文管理。这三个准备好，基本能应对 80% 的追问。
