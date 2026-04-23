---
title: LLM 上下文五层压缩机制详解
date: 2026-04-15
categories: 技术
tags:
  - LLM
  - Agent
  - 上下文压缩
  - 面试
---

做 Agent 项目时，对话持续进行，token 会不断累积，迟早超出模型的 context window。这篇文章整理一套五层上下文压缩机制，从轻到重依次触发，核心思路是"能少压就少压，实在不行再大压"。

<!-- more -->

## 整体逻辑

```
对话持续进行 → Token 接近上限 → 触发压缩 → 压缩成功继续 → 压缩失败升级到下一层
```

五层从左到右，压缩力度递增，成本也递增：

```
Tool Result Replace → Snip Compact → Micro Compact → Context Collapse → Auto Compact
      极低成本              低               低              高              最高
```

---

## 第一层：Tool Result Replace

**做什么：** 把工具调用的返回结果替换掉，只保留占位符。

```python
# 原来
tool_result = "这是一段很长的日志输出，有几千行..."

# 替换后
tool_result = "[已压缩，原始结果已清除]"
```

工具返回了大量原始数据（日志、代码、搜索结果），但后续对话其实不需要完整内容，只需要知道"调用过这个工具、拿到了结果"。

**成本：** 极低，只是字符串替换，不需要调用 LLM。

---

## 第二层：Snip Compact

**做什么：** 精准裁剪，找到对话里最占空间的片段，把它们缩短，但保留上下文结构。

比如一个很长的 `tool_result` 块，只保留前几行 + 末尾摘要，中间截断：

```
[原始内容前100行]
... [中间 2000 行已截断] ...
[原始内容最后20行]
```

**适用场景：** 有几个特别大的消息拖累了整体 token 数，其他消息都正常。

**成本：** 低，基于规则裁剪，不需要 LLM。

---

## 第三层：Micro Compact

**做什么：** 清理掉已经没用的工具调用记录。

比如 `cache_read`、`cache_write` 这类中间过程的工具调用，任务完成后这些记录对后续对话没有价值，直接标记为 `[tool cleared]` 删掉。

```
# 删除前
[tool_use: cache_write, params: {...}, result: "ok"]
[tool_use: cache_read, params: {...}, result: "...大量内容..."]

# 删除后
[tool cleared]
[tool cleared]
```

**适用场景：** 对话里有大量中间步骤的工具调用，但最终结果已经体现在后续消息里了。

**成本：** 低，规则删除，不需要 LLM。

---

## 第四层：Context Collapse

**做什么：** 不再保留原始消息，把整段历史对话喂给 LLM，让它生成一个 summary，用 summary 替代原始历史。

```python
def context_collapse(history):
    prompt = f"""
请将以下对话历史压缩为简洁摘要，保留：
- 用户的核心目标和需求
- 已完成的重要操作
- 关键决策和结论
- 当前任务状态

对话历史：
{format_history(history)}
"""
    summary = llm.call(prompt)
    
    # 用摘要替代原始历史
    return [{"role": "system", "content": f"[历史摘要]\n{summary}"}]
```

```
原来：[20轮完整对话，8000 token]
压缩后：[1条 summary 消息，500 token]
```

**适用场景：** 前三层都压不下去，token 还是超限。

**成本：** 高，需要调用一次 LLM 生成摘要，有延迟和费用。

**注意：** 这一层只压 Model-Facing 部分（发给模型的消息），不压 Raw Messages（原始记录），保留审计能力。

---

## 第五层：Auto Compact

**做什么：** 完全重建上下文。生成一个新的 compact prompt 作为全新对话的起点，原始历史全部丢弃。

新的起点包含三部分：

```python
compact_prompt = {
    "boundary": "当前任务边界和约束",      # 告诉模型它在做什么
    "summary": "历史摘要",                 # 之前发生了什么
    "assistant_context": "必要的角色设定"  # 模型应该以什么状态继续
}
```

**内部还分两档：**

- **Session Memory Compact（轻量）：** 用已有的 session memory 替代，成本低，但信息可能不完整，适合对话内容不太重要的场景
- **Full Compact（兜底）：** 完整重建，成本最高，但保证能继续运行，是最后的兜底手段

**适用场景：** 对话极长，前四层都无法把 token 压到窗口以内。

---

## 机制设计要点

**容量精准计算**

不用固定阈值，而是动态计算可用窗口：

```python
def get_effective_context_window():
    total = model.context_window          # 模型总窗口
    reserved_output = max_output_tokens   # 预留给输出的空间
    reserved_system = system_prompt_size  # 系统 prompt 占用
    return total - reserved_output - reserved_system
```

**缓存优先策略**

压缩时优先保留 `cache_read`/`cache_write` 相关内容，避免缓存失效导致重复计算费用。

**结构完整性保障**

压缩后保证 thinking/tool_use 配对完整，不出现孤立的工具调用（没有对应 tool_result 的 tool_use 会导致 API 报错）。

**失败兜底**

每层失败都有明确的升级路径：

```
第1层失败 → 升级到第2层
第2层失败 → 升级到第3层
...
第5层失败 → 报错，人工介入
```

---

## 五层对比

| 层级 | 方式 | 成本 | 信息损失 | 触发条件 |
|------|------|------|---------|---------|
| Tool Result Replace | 替换工具结果为占位符 | 极低 | 低 | 工具结果过大 |
| Snip Compact | 裁剪大消息中间部分 | 低 | 低 | 单条消息过大 |
| Micro Compact | 删除无用工具调用记录 | 低 | 极低 | 中间步骤堆积 |
| Context Collapse | LLM 生成历史摘要 | 高 | 中 | 前三层不够用 |
| Auto Compact | 完全重建上下文 | 最高 | 高 | 最后兜底 |

---

## 实际项目怎么选

大多数 Agent 项目不需要实现全部五层，按场景选：

**短会话 Agent（每次对话独立，如运维告警）：** 滑动窗口就够，不需要压缩机制。

**中等长度对话（如代码助手、问答）：** 实现第1层（Tool Result Replace）+ 第4层（Context Collapse）基本够用，覆盖了 80% 的场景。

**超长会话（如长期任务执行、个人助手）：** 需要完整五层，或者引入向量库做长期记忆。

面试被问到"你的 Agent 怎么处理长对话"，不要只说"我用了摘要压缩"，说清楚**在哪一层触发、为什么不需要更重的层**，才能体现你对整个方案空间的理解。
