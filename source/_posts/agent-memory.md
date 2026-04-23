---
title: Agent 对话记忆化：从原理到实现
date: 2026-04-15
categories: 技术
tags:
  - LLM
  - Agent
  - RAG
  - 面试
---

做 Agent 项目绕不开一个问题：大模型本身没有记忆，每次调用都是无状态的。所谓"记忆"，本质上是把历史信息塞进下一次请求的 prompt 里。这篇文章从原理出发，整理三种主流实现方案。

<!-- more -->

## 核心问题

大模型每次调用都是独立的，它不知道你上一轮说了什么。"记忆"的实现原理是这样的：

```
第1轮：[system] + [user: 你好]
第2轮：[system] + [user: 你好] + [assistant: 你好！] + [user: 我叫古恩豪]
第3轮：[system] + [前两轮历史] + [user: 我叫什么？]
                    ↑
              这就是"记忆"——把历史塞进 prompt
```

问题是 history 会无限增长，迟早超出 context window（Claude 200k token，GPT-4 128k token）。所以需要压缩策略。

---

## 方案一：滑动窗口

最简单的方案，只保留最近 N 轮，超出就丢掉最早的。

```python
class ConversationAgent:
    def __init__(self, max_turns=10):
        self.history = []
        self.max_turns = max_turns
    
    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        
        # 超出就截断，只保留最近 N 轮
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]
        
        response = llm.call(messages=[
            {"role": "system", "content": "你是一个助手"},
            *self.history
        ])
        
        self.history.append({"role": "assistant", "content": response})
        return response
```

**优点：** 实现极简，延迟低，token 消耗可控  
**缺点：** 早期信息完全丢失，用户说"你之前提到的那个方案"就找不到了

适合：客服、简单问答，对话之间关联性弱的场景。

---

## 方案二：摘要压缩

超过阈值时，把旧对话喂给 LLM 让它总结，摘要替代原始内容。

```python
class ConversationAgent:
    def __init__(self, max_turns=20):
        self.history = []
        self.summary = ""
        self.max_turns = max_turns
    
    def _compress(self):
        to_compress = self.history[:self.max_turns // 2]
        self.history = self.history[self.max_turns // 2:]
        
        prompt = f"""
已有摘要：{self.summary}

新增对话：
{self._format(to_compress)}

请将以上内容合并更新为简洁摘要，保留关键信息（用户说了什么、决定了什么、重要的上下文）：
"""
        self.summary = llm.call(prompt)
    
    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        
        if len(self.history) > self.max_turns:
            self._compress()
        
        messages = [{"role": "system", "content": "你是一个助手"}]
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"[对话历史摘要]\n{self.summary}"
            })
        messages.extend(self.history)
        
        response = llm.call(messages=messages)
        self.history.append({"role": "assistant", "content": response})
        return response
```

**优点：** 信息损失可控，关键内容保留  
**缺点：** 摘要本身消耗 token，有额外延迟；摘要质量依赖 LLM

这是 Claude Code 本身用的方案——你现在看到的对话 Summary 就是这个机制。

---

## 方案三：分层记忆

三层结构，各司其职：

```
短期记忆（最近5轮）  → 直接放进 prompt，保证对话连贯性
     ↓ 超出阈值
中期记忆（摘要）     → LLM 压缩，作为 system message
     ↓ 全量存储
长期记忆（向量库）   → 历史向量化，按当前问题检索相关片段
```

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class AgentWithMemory:
    def __init__(self):
        self.short_term = []     # 最近5轮，完整保留
        self.summary = ""        # 中期摘要
        self.long_term = []      # 所有历史的向量存储
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def _add_to_long_term(self, turn):
        text = f"用户：{turn['user']}\n助手：{turn['assistant']}"
        vec = self.embedder.encode(text)
        self.long_term.append({"text": text, "vec": vec})
    
    def _retrieve(self, query, top_k=2):
        if not self.long_term:
            return []
        q_vec = self.embedder.encode(query)
        scores = [np.dot(q_vec, item["vec"]) for item in self.long_term]
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.long_term[i]["text"] for i in top]
    
    def chat(self, user_input):
        # 检索相关历史
        relevant = self._retrieve(user_input)
        
        messages = [{"role": "system", "content": "你是一个助手"}]
        if self.summary:
            messages.append({"role": "system", "content": f"[历史摘要] {self.summary}"})
        if relevant:
            messages.append({"role": "system", "content": "[相关历史]\n" + "\n---\n".join(relevant)})
        messages.extend(self.short_term)
        messages.append({"role": "user", "content": user_input})
        
        response = llm.call(messages=messages)
        
        # 更新记忆
        self.short_term.append({"role": "user", "content": user_input})
        self.short_term.append({"role": "assistant", "content": response})
        
        if len(self.short_term) > 10:
            old_user = self.short_term.pop(0)
            old_asst = self.short_term.pop(0)
            self._add_to_long_term({
                "user": old_user["content"],
                "assistant": old_asst["content"]
            })
        
        return response
```

**优点：** 信息损失最小，既保连贯性又能召回远期细节  
**缺点：** 实现复杂，需要维护向量库，每次请求多一次检索

---

## 三种方案对比

| 方案 | 实现复杂度 | 信息损失 | 延迟 | 适合场景 |
|------|-----------|---------|------|---------|
| 滑动窗口 | 极简 | 高（早期全丢） | 最低 | 简单问答、客服 |
| 摘要压缩 | 中等 | 中（关键信息保留） | 中等 | 通用 Agent |
| 分层记忆 | 复杂 | 低 | 较高 | 长期陪伴、个人助手 |

实际项目里三层通常叠加用：**短期用滑动窗口保连贯性，中期用摘要保关键信息，长期用向量库按需召回**。

---

## 面试怎么答

面试官问"你的 Agent 怎么处理长对话"，不要只说"我用了分层记忆"，要说清楚**为什么选这个方案**：

- 对话 Agent（问答场景）：摘要压缩够用，分层记忆成本高但收益有限
- 运维 Agent（每次告警是独立会话）：滑动窗口就够，不需要跨会话记忆
- 个人助手（需要记住用户偏好）：分层记忆，向量库存用户画像

选型理由比实现细节更能体现思考深度。
