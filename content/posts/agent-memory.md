---
title: Agent 上下文工程与分层记忆架构：从 Observation 剪枝、滚动摘要到向量召回闭环
date: 2026-04-24
description: 结合生产级多约束 Agent 与超 100K Token 科研智能体实践，系统拆解长程会话中的上下文膨胀与注意力稀释问题：涵盖 Observation 结构化剪枝（降低约 40% Token 开销）、滚动工作摘要、向量检索长期记忆及 Prefix Cache 友好的上下文组装。
series: Agent 上下文工程
seriesOrder: 1
categories:
  - 技术
tags:
  - LLM
  - Agent
  - RAG
  - AI Infra
hidden: false
haloPublished: true
---

在构建多步工具调用（Tool-Use）与长周期自主规划（Autonomous Planning）的 Agent 系统时，**上下文管理（Context Engineering）**往往比单纯改写 System Prompt 更决定线上稳定性。大语言模型的每一次推理调用本质上都是无状态的，所谓“智能体的记忆与状态保持”，完全依赖于每一轮请求前如何向有限的上下文窗口组装历史交互与环境观测（Observation）。

即使当前主流模型普遍支持 128K～200K Token 的上下文窗口，如果不加节制地将每一轮工具返回的原始 JSON、终端完整日志或长文档直接追加到消息数组中，系统会在十余步交互后迅速遭遇三重瓶颈：

1. **注意力稀释（Lost-in-the-Middle）與指令遵循退化**：冗余工具字段淹没核心任务约束，导致模型在后续步骤出现实体混淆、参数漏传或时区/计费口径幻觉；
2. **首字延迟（TTFT）与推理成本非线性飙升**：Prefill 阶段的计算量与 KV Cache 显存占用随序列长度增长，长尾请求极易触发网关超时；
3. **破坏 Prefix Caching 命中率**：如果在消息序列头部频繁插入或修改非确定性内容，会导致底层推理引擎（如 SGLang RadixAttention / vLLM Prefix Caching）无法复用历史前缀 KV 块。

本文结合我在生产级多约束 Agent 与长上下文科研智能体中的工程实践，系统梳理**从单步 Observation 剪枝到三层分层记忆（短期滑动窗口 + 中期结构化摘要 + 长期向量/文件召回）**的设计原理与代码实现。

---

## 一、第一道防线：Observation 状态剪枝（入窗前降噪）

在实际生产场景（例如跨境酒旅多工具编排或代码沙箱执行）中，上下文膨胀的头号元凶通常不是用户的几句话，而是**工具返回的巨型 Observation 载荷**。一次房态/报价查询接口可能返回数十个多语言房型描述、冗长的税费拆分明细、无用的 HTML 样式标签以及长达数千字的退改政策原文；一次 `pytest` 或编译命令可能输出上千行堆栈。

如果在记忆层才去做事后压缩，第一轮超长 Observation 已经污染了当前轮次的注意力并消耗了大量 Prefill 算力。因此，最高效的策略是在**工具返回结果进入消息历史之前**执行结构化剪枝（Observation Pruning）：

```python
from typing import Any

def prune_hotel_rate_observation(raw_payload: dict[str, Any], target_locale: str = "zh-CN") -> dict[str, Any]:
    """裁剪高噪声接口返回字段，仅保留规划与决策强相关的结构化骨架（线上实测节省约 40% 单轮上下文 Token）。"""
    pruned_rooms = []
    for room in raw_payload.get("room_list", []):
        # 1. 剥离非目标语种的冗余长文本描述与营销文案
        localized_name = room.get("i18n_names", {}).get(target_locale) or room.get("default_name")
        
        # 2. 将冗长的退改原始条款折叠为结构化时间窗与罚金摘要
        cancel_policy = room.get("cancellation_policy", {})
        compact_cancel = {
            "free_cancel_until_utc": cancel_policy.get("free_before_utc"),
            "penalty_type": cancel_policy.get("penalty_rule_code"),
        }
        
        # 3. 聚合细碎税费项，仅保留决策必需的含税总价、币种与必缴到店税状态
        price_info = room.get("price_breakdown", {})
        pruned_rooms.append({
            "room_id": room["room_id"],
            "name": localized_name,
            "total_inclusive": price_info.get("total_amount_inclusive"),
            "currency": price_info.get("currency"),
            "pay_at_property_fee": price_info.get("resort_fee_amount", 0.0),
            "cancellation": compact_cancel,
        })
    return {
        "hotel_id": raw_payload.get("hotel_id"),
        "timezone": raw_payload.get("hotel_timezone"),
        "rooms": pruned_rooms,
    }
```

通过将“多语言营销描述、调试追踪 ID、UI 渲染样式字段与未命中的长尾规则”在入窗前剥离，仅保留带明确时区与币种标注的紧凑 JSON，单轮 Observation 的 Token 消耗可直接降低 **约 40%**，同时显著减少了模型在跨时区退改换算与计费口径上的幻觉。

---

## 二、三种跨轮记忆策略的演进与权衡

完成单步 Observation 降噪后，随着对话与工具调用轮数持续推进，历史消息仍会逼近上下文预算上限。工程上主流的跨轮记忆管理可分为三个演进阶段：

### 1. 策略一：带完整工具对约束的滑动窗口（Turn-Aligned Sliding Window）

最基础的方案是仅保留最近 $N$ 轮交互。但在包含 Tool-Use 的 Agent 循环中，**绝对不能按单纯的消息条数（`messages[-k:]`）粗暴截断**——如果截断位置恰好把某次 `assistant` 发起的 `tool_calls` 丢弃，而把紧随其后的 `role: "tool"` 结果留在了窗口开头，绝大多数模型 API 都会直接抛出协议校验错误（Orphaned Tool Result）。

因此，滑动窗口必须以**完整的交互原子单元（User Turn 或完整的 `Assistant ToolCall + Tool Result` 闭环）**为边界进行淘汰：

- **优势**：零额外模型调用开销，延迟最低，Token 上界严格可控；
- **局限**：一旦早期轮次滑出窗口，任务最初设定的全局约束（如预算上限、禁用某类外部接口、已排除的错误分支）会彻底丢失。

### 2. 策略二：增量工作状态摘要（Incremental Working-State Compaction）

当历史消息累积超过水位线（例如达到窗口预算的 75%）时，将最早的一批完整交互切片提取出来，合并到一段结构化的**工作状态摘要（Working Summary）**中，并用该摘要替换原始消息切片：

```python
class IncrementalSummaryMemory:
    def __init__(self, max_raw_turns: int = 16, compact_batch: int = 8):
        self.summary_state: str = ""
        self.active_turns: list[dict] = []
        self.max_raw_turns = max_raw_turns
        self.compact_batch = compact_batch

    def append_turn(self, turn_messages: list[dict], summarizer_llm) -> None:
        self.active_turns.append(turn_messages)
        if len(self.active_turns) > self.max_raw_turns:
            evicted = self.active_turns[:self.compact_batch]
            self.active_turns = self.active_turns[self.compact_batch:]
            self.summary_state = self._merge_summary(evicted, summarizer_llm)

    def _merge_summary(self, evicted_turns: list[list[dict]], summarizer_llm) -> str:
        prompt = (
            "你是一个状态归档器。请将【新增归档交互】合并入【当前工作状态台账】，严格保留：\n"
            "1. 用户的核心目标与硬性约束（预算/时间/格式/权限边界）；\n"
            "2. 已确认成功的工具调用结论与关键实体 ID；\n"
            "3. 已验证失败的路径及报错原因（防止后续重复踩坑）。\n\n"
            f"【当前工作状态台账】\n{self.summary_state or '（空）'}\n\n"
            f"【新增归档交互】\n{evicted_turns}"
        )
        return summarizer_llm.generate(prompt)
```

与普通闲聊摘要不同，Agent 的工作摘要必须显式保留**“关键实体精确标识符（ID/路径）”**与**“已证伪的失败路径（Negative Memory）”**，否则模型在摘要后极易重新尝试刚刚失败过的工具调用。

### 3. 策略三：三层混合记忆架构（短期窗口 + 中期台账 + 长期向量/文件按需召回）

对于跨越数十轮甚至跨会话的复杂系统（如科研智能体 DeepScientist 或终端编程引擎 MiniCode），单一摘要仍会在反复重写中丢失细粒度参数或代码片段。此时需要采用**三层分层记忆架构**：

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 静态前缀层 (System Prompt + 工具 Schema + 领域规则) ──► 锁定 Prefix Cache│
├─────────────────────────────────────────────────────────────────────────┤
│ 2. 中期工作台账 (Structured Ledger: 目标约束 / 已完成子图 / 失败避坑记录)   │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. 长期按需召回层 (Vector / Lexical Top-K 历史片段 或 外部 Artifact 索引)  │
├─────────────────────────────────────────────────────────────────────────┤
│ 4. 短期高保真窗口 (最近 K 轮完整 User / Assistant / Pruned Observation)   │
└─────────────────────────────────────────────────────────────────────────┘
```

其核心检索与组装闭环如下：

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class HierarchicalAgentMemory:
    def __init__(self, embed_model: str = "BAAI/bge-small-zh-v1.5", short_window: int = 6):
        self.short_term: list[dict] = []          # 最近 K 轮完整交互
        self.working_ledger: str = ""             # 结构化任务状态台账
        self.episodic_store: list[dict] = []      # 长期归档片段库 (含向量与元数据)
        self.embedder = SentenceTransformer(embed_model)
        self.short_window = short_window

    def archive_episode(self, turn_id: int, summary_text: str, raw_ref: str) -> None:
        vec = self.embedder.encode(summary_text, normalize_embeddings=True)
        self.episodic_store.append({
            "turn_id": turn_id,
            "text": summary_text,
            "raw_ref": raw_ref,
            "vec": vec,
        })

    def retrieve_relevant_episodes(self, current_query: str, top_k: int = 3, min_score: float = 0.55) -> list[dict]:
        if not self.episodic_store:
            return []
        q_vec = self.embedder.encode(current_query, normalize_embeddings=True)
        scored = [
            (float(np.dot(q_vec, item["vec"])), item)
            for item in self.episodic_store
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored[:top_k] if score >= min_score]
```

当某轮长工具输出（如完整的实验日志或论文片段）退短期窗口时，系统将其原文落盘为持久化文件（`raw_ref`），同时将高信息密度的结论编码存入 `episodic_store`。后续当模型遇到相关子问题时，先通过余弦相似度召回高相关历史摘要；若需要核对原始行号或精确数值，再通过 `read_file(raw_ref)` 按需拉取局部原文。

---

## 三、工程选型对比与 Prefix Cache 协同设计

| 记忆与上下文层级 | Token 占用 | 信息保真度 | 额外延迟开销 | 适用场景与核心职责 |
|---|---|---|---|---|
| **Observation 入窗前剪枝** | 降低 ~40% | 无损核心决策字段 | $< 1\text{ ms}$（规则裁剪） | 所有高频工具调用场景，消除多语言冗余描述与长尾噪声字段 |
| **短期滑动窗口（Turn-Aligned）** | $O(K)$ 常数上界 | 近期 100%，远期丢失 | 0 | 保持最近几步 ReAct 推理与工具参数修正的局部连贯性 |
| **中期结构化工作台账（Compaction）** | $O(1)$ 紧凑文本 | 保留全局约束与状态 | 仅在跨越水位线时触发一次 | 跨十余轮的多步规划任务，防止遗忘初始约束或重复踩同一错误 |
| **长期向量 / Artifact 索引召回** | 按需注入 Top-$K$ | 细粒度历史可追溯 | $5\sim 15\text{ ms}$（向量检索） | 100K+ 长周期科研探索、跨会话个人助手与代码库级重构 |

最后，在将上述四层内容拼接为最终发往推理引擎的 `messages` 数组时，**排列顺序对推理延迟至关重要**：

- 务必将 **System Prompt、全量工具声明（Tool Schemas）以及缓变的中期工作台账**固定在消息序列的最前端，确保连续多步 ReAct 循环在底层 SGLang / vLLM 引擎上最大化命中 **Prefix Cache（前缀 KV 缓存）**；
- 将每轮动态变化的**长期记忆检索片段（Retrieved Episodes）与当前步动态注入的 Few-Shot 规则**放在靠后的位置（紧邻最新 User / Observation 消息之前），避免因头部几个 Token 的变动导致整条长上下文的 KV Cache 全部失效重算。
