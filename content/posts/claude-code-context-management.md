---
title: Claude Code 上下文管理机制：从 Microcompact 到 Auto Compact
date: 2026-05-19 10:00:00
categories:
  - 技术
tags:
  - Claude Code
  - Agent
  - LLM
  - AI Infra
  - 上下文工程
---

这篇文章整理的是 Claude Code 上下文管理机制的源码阅读笔记。先给结论：

Claude Code 的上下文管理不是一个固定的“60% 用 A、70% 用 B、80% 用 C”的十层阶梯。更准确地说，它是三部分组合：

1. **请求前的轻量优化链路**：在每次请求发给模型之前，先尽量减少重复内容和旧工具结果。
2. **Auto Compact 高水位兜底**：当上下文接近上限时，自动触发 Session Memory Compact；如果不适用，再触发 Full Compact。
3. **API 报错后的应急恢复**：如果真正调用模型时出现 prompt too long / media-size 类可恢复错误，尝试 Context Collapse / Reactive Compact 等恢复路径；如果恢复不了，再把错误展示给用户。



> 说明：本文区分“源码中已经能完整确认的机制”和“只有接入点、但核心实现不在当前仓库中的机制”。后者只讨论接入位置、触发关系和可观察行为，不把未确认的内部算法写死。

所以更准确的关系是：

```text
Microcompact / Snip / Context Collapse
  = 请求前或请求中的上下文优化机制

Auto Compact
  = 高水位自动压缩调度器
  = 主要调度 Session Memory Compact 和 Full Compact

Full Compact
  = 最强的“创建新上下文基线”的兜底压缩

Reactive Compact
  = 主请求循环中 API 已经拒绝请求后才介入的应急恢复路径
  = 手动 /compact 的 reactive-only 模式也会复用这套机制
```

---

## 哪些机制已经实现

下面按“当前源码可见程度”分类。

### 源码中可确认完整实现的机制

#### Token Usage / Estimated Tail

已实现。

它负责估算当前上下文有多满。Claude Code 会优先使用最近一次真实 API 返回的 token usage，再加上之后新增消息的粗略估算，而不是简单把所有历史 token 累加一遍。

关键点：

- 真实 API usage 包括 input tokens、cache creation tokens、cache read tokens、output tokens。
- 如果最近一次 assistant 响应被拆成多个流式消息块，会往前找到同一个 API response 的第一个 assistant 块，避免漏算中间插入的 tool result。
- Auto Compact、blocking limit、session memory extraction 等路径依赖这个带 tail 估算的 token 计数。
- TUI 里的 Token Warning 组件主要使用最近一次真实 API usage，再交给同一个 warning state 计算函数判断；它不总是把最后一次 API 调用之后新增的 tail 再估进去。

#### Token Warning / Blocking State

已实现。

它负责告诉 UI / query loop：现在是否接近上下文上限，是否应该显示警告，是否已经到了不能继续普通请求的硬限制。

注意这里有两个取数入口：

- Auto Compact 和 blocking limit 用 `tokenCountWithEstimation(messages)`，会在最近一次 API usage 基础上估算 tail。
- TUI 的 Token Warning 当前主要用最近一次真实 API usage 作为显示依据，然后调用同一套 `calculateTokenWarningState` 判断。

真实逻辑不是百分比阶梯，而是基于几个 buffer：

- effective context window = 模型上下文窗口（可被环境变量覆盖）- 为 compact summary 预留的输出 token。
- auto compact threshold = effective context window - 13,000 tokens。
- warning/error threshold = 当前阈值 - 20,000 tokens。
- blocking limit = effective context window - 3,000 tokens。

如果 Auto Compact 关闭，blocking limit 会让用户还能手动 `/compact`，而不是直接把上下文塞满。

#### Read 去重

已实现。

当模型再次读取同一个文本/Notebook 文件、同一个范围，并且文件没有变化时，Read 工具不会再次把完整文件内容塞进上下文，而是返回一个 stub。图片/PDF 等媒体读取不走这条 dedup 路径。stub 的真实内容类似：

```text
File unchanged since last read. The content from the earlier Read tool_result in this conversation is still current — refer to that instead of re-reading.
```

这不是压缩摘要，而是“前一次完整 Read 仍然有效，请引用前面的内容”。

#### 大型 Tool Result 持久化与预算替换

已实现。

工具输出很大时，Claude Code 可以把完整输出保存到会话目录下的 tool-results 文件里，然后在上下文中只保留一个带预览和文件路径的替代文本。

另外还有“每条消息的工具结果总量预算”：

- 对某个 user message 内过大的 tool_result 做预算检查。
- 选择较大的新 tool_result 持久化到磁盘。
- 上下文里替换为固定的预览文本。
- 替换决策会记录下来，resume 后能复现同样的替换，避免 prompt cache 因内容不一致而失效。

#### Prompt Caching / API Cache Control

已实现。

Claude Code 会在 API 请求里放置 cache control，让系统提示词、工具定义、历史前缀等内容尽可能命中 prompt cache。

这不是“把上下文变短”，也通常不是让客户端请求体少发历史内容；更准确地说，它让重复前缀在服务端缓存里复用，从而减少重复计算、计费和延迟成本，并在 usage 中体现为 cache read / cache creation tokens。

#### API Context Management

已实现，但有开关和条件。

这里指 API 原生的 `context_management` 参数。源码里可见的策略包括：

- 通过 `clear_thinking_20251015` 管理 thinking blocks。默认并不是一律清旧 thinking；通常是 `keep: all`，只有在特定冷缓存条件下才只保留最近 1 个 thinking turn。
- 在特定 ant / 环境变量配置下，使用 API 的 `clear_tool_uses_20250919` 策略清理工具内容。

注意：这和 Cached Microcompact 的 `cache_reference` / `cache_edits` 是两条不同机制，不要混在一起讲。

#### Time-based Microcompact

已实现，但默认配置是关闭的。

它的真实触发不是“5 分钟或 20 条消息”，而是配置驱动：

- 默认 `enabled: false`。
- 默认时间间隔阈值是 60 分钟。
- 默认保留最近 5 个 compactable tool results。

它触发时做的事情也不是生成自然语言摘要，而是：

- 找到可 compact 的工具结果。
- 保留最近 N 个。
- 把更旧的 tool_result 内容替换为固定文本：

```text
[Old tool result content cleared]
```

它主要利用一个事实：如果距离上次 assistant 响应已经超过一段时间，服务端 prompt cache 大概率已经冷了。既然下一次请求本来就要重写前缀，不如在发送前清掉旧工具结果，减少重写成本。

#### Cached Microcompact 接入

接入逻辑已实现，核心模块在当前源码中不可见。

当前源码可确认：

- 它只在 feature gate 开启、模型支持、主线程请求等条件满足时运行。
- 它会登记可 compact 的 tool_result。
- 如果决定删除某些工具结果，会生成 pending `cache_edits`。
- 它不会修改本地消息内容。
- API 层会给旧 tool_result 加 `cache_reference`，再插入 `cache_edits` 删除指定引用。
- API 返回后，会用真实的 `cache_deleted_input_tokens` 生成 microcompact boundary。

但真正决定“删哪些 tool result”的 `cachedMicrocompact.js` 核心文件在当前仓库中不可见，所以不能把它内部策略讲得过于确定。

#### Session Memory Compact

已实现，但需要 feature / env 开关，并且依赖已有 session memory。

它不是在 compact 当下把早期对话重新发给 Claude 生成摘要。它使用已经异步抽取好的 session memory 内容作为摘要基础。

真实流程：

1. 检查 Session Memory Compact 是否启用。
2. 等待正在进行的 session memory extraction 完成。
3. 读取 session memory 文件。
4. 如果没有 session memory 或只是空模板，则放弃，回退到 Full Compact。
5. 根据 `lastSummarizedMessageId` 找到已经被 memory 覆盖到哪里。
6. 从这个位置之后开始保留最近消息。
7. 如果保留的消息太少，就向前扩展，直到满足：
   - 默认至少保留约 10,000 tokens。
   - 默认至少保留 5 条带文本块的消息。
   - 默认最多扩展到约 40,000 tokens。
8. 调整保留起点，避免切断 tool_use / tool_result 配对，也避免切断同一个 assistant response 的 thinking/tool 块。
9. 创建 compact boundary。
10. 插入 session memory summary。
11. 保留最近一段完整消息尾巴。
12. 恢复 plan 文件附件（如果有）。
13. 执行 session start hooks。
14. 如果压缩后的 token 仍然超过 Auto Compact 阈值，则放弃这条路径，改走 Full Compact。

Session Memory Compact 的价值是：比 Full Compact 更便宜、更快，因为它复用已经维护好的 session memory，并保留最近上下文原文。

#### Full Compact

已实现。

这是最强的上下文重置机制，也就是通常意义上的“压缩成 boundary + summary，开启新的上下文基线”。

真实流程：

1. 记录压缩前 token 数。
2. 执行 PreCompact hooks。
3. 构造 compact prompt。
4. 把当前 active conversation 发给一个 compact summary 请求。
5. 默认会先尝试 prompt-cache-sharing 的 forked-agent summary 路径；这条路径主要为了复用主会话缓存。
6. 如果 forked-agent 路径失败，会走 regular streaming fallback summarizer 路径；这条路径在发送前会移除图片/文档，替换成 `[image]` / `[document]`，避免 compact 请求自己爆上下文。
7. regular streaming fallback 路径还会去掉一些 compact 后会重新注入的附件，避免摘要被过时附件污染。
8. 如果 compact 请求本身 prompt too long，会最多重试 3 次。
9. 每次重试会从最老的 API round group 开始丢弃一部分历史，再尝试生成摘要。
10. 摘要成功后，清理 read file state 等缓存。
11. 恢复最近读取过的文件，默认最多 5 个，并受 token budget 限制。
12. 恢复 async agent、plan、plan mode、已调用 skills、deferred tools、agent listing、MCP instructions 等必要上下文。
13. 执行 session start hooks。
14. 插入 compact boundary。
15. 插入 compact summary message。
16. 执行 PostCompact hooks。
17. 做 post compact cleanup。

普通 Full Compact 压缩后的消息顺序是：

```text
compact boundary
summary message
恢复附件
hook results
```

通用 `CompactionResult` 支持 `messagesToKeep`。Session Memory Compact 和 Reactive Compact 这类 suffix-preserving 路径可以在 summary 后保留一段原文尾巴；Partial Compact 则会根据 `from` / `up_to` 保留前缀或后缀。普通 Full Compact 路径通常不保留最近消息原文尾巴。

Full Compact 是最彻底的压缩方式。它会丢失部分细节，但会最大幅度降低上下文占用。

#### Auto Compact

已实现。

但它不是“多档百分比策略选择器”。源码里的 Auto Compact 更像一个高水位兜底调度器。

它的真实判断：

1. 如果 compact 被整体禁用，直接不做。
2. 如果 Auto Compact 被禁用，直接不做。
3. 如果当前 query source 是 compact 或 session_memory，直接不做，避免递归死锁。
4. 如果 Context Collapse 正在接管上下文管理，Auto Compact 会被抑制。
5. 如果 Reactive-only 模式开启，主动 Auto Compact 会被抑制，让 API 的 prompt-too-long / media-size 等可恢复错误来触发 Reactive Compact。
6. 如果连续 Auto Compact 失败次数达到 3 次，熔断，不再反复尝试。
7. 计算 token usage。
8. 如果 token usage 没到 Auto Compact 阈值，不做。
9. 到阈值后，先尝试 Session Memory Compact。
10. Session Memory Compact 不可用、不满足条件、或压缩后仍超阈值，则走 Full Compact。

所以正确说法是：

```text
Auto Compact
  ├─ 先尝试 Session Memory Compact
  └─ 不行再 Full Compact
```

Microcompact 并不是 Auto Compact 内部按百分比选择出来的策略。它在 Auto Compact 之前已经作为请求前优化跑过了。

#### Compact Boundary / Active Context View

已实现。

Compact Boundary 是压缩后的分界线。代码层面的 active slice 会从最后一个 compact boundary 开始取，并包含 boundary 本身；但 boundary 是系统元消息，后续 API 归一化时不会作为普通对话内容直接喂给模型。因此从模型可见内容角度，可以近似理解为使用以最后一个 compact boundary 为基线的上下文。

它的作用：

- 标记发生过 compact。
- 记录触发方式：manual 或 auto。
- 记录压缩前 token 数。
- 记录 logical parent，保持会话链路。
- 在部分压缩或保留尾巴的场景中，记录 preserved segment，方便恢复消息链。

Active Context View 的真实含义是：

```text
从最后一个 compact boundary 开始的消息 slice（包含 boundary 元消息）
+ 如果 Snip 开启，则过滤掉 snipped messages
+ 如果 Context Collapse 开启，则投影成 collapsed view
+ 请求前再经过 tool result budget / microcompact 等处理
```

它不是简单的“UI 上能展开折叠的完整历史”。UI 可能仍保留滚动历史，但模型看到的是投影后的 active context。

#### Post Compact Cleanup

已实现。

压缩后会清理一些已经失效的缓存和状态，例如：

- microcompact state。
- context collapse state，主线程 compact 时才清。
- user context / memory file cache。
- system prompt sections。
- classifier approvals。
- speculative bash checks。
- session message cache。

这样做是为了避免压缩前的缓存污染压缩后的新上下文基线。

#### Compact Warning State

已实现。

压缩或 microcompact 刚成功后，token usage 可能还没等到下一次真实 API usage 校准。源码里已有 compact warning suppression 状态；当前可见调用点主要包括手动 `/compact` 成功、reactive-only 手动 compact 成功，以及 cached/time-based microcompact 成功。不要笼统理解为每一条 proactive Auto Compact 成功路径都会直接调用 suppress。

#### Message Grouping

已实现。

源码里按 API round 对消息分组，而不是只按用户回合分组。这主要服务于 prompt-too-long retry：如果 compact summary 请求自己太长，就按 API round 从头丢弃旧分组，避免切断工具调用配对。

#### Partial Compact

已实现。

这是手动局部 compact，不等同于 Snip。它可以围绕用户选择的消息做局部摘要：

- `up_to`：摘要选中点之前的消息（不含选中消息），保留选中消息及其后面的消息。
- `from`：摘要选中消息及其之后的消息，保留前面的消息。

它会生成 summary 和 boundary，并尽量保持 prompt cache 或消息链路。

### 有接入点，但当前仓库缺少核心实现的机制

#### Snip Compact

有接入点，但当前仓库缺少核心实现文件。

源码中可以看到：

- `HISTORY_SNIP` feature gate。
- query 前会调用 `snipCompactIfNeeded`。
- active context 会通过 `projectSnippedView` 过滤 snipped messages。
- UI 保留 scrollback 时可以 include snipped messages。
- `/force-snip` 等入口存在。

但 `snipCompact.js` / `snipProjection.js` 在当前源码树里不可见，所以不能确认具体算法。

可安全讲法：

```text
Snip 是一个 feature-gated 的历史裁剪机制。它会在 microcompact 之前运行，可能从模型可见上下文中移除一部分历史，并返回 tokensFreed 供 Auto Compact 的阈值判断参考。但核心裁剪策略在当前源码中不可见。
```

不要讲成“用户手动选择任意片段压缩成摘要”这样的确定功能，除非你能看到对应核心源码。

#### Context Collapse

有大量接入点，但当前仓库缺少核心实现文件。

源码中可见：

- `CONTEXT_COLLAPSE` feature gate。
- query 中在 Auto Compact 之前调用 `applyCollapsesIfNeeded`。
- `/context` 会通过 `projectView` 统计 collapse 后的上下文。
- UI 会显示多少 span summarized / staged。
- prompt too long 后会先尝试 `recoverFromOverflow`，也就是 drain staged collapses。
- Context Collapse 开启时，会抑制 proactive Auto Compact，让 Collapse 自己接管 headroom 管理。

但 `services/contextCollapse/index.js` / `operations.js` 等核心文件不在当前仓库中。

可安全讲法：

```text
Context Collapse 是一个 feature-gated 的上下文折叠系统。它在 Auto Compact 之前投影一个更小的上下文视图，并在 prompt too long 时优先尝试提交/排空 staged collapse。当前仓库只能验证接入点和整体位置，不能完整验证内部折叠算法。
```

不要讲成“用户可以像 IDE 折叠代码一样展开/折叠查看详情”，源码里当前可见部分不足以支持这个说法。

#### Reactive Compact

有接入点，但当前仓库缺少核心实现文件。

源码中可见：

- `REACTIVE_COMPACT` feature gate。
- query 中会把可恢复的 prompt-too-long / media-size error 暂时 withheld，不立刻展示给用户。
- 请求失败后会调用 `tryReactiveCompact`。
- 手动 `/compact` 在 reactive-only 模式下会走 `reactiveCompactOnPromptTooLong`。
- Reactive-only 模式会抑制 proactive Auto Compact，让真实 API 的 prompt-too-long 等可恢复错误触发恢复。

但 `reactiveCompact.js` 当前不可见。

可安全讲法：

```text
Reactive Compact 在主请求循环里是 API 已经拒绝请求后的应急恢复机制。它不是常规请求前优化，也不是 Auto Compact 的普通分支；另外，手动 `/compact` 在 reactive-only 模式下也会复用 reactive compact 路径。当前仓库只能验证它的调用位置和错误恢复框架，不能完整验证内部压缩策略。
```

---

## 正确的请求流程

下面是一次普通用户请求在上下文管理上的真实顺序。

### Step 1：构造 active context

先从当前消息历史里取以最后一次 compact boundary 为基线的 active slice。代码层面这个 slice 包含 boundary 元消息；模型可见内容可近似理解为以这个 boundary 为新基线的上下文。

```text
all messages
  -> slice from last compact boundary
  -> boundary is metadata, not ordinary model-visible dialogue
```

如果 Snip 开启，还会把 snipped messages 从模型可见视图中投影掉。

结果是：UI 里也许还能看到更早的滚动历史；模型真正收到的是以最后一次 compact boundary 为基线投影出来的 active context。

### Step 2：应用工具结果预算替换

在 microcompact 之前，先检查 tool_result 是否过大。

如果某条 user message 里的工具结果总量超过预算：

- 选择较大的新 tool_result。
- 持久化到磁盘。
- 上下文里替换成带预览的 persisted-output 文本。
- 记录这次替换，保证后续 resume 还能复现。

这一步通常对用户透明。

### Step 3：Snip，如果开启

如果 `HISTORY_SNIP` feature 开启：

- 在 microcompact 前运行。
- 可能移除一部分模型可见历史。
- 返回 `tokensFreed`。
- Auto Compact 后面判断阈值时，会把这部分 freed tokens 扣掉，避免刚被 Snip 降下来的上下文又被误判为超阈值。

当前仓库缺少核心实现，所以教学时只讲它的位置和作用，不讲具体算法。

### Step 4：Microcompact

Microcompact 在 Auto Compact 之前执行。

它先尝试 Time-based Microcompact。

如果触发：

- 说明距离上一次主线程 assistant 消息已经超过配置阈值。
- 清理旧 tool_result 内容。
- 保留最近 N 个可 compact 的工具结果。
- 返回修改后的 messages。
- 不再继续 cached microcompact。

如果 Time-based 没触发，再尝试 Cached Microcompact。

如果 Cached Microcompact 可用：

- 登记可 compact 的 tool_result。
- 判断是否有需要从 cache 中删除的 tool_result。
- 如果有，生成 `cache_edits`。
- 本地消息不变。
- API 层负责插入 `cache_reference` 和 `cache_edits`。
- API 返回后，用真实删除 token 数生成 microcompact boundary。

如果这些都不可用，Microcompact 直接返回原 messages。

### Step 5：Context Collapse，如果开启

Context Collapse 在 Auto Compact 之前执行。

它的设计意图是：如果能通过 collapse 投影把上下文降下来，就不要急着 Full Compact 成一个大摘要。

如果 Context Collapse 开启，它会接管 headroom 管理，所以 proactive Auto Compact 会被抑制，避免两套系统互相抢控制权。

当前仓库能验证这个顺序和互斥关系，但不能验证具体 collapse 算法。

### Step 6：Auto Compact 高水位判断

Auto Compact 这时才开始判断。

它先做一堆保护：

- compact 整体禁用则退出。
- Auto Compact 禁用则退出。
- 当前就是 compact/session_memory query 则退出。
- Context Collapse 接管则退出。
- Reactive-only 模式则退出。
- 连续失败次数达到 3 次则退出。

然后计算：

```text
tokenCountWithEstimation(messages) - snipTokensFreed
```

如果没到 auto compact threshold，则继续正常请求。

如果到了阈值，进入真正压缩。

### Step 7：Auto Compact 先尝试 Session Memory Compact

Auto Compact 到阈值后，不是先做 microcompact，因为 microcompact 已经做过了。

它第一选择是 Session Memory Compact。

Session Memory Compact 成功的条件比较多：

- feature/env 允许。
- session memory 文件存在。
- session memory 不是空模板。
- 能找到 last summarized message 或符合 resumed session 情况。
- 保留尾巴后不会切断工具调用配对。
- 压缩后的整体 token 低于 Auto Compact 阈值。

成功后，新的上下文形态是：

```text
compact boundary
session memory summary
recent messages kept verbatim
plan attachment, if any
session start hook results
```

如果失败或不适用，回退到 Full Compact。

### Step 8：Auto Compact 回退到 Full Compact

Full Compact 会调用模型生成完整摘要。

它会把当前 active conversation 总结成一个 summary，然后建立新的上下文基线：

```text
compact boundary
compact summary
post-compact restored file attachments
plan / plan mode / skills
tool / agent / MCP delta attachments
hook results
```

Full Compact 是 Auto Compact 的最强兜底。它牺牲更多细节，但能最大幅度释放上下文。

### Step 9：调用模型 API

如果没有 compact，或者 compact 后构造好了新上下文，就进入真实 API 调用。

API 层还会做几件事：

- 加 prompt cache breakpoints。
- 插入 cache control。
- 如果 Cached Microcompact 有 pending edits，插入 `cache_edits`。
- 给旧 tool_result 添加 `cache_reference`。
- 如果启用了 API context management，把 `context_management` 参数传给 API。

模型返回后，如果 Cached Microcompact 刚删了缓存内容，会根据 API usage 里的 `cache_deleted_input_tokens` 生成 microcompact boundary。

### Step 10：如果 API 返回 prompt too long / media-size 类错误

如果真实 API 请求被拒绝，错误不会立刻全部显示给用户。部分可恢复错误会先被 withheld。

恢复顺序大致是：

1. 对 prompt-too-long：如果 Context Collapse 开启，先尝试 drain staged collapses。
2. 如果 Reactive Compact 可用，再尝试 Reactive Compact。
3. 对 media-size 类错误：Context Collapse 不会 strip 图片/文档，通常直接进入 Reactive Compact 的恢复路径。
4. 如果恢复成功，构造压缩后的 messages，然后重新进入请求循环。
5. 如果恢复失败，才把 prompt too long 或 media-size error 显示给用户。

这就是 Reactive Compact 的位置：它是失败后的恢复机制，不是请求前的常规压缩层。

---

## Auto Compact 的正确理解

错误理解：

```text
60% -> Microcompact
70% -> Session Memory Compact
85% -> Full Compact
90% -> Emergency Compact
```

这个说法不符合当前源码。

正确理解：

```text
每次请求前：
  先跑请求前优化：
    - tool result budget
    - Snip, if enabled
    - Microcompact
    - Context Collapse, if enabled

然后 Auto Compact 只在高水位触发：
  if token usage >= auto compact threshold:
    try Session Memory Compact
    if not usable:
      run Full Compact
```

Auto Compact 是“高水位自动压缩调度器”，不是“所有压缩策略的总容器”。

Microcompact 是 Auto Compact 前面的轻量优化。

Session Memory Compact 和 Full Compact 才是 Auto Compact 直接调度的两个主要压缩执行路径。

---

## Full Compact 为什么是终极上下文基线

Full Compact 的本质不是简单删历史，而是重新建立上下文基线。

压缩前：

```text
大量历史对话
大量工具调用
大量文件读取
多轮计划和修改
```

压缩后：

```text
Conversation compacted boundary
一段结构化 summary
少量必要附件
最近文件/计划/技能/工具上下文
```

这相当于把“整本施工日志”整理成“当前项目交接文档”。模型不再逐字看到所有旧消息，但能看到：

- 当前目标是什么。
- 已经完成了什么。
- 修改过哪些文件。
- 关键决策是什么。
- 当前还在做什么。
- 接下来应该怎么继续。

所以 Full Compact 是最强兜底，但不是最优先使用的方式。因为它会损失细节，也需要额外模型调用。

---

## 一个真实长会话示例

假设用户进行 2 小时编程会话。

### 0-30 分钟

主要发生的是透明优化：

- Read 同文件去重。
- 大工具输出持久化到磁盘。
- prompt caching 复用系统提示词、工具定义、历史前缀。
- token warning 持续监控。

用户通常无感。

### 30-90 分钟

如果工具输出很多：

- tool result budget 会把过大的结果替换成 persisted-output 预览。
- 如果 Cached Microcompact 开启，可能通过 cache_edits 删除旧 cached tool_result。
- 如果会话中断很久再回来，并且 Time-based Microcompact 开启，旧 tool_result 可能被替换为 `[Old tool result content cleared]`。

注意，这些都不是 Auto Compact 的分档策略，而是 Auto Compact 之前的轻量优化链路。

### 接近上下文上限

当 token usage 达到 Auto Compact 阈值：

1. Auto Compact 触发。
2. 先尝试 Session Memory Compact。
3. 如果 session memory 可用，并且压缩后低于阈值，就保留 session memory summary + 最近消息尾巴。
4. 如果不可用或压缩效果不够，就触发 Full Compact。

### Full Compact 后

上下文会变成：

```text
compact boundary
conversation summary
recent restored files
plan / skills / hooks / tool context
```

token 使用大幅下降，用户可以继续工作。

### 如果 API 仍然报 prompt too long

这是应急阶段：

- Context Collapse 可能先尝试恢复。
- Reactive Compact 如果启用，尝试对失败请求做恢复压缩。
- 如果都不行，错误才会显示给用户。

---

## 教学视频推荐讲法

建议不要按“十层楼”讲。更推荐按“三段式”讲。

### 第一段：日常省 token

主题：大多数时候，Claude Code 不需要大压缩，而是在悄悄减少浪费。

包括：

- Read 去重。
- 大工具输出持久化。
- prompt caching。
- tool result budget。
- time-based / cached microcompact。

一句话：

```text
这一层不是总结对话，而是减少重复内容和旧工具输出。
```

### 第二段：接近上限时自动建新基线

主题：Auto Compact 是高水位兜底。

包括：

- token usage 估算。
- auto compact threshold。
- circuit breaker。
- 先 Session Memory Compact。
- 再 Full Compact。
- compact boundary + summary。

一句话：

```text
Auto Compact 不是每个百分比切换一个策略，而是在快满时自动选择“能否用 session memory”，不行就完整 compact。
```

### 第三段：已经爆了怎么办

主题：API 拒绝请求后的恢复。

包括：

- prompt too long withheld。
- Prompt-too-long 时的 Context Collapse recovery。
- Reactive Compact recovery（prompt-too-long / media-size 等）。
- 恢复失败后错误浮出。

一句话：

```text
Reactive Compact 在主请求循环里是错误后的救援队，不是平时主动跑的清理器；手动 /compact 的 reactive-only 复用路径是例外入口。
```

---

## 最容易讲错的点

### 错误点 1：Auto Compact 包含所有压缩策略

更正：

Auto Compact 直接调度的是 Session Memory Compact 和 Full Compact。Microcompact 在它之前运行，Context Collapse 开启时甚至会抑制 Auto Compact。

### 错误点 2：Microcompact 会总结旧对话

更正：

当前可见源码里的 Time-based Microcompact 是清空旧 tool_result 内容，不生成摘要。Cached Microcompact 是通过 API cache editing 删除 cached tool_result，也不是自然语言摘要。

### 错误点 3：Session Memory Compact 现场生成摘要

更正：

它使用已经存在的 session memory 内容，并保留最近消息尾巴。现场生成完整摘要的是 Full Compact。

### 错误点 4：Full Compact 只剩 summary

更正：

Full Compact 后不只是 summary。它还会恢复最近文件、plan、plan mode、skills、工具/agent/MCP 附件、hook results 等必要上下文。

### 错误点 5：Context Collapse 已能完整确认算法

更正：

当前源码只能确认 Context Collapse 的接入位置、统计/恢复路径和对 Auto Compact 的抑制关系。核心实现文件不在当前仓库中，不能完整讲算法细节。

### 错误点 6：Reactive Compact 是主动压缩层

更正：

在主请求循环中，Reactive Compact 是 API 返回 prompt too long / media-size 类错误后的恢复路径。Reactive-only 模式下，它会故意抑制 proactive Auto Compact，等待真实 API 错误触发恢复；手动 `/compact` 在 reactive-only 模式下也会复用 reactive compact 机制。

---

## 最终版总流程图

```text
用户输入
  |
  v
取以最后一个 compact boundary 为基线的 active context
  |
  v
应用 tool result budget / 大结果持久化
  |
  v
Snip Compact, if feature enabled
  |
  v
Microcompact
  |-- Time-based: 清空旧 tool_result 内容
  |-- Cached: 生成 cache_edits, 本地消息不变
  |
  v
Context Collapse, if feature enabled
  |
  v
Auto Compact 高水位判断
  |-- 未到阈值：继续
  |-- 到阈值：
        |-- 尝试 Session Memory Compact
        |-- 不适用则 Full Compact
  |
  v
构造 API 请求
  |-- cache_control
  |-- cache_reference / cache_edits, if cached MC
  |-- context_management, if enabled
  |
  v
模型响应
  |
  |-- 成功：正常继续
  |
  |-- prompt too long:
        |-- Context Collapse recovery, if available
        |-- Reactive Compact, if available
        |-- 仍失败则显示错误

  |-- media size error:
        |-- Reactive Compact, if available
        |-- 仍失败则显示错误
```

---

## 一句话总结

Claude Code 的上下文管理不是单一 compact，也不是固定百分比阶梯；它是“请求前轻量减负 + 高水位 Auto Compact + 错误后 Reactive Recovery”的组合系统。

其中：

- Microcompact 负责清理旧工具结果。
- Session Memory Compact 负责用已有 session memory 替换早期历史，同时保留最近尾巴。
- Full Compact 负责生成 summary 并创建新的 compact boundary 基线。
- Context Collapse 和 Reactive Compact 是 feature-gated 的高级/实验路径，当前仓库只能确认接入点，不能完整确认内部算法。
