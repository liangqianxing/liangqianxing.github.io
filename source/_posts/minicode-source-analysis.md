---
title: MiniCode 源码解析：用 5000 行 TypeScript 实现一个 AI 编程助手
date: 2026-04-06
categories: 技术
cover: /images/posts/minicode.jpg
tags:
  - TypeScript
  - CLI
  - LLM
  - 源码分析
  - 面试
---

MiniCode 是一个轻量级终端 AI 编程助手，类 Claude Code 工作流，181 ⭐。整个核心只有 5000 行 TypeScript，依赖极简（只有 `diff` 和 `zod` 两个运行时依赖），非常适合学习 AI Agent 的架构设计。这篇文章从源码角度拆解它的实现，作为面试准备。

<!-- more -->

## 整体架构

```
用户输入
   ↓
index.ts（入口：初始化配置、工具注册、权限管理）
   ↓
agent-loop.ts（核心循环：发送消息 → 解析工具调用 → 执行工具 → 继续）
   ↙              ↘
tool.ts           tty-app.ts
（工具注册表）      （终端 UI）
   ↓
tools/（内置工具：读写文件、执行命令、搜索、网络请求）
mcp.ts（外部 MCP 工具服务器）
```

这是标准的 **ReAct（Reasoning + Acting）** 模式：模型思考 → 调用工具 → 观察结果 → 继续思考，循环直到任务完成。

## 核心模块拆解

### 1. Agent Loop（`src/agent-loop.ts`）

这是整个项目最核心的文件，实现了 LLM 的工具调用循环：

```typescript
// 伪代码，展示核心逻辑
async function agentLoop(messages, tools, model) {
  while (true) {
    // 1. 调用模型
    const response = await model.call(messages, tools)

    // 2. 如果模型只返回文本（没有工具调用），结束循环
    if (!response.hasToolCalls) {
      return response
    }

    // 3. 执行工具调用
    const toolResults = await executeTools(response.toolCalls)

    // 4. 把工具结果加入消息历史，继续循环
    messages.push(response, toolResults)
  }
}
```

关键细节：
- **重试逻辑**：模型返回空响应时自动重试
- **权限检查**：每次工具调用前先过权限管理器
- **错误处理**：工具执行失败时把错误信息反馈给模型，让它自行纠正

### 2. 工具系统（`src/tool.ts` + `src/tools/`）

所有工具（内置 + MCP 外部工具）遵循同一个接口：

```typescript
interface Tool {
  name: string
  description: string
  parameters: ZodSchema  // 用 Zod 做运行时参数校验
  execute: (params, context) => Promise<ToolResult>
}
```

内置工具清单：

| 工具 | 功能 |
|---|---|
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件（需权限审批） |
| `edit_file` | 局部编辑（生成 diff 预览） |
| `list_files` | 列出目录文件 |
| `grep_files` | 正则搜索文件内容 |
| `run_command` | 执行 Shell 命令（需权限审批） |
| `web_fetch` | 抓取网页内容 |
| `web_search` | 网络搜索 |
| `ask_user` | 向用户提问 |
| `load_skill` | 加载本地 Skill 文件 |

**Zod 的作用**：模型返回的工具参数是 JSON，Zod 在运行时校验参数类型，防止模型幻觉导致的非法参数进入执行层。

### 3. 权限系统（`src/permissions.ts`）

这是 AI Agent 安全性的核心设计。所有"危险操作"（写文件、执行命令）都需要用户审批：

```
模型想执行 run_command("rm -rf /tmp/xxx")
         ↓
权限管理器拦截，展示给用户：
  ┌─────────────────────────────┐
  │ Allow: rm -rf /tmp/xxx ?    │
  │ [Y] Yes  [N] No  [A] Always │
  └─────────────────────────────┘
         ↓
用户选择 → 执行或拒绝（拒绝时把原因反馈给模型）
```

审批模式：
- **Allow this turn** — 本次会话允许
- **Allow always** — 加入白名单，后续不再询问
- **Reject with guidance** — 拒绝并告诉模型为什么，让它换个方案

文件修改会先生成 unified diff 预览，用户看到具体改了什么再决定是否允许。

### 4. MCP 集成（`src/mcp.ts`）

MCP（Model Context Protocol）是 Anthropic 定义的工具服务器协议，允许外部进程提供工具给模型使用。

MiniCode 的 MCP 实现亮点是**自动协商帧协议**：

```
启动 MCP 服务器进程
         ↓
尝试标准 Content-Length 帧格式（MCP 规范）
         ↓ 失败
回退到 newline-JSON 格式（轻量级）
         ↓ 失败
尝试 HTTP streaming
```

这样可以兼容不严格遵循 MCP 规范的服务器，实用性更强。

外部 MCP 工具被包装成和内置工具相同的接口，Agent Loop 不需要区分工具来源。

### 5. 终端 UI（`src/tty-app.ts`）

这是代码量最大的文件（39KB），实现了一个全屏 TUI（Terminal User Interface）：

```
┌─────────────────────────────────────────┐
│ 对话历史（用户消息 / 模型回复 / 工具调用）  │
│                                         │
│ > 用户: 帮我重构这个函数                  │
│ ✓ read_file: src/utils.ts               │
│ ✓ edit_file: src/utils.ts (diff 预览)   │
│ ◎ 模型: 已完成重构，主要改动是...          │
│                                         │
├─────────────────────────────────────────┤
│ > 输入框                    [tokens: 1.2k]│
└─────────────────────────────────────────┘
```

技术细节：
- 直接操作 TTY 转义序列（不依赖 ncurses 或 blessed）
- 输入历史持久化（`src/history.ts`）
- Slash 命令：`/help`、`/tools`、`/skills`、`/mcp`、`/model`

### 6. Skills 系统（`src/skills.ts`）

Skills 是存储在本地的 Markdown 文件，模型可以通过 `load_skill` 工具加载：

```
发现路径（优先级从高到低）：
./.mini-code/skills/<name>/SKILL.md   # 项目级
~/.mini-code/skills/<name>/SKILL.md   # 用户级
./.claude/skills/<name>/SKILL.md      # 兼容 Claude Code
~/.claude/skills/<name>/SKILL.md
```

这个设计让用户可以把常用的工作流（比如"提交代码"、"写测试"）封装成 Skill，模型按需加载，不需要每次都在 System Prompt 里塞满指令。

## 依赖极简的设计哲学

整个项目运行时只依赖两个包：

| 包 | 用途 | 为什么不自己实现 |
|---|---|---|
| `diff` | 生成 unified diff | diff 算法（Myers diff）实现复杂，有成熟库直接用 |
| `zod` | 运行时 Schema 校验 | 工具参数校验是安全关键路径，用成熟库更可靠 |

其他所有功能（HTTP 请求、文件操作、终端控制）全部用 Node.js 内置模块实现。这让整个项目安装极快，没有依赖地狱。

## 面试常见问题

**Q：什么是 ReAct 模式？**

ReAct = Reasoning + Acting。模型不是一次性给出答案，而是交替进行"思考"和"行动"：
1. 思考：分析当前状态，决定下一步做什么
2. 行动：调用工具获取信息或执行操作
3. 观察：把工具结果加入上下文
4. 重复，直到任务完成

相比单次推理，ReAct 能处理需要多步骤、需要外部信息的复杂任务。

**Q：为什么 AI Agent 需要权限系统？**

LLM 会产生幻觉，可能调用错误的工具参数（比如删错文件）。权限系统在执行层做最后一道防线：
- 危险操作（写文件、执行命令）必须人工确认
- 展示 diff 让用户看到具体改动
- 拒绝时把原因反馈给模型，让它自我纠正

这是"Human in the loop"设计原则的体现。

**Q：MCP 协议是什么？**

Model Context Protocol，Anthropic 提出的开放标准，定义了 AI 模型和外部工具服务器之间的通信协议。类似 LSP（Language Server Protocol）之于编辑器，MCP 让工具服务器和 AI 客户端解耦，任何支持 MCP 的工具都能被任何支持 MCP 的模型使用。

**Q：Zod 在这里解决什么问题？**

模型返回的工具调用参数是 JSON 字符串，TypeScript 的类型系统在运行时不起作用。Zod 在运行时校验参数结构，如果模型返回了错误类型的参数（比如把数字传成了字符串），Zod 会抛出详细的错误信息，可以反馈给模型让它重试，而不是让错误参数进入执行层导致不可预期的行为。

**Q：为什么只用两个运行时依赖？**

依赖越少，安全风险越低（供应链攻击），安装越快，代码越容易理解。Node.js 内置模块已经足够实现 HTTP、文件操作、进程管理。只在"自己实现成本高且风险大"的地方（diff 算法、Schema 校验）引入外部依赖，这是工程上的克制。

## 如何本地运行

```bash
git clone https://github.com/LiuMengxuan04/MiniCode
cd MiniCode
npm install
npm run install-local   # 安装到 ~/.local/bin/minicode

# 设置 API Key
export ANTHROPIC_API_KEY=your_key

minicode                # 启动交互模式

# 离线 demo 模式（不需要 API Key）
MINI_CODE_MODEL_MODE=mock npm run dev
```

## 值得学习的设计

1. **统一工具接口**：内置工具和 MCP 外部工具用同一套接口，Agent Loop 不感知来源
2. **权限前置**：危险操作在执行前拦截，而不是执行后补救
3. **错误反馈给模型**：工具失败不直接报错，而是把错误信息加入消息历史让模型自行纠正
4. **Skills 本地优先**：不依赖云端 marketplace，项目级和用户级 Skill 都支持
5. **协议自动协商**：MCP 帧格式自动降级，兼容性更好
