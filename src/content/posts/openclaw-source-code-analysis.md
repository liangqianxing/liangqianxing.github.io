---
title: OpenClaw 源码导读：个人 AI 助手的网关、通道、插件与运行时架构
date: 2026-05-08 16:30:00
tags:
  - OpenClaw
  - Agent
  - AI Infra
  - TypeScript
  - 源码分析
categories:
  - 技术
---

OpenClaw 是一个开源的个人 AI 助手项目。它的定位不是单纯的聊天网页，也不是只有一个 CLI，而是一个“运行在自己设备上的多通道 AI 助手”：用户可以通过 WhatsApp、Telegram、Slack、Discord、Google Chat、Signal、iMessage、IRC、Teams、Matrix、飞书、LINE、Mattermost、Nextcloud Talk、Nostr、Twitch、Zalo、WeChat、QQ、WebChat 等通道接入，同时通过 Gateway、插件系统、模型 provider、技能、任务和控制台 UI 组合成一个完整的个人 AI 操作系统。

这篇文章按源码结构来讲 OpenClaw：从启动入口开始，拆解 CLI、Gateway、Channel、Plugin SDK、Agent Runtime、Web UI、Daemon、配置与安全边界，帮助你快速建立对这个项目的整体理解。



## 1. 项目整体定位

OpenClaw README 里给出的核心定义是：

> OpenClaw is a personal AI assistant you run on your own devices.

这个定义很关键。它不是一个“云端 SaaS 助手”，而是偏个人部署、个人设备、个人通道聚合的 AI 助手框架。

从源码看，它主要解决四类问题：

1. 多通道接入：把不同聊天平台、语音、网页、移动端接入统一入口。
2. 模型与工具编排：把 LLM provider、工具、技能、记忆和任务运行时组合起来。
3. 本地控制平面：通过 Gateway 和 UI 管理会话、配置、插件、通道和运行状态。
4. 可扩展生态：通过 plugin-sdk、extensions、skills 让第三方扩展接入。

如果用一句话概括源码架构：

> OpenClaw 是一个 TypeScript/Node.js 实现的多通道 Agent Gateway，核心是把外部消息事件规范化为内部会话与任务，再通过插件化模型运行时和工具系统生成回复，最后回写到原始通道。

## 2. 仓库目录总览

仓库顶层结构大致如下：

```text
openclaw/
├── openclaw.mjs          # npm bin 启动包装器
├── src/                  # Node.js 后端核心源码
├── ui/                   # 控制台 Web UI
├── extensions/           # 官方内置/示例扩展
├── packages/             # SDK 与包契约
├── apps/                 # Android 等应用端
├── docs/                 # 文档站内容
├── scripts/              # 构建、检查、生成脚本
├── deploy/               # 部署相关
├── security/             # 安全扫描与策略
├── skills/               # 技能资源
└── test/ qa/             # 测试与质量保障
```

最值得看的目录是：

- `src/entry.ts`：真正的 CLI 入口。
- `src/cli/`：命令行命令注册与参数解析。
- `src/gateway/`：Gateway 控制平面、WebSocket 客户端、控制 UI。
- `src/channels/`：通道抽象、消息处理、turn kernel。
- `src/plugins/`：插件加载、运行时、provider 与能力抽象。
- `src/plugin-sdk/`：对外暴露的插件 SDK 类型和工具。
- `src/config/`：配置加载、校验、写入、运行时快照。
- `src/daemon/`：systemd、Windows schtasks 等守护进程管理。
- `ui/src/`：前端控制台入口、Web Components、样式和交互。
- `extensions/`：大量具体 channel/provider/tool 插件实现。

这类项目看源码时，不建议一头扎进某个超长文件。更好的顺序是：启动入口 → 命令系统 → Gateway 协议 → Channel 抽象 → Plugin Runtime → UI 控制台 → 具体 extension。

## 3. 技术栈判断

从 `package.json` 和源码形态可以看出：

- 语言：TypeScript + ESM。
- 运行时：Node.js，启动器要求 Node.js 22.12+。
- 包管理：pnpm。
- 前端：Vite 风格的 `ui/src/main.ts` 入口，基于 Web Components / 原生组件组织。
- CLI：Node bin，通过 `openclaw.mjs` 暴露 `openclaw` 命令。
- 通信：Gateway 客户端使用 WebSocket，默认地址类似 `ws://127.0.0.1:18789`。
- 扩展：`extensions/` + `plugin-sdk`，用类型约束插件能力。
- 测试：大量 `*.test.ts`、`*.e2e.test.ts`、集成测试和架构检查脚本。

它不像一个普通 Next.js 应用，而更像“本地 daemon + CLI + Web 控制台 + 插件生态”的组合体。

## 4. 启动入口：openclaw.mjs

`openclaw.mjs` 是 npm bin 入口，在 `package.json` 里这样暴露：

```json
"bin": {
  "openclaw": "openclaw.mjs"
}
```

这个文件不是业务主逻辑，而是启动包装器。它做了几件事：

1. 检查 Node.js 版本，要求 Node.js 22.12+。
2. 判断当前是源码 checkout 还是已打包安装。
3. 处理 Node compile cache。
4. 必要时 respawn 子进程。
5. 找到真正的 `entry.js` / `src/entry.ts`。
6. 把控制权交给后端入口。

为什么要单独做一层启动器？

因为 CLI 项目经常要处理：

- 用户 Node 版本不符合。
- npm 包安装路径和源码路径不同。
- Windows/macOS/Linux 启动差异。
- compile cache 的兼容性。
- 信号转发和子进程退出码。
- 打包后 dist 入口和源码 tsx 入口差异。

所以 `openclaw.mjs` 本质是“稳定启动壳”，真正业务入口在 `src/entry.ts`。

## 5. 真正入口：src/entry.ts

`src/entry.ts` 是 CLI 后端入口。它的职责更像启动编排器：

- 解析 argv。
- 处理 root help / version fast path。
- 标准化 Windows argv。
- 应用 profile 环境变量。
- 检查 root 运行保护。
- 处理 container target。
- 设置 warning filter。
- 安装 unhandled rejection / exception handler。
- 初始化运行环境。
- 进入具体 CLI 命令分发。

这说明 OpenClaw 对 CLI 启动体验很重视。大型 CLI 的一个常见问题是：还没进入业务逻辑，就可能因为环境、权限、路径、Node 版本、配置或依赖问题崩掉。OpenClaw 把这些前置问题集中在 entry 层处理。

可以把入口链路理解为：

```text
openclaw command
  ↓
openclaw.mjs
  ↓
src/entry.ts
  ↓
src/cli/*
  ↓
具体命令 / Gateway / Daemon / Channel / UI
```

## 6. CLI 模块：src/cli

`src/cli/` 下面文件非常多，说明 OpenClaw 的命令行能力很完整。它不只是 `openclaw start`，而是覆盖配置、通道、密钥、补全、daemon、能力查询、ACP、onboard 等多个命令面。

从文件名可以看到几个重点：

- `argv.ts`：命令参数基础处理。
- `config-cli.ts`：配置命令。
- `channels-cli.ts`：通道相关命令。
- `channel-auth.ts`：通道认证。
- `command-options.ts`：命令选项抽象。
- `completion-cli.ts`：shell completion。
- `capability-cli.ts`：能力查询。
- `acp-cli.ts`：ACP 相关 CLI。
- `clawbot-cli.ts`：兼容或历史命名入口。

CLI 层的设计重点不是“调用函数”，而是把命令行输入规范化为内部操作。例如：

```text
用户输入 openclaw channels auth feishu
  ↓
cli 解析命令和参数
  ↓
读取配置和 secret
  ↓
调用 channel auth runtime
  ↓
输出认证状态或下一步指引
```

命令行模块通常不应该直接塞太多业务逻辑，而应该负责参数解析、交互提示、错误格式化、调用底层 runtime。OpenClaw 从目录拆分上基本遵循这个思路。

## 7. Gateway：控制平面核心

`src/gateway/` 是理解 OpenClaw 的关键目录。

Gateway 可以理解为本地控制平面，它负责：

- 接受 UI / CLI / 外部客户端连接。
- 管理 WebSocket 协议。
- 做认证和设备授权。
- 暴露控制 UI。
- 提供会话、配置、通道状态、任务等控制接口。
- 把用户操作转成后端 runtime 请求。

源码里能看到这些文件：

- `client.ts`：Gateway WebSocket 客户端。
- `control-ui.ts`：控制台 UI 服务。
- `auth.ts`、`device-auth.ts`：认证与设备授权。
- `credentials.ts`：凭据处理。
- `control-plane-rate-limit.ts`：控制平面限流。
- `control-ui-csp.ts`：控制台 CSP 安全策略。
- `protocol/`：Gateway 协议定义和校验。

`GatewayClient` 默认地址类似：

```text
ws://127.0.0.1:18789
```

这表明 OpenClaw 采用了“本地服务 + 客户端连接”的架构，而不是每个 CLI 命令都单独起一套完整 runtime。

一个典型控制链路是：

```text
Web UI / CLI
  ↓ WebSocket request frame
Gateway
  ↓ runtime dispatch
Session / Channel / Plugin / Agent
  ↓ event frame / response frame
Web UI / CLI
```

## 8. Gateway 协议设计

`src/gateway/client.ts` 中可以看到对 `EventFrame`、`RequestFrame`、`ResponseFrame`、`HelloOk`、`PROTOCOL_VERSION` 的引用，说明 Gateway 通信不是随便发 JSON，而是有版本化协议。

这类设计有几个好处：

1. UI、CLI、移动端可以共享协议。
2. 协议升级时可以做版本兼容。
3. 请求、响应、事件可以统一校验。
4. 错误码、retry、startup unavailable 等状态可以标准化。
5. 客户端可以实现 pending request map 和 timeout。

`GatewayClientRequestError` 这类错误类型也说明它不是简单 `throw Error`，而是携带：

- gatewayCode。
- details。
- retryable。
- retryAfterMs。

这对于控制台体验很重要：用户看到的不是“连接失败”，而是“服务正在启动、可重试、多久后再试、如何恢复”。

## 9. Channel 层：把外部平台变成统一消息

`src/channels/` 是 OpenClaw 的核心抽象之一。

OpenClaw 支持非常多聊天平台。如果每个平台都直接调用 agent runtime，系统会变得混乱。因此它需要一个 Channel 抽象，把不同平台的事件统一成内部消息和会话。

可以从目录看到：

- `typing.ts`、`typing-lifecycle.ts`：输入中状态管理。
- `turn/kernel.ts`：一轮消息处理核心。
- `turn/durable-delivery.ts`：可靠投递。
- `plugins/`：通道插件类型与适配器。
- `routing/`：消息路由。
- `bindings/`：通道会话与目标绑定。

一个外部消息进入系统，大概会经历：

```text
平台 webhook / websocket / polling
  ↓
具体 extension 适配器
  ↓
Channel normalized message
  ↓
Turn kernel
  ↓
Session / Agent runtime
  ↓
Channel reply adapter
  ↓
原平台回复消息
```

这里的关键是“turn”。一次用户输入到一次模型回复，可以看成一个 turn。Turn kernel 要处理的不只是调用模型，还包括：

- 消息去重。
- 输入中状态。
- 会话解析。
- 路由到哪个 agent / skill。
- 是否需要自动回复。
- 回复投递是否成功。
- 失败重试和 durable delivery。

这也是多通道 AI 助手比普通网页聊天复杂得多的地方。

## 10. Plugin SDK：扩展能力的边界

`src/plugin-sdk/index.ts` 非常值得看。这个文件本身代码不复杂，主要是类型导出，但它暴露了 OpenClaw 插件生态的“公共契约”。

它导出了几类核心类型：

- `ChannelPlugin`：通道插件。
- `OpenClawPluginApi`：插件可调用 API。
- `CliBackendPlugin`：CLI 后端插件。
- `MediaUnderstandingProviderPlugin`：媒体理解 provider。
- `SpeechProviderPlugin`：语音 provider。
- `RealtimeTranscriptionProviderPlugin`：实时转写 provider。
- `ProviderRuntimeModel`：模型运行时描述。
- `PluginRuntime`：插件运行时。
- `SubagentRunParams` / `SubagentRunResult`：子 Agent 运行。
- `LlmCompleteParams` / `LlmCompleteResult`：LLM completion 调用。
- `TaskFlowView` / `TaskRunView`：任务流和任务运行。
- `OpenClawConfig`：配置类型。

这说明 OpenClaw 的插件系统不只是“加几个工具函数”，而是覆盖了：

1. 通道接入。
2. 模型 provider。
3. 媒体理解。
4. 语音能力。
5. CLI 后端。
6. 任务流。
7. 子 Agent。
8. 记忆能力。
9. 配置 schema。

从架构上看，`plugin-sdk` 是 OpenClaw 保持核心稳定、扩展灵活的关键。

## 11. extensions：官方插件实现仓库

`extensions/` 目录非常大，里面有很多官方扩展，例如：

- 模型 provider：anthropic、deepseek、google、groq、huggingface、fireworks、amazon-bedrock、azure 等。
- 通道：discord、feishu、googlechat、line、msteams、whatsapp、zalo、qqbot、wechat 等。
- 工具与服务：brave、browser、duckduckgo、exa、firecrawl、document-extract、file-transfer。
- 诊断与监控：diagnostics-otel、diagnostics-prometheus。
- 媒体生成/理解：comfy、fal、elevenlabs、deepgram 等。

这是一种很典型的“核心内核 + 官方扩展包”结构：

```text
src/             # 核心抽象和运行时
src/plugin-sdk/  # 对外契约
extensions/      # 具体插件实现
```

好处是：

- 核心不用知道每个平台的细节。
- 新平台可以以插件形式接入。
- 官方插件可以跟随主仓库一起测试。
- 第三方插件有明确接口可以实现。

## 12. Agent 与 Runtime

`src/agents/`、`src/tasks/`、`src/tools/`、`src/routing/`、`src/memory/`、`src/context-engine/` 等目录共同组成 OpenClaw 的 agent runtime。

可以把 agent runtime 理解为：

```text
输入消息
  ↓
构造上下文
  ↓
选择模型和工具
  ↓
调用 LLM provider
  ↓
执行工具 / 子任务 / 子 agent
  ↓
生成回复
  ↓
写回 channel
```

OpenClaw 这种项目的难点不是“调用一次 OpenAI API”，而是如何管理完整上下文：

- 当前会话是谁。
- 来自哪个通道。
- 是否有历史消息。
- 是否绑定某个 workspace。
- 可用工具有哪些。
- 用户权限是什么。
- 回复要投递到哪里。
- 失败后怎么恢复。

所以你会看到 `sessions`、`routing`、`bindings`、`context-engine`、`memory` 等模块。这些模块一起解决“模型调用前后”的工程问题。

## 13. 配置系统：src/config

`src/config/config.ts` 是一个 re-export 聚合文件，背后核心在 `io.ts`、`validation.ts`、`mutate.ts`、`runtime-snapshot.ts` 等文件。

配置系统提供的能力包括：

- 读取配置文件。
- 解析 JSON5。
- 运行时配置快照。
- 配置 hash。
- 配置写入监听。
- 配置恢复。
- last known good。
- 插件元数据参与校验。
- 配置 mutation conflict 处理。
- Nix mode 下写保护。

这说明 OpenClaw 的配置不是简单 `JSON.parse`，而是一个可恢复、可校验、可热更新、支持插件扩展的配置系统。

为什么复杂？因为它面对的是长期运行的个人助手：

- 用户可能通过 UI 改配置。
- CLI 也可能改配置。
- 插件会扩展配置 schema。
- daemon 运行中要热更新。
- 错配置不能把整个服务永久弄坏。
- Nix / Docker / 本机安装模式对写入权限要求不同。

因此配置系统需要“运行时快照 + 校验 + 恢复策略”。

## 14. Daemon：让助手长期运行

`src/daemon/` 负责把 OpenClaw 作为系统服务运行。

它支持的方向包括：

- Linux systemd。
- Windows schtasks。
- service install / uninstall / start / stop。
- runtime paths。
- managed env。
- restart logs。
- service audit。

个人 AI 助手要真正可用，不能每次手动开终端。因此 daemon 层非常重要。

可以把 daemon 视为：

```text
用户 openclaw daemon install
  ↓
根据平台生成 systemd unit 或 schtasks
  ↓
设置环境变量和运行路径
  ↓
启动 gateway / channel runtime
  ↓
保持后台运行并记录状态
```

这也是 OpenClaw 和普通“命令行聊天机器人”的区别之一：它追求长期在线、可管理、可恢复。

## 15. UI：控制台前端

`ui/src/main.ts` 很短：

```ts
import "./styles.css";
import "./ui/app.ts";
```

同时它在生产环境注册 service worker，开发环境清理旧 service worker。这说明 UI 是一个前端控制台应用，入口很轻，主要逻辑在 `ui/src/ui/app.ts` 和其他组件中。

从 `ui/src/` 可以看到：

- `ui/`：组件和交互逻辑。
- `styles/`：布局、聊天、配置、usage 等样式。
- `i18n/`：国际化。
- `types/`：类型。
- `local-storage.ts`：本地状态。

UI 不是业务内核，而是 Gateway 的操作面。它通常通过 Gateway 协议调用后端，例如：

```text
用户在 UI 输入消息
  ↓
UI 调用 Gateway request
  ↓
Gateway 转给 chat/session runtime
  ↓
后端流式返回 event
  ↓
UI 更新消息列表和状态
```

UI 中有大量测试，例如 `app-chat.test.ts`，说明项目对前端交互状态也做了细致验证：草稿保存、模型切换、发送队列、run id、stream 状态等。

## 16. 安全边界

OpenClaw 涉及本地网关、外部平台、密钥、模型 provider、插件和工具执行，因此安全边界很重要。

源码中能看到多个安全相关模块：

- `src/security/`。
- `src/secrets/`。
- `src/gateway/auth.ts`。
- `src/gateway/device-auth.ts`。
- `src/gateway/control-ui-csp.ts`。
- `security/` 目录下的扫描规则。
- `src/cli/root-guard.ts`。
- `src/config/nix-mode-write-guard.ts`。

几个明显的设计点：

1. Gateway 连接需要认证和设备授权。
2. 控制 UI 有 CSP 策略。
3. CLI 有 root guard，避免用户以 root 运行造成权限和安全问题。
4. secrets 单独抽象，不把密钥当普通配置随意处理。
5. 插件 API 有类型边界，避免插件直接随意侵入核心。
6. 配置写入有 guard 和 conflict 处理。

对这种本地 AI 助手来说，安全风险主要包括：

- 外部通道伪造消息。
- Gateway 未授权访问。
- 插件滥用权限。
- 工具调用执行危险命令。
- 密钥泄露。
- Web UI XSS。
- 配置被恶意修改。

OpenClaw 的源码结构显示它对这些风险做了分层处理。

## 17. 消息处理主链路

综合源码结构，一个用户从外部聊天软件发消息到收到 AI 回复，大致链路如下：

```text
外部平台消息
  ↓
extensions/<channel>
  ↓
Channel adapter
  ↓
Channel normalized event
  ↓
turn/kernel.ts
  ↓
session resolution / binding / routing
  ↓
context-engine / memory / tools
  ↓
plugin runtime selects provider model
  ↓
LLM completion / tool call / subagent / task
  ↓
reply formatting
  ↓
durable delivery
  ↓
channel adapter sends response
  ↓
外部平台收到回复
```

这个链路里最值得关注的是中间三层：

- Turn kernel：负责“一轮对话”的生命周期。
- Routing/binding/session：负责“这条消息属于谁、该交给谁”。
- Plugin runtime：负责“用什么模型、什么工具、什么 provider”。

## 18. 控制台链路

如果用户不是从外部平台，而是在 Web UI 里操作，链路大致是：

```text
浏览器 UI
  ↓
Gateway WebSocket client
  ↓
Gateway protocol request frame
  ↓
Gateway control handlers
  ↓
chat/session/config/plugin runtime
  ↓
EventFrame streaming back
  ↓
UI 更新消息、状态、配置页
```

这和外部 channel 的区别是：UI 更像控制平面客户端；channel 更像用户消息入口。两者最终都会进入会话、模型和任务 runtime。

## 19. 为什么源码这么细碎

OpenClaw 的 `src/` 目录非常细，很多模块都有对应测试。原因不是“过度工程”，而是它的复杂度来自四个方向：

1. 平台多：不同 channel 的协议和状态差异巨大。
2. 能力多：文本、语音、图像、视频、网页、搜索、任务、记忆都要接入。
3. 运行形态多：CLI、daemon、gateway、UI、移动端、Docker、Nix。
4. 安全要求高：本地凭据、外部消息、插件、工具执行都要隔离。

所以它不能写成一个 `bot.ts`。如果要长期维护，必须有清晰边界：core、gateway、channels、plugins、config、daemon、ui。

## 20. 和普通聊天机器人的区别

普通聊天机器人通常是：

```text
收到消息 → 调 API → 回复
```

OpenClaw 更接近：

```text
多通道消息
  → 统一通道抽象
  → 会话和身份解析
  → 上下文和记忆检索
  → 模型/工具/插件编排
  → 可靠投递
  → 控制台可观测和配置管理
```

它解决的不是单点模型能力，而是“个人 AI 助手系统”问题。

## 21. 适合重点阅读的源码文件

如果你想继续深入，不建议从所有文件开始读。可以按这个顺序：

1. `openclaw.mjs`：理解 npm bin 启动壳。
2. `src/entry.ts`：理解真正入口和启动前置检查。
3. `src/cli/argv.ts`、`src/cli/config-cli.ts`、`src/cli/channels-cli.ts`：理解 CLI 命令组织。
4. `src/gateway/client.ts`：理解 Gateway 客户端协议。
5. `src/gateway/control-ui.ts`：理解控制台如何接入。
6. `src/channels/turn/kernel.ts`：理解一轮消息处理。
7. `src/channels/plugins/types.plugin.ts`：理解通道插件契约。
8. `src/plugin-sdk/index.ts`：理解对外 SDK 暴露面。
9. `src/plugins/types.ts`：理解插件能力模型。
10. `src/config/io.ts`、`src/config/validation.ts`：理解配置生命周期。
11. `src/daemon/service.ts`、`src/daemon/systemd.ts`、`src/daemon/schtasks.ts`：理解后台服务。
12. `ui/src/ui/app.ts`：理解前端控制台状态。
13. `extensions/feishu` 或 `extensions/discord`：选一个通道插件看完整实现。
14. `extensions/anthropic` 或 `extensions/deepseek`：选一个 provider 插件看模型接入。

读完这些，基本就能掌握主干。

## 22. 可以学习的工程设计

OpenClaw 源码里有几个值得借鉴的工程点。

第一，启动层和业务层分离。`openclaw.mjs` 只做环境和入口适配，`src/entry.ts` 才进入业务启动。

第二，Gateway 协议化。UI/CLI/客户端不是随便调函数，而是通过版本化 request/response/event frame 通信。

第三，核心抽象和官方扩展分离。`src/` 提供内核，`extensions/` 实现具体平台。

第四，插件 SDK 类型先行。对外扩展用类型定义边界，避免核心和插件互相污染。

第五，配置系统可恢复。对长期运行服务来说，last-known-good 和运行时快照非常重要。

第六，安全模块显式存在。root guard、CSP、device auth、secrets、Nix write guard 都是长期运行工具必须考虑的东西。

第七，测试密度高。大量模块都有单测、集成测试、e2e 测试和架构检查脚本，这对多平台项目非常关键。

## 23. 如果要给 OpenClaw 加一个新通道

假设要接入一个新的聊天平台，源码层面大概需要：

1. 在 `extensions/<new-channel>/` 新建插件。
2. 实现 `ChannelPlugin` 类型要求的接口。
3. 处理平台认证和配置 schema。
4. 把平台消息转换成 OpenClaw 的 channel message。
5. 实现回复发送 adapter。
6. 处理 typing、附件、媒体、错误和重试。
7. 加入配置元数据和文档。
8. 写单元测试和必要的 e2e 测试。

核心原则是：新通道不要修改核心 turn kernel，而是通过插件契约接入。

## 24. 如果要加一个新模型 Provider

接入新模型 provider 通常需要：

1. 在 `extensions/<provider>/` 新建 provider 插件。
2. 定义认证方式，例如 API key、OAuth、本地服务地址。
3. 暴露模型列表或静态模型 catalog。
4. 实现 completion / streaming completion。
5. 映射 OpenClaw 的消息格式到 provider API。
6. 处理 tool call、usage、错误码、rate limit。
7. 将 provider 能力注册到 plugin runtime。
8. 增加配置 schema 和测试。

这里最容易出错的是 streaming、tool call 和错误恢复，因为不同 provider API 事件格式差异很大。

## 25. 总结

OpenClaw 的源码主线可以用四个词概括：Gateway、Channel、Plugin、Runtime。

- Gateway 负责控制平面和客户端连接。
- Channel 负责把不同平台消息统一成内部 turn。
- Plugin 负责扩展通道、模型、工具、媒体和任务能力。
- Runtime 负责会话、上下文、模型调用、工具执行和回复投递。

它不是一个简单 chatbot，而是一个可长期运行、可多端接入、可插件扩展、可本地控制的个人 AI 助手系统。

如果你想读这类大型 TypeScript Agent 项目，OpenClaw 是一个很好的案例：它把“LLM 应用”从单次 API 调用，推进到了完整的本地 AI 操作系统工程。
