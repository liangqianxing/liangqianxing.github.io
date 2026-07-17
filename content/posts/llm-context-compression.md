---
title: Agent 上下文压缩机制已合并
date: 2026-04-24
description: 早期的五层压缩笔记已并入更完整的 Claude Code 上下文管理源码导读。
hidden: true
legacy: true
categories:
  - 技术
tags:
  - LLM
  - Agent
  - 上下文压缩
---

早期版本把上下文管理概括为固定的“五层阶梯”，容易把请求前优化、自动压缩调度和 API 报错恢复混在一起。重新核对源码后，内容已整理到：

## [Claude Code 上下文管理：请求前减负、Auto Compact 与应急恢复](/posts/claude-code-context-management)

新文章明确区分了三条链路：

1. 请求前的轻量减负：Read 去重、大型工具结果持久化、Snip 与 Microcompact。
2. 接近窗口上限时的 Auto Compact：Session Memory Compact 与 Full Compact。
3. API 拒绝请求后的应急恢复：Context Collapse 与 Reactive Compact。

旧地址继续保留，用于兼容已有书签和外部链接。
