---
title: 从草图到上线：周末里的 Design to Code 流程
date: 2025-06-02 21:10:00
categories:
  - 创作
tags:
  - Design System
  - 前端
  - Workflow
cover: https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=1400&q=80
---

这个小实验源自一个想法：能不能在 48 小时里，从一张草图走到线上 Demo？我给自己定了下面这条流水线：

1. Figma 里搭建一个极简组件库，限制色板（3 种主色 + 1 种强调色）。
2. 用 Auto‑layout 输出高保真稿，然后导出 Tokens。
3. 在代码端用 UnoCSS + Vue 3 还原，所有样式只读 token config。
4. 利用 Cloudflare Pages 直接部署，免去了传统服务器设置。

<!-- more -->

### 过程中的几个心得

- **别急着写代码。** 先把交互状态和组件语义在设计稿里讲清楚，编码速度会快很多。
- **Tokens 要能被脚本消费。** 这次用 JSON + pnpm scripts，把颜色 / 阴影 / 间距自动转成 TS 类型，减少手抄错误。
- **自动化比手动调色靠谱。** 一个 `pnpm generate-theme` 就能同步 Figma 的变更。

最后的 Demo 虽然只有 3 个页面，但所有细节都能被复用。接下来我会把这套流程沉淀成模版，给之后的 side project 套用。
