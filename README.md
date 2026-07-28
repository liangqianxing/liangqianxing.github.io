# gu.log

个人技术博客，记录 LLM、Agent、AI Infra、后端系统、源码阅读和学习复盘。

基于 [Nuxt 4](https://nuxt.com/) 与 [Nuxt Content 3](https://content.nuxt.com/) 构建，静态生成后部署到 GitHub Pages。

[访问博客](https://liangqianxing.github.io) · [文章库](https://liangqianxing.github.io/posts) · [主题索引](https://liangqianxing.github.io/tags)

## 内容地图

- **AI 系统基础课**：Autograd、GPU 推理、分布式系统、数据库与 RAG。
- **LLM / Agent**：Transformer、模型训练、推理服务、记忆与上下文工程。
- **源码导读**：从入口、数据流和关键抽象拆解 AI 项目与开源框架。
- **后端工程**：Go、缓存、消息队列、高并发、限流与系统设计。
- **项目与面试复盘**：工程项目、岗位准备、算法与 CS 基础。

文章通过 `series` 与 `seriesOrder` 组织为连续阅读路径，首页提供知识地图，文章库支持关键词搜索和标签筛选。

文章页提供浅色、深色与 Cyber 三套主题联动的代码高亮，并包含固定目录、阅读进度、作者轨道和响应式长文排版。

## 三大会论文精读

每天从 NeurIPS、ICML、ICLR 选择一篇论文进行中文精读。文章会核对官方原文、实验结果与开源资源，优先使用论文原图并注明论文、图号与官方来源；仅在原图不适合引用时进行原创重绘。

<!-- PAPER_READING_START -->
- [Dynamic-LLaVA 精读：同时压缩视觉 Token 与生成上下文](https://diycv.top/archives/dynamic-llava-context-sparsification) · ICLR 2025 · 2026-07-28
- [VideoLISA 精读：多模态模型的训练侧视频分割适配](https://diycv.top/archives/videolisa-video-reasoning-segmentation) · NeurIPS 2024 · 2026-07-27
- [SparseVLM 精读：让问题决定保留哪些视觉 Token](https://diycv.top/archives/sparsevlm-text-guided-visual-token-sparsification) · ICML 2025 · 2026-07-23
- [InstructBLIP 精读：让视觉特征听懂任务指令](https://diycv.top/archives/instructblip-vision-language-instruction-tuning) · NeurIPS 2023 · 2026-07-23
- [DeeR-VLA 精读：用动态早退加速多模态机器人推理](https://diycv.top/archives/deervla-dynamic-early-exit-inference-acceleration) · NeurIPS 2024 · 2026-07-22
- [TD-MPC2 精读：用隐式世界模型统一 104 个连续控制任务](https://diycv.top/archives/tdmpc2-scalable-world-models) · ICLR 2024 · 2026-07-22
- [Yo'LLaVA 精读：用 16 个软 Token 记住你的专属视觉概念](https://diycv.top/archives/yollava-personalized-multimodal-assistant) · NeurIPS 2024 · 2026-07-21
- [FlashAttention-3 精读：用异步流水与 FP8 加速 Hopper Attention](https://diycv.top/archives/flashattention3-hopper-asynchronous-attention) · NeurIPS 2024 · 2026-07-21
- [M3 精读：可伸缩视觉 Token 如何加速多模态推理](https://diycv.top/archives/matryoshka-multimodal-models-inference-acceleration) · ICLR 2025 · 2026-07-20
- [BLIP-2 精读：用 Q-Former 接通冻结视觉编码器与大语言模型](https://diycv.top/archives/blip2-q-former-multimodal) · ICML 2023 · 2026-07-20
- [DPO 精读：不用 PPO，如何直接从偏好数据对齐语言模型](https://diycv.top/archives/dpo-direct-preference-optimization) · NeurIPS 2023 · 2026-07-19
- [QLoRA 精读：4-bit 量化如何把 65B 微调压进单卡](https://diycv.top/archives/qlora-efficient-finetuning) · NeurIPS 2023 · 2026-07-18
<!-- PAPER_READING_END -->

## 技术栈

- Nuxt 4、Vue 3、TypeScript
- Nuxt Content 3、Markdown
- Tailwind CSS 4
- Nitro 静态生成、GitHub Pages
- GitHub Actions、Dependabot

## 项目结构

```text
.
├── assets/css/main.css       # 全站主题、排版和响应式样式
├── components/               # 导航、文章卡片、知识地图等组件
├── content/posts/            # Markdown 文章
├── content.config.ts         # Nuxt Content collection schema
├── pages/                    # 首页、文章、标签、友链和关于页面
├── public/images/posts/      # 文章配图，按 slug 分目录
├── server/api/posts.get.ts   # 静态文章元数据接口
├── utils/blog.ts             # 日期、阅读时长和标签工具
├── app.config.ts             # 作者、导航和站点信息
└── nuxt.config.ts            # 构建、Markdown 与预渲染配置
```

## 本地开发

要求 Node.js 22，和 GitHub Actions 的构建环境保持一致。

```bash
npm ci
npm run dev       # 默认 http://localhost:3000
npm run build     # 静态产物输出到 .output/public
npm run preview   # 预览生产构建
```

## 写新文章

在 [`content/posts/`](content/posts/) 下新建 Markdown 文件。文件名即 URL slug，例如 `agent-memory.md` 对应 `/posts/agent-memory`。

```yaml
---
title: 文章标题
date: 2026-07-17 09:00:00
description: 用一句话说明文章解决的问题和覆盖范围。
categories:
  - AI
tags:
  - LLM
  - AI Infra
series: 系列名称
seriesOrder: 1
---
```

可见性字段：

- `draft: true`：草稿，不进入公开列表。
- `hidden: true`：保留页面内容，但不进入文章列表。
- `published: false`：暂不发布。
- `legacy: true`：隐藏文章仍生成静态路由，用于兼容旧链接。

## 文章配图

图片放在 `public/images/posts/<slug>/`，正文使用 `/images/posts/<slug>/image.svg` 引用。

- 优先使用带 `title`、`desc` 和 `viewBox` 的 SVG，保证清晰度和可访问性。
- 论文原图只有在许可明确允许时才收录，并保留 caption、来源和许可说明。
- 无法确认复用许可时，根据论文机制原创重绘，并在正文标注“本文原创重绘”和论文链接。
- 禁止热链、来源不明图片和未经核对的实验图表。

## 部署

推送到 `main` 后，[GitHub Actions](.github/workflows/deploy.yml) 会执行：

1. 使用 Node.js 22 安装锁定依赖。
2. 运行 `npm run build` 生成静态站点。
3. 上传 `.output/public` 并部署到 GitHub Pages。

## 安全与协议

- `.env` 已被 Git 忽略，环境变量格式参考 [`.env.example`](.env.example)。
- 不要提交 cookie、token、私钥或包含个人数据的调试文件。
- 文章内容版权归作者所有，仓库代码遵循 [MIT License](LICENSE)。
