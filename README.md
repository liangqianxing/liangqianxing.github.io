# qx.log

> 个人技术博客 · 基于 [Astro](https://astro.build) 构建，托管在 GitHub Pages。

记录关于 LLM、Agent、AI Infra、后端工程与日常的笔记。

[在线访问 →](https://liangqianxing.github.io)

## 项目结构

```
.
├── astro.config.mjs        # Astro 配置（站点、集成、Markdown）
├── tailwind.config.mjs     # Tailwind 主题
├── src/
│   ├── content/
│   │   ├── config.ts       # 文章 collection schema
│   │   └── posts/          # 所有 Markdown 文章
│   ├── layouts/
│   │   ├── Base.astro      # 全站布局（导航、页脚、主题切换）
│   │   └── Post.astro      # 文章排版与正文样式
│   ├── lib/
│   │   ├── posts.ts        # 列表排序 / 阅读时长 / 标签 slug
│   │   └── site.ts         # 站点元数据（标题、导航、社交）
│   ├── pages/
│   │   ├── index.astro     # 首页
│   │   ├── 404.astro
│   │   ├── about.astro
│   │   ├── posts/          # 文章列表 + 详情
│   │   ├── tags/           # 标签索引 + 详情
│   │   └── rss.xml.js      # RSS 输出
│   └── styles/
│       └── global.css      # 主题变量、组件样式
├── public/                 # 静态资源（favicon、图片）
└── scripts/                # 旧 Hexo 辅助脚本
```

## 本地开发

要求 Node.js 18+。

```bash
npm install
npm run dev          # http://localhost:4321
npm run build        # 输出到 dist/
npm run preview      # 预览构建结果
```

## 写新文章

在 [src/content/posts/](src/content/posts/) 下新建 `.md` 文件，frontmatter 字段如下：

```yaml
---
title: 文章标题
date: 2026-05-24
description: 一句话摘要（可选，会用于首页和 RSS）
tags: [LLM, Agent]            # 可选
categories: [AI]              # 可选
updated: 2026-05-25           # 可选
---

正文内容…
```

文件名即 URL slug。例如 `agent-memory.md` → `/posts/agent-memory`。

## 自定义站点信息

编辑 [src/lib/site.ts](src/lib/site.ts) 即可修改站点标题、描述、导航和社交链接，无需在多处替换。

## 隐私 & Git

- [.env](.env) 已在 [.gitignore](.gitignore) 中忽略，绝不会被推送。可参考 [.env.example](.env.example) 了解格式。
- 不要将带有 cookie / token 的文件加入版本控制。新增脚本时请通过环境变量读取敏感值。

## 部署

push 到 `main` 分支后，[.github/workflows/deploy.yml](.github/workflows/deploy.yml) 会自动构建并发布到 GitHub Pages。

## 协议

文章内容版权归作者所有；模板代码以 MIT 协议开放。
