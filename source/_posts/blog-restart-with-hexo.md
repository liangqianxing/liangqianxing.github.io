---
title: 博客重启：用 Hexo 重新搭建个人站点
date: 2026-04-06 10:00:00
tags:
  - Hexo
  - 博客
  - GitHub Pages
categories:
  - 折腾记录
---

旧博客跑了一段时间，积累了不少历史包袱，索性趁这次机会彻底重置，重新搭一个干净的站点。

<!-- more -->

## 为什么重置

- 旧主题年久失修，移动端体验差
- 文章结构混乱，分类标签一团糟
- 部署流程繁琐，每次发文都要手动操作

这次重建的目标很简单：**轻量、自动化、好维护**。

## 技术选型

依然选择 [Hexo](https://hexo.io)，理由：

- Node.js 生态，上手成本低
- 插件丰富，社区活跃
- 与 GitHub Pages 集成方便

主题换成了 [Memory](https://github.com/Dreamer-Paul/Hexo-Theme-Memory)，风格简洁，没有多余的东西。

## 自动化部署

用 GitHub Actions 实现推送即部署，核心配置如下：

```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci && npm run build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./public
```

推送到 `main` 分支后，Actions 自动构建并发布到 GitHub Pages，整个流程大概 1-2 分钟。

## 后续计划

- 补充技术笔记类文章
- 完善友链页面
- 考虑接入评论系统

先把架子搭好，内容慢慢填。
