# Liang Qianxing · Blog

基于 Hexo 8 + NexT Mist 方案定制的个人博客，用来记录代码、设计与 side project 的全过程。项目托管在 GitHub Pages，通过 Actions 自动构建。

## 技术栈

- **Hexo 8**：内容管理、Markdown 渲染。
- **NexT 主题**：定制首页 Hero、色板与排版，样式集中在 `source/_data/styles.styl`。
- **插件**：RSS（`hexo-generator-feed`）、站内搜索（`hexo-generator-searchdb`）。
- **CI/CD**：GitHub Actions -> upload-pages-artifact -> deploy-pages。

## 本地开发

```bash
npm install
npm run dev  # http://localhost:4000
```

发布前可以运行：

```bash
npm run clean && npm run build
```

## 内容结构

- `source/_posts`：日常博客文章。
- `source/about`：关于页面。
- `source/projects`：Side Project 集合。
- `themes/next/layout/index.njk`：注入首页 Hero 的自定义内容。

欢迎提出 Issue 或 PR，一起让这里更好看。
