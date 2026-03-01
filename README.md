# Gu EnHao Blog

基于 Hexo 8 + NexT Pisces 的个人博客，托管在 GitHub Pages，使用 GitHub Actions 自动构建并发布。

## 技术栈

- Hexo 8: 内容管理与静态站点生成
- NexT (Pisces): 主题与页面布局
- 定制样式: `source/_data/styles.styl`
- 主题脚本: `source/_data/head.njk` 与 `source/_data/body-end.njk`
- 插件: `hexo-generator-feed`、`hexo-generator-searchdb`、`hexo-filter-mathjax`

## 本地开发

```bash
npm install
npm run dev
```

默认访问地址: `http://localhost:4000`

发布前建议先执行:

```bash
npm run clean
npm run build
```

## 部署到 GitHub Pages

1. 推送到 `main` 分支会自动触发 `.github/workflows/deploy.yml`
2. 在仓库设置中确认 `Settings -> Pages -> Build and deployment` 选择 `GitHub Actions`
3. 日常发布命令:

```bash
git add .
git commit -m "update blog"
git push origin main
```

## 内容目录

- `source/_posts`: 博客文章
- `source/about`: 关于页面
- `source/projects`: 项目页面
- `themes/next/layout/index.njk`: 首页 Hero 模板

