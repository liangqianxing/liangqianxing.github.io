# Gu EnHao Blog (Fresh Start)

这是一个重新起步的 Hexo 博客仓库。

## 入口链接

- 博客: https://liangqianxing.github.io/posts/%E5%A6%82%E4%BD%95%E6%90%AD%E5%BB%BA%E8%87%AA%E5%B7%B1%E7%9A%84codex%E4%B8%8D%E9%99%90%E9%87%8F%E5%8F%B7%E6%B1%A0/
- 中转站: http://81.70.32.82/
- 卡密商店: https://pay.ldxp.cn/shop/TUQKNUNV

## 当前状态

- 框架: Hexo 8
- 主题: Memory (`hexo-theme-memory`)
- 内容: 已清空，仅保留 `source/_posts/.gitkeep`

## 友链备份

旧站友链已保存到:

- `friend-links.backup.yml`

后续换主题或重新加友链时，可直接复制该文件中的 `links`。

## 本地启动

```bash
npm install
npm run dev
```

## 构建

```bash
npm run clean
npm run build
```

## 发布

推送到 `main` 后会自动触发 GitHub Actions 部署到 GitHub Pages。

## 文章图片放置规范（可直接用于 Actions 部署）

- 图片统一放在 `source/images/posts/` 下，建议按文章分子目录。
  - 例如：`source/images/posts/my-first-post/cover.png`
- 在 Markdown 中使用以 `/` 开头的站点绝对路径引用：
  - `![封面](/images/posts/my-first-post/cover.png)`
- 不要把图片放到 `public/`，该目录是构建产物，每次构建会被覆盖。
