# Gu EnHao Blog (Fresh Start)

这是一个重新起步的 Hexo 博客仓库。

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
