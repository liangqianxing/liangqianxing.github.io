---
title: How to push new post
date: 2025-11-08 10:34:09
tags:
    others
cover: https://i.pximg.net/img-original/img/2025/11/07/20/38/28/137200740_p0.jpg
---
## 创建 Markdown 文件

运行 Hexo 命令自动生成草稿：
`npx hexo new post "你的标题"`

它会在 source/_posts/ 下生成 你的标题.md，带好 front‑matter。
或者直接在 source/_posts/ 里手动新建 yyyy-mm-dd-xxx.md，内容格式如下：
```
title: 新文章标题
date: 2025-06-06 10:00:00
categories:
  - 分类名
tags:
  - 标签1
  - 标签2
cover: https://你的封面图 (可选)
sticky: 0          # 可选，越大越靠前
```
正文从这里开始……
## 本地预览/构建

npm run dev → 浏览器开 http://localhost:4000 看效果。
确认 OK 后 npm run build（可选，用来检查生成有没有报错）。
## 提交并推送
```
git add source/_posts/xxx.md
git commit -m "post: xxx"
git push origin main
```
推送后 GitHub Actions 会触发 “Build & Deploy Blog”，几分钟内博客自动更新。