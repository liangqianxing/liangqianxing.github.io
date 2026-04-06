---
title: 我是怎么从零实现 Nova 主题的
date: 2026-04-06
categories: 技术
tags:
  - Hexo
  - 前端
  - 开源
---

这篇文章记录 Nova 主题的实现思路。Nova 是我为这个博客定制的 Hexo 主题，目标是把学术个人主页和博客合二为一——进来先看到个人介绍，再从里面跳转到博客内容。

<!-- more -->

## 整体架构

Hexo 的工作流程很简单：

```
Markdown 文章 → 渲染器 → layout 模板 → 静态 HTML
```

每个页面对应一个 layout，Hexo 按页面类型在 `layout/` 目录下找对应的 `.ejs` 文件。Nova 主题的 layout 分工如下：

| 文件 | 负责的页面 |
|---|---|
| `layout.ejs` | 所有页面的外壳（`<html>/<head>/<body>`） |
| `index.ejs` | 首页学术主页 |
| `archive.ejs` | `/blog/` 博客列表 |
| `post.ejs` | 文章详情页 |
| `page.ejs` | 关于、友链等静态页 |

## 首页的路由问题

这是整个主题里最绕的地方。

Hexo 有一个内置的 `index_generator` 插件，它会把所有文章聚合，生成 `index.html`，用的是 `index.ejs`。默认行为是渲染文章列表——但我们想要的首页是学术主页，不是文章列表。

解决方案是：**`index.ejs` 直接写学术主页的完整 HTML，完全绕过文章列表逻辑**。

```
/          → index.ejs       → 学术主页（两栏布局）
/blog/     → archive.ejs     → 博客卡片列表
/posts/xxx → post.ejs        → 文章详情
```

`/blog/` 对应的是 `source/blog/index.md`，里面只有一行 front-matter：

```yaml
---
layout: archive
---
```

这样 Hexo 就会用 `archive.ejs` 渲染它，而不是默认的 `index.ejs`。

## 两栏学术布局

`index.ejs` 的 HTML 结构：

```
.academic-wrapper  (CSS Grid: 280px + 1fr)
├── .academic-sidebar  (position: sticky)
│   └── 头像、姓名、机构、社交链接
└── .academic-content  (右侧滚动区)
    ├── Biography
    ├── 技术栈
    ├── 教育经历
    └── 最新文章
```

CSS 核心只有两行：

```css
.academic-wrapper {
  display: grid;
  grid-template-columns: 280px 1fr;
}
.profile-sticky {
  position: sticky;
  top: calc(var(--nav-height) + 2rem);
}
```

左栏固定不动，右栏随页面滚动。响应式处理也很简单，小屏幕下把 grid 改成单列就行。

## 数据配置

所有个人信息存在 `themes/nova/_config.yml`，在 EJS 里通过 `theme.*` 访问：

```yaml
profile:
  name: Gu EnHao
  name_cn: 古恩豪
  avatar: /images/avatar.jpg
  bio: 新疆大学软件工程在读...

education:
  - school: 西湖大学
    school_en: Westlake University
    school_logo: /images/logos/westlake.png
    period: 2025.12 – 2026.03
    degree: 访问学生
    dept: 自然语言处理实验室
```

模板里这样用：

```ejs
<h1><%= theme.profile.name %></h1>

<% theme.education.forEach(edu => { %>
<div class="ac-edu-item">
  <img src="<%- url_for(edu.school_logo) %>" alt="<%= edu.school %>">
  <span><%= edu.school %></span>
</div>
<% }) %>
```

文章数据来自 `site.posts`（Hexo 全局对象），不需要手动配置。

## 暗色模式（无闪烁）

这是个经典的 FOUC（Flash of Unstyled Content）问题。如果在 JS 里切换主题，页面加载时会先白屏再变暗，体验很差。

解决方案是在 `<head>` 里放一段**同步执行**的内联脚本，在浏览器渲染任何内容之前就把主题设好：

```html
<script>
(function(){
  var t = localStorage.getItem('nova-theme');
  if (t) document.documentElement.setAttribute('data-theme', t);
  // 禁用所有 transition，防止初始化时的动画闪烁
  document.documentElement.classList.add('no-transition');
  window.addEventListener('DOMContentLoaded', function(){
    requestAnimationFrame(function(){
      requestAnimationFrame(function(){
        document.documentElement.classList.remove('no-transition');
      });
    });
  });
})();
</script>
<style>.no-transition, .no-transition * { transition: none !important; }</style>
```

`requestAnimationFrame` 嵌套两层是为了确保浏览器完成一帧渲染后再恢复 transition，避免初始化动画。

CSS 主题切换靠 `[data-theme]` 属性选择器 + CSS 变量：

```css
:root {
  --bg: #ffffff;
  --text: #1a1a1a;
  --accent: #3b82f6;
}
[data-theme="dark"] {
  --bg: #121212;
  --text: #e5e5e5;
}
```

## TOC 目录的坑

Hexo 内置的 `toc()` helper 对中文标题生成的锚点有问题，`href` 经常是空的，点了没反应。

所以 TOC 完全用 JS 动态构建，读取页面里实际存在的标题元素的 `id`：

```js
function buildToc() {
  const tocList = document.getElementById('tocList');
  const headings = document.querySelectorAll('.post-content h1, .post-content h2, .post-content h3');

  headings.forEach(h => {
    // Hexo 把 id 放在 <h2> 内部的 <span> 上，不是 <h2> 本身
    const idEl = h.id ? h : h.querySelector('[id]');
    if (!idEl) return;

    const a = document.createElement('a');
    a.href = '#' + idEl.id;
    a.textContent = h.textContent.trim();
    tocList.appendChild(a);
  });
}
```

滚动高亮也是 JS 实现的，监听 `scroll` 事件，找到当前视口内最近的标题，给对应的 TOC 链接加 `active` class。

## 博客卡片的渐变色条

每张卡片顶部有一条 4px 的彩色渐变，6 张一循环，用 `nth-child` + `::before` 伪元素实现：

```css
.post-card::before {
  content: '';
  display: block;
  height: 4px;
}
.post-card:nth-child(6n+1)::before { background: linear-gradient(90deg, #3b82f6, #06b6d4); }
.post-card:nth-child(6n+2)::before { background: linear-gradient(90deg, #8b5cf6, #ec4899); }
.post-card:nth-child(6n+3)::before { background: linear-gradient(90deg, #10b981, #3b82f6); }
/* ... */
```

不需要额外 HTML，纯 CSS 搞定。

## 总结

整个主题的核心思路：

- **路由**：用 Hexo 的 layout 系统做页面分发，`index.ejs` 劫持首页渲染学术主页
- **数据**：个人信息放 `_config.yml`，文章数据用 Hexo 全局 `site.posts`
- **主题切换**：CSS 变量 + `[data-theme]` 属性，内联脚本防 FOUC
- **交互**：JS 只处理 TOC、暗色模式切换、导航滚动阴影，没有引入任何框架

主题已开源在 [hexo-theme-nova](https://github.com/liangqianxing/hexo-theme-nova)，欢迎 star。
