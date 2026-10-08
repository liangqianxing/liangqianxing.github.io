# 学术主页编辑指南

## 页面与文件

| 地址 | 用途 | 文件 |
| --- | --- | --- |
| `/` | 选择学术主页或个人博客 | `pages/index.vue` |
| `/academic` | 学术个人主页 | `pages/academic.vue` |
| `/blog` | 原来的博客首页 | `pages/blog.vue` |

**日常修改只需编辑 `data/academic.ts`。** 保存为 UTF-8。该文件的字段决定页面内容，布局无需跟着修改。

## 个人信息

- `name`、`nameCN`：英文与中文姓名。
- `avatar`：头像路径。默认沿用原站的 `/avatar.jpg`；可以把新照片放进 `public/` 后改成对应路径。
- `tagline`：姓名下的一句话。
- `affiliation`、`affiliationNote`：单位与当前身份。未来入学和已经入学应明确区分。
- `bio`：个人简介，每个字符串显示为一个段落。
- `interests`：研究兴趣，包含中英文名称和简短说明。
- `email`、`scholar`、`cv`：邮箱、Google Scholar 和简历链接，留空就不显示。
- `location`：所在地，留空就不显示。
- `lastUpdated`：最近更新时间，需要修改时手动更新。

简历文件可以放在 `public/files/cv.pdf`，将 `cv` 设置为 `/files/cv.pdf`。不要放入不准备公开的申请材料。

## 添加论文

在 `publications` 列表中添加或修改条目。下面是字段示例，务必替换成真实论文信息后再使用：

```ts
publications: [
  {
    title: '替换为真实论文标题',
    authors: [
      { name: 'Enhao Gu', self: true },
      { name: '替换为其他作者姓名' },
    ],
    venue: '替换为会议、期刊或预印本名称',
    year: '替换为真实年份',
    badge: '替换为简短会议 / 预印本标签与年份',
    summary: '可选：一句话概括研究问题与方法。',
    // image: '/media/publications/paper-overview.png',
    // imageAlt: '可选：简短描述框架图的内容。',
    // imageSource: { label: 'Figure 1 · arXiv', url: '替换为原图出处' },
    links: [
      { label: 'Paper', url: '替换为真实论文链接' },
      { label: 'Code', url: '替换为真实代码链接' },
    ],
  },
] as Publication[],
```

`self: true` 会加粗自己的姓名；标题使用 `links` 中标签为 `Paper` 的链接，没有时显示普通标题。
`badge` 是简短的会议 / 预印本标签，不填则显示年份。`image`、`imageAlt`、`imageSource` 和 `summary` 均可不填，
无图的论文直接显示文字列表。`imageSource` 是独立的图源链接，不替代论文的 `Paper` 链接。
不需要的资源链接直接删除，按你希望展示的顺序排列论文即可。预印本不能标为已录用会议论文。

### 出版物框架图

桌面左图右文，760px 及以下单列。图保持原比例，点击图片可在新标签页查看本站原图。
图源链接位于图片下方，完整作者串、论文简介与资源链接直接显示。

当前两张图来自论文原文，核实于 2026-10-08：

| 论文 | 本地原图 | 图号与官方来源 |
| --- | --- | --- |
| AutoFigure-Edit | `public/media/publications/autofigure-edit-overview.png` | [Figure 1 · arXiv:2603.06674v1](https://arxiv.org/html/2603.06674v1/method_v1.png) |
| DeepReviewer 2.0 | `public/media/publications/deepreviewer-v2-overview.png` | [Figure 2 · arXiv:2604.09590v1](https://arxiv.org/html/2604.09590v1/final.png) |

新增框架图时，将原图保存到 `public/media/publications/`，并在 `scripts/optimize-images.py` 的
`RASTER_SOURCES` 注册站内路径与 320 / 640 / 960px 宽度，运行 `python scripts/optimize-images.py`。
把原图、生成的 WebP 和 `data/image-assets.json` 一起提交；`SiteImage` 自动选择适合显示宽度的变体，
变体失败回退原图，原图也失败则保留稳定占位。原 PNG 和旧哈希变体均保留。
这一目录只用于 Pages，不放入会触发 Halo 同步的 `public/images/`；论文图修改不运行 `sync:halo`。

## 动态与项目

`news` 和 `projects` 初始为空。填入后会自动显示对应栏目与栏目导航。

```ts
news: [
  { date: '2026.10', text: '替换为真实动态。' },
] as { date: string; text: string; url?: string }[],

projects: [
  {
    name: '替换为项目名称',
    description: '替换为项目介绍。',
    tags: ['Python', 'LLM'],
    links: [{ label: 'Code', url: '替换为真实项目链接' }],
  },
] as AcademicProject[],
```

教育和经历分别修改 `education`、`experience`。`logo`、`description` 留空时不显示。添加本科、奖项或研究经历前，先确认名称、时间和身份。

## 本地预览与部署

```bash
npm ci
npm run dev
npm run build
```

开发预览同时检查 `/`、`/academic`、`/blog`。构建产物位于 `.output/public`。推送到 `main` 后，原有 GitHub Actions 会部署 GitHub Pages；本地编辑不会自动上线。

博客文章仍按仓库 `AGENTS.md` 中的可见性规则维护。本次新增入口不改变文章内容、`hidden` / `haloPublished` 字段，也不改变 Halo 同步。
