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
    summary: '可选：一句话概括研究问题与方法。',
    // image: '/images/publications/paper-overview.png',
    links: [
      { label: 'Paper', url: '替换为真实论文链接' },
      { label: 'Code', url: '替换为真实代码链接' },
    ],
  },
] as Publication[],
```

`self: true` 会加粗自己的姓名。`image` 和 `summary` 可以不填；不需要的链接直接删除。按你希望展示的顺序排列论文即可。

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
