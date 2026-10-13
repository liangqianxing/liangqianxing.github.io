# 学习专题维护

学习专题支持多门课程。`/learning/` 是课程总览，每门课程在 `/learning/<slug>/` 展示自己的资料和学习记录。目前只建立 CMU 15-779 专题，内容由本人后续添加。

所有资料与 Markdown 都在本地仓库中维护，通过 Git 提交并推送到 `main` 后部署。专题网页仅展示内容。

## 新增课程

在 `data/learning/courses.ts` 的 `learningCourses` 数组中加入一个课程对象，即可创建专题。课程页面和资料接口共用这份登记表，无需复制页面或新增课程专用接口。

| 字段 | 用途 |
| --- | --- |
| `slug` | 唯一标识，用于页面地址与资料目录，例如 `cmu-15-779` |
| `code` | 课程编号 |
| `institution` | 学校或机构名称 |
| `series` | 学习记录的系列名；文章中的 `series` 必须完全一致 |
| `title` | 专题显示名称 |
| `courseTitle` | 官方课程名称 |
| `courseUrl` | 官方课程主页 |
| `description` | 专题简介 |
| `topics` | 主题关键词数组 |

以后需要加入 CMU 15-445 时，可以参考以下对象。此示例仅存在于维护文档中，当前没有登记或发布该课程。

```ts
{
  slug: 'cmu-15-445',
  code: '15-445',
  institution: 'CMU',
  series: 'CMU 15-445',
  title: 'CMU 15-445',
  courseTitle: 'Database Systems',
  courseUrl: 'https://15445.courses.cs.cmu.edu/',
  description: '用于收录课程资料、实验附件和学习记录。',
  topics: ['数据库', '系统'],
},
```

构建时会根据 `learningCourses` 自动预渲染课程总览、每门课程的详情页和资料索引。没有资料或记录的课程会保留空板块，不需要先写文章或添加占位内容。

## 添加课程资料

在本地仓库中将文件放入 `public/materials/<slug>/`。可以按主题建立子目录，例如 `notes/`、`code/`、`experiments/`。课程资料列表会在构建时自动归集，无需手动维护清单。

资料访问地址是 `/materials/<slug>/<文件相对路径>`，文章可以直接链接到该地址。README、隐藏文件和顶层 `templates/` 目录不会进入资料索引，符号链接不会被索引。

资料列表按「中文资料」「英文资料」「实验附件」「其他资料」分组，仅显示有文件的分组。新增课程也使用相同规则：

- `experiments/`、`code/` 下的文件归入实验附件。
- 已登记的在线阅读稿和对应原文分别使用 `language` 与 `sourceLanguage` 指定语言，值为 `zh` 或 `en`。
- 其他资料放在课程目录的 `zh/` 或 `en/` 下，或使用 `-zh`、`-en` 文件名后缀，例如 `lecture-01-zh.pdf`。文件格式不会决定语言。
- 需要保留旧文件路径时，可在 `data/learning/materials.ts` 的 `learningMaterialLanguages` 中登记完整公开路径及语言。例如现有中文译稿压缩包保持原地址并归入中文资料。
- 未标明语言的文件归入其他资料，仍保留访问和下载入口。

## 在线阅读资料

Markdown 资料可在 `data/learning/materials.ts` 的 `learningMaterialArticles` 中登记 `courseSlug`、`slug`、`fileName`、`title`、`description`、`language`、`sourcePdf` 和 `sourceLanguage`。文件名对应课程资料目录中的译稿，`sourcePdf` 对应公开的原始 PDF。登记后课程资料列表会显示资料标题和在线阅读入口，构建自动生成 `/learning/<courseSlug>/materials/<slug>` 及内容接口。

阅读页面以博客样式呈现正文、目录、表格、代码和 KaTeX 公式，保留译稿下载与原文 PDF 对照。插图在 Markdown 中使用公开绝对路径，例如 `![原文图 1](/images/materials/<courseSlug>/<topic>/figure-01.webp)`。原图资源放在 `public/images/materials/`，避免让图片逐项进入课程资料列表。资料阅读页独立于 `content/posts/`，不会作为学习记录计数或同步到 Halo。

公式使用 LaTeX 定界符：行内公式写成 `$d_{\mathrm{model}} = 512$`；独立公式使用单独成行的 `$$`，上下留空行。例如：

```markdown
$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
```

反引号用于代码和 API 名称，公式放在反引号中会按普通代码显示。下标、幂次、根号和矩阵维度使用 `d_{\mathrm{model}}`、`10^{-9}`、`\sqrt{d_k}`、`\mathbb{R}^{M \times K}` 等标准 LaTeX 写法；更新译稿后，同步更新对应下载压缩包中的 Markdown。

此目录内容会公开访问，提交前清理私人信息、访问令牌和密码。课程官方资料可以在学习记录中链接到原站。

## 添加学习记录

推荐复制通用空模板 `public/materials/templates/learning-note.md` 到 `content/posts/<课程slug>-<主题>.md`，例如 `cmu-15-779-<主题>.md`。模板公开地址是 `/materials/templates/learning-note.md`，可在课程详情页下载。填写模板里的标题、日期、描述和 `series` 占位字段，再编写正文。文章的 `series` 必须与课程登记表中的值完全一致，专题才能自动归集。

- `seriesOrder` 从 `1` 开始，按实际学习顺序递增。
- 如果记录有配套代码或附件，可在 front matter 中填写 `codeUrl` 和 `attachmentUrl`；课程页会在学习记录卡片下显示对应入口。
- 填写真实的标题、日期、描述与正文。
- 未完成记录保持 `draft: true`，准备公开时改为 `draft: false`。
- 当前 GitHub Pages 学习专题使用 `hidden: false`、`haloSync: false`、`haloPublished: false`。

通用模板默认 `draft: true`，没有预填任何课程内容。模板中的日期和系列名占位符不用于正式记录。

现有 CMU 15-779 模板仍保留在 `public/materials/cmu-15-779/templates/learning-note.md`，公开地址保持 `/materials/cmu-15-779/templates/learning-note.md`，已有链接可以继续使用。

## 通过 Git 发布

1. 在本地仓库中修改课程登记信息、资料文件或 Markdown。
2. 运行 `npm run build`，检查课程总览、详情页、资料链接和学习记录。
3. 用 `git status` 检查改动，再用 `git add` 暂存本次需要发布的文件。
4. 在 `main` 分支提交，执行 `git push origin main`，由现有 GitHub Pages 流程构建和部署。

添加、替换或删除资料后，也需要提交并推送。部署完成后，课程内容和资料索引随构建结果更新。
