# CMU 15-779 专题维护

本专题发布在 `liangqianxing.github.io`，预留 CMU 15-779 的学习资料归档和学习记录板块。目前没有预填学习计划、课程目录或启动文章，内容由本人后续添加。新增内容仅发布到当前 GitHub Pages 站点，使用 `haloSync: false` 避免同步到 Halo。

## 新增学习记录

1. 复制 `public/materials/cmu-15-779/templates/learning-note.md` 到 `content/posts/cmu-15-779-<主题>.md`。
2. 填写标题、真实日期、描述和正文。保持 `series: CMU 15-779`，第一篇使用 `seriesOrder: 1`，之后依次递增。
3. 准备完成后设置 `draft: false`，保持 `hidden: false`、`haloSync: false`、`haloPublished: false`。
4. 运行 `npm run build` 并检查专题和文章页面。提交并推送后由现有 GitHub Pages 流程发布。

模板默认 `draft: true`，未完成的正文不会作为公开学习记录展示。不要将 `YYYY-MM-DD` 日期占位符用于正式记录。

建议文件名按主题命名，例如 `cmu-15-779-pytorch-systems.md`、`cmu-15-779-cuda-reduction.md`。专题会按 `seriesOrder` 自动归集公开文章。系列名必须完全一致，不需要手动维护文章列表。

## 添加学习资料

将资料放在 `public/materials/cmu-15-779/` 下，可以创建子目录。构建时自动生成资料索引，无需另写清单。

```text
public/materials/cmu-15-779/
├── notes/
│   └── reading-notes.pdf
└── experiments/
    ├── kernel-experiment.zip
    └── benchmark.csv
```

文件可通过 `/materials/cmu-15-779/<相对路径>` 访问。文章中可以直接链接附件，例如 `[实验代码](/materials/cmu-15-779/experiments/kernel-experiment.zip)`。README、隐藏文件和 `templates/` 不会显示在资料列表中，符号链接不会被索引。

添加、修改或删除附件后重新构建并发布，公开索引随构建结果更新。此站点是静态站点，资料通过仓库上传和发布；专题本身没有登录或网页上传后台。请在上传前清理私人信息和凭据。

## 更新专题信息

专题名称、简介与课程主页链接集中在 `data/learning/cmu-15-779.ts`。这里仅保留专题元数据，不预填学习内容。

课程主页：<https://www.cs.cmu.edu/~zhihaoj2/15-779/>。后续资料和记录可以持续添加，无需重新创建专题。
