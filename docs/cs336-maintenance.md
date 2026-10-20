# Stanford CS336 专栏维护

专栏地址：<https://liangqianxing.github.io/learning/cs336/>。
配套代码：<https://github.com/liangqianxing/cs336-labs>。

## 内容与进度

课程登记位于 `data/learning/courses.ts`，固定使用 `slug: cs336`、`series: Stanford CS336`。
学习记录放在 `content/posts/cs336-*.md`，按 `seriesOrder` 从 1 递增。当前四篇整理自实际学习过程，依次覆盖 Linear / Embedding、RMSNorm / SwiGLU、Softmax / Scaled Dot-Product Attention，以及多头自注意力的拆头、合头与因果 mask。

2026-10-20 的前三篇保留首次发布时的学习状态及代码提交；同日追加的第四篇记录已通过官方测试的无 RoPE 多头自注意力。当前已完成组件对应 9 项官方测试。RoPE、完整 Transformer、训练循环等后续内容需要在学习并验证后另行补充。本轮跳过 BPE，不把其他项目中已学过 BPE 写成本仓库通过 tokenizer 测试。

课程学习记录使用 GitHub Pages 的可见性约定：`draft: false`、`hidden: false`、`haloSync: false`、`haloPublished: false`。这是当前公开学习专栏的发布配置，与普通 Halo 文章区分。

## 代码版本

`cs336-labs` 保留官方作业的来源与许可，收录已写代码及官方测试脚手架；它是学习进度快照，不是全部作业的完成声明。更新时先在原学习目录完成练习，使用该仓库的显式同步命令导入，再验证已完成测试、检查差异并提交。博客中的验证结果应链接到具体提交，避免随着默认分支更新而失去依据。

## 配图

配图使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) skill。文章图片位于 `public/images/posts/<slug>/`，文件名包含 slug 保证唯一。生成提示词、任务标识和视觉核对记录存放在 `docs/illustrations/cs336-20261020/`；API 凭据、完整原始响应和本机配置不入库。

封面是概念插画，不能作为矩阵尺寸、运算顺序或实验数据的依据。公式、张量形状与 mask 语义以正文和可复现测试为准。图片准备完成后，在 `scripts/optimize-images.py` 的白名单登记，生成响应式 WebP 和 `data/image-assets.json`。

## 发布

运行 `npm run build`，检查课程总览、CS336 专题、新增记录及代码入口；确认图片和公式渲染后提交到 `main`。GitHub Actions 完成 Pages 部署后，再检查线上 URL。后续有新的学习内容时，追加记录并更新代码仓库与当前进度，不预先宣称未完成的模块通过测试。
