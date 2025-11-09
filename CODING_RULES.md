# Codex 协作规则

1. **命令与工具**
   - 默认使用 `powershell.exe -Command "<cmd>"` 并显式设置 `workdir`；必要时可用 `bash -lc`.
   - 文件/文本检索优先 `rg`；无则再用 `findstr`、`find` 等。
   - 命令输出只摘取关键信息，避免整段粘贴，失败需解释原因与替代方案。

2. **编辑流程**
   - 文件编码使用 UTF-8，保持现有代码风格；注释仅在复杂逻辑前简单说明。
   - 小范围修改优先 `apply_patch`，避免覆盖现有未提交改动；切勿回滚他人变更。
   - 大多数任务需先列出≥2步计划并在执行后更新；仅对显然简单的操作可跳过计划。

3. **Markdown 与 LaTeX**
   - 所有行内公式使用 `$...$`；标题或正文中的公式亦需 `$` 包裹。
   - 块级公式使用 `$$`。矩阵或多行推导可用 `aligned`/`bmatrix`，避免孤立等号变成 Markdown 标题。

4. **Hexo / MathJax 配置**
   - 文章需启用 `mathjax: true`；全局使用 `hexo-filter-mathjax`，复杂场景采用 `hexo-renderer-pandoc`。
   - 相关路径、参数写入 `_config.yml` 中的 `mathjax`、`pandoc` 节，必要时提供完整 `pandocPath`。

5. **外部工具**
   - 安装 Pandoc 等工具后若命令未在 PATH 中，可直接在配置里引用绝对路径，避免依赖用户环境变量。
