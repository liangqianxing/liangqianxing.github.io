# CS336 多头注意力封面生成记录

使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 技能，按结构化中文提示词提交。技能文件版本 Git blob SHA：`0ac6e811f7ce24a41e9d1a86b35c513dac147ca9`。

- Prompt：`prompts/cs336-multihead-attention-shapes-cover.txt`
- 请求比例 / 尺寸：16:9 / 2K
- Task ID：`6ad73dd1-e4b086bd-aacf482a-1792490961828`
- 发布文件：`cs336-multihead-attention-shapes-cover.webp`，1600 × 900
- 原 PNG：服务返回 2752 × 1536，保留于本地；发布版居中裁去极窄的左右边缘后导出 WebP。

已逐一打开原 PNG 与发布 WebP 检查。奶白纸面、深蓝与青绿色半透明结构、少量暖橙的风格与前三张封面一致。完整的单宽束展开为几组等长平行窄轨道，再并拢为单宽束；每条窄轨道都保留贯穿全长的重复序列片段，表达同一序列的不同特征子空间。没有 RoPE、数字、文字、人物或 Logo。

本图是多头并行与合并的概念插画；轨道条数、片段数和薄片宽度不代表精确头数、序列长度或真实张量维度。正文中的形状推导和代码独立承担精确解释。

原始 API 响应与原 PNG 保留于本地工作目录，不随博客发布。生成令牌未写入提示词、图片、博客代码或生成记录。
