# CS336 专栏封面生成记录

三张封面均使用用户指定的 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 技能，按照结构化中文提示词提交。技能文件版本 Git blob SHA：`0ac6e811f7ce24a41e9d1a86b35c513dac147ca9`。

请求规格：16:9，2K。服务返回原 PNG 尺寸为 2752 × 1536；发布版本居中裁去极窄的左右边缘，导出精确 1600 × 900 WebP。三张发布文件共约 181 KiB。

| 主题 | Prompt | Task ID | 发布文件 | QA |
| --- | --- | --- | --- | --- |
| token、Embedding 与 Linear | `prompts/cs336-linear-embedding-shapes-cover.txt` | `6ad730c1-e4b0d5de-7ed129e7-1792487617479` | `cs336-linear-embedding-shapes-cover.webp` | 已打开逐张检查：符号片、连续向量条与映射平面清楚，主体完整，无文字、人物、Logo |
| RMSNorm 与 SwiGLU | `prompts/cs336-rmsnorm-swiglu-cover.txt` | `6ad730c1-e4b0c80c-6f8a21ce-1792487617758` | `cs336-rmsnorm-swiglu-cover.webp` | 已打开逐张检查：尺度整理后仍保留差异，双支路与橙色门控片可辨，无文字、人物、Logo |
| Softmax、Attention 与因果遮罩 | `prompts/cs336-softmax-attention-mask-cover.txt` | `6ad730c2-e4b0fec5-2e5f5b5b-1792487618024` | `cs336-softmax-attention-mask-cover.webp` | 已打开逐张检查：面板层次清楚，左下区域可见、右上区域遮住，无虚构数字、文字、人物、Logo |

这些封面是概念插画。正文中的公式、张量形状和因果遮罩示例独立承担精确解释；插画中的块数、长度和连线不代表真实数据或计算结果。

原始 API 响应与原 PNG 保留于本地工作目录，不随博客发布。生成令牌未写入提示词、图片、博客代码或生成记录。
