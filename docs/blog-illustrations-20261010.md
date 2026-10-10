# 博客插图维护记录 · 2026-10-10

本次为“LLM 从零实现”系列的三篇公开文章制作了 3 张封面、4 张正文示意图。以下只记录最终采用的 7 张图片。生成后端为 Sankuai AIGC，模型 `gemini-3-pro-image`，请求尺寸均为 `2K`。

记录依据为本次生成目录中的 `asset-records.json`、`jobs.json` 和最终 WebP 文件。任务 ID 对应实际采用的生成任务；原始生成响应、下载地址和服务凭据不纳入此文件。

## 最终资产

以下文件均位于 `public/images/posts/`，正文或封面字段使用去掉 `public` 前缀的 URL。尺寸单位为像素；请求比例不代表裁切后的最终图片比例。

| 最终文件 | 用途 | 请求比例 | 原图尺寸 | 最终尺寸 | 文件大小 |
| --- | --- | --- | --- | --- | --- |
| `bpe-tokenizer-from-scratch/bpe-cover.webp` | BPE 文章封面 | 16:9 | 2752 × 1536 | 1600 × 900 | 46,508 B |
| `transformer-from-scratch/transformer-cover.webp` | Transformer 文章封面 | 16:9 | 2752 × 1536 | 1600 × 900 | 103,088 B |
| `lm-lab-project-complete/lm-lab-cover.webp` | LM Lab 项目复盘封面 | 16:9 | 2752 × 1536 | 1600 × 900 | 68,410 B |
| `bpe-tokenizer-from-scratch/bpe-merge-process.webp` | BPE 三阶段合并示例 | 16:9 | 2752 × 1536 | 1800 × 513 | 31,330 B |
| `transformer-from-scratch/transformer-architecture.webp` | Decoder-only 模型前向总览 | 16:9 | 2752 × 1536 | 1800 × 462 | 50,940 B |
| `transformer-from-scratch/transformer-causal-mask.webp` | 4 × 4 因果掩码 | 4:3 | 2400 × 1792 | 1800 × 1344 | 43,438 B |
| `lm-lab-project-complete/lm-lab-training-cycle.webp` | 单步训练的目标数据流 | 16:9 | 2752 × 1536 | 1800 × 854 | 63,384 B |

## 生成任务

| 资产名称 | 实际生成任务 ID |
| --- | --- |
| `bpe-cover` | `6aca2ca7-e4b066dd-5e64cc57-1791634599803` |
| `transformer-cover` | `6aca2ca7-e4b0d5de-7ed11e45-1791634599788` |
| `lm-lab-cover` | `6aca2d8d-e4b0f311-02f92033-1791634829167` |
| `bpe-merge-process` | `6aca2cca-e4b0b78d-fe339592-1791634634658` |
| `transformer-architecture` | `6aca2cfb-e4b0c80c-6f8a1636-1791634683851` |
| `transformer-causal-mask` | `6aca2cfb-e4b0686c-92836fc3-1791634683897` |
| `lm-lab-training-cycle` | `6aca2d2c-e4b032d1-85e9d0da-1791634732768` |

## 提示词摘要与关键限制

统一视觉要求：温暖纸白背景，低饱和鼠尾草绿、浅紫和少量暖橙，深灰细线；采用克制的技术出版物编辑插画，轻微纸质纹理、柔和漫射光和充足留白。不要水印、品牌 Logo、装饰性乱码、照片、夸张霓虹或重复主体，不直接模仿在世艺术家。

| 资产 | 简洁提示词 | 特定限制 |
| --- | --- | --- |
| BPE 封面 | 细小字节积木从左进入，经组合后变成较少而更长的子词积木；中央横向构图 | 完全无文字、字母或数字；表达组合，不表现数据丢失；四周留白 |
| Transformer 封面 | 输入 token 积木进入分层计算模块，以矩阵片连结，输出概率柱 | 无文字、公式或数学标签；矩阵仅作机制比喻，避免信息过密 |
| LM Lab 封面 | 收纳盒与积木、层叠计算板、机械齿轮和空白纸页构成微型工程工作台 | 纯图形，无任何文字、标签、数字或符号；纸页必须空白；不绘制指标曲线 |
| BPE 合并图 | 三阶段从左到右：`97 98 97 98` → `256 256` → `257`，配相应合并规则 | 严格 4 块、2 块、1 块；标签 `UTF-8 bytes`、`ab + ab`、`abab`；无额外阶段或重叠合并 |
| Transformer 架构图 | `Token IDs` → `Embedding + Position` → `Transformer Block` × N → `Linear` → `Probabilities` | Block 内为 `Causal Attention` 和 `Feed Forward`；末端箭头标 `Softmax`；无 Encoder 或 cross-attention；简化图省略残差、归一化和尺寸 |
| 因果掩码图 | 4 行 4 列，列为 key、行为 query，标 `t1` 至 `t4`；可见格绿色，遮挡格灰色 | 对角线及以下 10 格可见，上三角 6 格遮挡；格子无数字；图例只写 `Visible` 和 `Masked` |
| 训练闭环图 | 上行向右：`Token batch` → `Transformer` → `Logits` → `Loss`；下行向左：`Gradients` → `AdamW` → `Weights` | Loss 向下接 Gradients，Weights 返回 Transformer；批次另以 `Targets` 线接 Loss；只有 7 个模块，不画虚构测试或性能结果 |

## 技术核对点

### BPE 合并

`abab` 的 UTF-8 字节为 `[97, 98, 97, 98]`。以新增 token ID 从 256 开始的示例规则，第一次将 `(97, 98)` 从左到右非重叠合并为 256，得到 `[256, 256]`；第二次将 `(256, 256)` 合并为 257，得到 `[257]`。ID 256 和 257 是这组示例词表的分配结果，不是所有 BPE 词表通用的编号。

### Decoder-only 前向架构

架构图表达因果自注意力语言模型，没有 Encoder 或交叉注意力。实际模型前向输出 logits；图中继续通过 Softmax 展示其对应的词表概率。正文应说明这是简化总览，归一化、残差和输出权重共享等实现细节以正文及代码为准。

### 因果掩码

读取矩阵时，行对应 query 位置，列对应 key 位置。第 1 至第 4 行分别允许看到 1、2、3、4 个位置，总计 `1 + 2 + 3 + 4 = 10` 个可见格；严格上三角为 6 个遮挡格。对角线可见，因为每个位置允许关注自身。

### 目标训练闭环

输入与右移一位的 Targets 来自同一批次。模型产出 logits，损失结合 logits 与 Targets，梯度经过反向链路交给 AdamW，更新权重后用于下一次前向。图中 `Gradients` 是反向过程的概括，不能理解为 Loss 直接跳过模型就产生了全部参数梯度。

这张图描述应当完成的训练流程，不能当作当前 LM Lab 源码已完成端到端训练验证的证据。文章中保留的实现边界说明和源码版本记录仍然适用。

## 文件处理与后续维护

生成原图转换为 WebP，用于减小网页传输体积。封面统一为 1600 × 900；正文图统一到 1800 像素宽。横向合并图、架构图和训练闭环图裁去了外围多余空白，因此最终比例与请求比例不同；因果掩码图保留了原图的矩阵布局比例。裁切不能截断标签、箭头或图例。

记录尺寸已对最终文件逐张读取核验，格式均为 WebP。后续重新生成时，应再次检查数字、英文标签、箭头方向与矩阵填色，并在文章显示宽度下确认可读性；不能仅凭提示词正确就认定生成图正确。

图片更换后同步维护文章的 `cover`、`coverAlt`、正文图片替代文本与说明。技术图说明应区分简化机制、示例数据和实测结果。生成任务更新时，只记录最终采用的任务及资产，继续避免把凭据、下载地址或原始响应加入仓库。
