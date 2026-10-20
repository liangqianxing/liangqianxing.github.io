# Stanford CS336 学习索引

更新于 2026-10-20。

CS336（Language Modeling from Scratch）围绕语言模型的基础实现、系统优化、规模规律、数据和对齐展开。本专栏记录自学过程，按实际完成顺序整理，不代表课程官方解答。

## 课程与代码

- [课程官网](https://cs336.stanford.edu/)
- [Stanford CS336 官方 GitHub](https://github.com/stanford-cs336)
- [Assignment 1：Basics 官方仓库](https://github.com/stanford-cs336/assignment1-basics)
- [个人学习代码与验证记录](https://github.com/liangqianxing/cs336-labs)
- [专栏主页](https://liangqianxing.github.io/learning/cs336/)

课程的五部分作业分别涉及基础语言模型、系统优化、Scaling Laws、数据处理与对齐；具体题目、版本和要求以课程官网及对应年度仓库为准。

## 当前阅读顺序

| 顺序 | 学习记录 | 关注的问题 |
| --- | --- | --- |
| 01 | [Linear 与 Embedding](https://liangqianxing.github.io/posts/cs336-linear-embedding-shapes/) | 类、参数与 forward，矩阵转置和张量形状 |
| 02 | [RMSNorm 与 SwiGLU](https://liangqianxing.github.io/posts/cs336-rmsnorm-swiglu/) | 最后一维归一化、广播、SiLU 与门控 |
| 03 | [Softmax 与 Attention Mask](https://liangqianxing.github.io/posts/cs336-softmax-attention-mask/) | 数值稳定、Q/K/V、负无穷与可见位置 |

本轮从基础模型组件开始，跳过已经接触过的 BPE。当前记录中的已完成组件为 Linear、Embedding、RMSNorm、SiLU、SwiGLU、Softmax 与缩放点积注意力，对应官方测试共 8 项。具体命令和结果见代码仓库的进度记录。

下一步是多头自注意力的拆头与合头。RoPE、完整 Transformer、训练与 tokenizer 的测试状态，以后续代码提交为准。

## 参考文档

- [PyTorch nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)
- [PyTorch 广播语义](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

注意：这里已实现的 Attention 接收外部 mask；不传 mask 时不会自动启用因果约束。布尔 mask 的语义为 `True` 允许关注，`False` 屏蔽。
