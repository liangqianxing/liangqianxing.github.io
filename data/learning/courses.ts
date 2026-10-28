export interface LearningCourse {
  slug: string
  code: string
  institution: string
  series: string
  title: string
  courseTitle: string
  courseUrl: string
  codeUrl?: string
  description: string
  topics: string[]
  cover?: { src: string, alt: string, caption?: string }
}

export const learningCourses: LearningCourse[] = [
  {
    slug: 'cmu-15-779',
    code: '15-779',
    institution: 'CMU',
    series: 'CMU 15-779',
    title: 'CMU 15-779',
    courseTitle: 'Advanced Topics in Machine Learning Systems (LLM Edition)',
    courseUrl: 'https://www.cs.cmu.edu/~zhihaoj2/15-779/',
    codeUrl: 'https://github.com/liangqianxing/cmu-15-779-labs',
    description: '围绕大模型机器学习系统核心脉络，收录 6 篇经典系统论文中文精译（PyTorch / TensorFlow / Transformer / FlashAttention / Triton / TVM）与 14 篇从零实现的 GPU 算子及服务实验记录（共享内存分块 GEMM、Online Softmax、KV Cache 粒度建模、Speculative Sampling、LoRA 量化与 MoE 路由）。',
    topics: ['ML Systems', 'CUDA / Triton', 'FlashAttention', 'KV Cache Serving'],
    cover: {
      src: '/images/learning/cmu-15-779-flashattention-figure1.webp',
      alt: 'FlashAttention 论文图 1 的左、中子图：GPU 内存层次与注意力分块计算',
      caption: 'FlashAttention · 图 1（局部）',
    },
  },
  {
    slug: 'cs336',
    code: 'CS336',
    institution: 'Stanford',
    series: 'Stanford CS336',
    title: 'Stanford CS336',
    courseTitle: 'Language Modeling from Scratch',
    courseUrl: 'https://cs336.stanford.edu/',
    codeUrl: 'https://github.com/liangqianxing/cs336-labs',
    description: '不依赖 PyTorch 高层黑盒 API，从底层张量契约与算子原语完整实现 Stanford CS336 Assignment 1 全栈语言模型：涵盖 Byte-Level BPE 分词器（倒排索引增量训练 + 流式编码）、Pre-Norm Transformer LM（Linear / Embedding / RMSNorm / SwiGLU / RoPE / 因果多头自注意力）、数值稳定 Cross-Entropy、AdamW 优化器、余弦退火调度、Checkpoint 序列化与 Top-p 采样解码，100% 通过官方全部 48 项单元测试。',
    topics: ['Byte-Level BPE', 'Pre-Norm Transformer & RoPE', 'AdamW & Cosine Schedule', 'FLOPs & Memory Accounting'],
    cover: {
      src: '/images/posts/cs336-pre-norm-transformer-block-lm/cs336-pre-norm-transformer-block-lm-cover.webp',
      alt: '干净残差流主干穿过多个 Pre-Norm 子层并汇聚到词表投影层的 AI 辅助概念插画',
      caption: 'CS336 Assignment 1 · 全栈从零实现（48/48 测试通过）',
    },
  },
]

export function getLearningCourse(slug: string): LearningCourse | undefined {
  return learningCourses.find((course) => course.slug === slug)
}
