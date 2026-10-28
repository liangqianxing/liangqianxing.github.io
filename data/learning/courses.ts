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
    description: '不依赖高层黑盒 API，从底层张量契约与算子原语手写实现现代语言模型核心组件：涵盖 Linear / Embedding 参数初始化、RMSNorm + SwiGLU 门控前馈网络、数值稳定 Softmax + 因果遮罩、多头自注意力（MHA）与 RoPE 旋转位置编码，并通过课程官方 pytest 单元测试。',
    topics: ['PyTorch', 'Transformer', 'RMSNorm / SwiGLU', 'RoPE & Attention'],
    cover: {
      src: '/images/posts/cs336-softmax-attention-mask/cs336-softmax-attention-mask-cover.webp',
      alt: '注意力分数、权重与因果遮罩的 AI 辅助概念插画',
      caption: '从基础算子到 Attention · 从零实现',
    },
  },
]

export function getLearningCourse(slug: string): LearningCourse | undefined {
  return learningCourses.find((course) => course.slug === slug)
}
