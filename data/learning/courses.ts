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
    description: '用于收录课程资料、实验附件和学习记录。',
    topics: ['ML Systems', 'LLM'],
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
    description: '从张量形状和 PyTorch 基础层开始，记录从零搭建语言模型时的理解、踩坑与测试。已完成带因果 mask 的多头自注意力，下一步学习 RoPE。',
    topics: ['PyTorch', 'Transformer', '语言模型'],
    cover: {
      src: '/images/posts/cs336-softmax-attention-mask/cs336-softmax-attention-mask-cover.webp',
      alt: '注意力分数、权重与因果遮罩的 AI 辅助概念插画',
      caption: '从基础层到 Attention · 自学记录',
    },
  },
]

export function getLearningCourse(slug: string): LearningCourse | undefined {
  return learningCourses.find((course) => course.slug === slug)
}
