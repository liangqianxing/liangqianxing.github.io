export interface LearningMaterialArticle {
  courseSlug: string
  slug: string
  fileName: string
  title: string
  description: string
  sourcePdf: string
}

export const learningMaterialArticles: LearningMaterialArticle[] = [
  {
    courseSlug: 'cmu-15-779',
    slug: 'pytorch-paper-zh',
    fileName: 'pytorch-paper-zh.md',
    title: 'PyTorch：一种命令式风格的高性能深度学习库',
    description: 'PyTorch 论文中文学习译稿。',
    sourcePdf: '/materials/cmu-15-779/originals/pytorch-paper-original.pdf',
  },
  {
    courseSlug: 'cmu-15-779',
    slug: 'tensorflow-paper-zh',
    fileName: 'tensorflow-paper-zh.md',
    title: 'TensorFlow：面向大规模机器学习的系统',
    description: 'TensorFlow 论文中文学习译稿。',
    sourcePdf: '/materials/cmu-15-779/originals/tensorflow-paper-original.pdf',
  },
  {
    courseSlug: 'cmu-15-779',
    slug: 'transformer-paper-zh',
    fileName: 'transformer-paper-zh.md',
    title: 'Attention Is All You Need：只需注意力机制',
    description: 'Transformer 原论文中文学习译稿。',
    sourcePdf: '/materials/cmu-15-779/originals/transformer-paper-original.pdf',
  },
  {
    courseSlug: 'cmu-15-779',
    slug: 'flashattention-paper-zh',
    fileName: 'flashattention-paper-zh.md',
    title: 'FlashAttention：具有 IO 感知的快速、内存高效精确注意力',
    description: 'FlashAttention 论文中文学习译稿。',
    sourcePdf: '/materials/cmu-15-779/originals/flashattention-paper-original.pdf',
  },
  {
    courseSlug: 'cmu-15-779',
    slug: 'triton-paper-zh',
    fileName: 'triton-paper-zh.md',
    title: 'Triton：用于分块神经网络计算的中间语言和编译器',
    description: 'Triton 论文中文学习译稿。',
    sourcePdf: '/materials/cmu-15-779/originals/triton-paper-original.pdf',
  },
  {
    courseSlug: 'cmu-15-779',
    slug: 'tvm-slides-zh',
    fileName: 'tvm-slides-zh.md',
    title: 'TVM：面向深度学习的自动化端到端优化编译器',
    description: 'TVM OSDI 2018 幻灯片中文译稿。',
    sourcePdf: '/materials/cmu-15-779/originals/tvm-slides-original.pdf',
  },
]

export function getLearningMaterialArticle(slug: string, courseSlug?: string): LearningMaterialArticle | undefined {
  return learningMaterialArticles.find(material => material.slug === slug && (!courseSlug || material.courseSlug === courseSlug))
}

export function getLearningMaterialArticleByFileName(fileName: string, courseSlug?: string): LearningMaterialArticle | undefined {
  return learningMaterialArticles.find(material => material.fileName === fileName && (!courseSlug || material.courseSlug === courseSlug))
}
