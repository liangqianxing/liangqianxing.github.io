export type LearningMaterialLanguage = 'zh' | 'en'
export type LearningMaterialCategory = LearningMaterialLanguage | 'attachments' | 'other'

export const learningMaterialCategories: { id: LearningMaterialCategory, title: string }[] = [
  { id: 'zh', title: '中文资料' },
  { id: 'en', title: '英文资料' },
  { id: 'attachments', title: '实验附件' },
  { id: 'other', title: '其他资料' },
]

export interface LearningMaterialArticle {
  courseSlug: string
  slug: string
  fileName: string
  title: string
  description: string
  language: LearningMaterialLanguage
  sourcePdf: string
  sourceLanguage: LearningMaterialLanguage
}

export const learningMaterialArticles: LearningMaterialArticle[] = [
  {
    courseSlug: 'cmu-15-779',
    slug: 'pytorch-paper-zh',
    fileName: 'pytorch-paper-zh.md',
    title: 'PyTorch：一种命令式风格的高性能深度学习库',
    description: 'PyTorch 论文中文学习译稿。',
    language: 'zh',
    sourcePdf: '/materials/cmu-15-779/originals/pytorch-paper-original.pdf',
    sourceLanguage: 'en',
  },
  {
    courseSlug: 'cmu-15-779',
    slug: 'tensorflow-paper-zh',
    fileName: 'tensorflow-paper-zh.md',
    title: 'TensorFlow：面向大规模机器学习的系统',
    description: 'TensorFlow 论文中文学习译稿。',
    language: 'zh',
    sourcePdf: '/materials/cmu-15-779/originals/tensorflow-paper-original.pdf',
    sourceLanguage: 'en',
  },
  {
    courseSlug: 'cmu-15-779',
    slug: 'transformer-paper-zh',
    fileName: 'transformer-paper-zh.md',
    title: 'Attention Is All You Need：只需注意力机制',
    description: 'Transformer 原论文中文学习译稿。',
    language: 'zh',
    sourcePdf: '/materials/cmu-15-779/originals/transformer-paper-original.pdf',
    sourceLanguage: 'en',
  },
  {
    courseSlug: 'cmu-15-779',
    slug: 'flashattention-paper-zh',
    fileName: 'flashattention-paper-zh.md',
    title: 'FlashAttention：具有 IO 感知的快速、内存高效精确注意力',
    description: 'FlashAttention 论文中文学习译稿。',
    language: 'zh',
    sourcePdf: '/materials/cmu-15-779/originals/flashattention-paper-original.pdf',
    sourceLanguage: 'en',
  },
  {
    courseSlug: 'cmu-15-779',
    slug: 'triton-paper-zh',
    fileName: 'triton-paper-zh.md',
    title: 'Triton：用于分块神经网络计算的中间语言和编译器',
    description: 'Triton 论文中文学习译稿。',
    language: 'zh',
    sourcePdf: '/materials/cmu-15-779/originals/triton-paper-original.pdf',
    sourceLanguage: 'en',
  },
  {
    courseSlug: 'cmu-15-779',
    slug: 'tvm-slides-zh',
    fileName: 'tvm-slides-zh.md',
    title: 'TVM：面向深度学习的自动化端到端优化编译器',
    description: 'TVM OSDI 2018 幻灯片中文译稿。',
    language: 'zh',
    sourcePdf: '/materials/cmu-15-779/originals/tvm-slides-original.pdf',
    sourceLanguage: 'en',
  },
]

// Use explicit metadata for existing files whose paths have no language marker.
export const learningMaterialLanguages: Record<string, LearningMaterialLanguage> = {
  '/materials/cmu-15-779/cmu-15-779-translations.zip': 'zh',
}

export function getLearningMaterialCategory(file: { name: string, path: string }, courseSlug: string): LearningMaterialCategory {
  const relativePath = file.path.slice(`/materials/${encodeURIComponent(courseSlug)}/`.length)
  const directory = relativePath.split('/')[0]
  if (directory === 'experiments' || directory === 'code') return 'attachments'

  const article = getLearningMaterialArticleByFileName(file.name, courseSlug)
  if (article) return article.language
  const source = learningMaterialArticles.find(material => material.courseSlug === courseSlug && material.sourcePdf === file.path)
  if (source) return source.sourceLanguage
  if (learningMaterialLanguages[file.path]) return learningMaterialLanguages[file.path]

  if (directory === 'zh' || directory === 'en') return directory
  const languageSuffix = file.name.match(/-(zh|en)\.[^.]+$/i)?.[1]?.toLowerCase()
  if (languageSuffix === 'zh' || languageSuffix === 'en') return languageSuffix
  return 'other'
}

export function getLearningMaterialArticle(slug: string, courseSlug?: string): LearningMaterialArticle | undefined {
  return learningMaterialArticles.find(material => material.slug === slug && (!courseSlug || material.courseSlug === courseSlug))
}

export function getLearningMaterialArticleByFileName(fileName: string, courseSlug?: string): LearningMaterialArticle | undefined {
  return learningMaterialArticles.find(material => material.fileName === fileName && (!courseSlug || material.courseSlug === courseSlug))
}
