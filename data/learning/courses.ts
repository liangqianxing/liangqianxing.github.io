export interface LearningCourse {
  slug: string
  code: string
  institution: string
  series: string
  title: string
  courseTitle: string
  courseUrl: string
  description: string
  topics: string[]
  cover?: { src: string, alt: string }
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
      src: '/images/learning/cmu-15-779-cover.svg',
      alt: '计算芯片连接张量矩阵与神经网络的机器学习系统插图',
    },
  },
]

export function getLearningCourse(slug: string): LearningCourse | undefined {
  return learningCourses.find((course) => course.slug === slug)
}
