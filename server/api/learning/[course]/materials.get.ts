import { join } from 'node:path'
import { getLearningCourse } from '../../../../data/learning/courses'
import { readLearningMaterials } from '../../../utils/learning-materials'

export type { LearningMaterial } from '../../../utils/learning-materials'

export default defineEventHandler((event) => {
  const course = getLearningCourse(getRouterParam(event, 'course', { decode: true }) ?? '')
  if (!course) {
    throw createError({ statusCode: 404, statusMessage: 'Learning course not found' })
  }

  return readLearningMaterials(
    join(process.cwd(), 'public', 'materials', course.slug),
    `/materials/${encodeURIComponent(course.slug)}/`,
  )
})
