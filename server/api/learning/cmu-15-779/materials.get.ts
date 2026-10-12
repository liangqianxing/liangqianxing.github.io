import { join } from 'node:path'
import { readLearningMaterials } from '../../../utils/learning-materials'

export type { LearningMaterial } from '../../../utils/learning-materials'

export default defineEventHandler(() =>
  readLearningMaterials(join(process.cwd(), 'public', 'materials', 'cmu-15-779')),
)
