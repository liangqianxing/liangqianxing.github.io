import { lstat, readdir } from 'node:fs/promises'
import { extname, join } from 'node:path'

export interface LearningMaterial {
  name: string
  path: string
  bytes: number
  extension: string
}

const materialBasePath = '/materials/cmu-15-779/'

/** Build the public index without following links or exposing hidden files. */
export async function readLearningMaterials(directory: string): Promise<LearningMaterial[]> {
  const materials: LearningMaterial[] = []

  async function visit(parts: string[]): Promise<void> {
    const currentDirectory = join(directory, ...parts)
    let directoryStats
    try {
      directoryStats = await lstat(currentDirectory)
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return
      throw error
    }
    if (!directoryStats.isDirectory() || directoryStats.isSymbolicLink()) return

    const entries = await readdir(currentDirectory, { withFileTypes: true })
    for (const entry of entries) {
      if (entry.name.startsWith('.') || entry.name.toLowerCase() === 'readme.md' || entry.isSymbolicLink()) continue
      if (parts.length === 0 && entry.name === 'templates' && entry.isDirectory()) continue
      const entryParts = [...parts, entry.name]
      if (entry.isDirectory()) {
        await visit(entryParts)
      } else if (entry.isFile()) {
        const fileStats = await lstat(join(directory, ...entryParts))
        if (!fileStats.isFile() || fileStats.isSymbolicLink()) continue
        materials.push({
          name: entry.name,
          path: materialBasePath + entryParts.map((part) => encodeURIComponent(part)).join('/'),
          bytes: fileStats.size,
          extension: extname(entry.name).slice(1).toLowerCase(),
        })
      }
    }
  }

  await visit([])
  return materials.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0)
}
