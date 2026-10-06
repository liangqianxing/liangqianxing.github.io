import { formatDate, tagSlug } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

export interface ArchiveTag {
  tag: string
  slug: string
  count: number
  latestPost: PostMeta
}

export function tagPath(tag: string): string {
  return `/tags/${encodeURIComponent(tagSlug(tag))}`
}

export function archiveTone(value: string): string {
  let hash = 0
  for (const character of value) hash = (hash * 31 + character.codePointAt(0)!) >>> 0
  return ['purple', 'green', 'blue', 'orange'][hash % 4]
}

export function collectArchiveTags(posts: PostMeta[]): ArchiveTag[] {
  const entries = new Map<string, ArchiveTag>()
  const countedPaths = new Map<string, Set<string>>()
  for (const post of posts) {
    for (const originalTag of post.tags ?? []) {
      const tag = originalTag.trim()
      const slug = tagSlug(tag)
      if (!slug) continue
      const entry = entries.get(slug)
      if (!entry) {
        entries.set(slug, { tag, slug, count: 1, latestPost: post })
        countedPaths.set(slug, new Set([post.path]))
        continue
      }
      const paths = countedPaths.get(slug)!
      if (paths.has(post.path)) continue
      paths.add(post.path)
      entry.count += 1
      const candidateDate = new Date(post.date).getTime()
      const latestDate = new Date(entry.latestPost.date).getTime()
      if (Number.isFinite(candidateDate) && (!Number.isFinite(latestDate) || candidateDate > latestDate)) entry.latestPost = post
    }
  }
  return [...entries.values()].sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag, 'zh-CN'))
}

export function groupArchiveByYear(posts: PostMeta[]): [string, PostMeta[]][] {
  const years = new Map<string, PostMeta[]>()
  for (const post of posts) {
    const year = formatDate(post.date).slice(0, 4) || '未标日期'
    if (!years.has(year)) years.set(year, [])
    years.get(year)!.push(post)
  }
  return [...years.entries()].sort((a, b) => {
    if (a[0] === '未标日期') return 1
    if (b[0] === '未标日期') return -1
    return b[0].localeCompare(a[0])
  })
}
