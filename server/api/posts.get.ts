/**
 * server/api/posts.get.ts
 *
 * 直接从文件系统读取 markdown 文件并解析 frontmatter。
 * 绕过 @nuxt/content SQLite WASM 在静态构建 SSR 阶段未初始化的问题。
 * nuxt generate prerender 时 Nitro server 可访问文件系统，此方式完全可靠。
 */
import { readdirSync, readFileSync } from 'node:fs'
import { join, basename } from 'node:path'

export interface PostMeta {
  slug: string
  path: string
  title: string
  date: string
  tags: string[]
  description: string
  draft: boolean
  hidden: boolean
  published: boolean
  readingTime: number
}

/** 简单 frontmatter 解析（YAML only） */
function parseFrontmatter(source: string): { data: Record<string, unknown>; body: string } {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  if (!match) return { data: {}, body: source }
  const yaml = match[1]
  const body = source.slice(match[0].length)
  const data: Record<string, unknown> = {}

  for (const line of yaml.split('\n')) {
    const colon = line.indexOf(':')
    if (colon < 0) continue
    const key = line.slice(0, colon).trim()
    const val = line.slice(colon + 1).trim()
    if (!key || key.startsWith('-')) continue

    if (val === 'true') { data[key] = true; continue }
    if (val === 'false') { data[key] = false; continue }
    if (val === '' || val === null || val === 'null') { data[key] = null; continue }
    // numeric
    if (/^\d+$/.test(val)) { data[key] = parseInt(val, 10); continue }
    // array start (next lines are - items)
    if (val === '') { data[key] = []; continue }
    data[key] = val.replace(/^['"]|['"]$/g, '')
  }

  // Parse inline arrays: tags: [LLM, Agent]
  const arrayInline = yaml.match(new RegExp(`(\\w+):\\s*\\[([^\\]]+)\\]`, 'g'))
  if (arrayInline) {
    for (const m of arrayInline) {
      const [, k, v] = m.match(/([\w-]+):\s*\[([^\]]+)\]/) ?? []
      if (k && v) data[k] = v.split(',').map((s: string) => s.trim().replace(/^['"]|['"]$/g, ''))
    }
  }

  // Parse multi-line arrays:  - item
  const mlArray = yaml.matchAll(/^(\w+):\s*\n((?:\s+-\s+.+\n?)*)/gm)
  for (const m of mlArray) {
    const [, k, block] = m
    data[k] = block.split('\n')
      .filter((l) => l.trim().startsWith('-'))
      .map((l) => l.trim().replace(/^-\s*/, '').replace(/^['"]|['"]$/g, ''))
  }

  return { data, body }
}

/** 估算阅读时间 */
function readingTime(body: string): number {
  const zh = body.match(/[一-鿿]/g)?.length ?? 0
  const en = body.replace(/[一-鿿]/g, ' ').match(/[A-Za-z0-9]+/g)?.length ?? 0
  return Math.max(1, Math.ceil(zh / 300 + en / 200))
}

export default defineEventHandler((_event): PostMeta[] => {
  const dir = join(process.cwd(), 'content', 'posts')
  const files = readdirSync(dir).filter((f) => f.endsWith('.md'))

  const posts: PostMeta[] = []

  for (const file of files) {
    const source = readFileSync(join(dir, file), 'utf-8')
    const { data, body } = parseFrontmatter(source)
    const slug = basename(file, '.md')

    // 过滤隐藏 / 草稿
    if (data.draft === true) continue
    if (data.hidden === true) continue
    if (data.published === false) continue

    posts.push({
      slug,
      path: `/posts/${slug}`,
      title: String(data.title ?? slug),
      date: String(data.date ?? '2020-01-01'),
      tags: Array.isArray(data.tags) ? (data.tags as string[]) : [],
      description: String(data.description ?? ''),
      draft: Boolean(data.draft),
      hidden: Boolean(data.hidden),
      published: data.published !== false,
      readingTime: readingTime(body),
    })
  }

  // 按日期降序
  posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  return posts
})
