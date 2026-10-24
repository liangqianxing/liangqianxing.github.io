import { readdirSync, readFileSync } from 'node:fs'
import { join, basename } from 'node:path'
import { isVisible } from '~/utils/blog'

interface FeedPost {
  slug: string
  path: string
  title: string
  date: string
  description: string
  tags: string[]
}

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

function parsePostMeta(source: string, slug: string): FeedPost | null {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  if (!match) return null
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
    if (val) data[key] = val.replace(/^['"]|['"]$/g, '')
  }

  const arrayInline = yaml.match(/([\w-]+):\s*\[([^\]]+)\]/g)
  if (arrayInline) {
    for (const m of arrayInline) {
      const [, k, v] = m.match(/([\w-]+):\s*\[([^\]]+)\]/) ?? []
      if (k && v) data[k] = v.split(',').map((s: string) => s.trim().replace(/^['"]|['"]$/g, ''))
    }
  }

  const mlArray = yaml.matchAll(/^(\w+):\s*\n((?:\s+-\s+.+\n?)*)/gm)
  for (const m of mlArray) {
    const [, k, block] = m
    data[k] = block.split('\n')
      .filter(l => l.trim().startsWith('-'))
      .map(l => l.trim().replace(/^-\s*/, '').replace(/^['"]|['"]$/g, ''))
  }

  if (!isVisible(data)) return null

  const rawExcerpt = body
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/!\[[^\]]*]\([^)]*\)/g, ' ')
    .replace(/\[[^\]]+]\([^)]*\)/g, m => m.replace(/^\[([^\]]+)].*$/, '$1'))
    .replace(/^#+\s+/gm, '')
    .replace(/[*_`~>-]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 160)

  return {
    slug,
    path: `/posts/${slug}`,
    title: String(data.title ?? slug),
    date: String(data.date ?? '2026-01-01'),
    description: String(data.description || rawExcerpt),
    tags: Array.isArray(data.tags) ? (data.tags as string[]) : [],
  }
}

export default defineEventHandler((event) => {
  const siteUrl = 'https://liangqianxing.github.io'
  const dir = join(process.cwd(), 'content', 'posts')
  const files = readdirSync(dir).filter(f => f.endsWith('.md'))
  const posts: FeedPost[] = []

  for (const file of files) {
    const source = readFileSync(join(dir, file), 'utf-8')
    const meta = parsePostMeta(source, basename(file, '.md'))
    if (meta) posts.push(meta)
  }

  posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())

  const latestDate = posts[0]?.date ? new Date(posts[0].date).toUTCString() : new Date().toUTCString()

  const itemsXml = posts.slice(0, 40).map((post) => {
    const link = `${siteUrl}${post.path}`
    const pubDate = new Date(post.date).toUTCString()
    const categories = post.tags.map(t => `      <category>${escapeXml(t)}</category>`).join('\n')
    return [
      '    <item>',
      `      <title>${escapeXml(post.title)}</title>`,
      `      <link>${escapeXml(link)}</link>`,
      `      <guid isPermaLink="true">${escapeXml(link)}</guid>`,
      `      <pubDate>${pubDate}</pubDate>`,
      `      <description>${escapeXml(post.description)}</description>`,
      ...(categories ? [categories] : []),
      '    </item>',
    ].join('\n')
  }).join('\n')

  const xml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">',
    '  <channel>',
    '    <title>gu.log · 古恩豪</title>',
    `    <link>${siteUrl}/blog</link>`,
    `    <atom:link href="${siteUrl}/rss.xml" rel="self" type="application/rss+xml" />`,
    '    <description>写 LLM、Agent、后端工程，偶尔写比赛、实习和旅行。这里是古恩豪的个人笔记与工程记录。</description>',
    '    <language>zh-CN</language>',
    `    <lastBuildDate>${latestDate}</lastBuildDate>`,
    itemsXml,
    '  </channel>',
    '</rss>',
    '',
  ].join('\n')

  setResponseHeader(event, 'content-type', 'application/rss+xml; charset=utf-8')
  return xml
})
