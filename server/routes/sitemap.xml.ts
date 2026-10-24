import { readdirSync, readFileSync } from 'node:fs'
import { join, basename } from 'node:path'
import { isVisible, tagSlug } from '~/utils/blog'
import { learningCourses } from '~/data/learning/courses'
import { learningMaterialArticles } from '~/data/learning/materials'

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

export default defineEventHandler((event) => {
  const siteUrl = 'https://liangqianxing.github.io'
  const dir = join(process.cwd(), 'content', 'posts')
  const files = readdirSync(dir).filter(f => f.endsWith('.md'))

  const urls: { loc: string; lastmod?: string; priority?: string }[] = [
    { loc: `${siteUrl}/`, priority: '1.0' },
    { loc: `${siteUrl}/blog`, priority: '1.0' },
    { loc: `${siteUrl}/academic`, priority: '0.9' },
    { loc: `${siteUrl}/posts`, priority: '0.9' },
    { loc: `${siteUrl}/learning`, priority: '0.9' },
    { loc: `${siteUrl}/tags`, priority: '0.8' },
    { loc: `${siteUrl}/about`, priority: '0.8' },
    { loc: `${siteUrl}/friends`, priority: '0.6' },
  ]

  for (const course of learningCourses) {
    urls.push({
      loc: `${siteUrl}/learning/${encodeURIComponent(course.slug)}`,
      priority: '0.8',
    })
  }

  for (const material of learningMaterialArticles) {
    urls.push({
      loc: `${siteUrl}/learning/${encodeURIComponent(material.courseSlug)}/materials/${encodeURIComponent(material.slug)}`,
      priority: '0.7',
    })
  }

  const tagSet = new Set<string>()
  for (const file of files) {
    const source = readFileSync(join(dir, file), 'utf-8')
    const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---/)
    if (!match) continue
    const yaml = match[1]
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
    if (!isVisible(data)) continue

    const slug = basename(file, '.md')
    const date = typeof data.date === 'string' ? data.date : undefined
    urls.push({
      loc: `${siteUrl}/posts/${slug}`,
      lastmod: date,
      priority: '0.8',
    })

    if (Array.isArray(data.tags)) {
      for (const t of data.tags) {
        if (typeof t === 'string' && t) tagSet.add(tagSlug(t))
      }
    }
  }

  for (const t of tagSet) {
    urls.push({
      loc: `${siteUrl}/tags/${encodeURIComponent(t)}`,
      priority: '0.6',
    })
  }

  const urlNodes = urls.map(entry => [
    '  <url>',
    `    <loc>${escapeXml(entry.loc)}</loc>`,
    ...(entry.lastmod ? [`    <lastmod>${escapeXml(entry.lastmod)}</lastmod>`] : []),
    ...(entry.priority ? [`    <priority>${entry.priority}</priority>`] : []),
    '  </url>',
  ].join('\n')).join('\n')

  const xml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    urlNodes,
    '</urlset>',
    '',
  ].join('\n')

  setResponseHeader(event, 'content-type', 'application/xml; charset=utf-8')
  return xml
})
