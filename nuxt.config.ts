import tailwindcss from '@tailwindcss/vite'
import { defineNuxtConfig } from 'nuxt/config'
import { readFileSync, readdirSync } from 'node:fs'
import { join, basename } from 'node:path'

function parsePostFrontmatter(source: string): Record<string, unknown> {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  if (!match) return {}
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

  const mlArray = yaml.matchAll(/^(\w+):\s*\n((?:\s+-\s+.+\n?)*)/gm)
  for (const m of mlArray) {
    const [, k, block] = m
    data[k] = block.split('\n')
      .filter((l) => l.trim().startsWith('-'))
      .map((l) => l.trim().replace(/^-\s*/, '').replace(/^['"]|['"]$/g, ''))
  }

  return data
}

function isVisiblePost(data: Record<string, unknown>) {
  return data.draft !== true && data.hidden !== true && data.published !== false
}

function toTagSlug(tag: string) {
  return tag
    .trim()
    .replace(/\s*\/\s*/g, '-')
    .replace(/\s+/g, '-')
    .toLowerCase()
}

// 在构建时静态读取所有公开 post 文件，提取 slug
function getPostSlugs() {
  try {
    const dir = join(process.cwd(), 'content', 'posts')
    return readdirSync(dir)
      .filter((f) => f.endsWith('.md'))
      .filter((f) => isVisiblePost(parsePostFrontmatter(readFileSync(join(dir, f), 'utf-8'))))
      .map((f) => `/posts/${basename(f, '.md')}`)
  } catch {
    return []
  }
}

// 提取所有 tag，用于预渲染标签页
function getTagRoutes() {
  try {
    const dir = join(process.cwd(), 'content', 'posts')
    const tags = new Set<string>()
    for (const f of readdirSync(dir).filter((f) => f.endsWith('.md'))) {
      const text = readFileSync(join(dir, f), 'utf-8')
      const data = parsePostFrontmatter(text)
      if (!isVisiblePost(data)) continue
      if (Array.isArray(data.tags)) {
        for (const tag of data.tags) {
          if (typeof tag === 'string' && tag) tags.add(toTagSlug(tag))
        }
      }
    }
    return [...tags].map((t) => `/tags/${encodeURIComponent(t)}`)
  } catch {
    return []
  }
}

export default defineNuxtConfig({
  compatibilityDate: '2025-01-01',
  modules: ['@nuxt/content'],
  css: ['~/assets/css/main.css'],
  vite: { plugins: [tailwindcss()] },
  content: {
    build: {
      markdown: {
        highlight: {
          theme: { default: 'github-dark-dimmed', light: 'github-light' },
          langs: ['typescript', 'javascript', 'python', 'go', 'rust', 'bash', 'json', 'yaml', 'markdown', 'sql', 'cpp', 'java', 'vue', 'css', 'html'],
        },
        toc: { depth: 3 }
      }
    }
  },
  nitro: {
    preset: 'github-pages',
    prerender: {
      crawlLinks: true,
      // 直接从文件系统注入所有 post 和 tag 路由，
      // 绕过 Content v3 SQLite WASM 在 SSR 阶段未初始化的问题
      routes: ['/', '/posts', '/tags', '/about', '/friends', ...getPostSlugs(), ...getTagRoutes()],
    },
  },
  app: {
    head: {
      meta: [
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { charset: 'utf-8' },
        { name: 'theme-color', content: '#101214' },
      ],
      link: [
        { rel: 'icon', type: 'image/svg+xml', sizes: 'any', href: '/favicon.svg?v=2' },
        { rel: 'shortcut icon', type: 'image/svg+xml', href: '/favicon.svg?v=2' },
        { rel: 'apple-touch-icon', href: '/favicon.svg?v=2' },
        { rel: 'manifest', href: '/site.webmanifest' },
        { rel: 'alternate', type: 'application/rss+xml', title: 'gu.log', href: '/rss.xml' },
      ],
      // 防主题闪烁：在 DOM 渲染前读取 localStorage 并立即应用主题 class
      script: [
        {
          innerHTML: `(function(){var t=localStorage.getItem('theme');var h=document.documentElement;t=t==='light'||t==='cyber'||t==='dark'?t:'dark';h.dataset.theme=t;h.classList.toggle('light',t==='light');h.classList.toggle('dark',t!=='light');h.classList.toggle('cyber',t==='cyber');var m=document.querySelector('meta[name=theme-color]');if(m)m.setAttribute('content',t==='light'?'#f5f1e8':t==='cyber'?'#07110f':'#101214');})();`,
          type: 'text/javascript',
        },
      ],
    }
  }
})
