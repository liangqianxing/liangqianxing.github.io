import tailwindcss from '@tailwindcss/vite'
import { defineNuxtConfig } from 'nuxt/config'
import { readFileSync, readdirSync } from 'node:fs'
import { join, basename } from 'node:path'

// 在构建时静态读取所有 post 文件，提取 slug
function getPostSlugs() {
  try {
    const dir = join(process.cwd(), 'content', 'posts')
    return readdirSync(dir)
      .filter((f) => f.endsWith('.md'))
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
      const m = text.match(/^tags:\s*\n((?:\s+-\s+.+\n)*)/m)
      if (m) {
        for (const line of m[1].split('\n')) {
          const tag = line.trim().replace(/^-\s*/, '')
          if (tag) tags.add(tag.replace(/\//g, '-'))
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
  modules: ['@nuxt/content', '@nuxt/fonts'],
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
  fonts: {
    families: [
      { name: 'JetBrains Mono', provider: 'google', weights: ['400', '500', '600'] },
      { name: 'Inter', provider: 'google', weights: ['400', '500', '600'] },
    ]
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
      ],
      link: [
        { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' },
        { rel: 'alternate', type: 'application/rss+xml', title: 'gu.log', href: '/rss.xml' },
      ]
    }
  }
})
