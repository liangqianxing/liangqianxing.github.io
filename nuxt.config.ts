import tailwindcss from '@tailwindcss/vite'
import { defineNuxtConfig } from 'nuxt/config'

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
  nitro: { preset: 'github-pages' },
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
