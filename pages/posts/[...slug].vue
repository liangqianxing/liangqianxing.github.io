<template>
  <div>
    <!-- TOC Sidebar -->
    <aside
      v-if="toc && toc.links && toc.links.length > 2"
      class="toc-sidebar"
      aria-label="目录"
    >
      <div class="toc-label">目录</div>
      <template v-for="link in toc.links" :key="link.id">
        <a
          :href="`#${link.id}`"
          class="toc-link"
          :class="{ 'toc-link-active': activeId === link.id }"
          @click.prevent="scrollToHeading(link.id)"
        >{{ link.text }}</a>
        <template v-if="link.children">
          <a
            v-for="child in link.children"
            :key="child.id"
            :href="`#${child.id}`"
            class="toc-link toc-link-h3"
            :class="{ 'toc-link-active': activeId === child.id }"
            @click.prevent="scrollToHeading(child.id)"
          >{{ child.text }}</a>
        </template>
      </template>
    </aside>

    <!-- Post content -->
    <div class="page-wrapper-narrow">
      <!-- Back link -->
      <NuxtLink to="/posts" class="post-header-back">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="15 18 9 12 15 6"/>
        </svg>
        所有文章
      </NuxtLink>

      <template v-if="page">
        <!-- Post header -->
        <header class="post-header">
          <h1 class="post-title">{{ page.title }}</h1>
          <p v-if="page.description" class="post-desc">{{ page.description }}</p>
          <div class="post-meta">
            <span>{{ formatDate(page.date) }}</span>
            <span class="post-meta-sep">·</span>
            <span>{{ postReadingTime }} min read</span>
            <template v-if="page.categories?.length">
              <span class="post-meta-sep">·</span>
              <span>{{ page.categories.join(', ') }}</span>
            </template>
          </div>
          <div v-if="page.tags?.length" class="post-tags">
            <NuxtLink
              v-for="tag in page.tags"
              :key="tag"
              :to="`/tags/${tagSlug(tag)}`"
              class="tag-chip"
            >{{ tag }}</NuxtLink>
          </div>
        </header>

        <!-- Article body -->
        <article class="prose" ref="articleRef">
          <ContentRenderer :value="page" />
        </article>

        <!-- Prev/Next navigation -->
        <nav class="post-nav" aria-label="文章导航">
          <NuxtLink
            v-if="prevPost"
            :to="prevPost.path"
            class="post-nav-item prev"
          >
            <span class="post-nav-label">← 上一篇</span>
            <span class="post-nav-title">{{ prevPost.title }}</span>
          </NuxtLink>
          <div v-else />
          <NuxtLink
            v-if="nextPost"
            :to="nextPost.path"
            class="post-nav-item next"
          >
            <span class="post-nav-label">下一篇 →</span>
            <span class="post-nav-title">{{ nextPost.title }}</span>
          </NuxtLink>
          <div v-else />
        </nav>
      </template>

      <!-- 404 state -->
      <template v-else>
        <div style="text-align: center; padding: 4rem 0; color: var(--text-muted)">
          <p style="font-size: 1.25rem; margin-bottom: 1rem">文章不存在</p>
          <NuxtLink to="/posts" class="pill" style="display: inline-flex">返回文章列表</NuxtLink>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { formatDate, tagSlug, readingTime } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

const route = useRoute()
const appConfig = useAppConfig()

// Build path from slug
const path = computed(() => {
  const slug = route.params.slug
  const slugStr = Array.isArray(slug) ? slug.join('/') : slug
  return `/posts/${slugStr}`
})

// Fetch current post
const { data: page } = await useAsyncData(`post-${path.value}`, () =>
  queryCollection('posts').path(path.value).first()
)

// 用轻量 API 获取导航用的文章列表（只含 metadata，无 body AST）
// 替换原来的 queryCollection('.all()') 避免把 35 篇 body AST 塞进 payload
const { data: navPosts } = await useAsyncData<PostMeta[]>('all-posts-nav', () =>
  $fetch('/api/posts')
)

const currentIndex = computed(() =>
  (navPosts.value ?? []).findIndex(p => p.path === path.value)
)

const prevPost = computed(() => {
  const posts = navPosts.value ?? []
  return currentIndex.value > 0 ? posts[currentIndex.value - 1] : null
})

const nextPost = computed(() => {
  const posts = navPosts.value ?? []
  return currentIndex.value < posts.length - 1 ? posts[currentIndex.value + 1] : null
})

const toc = computed(() => page.value?.body?.toc ?? null)

// 优先从轻量 API 返回的预计算值取，fallback 再从 body AST 计算
const postReadingTime = computed(() => {
  const meta = (navPosts.value ?? []).find(p => p.path === path.value)
  if (meta?.readingTime) return meta.readingTime
  if (!page.value) return 1
  return readingTime(JSON.stringify(page.value.body ?? ''))
})

// TOC active heading tracking
const activeId = ref('')
const articleRef = ref<HTMLElement | null>(null)

function scrollToHeading(id: string) {
  const el = document.getElementById(id)
  if (el) {
    const offset = 72 // nav height + some padding
    const top = el.getBoundingClientRect().top + window.scrollY - offset
    window.scrollTo({ top, behavior: 'smooth' })
  }
}

onMounted(() => {
  // Add copy buttons to code blocks
  const addCopyButtons = () => {
    const pres = document.querySelectorAll('.prose pre')
    pres.forEach(pre => {
      if (pre.querySelector('.code-copy-btn')) return
      const wrapper = document.createElement('div')
      wrapper.className = 'code-block-wrapper'
      wrapper.style.position = 'relative'
      pre.parentNode?.insertBefore(wrapper, pre)
      wrapper.appendChild(pre)

      const btn = document.createElement('button')
      btn.className = 'code-copy-btn'
      btn.textContent = 'copy'
      btn.addEventListener('click', async () => {
        const code = pre.querySelector('code')?.textContent ?? ''
        try {
          await navigator.clipboard.writeText(code)
          btn.textContent = 'copied!'
          btn.classList.add('copied')
          setTimeout(() => {
            btn.textContent = 'copy'
            btn.classList.remove('copied')
          }, 2000)
        } catch {
          btn.textContent = 'error'
          setTimeout(() => { btn.textContent = 'copy' }, 2000)
        }
      })
      wrapper.appendChild(btn)
    })
  }

  addCopyButtons()

  // TOC intersection observer
  const headings = document.querySelectorAll('.prose h2, .prose h3')
  if (headings.length === 0) return

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          activeId.value = entry.target.id
        }
      }
    },
    { rootMargin: '-72px 0px -60% 0px', threshold: 0 }
  )

  headings.forEach(h => observer.observe(h))
  onUnmounted(() => observer.disconnect())
})

// SEO
useHead(() => ({
  title: page.value?.title ?? '文章',
  meta: [
    { name: 'description', content: page.value?.description ?? appConfig.description },
    { property: 'og:title', content: page.value?.title ?? '' },
    { property: 'og:description', content: page.value?.description ?? appConfig.description },
    { property: 'og:type', content: 'article' },
    { property: 'article:published_time', content: page.value?.date ?? '' },
    { property: 'article:author', content: appConfig.authorCN },
  ],
  script: page.value
    ? [
        {
          type: 'application/ld+json',
          innerHTML: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'BlogPosting',
            headline: page.value.title,
            description: page.value.description ?? '',
            datePublished: page.value.date,
            author: {
              '@type': 'Person',
              name: appConfig.authorCN,
              url: appConfig.url,
            },
            url: `${appConfig.url}${page.value.path}`,
          }),
        },
      ]
    : [],
}))
</script>
