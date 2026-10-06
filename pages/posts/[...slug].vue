<template>
  <div>
    <aside v-if="page" class="author-sidebar" aria-label="关于作者">
      <PostAuthorNote :reading-progress="readingProgress" />
    </aside>

    <!-- TOC Sidebar -->
    <aside
      v-if="hasToc"
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
          <div class="post-header-overline">
            <span>{{ primaryTopic }}</span>
            <time :datetime="page.date">{{ formatDate(page.date) }}</time>
          </div>
          <h1 class="post-title">{{ page.title }}</h1>
          <p v-if="page.description" class="post-desc">{{ page.description }}</p>
          <div class="post-header-footer">
            <div class="post-meta">
              <span>{{ postReadingTime }} min read</span>
              <template v-if="page.series">
                <span class="post-meta-sep">·</span>
                <span>{{ page.series }}<template v-if="page.seriesOrder"> · 第 {{ page.seriesOrder }} 篇</template></span>
              </template>
              <template v-else-if="page.tags?.length">
                <span class="post-meta-sep">·</span>
                <span>{{ page.tags.length }} keywords</span>
              </template>
            </div>
            <a class="post-header-jump" href="#article-start" aria-label="开始阅读" title="开始阅读">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M12 4v15" />
                <path d="m6.5 13.5 5.5 5.5 5.5-5.5" />
              </svg>
            </a>
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

        <details v-if="hasToc" class="post-inline-toc">
          <summary>文章目录</summary>
          <nav class="post-inline-toc-links" aria-label="文章目录">
            <template v-for="link in toc?.links" :key="link.id">
              <a
                :href="`#${link.id}`"
                :aria-current="activeId === link.id ? 'location' : undefined"
                @click.prevent="scrollToHeading(link.id)"
              >{{ link.text }}</a>
              <template v-if="link.children">
                <a
                  v-for="child in link.children"
                  :key="child.id"
                  :href="`#${child.id}`"
                  class="post-inline-toc-child"
                  :aria-current="activeId === child.id ? 'location' : undefined"
                  @click.prevent="scrollToHeading(child.id)"
                >{{ child.text }}</a>
              </template>
            </template>
          </nav>
        </details>

        <!-- Article body -->
        <article id="article-start" class="prose" ref="articleRef">
          <ContentRenderer :value="page" />
        </article>

        <section v-if="seriesPosts.length > 1" class="post-series" aria-labelledby="series-title">
          <div>
            <p>Series</p>
            <h2 id="series-title">{{ page.series }}</h2>
          </div>
          <ol>
            <li v-for="post in seriesPosts" :key="post.path" :class="{ current: post.path === path }">
              <NuxtLink :to="post.path">
                <span>{{ String(post.seriesOrder ?? 0).padStart(2, '0') }}</span>
                {{ post.title }}
              </NuxtLink>
            </li>
          </ol>
        </section>

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

const seriesPosts = computed(() => {
  if (!page.value?.series) return []
  return (navPosts.value ?? [])
    .filter(post => post.series === page.value?.series)
    .sort((a, b) => (a.seriesOrder ?? 999) - (b.seriesOrder ?? 999))
})

const toc = computed(() => page.value?.body?.toc ?? null)
const hasToc = computed(() => (toc.value?.links?.length ?? 0) > 2)
const primaryTopic = computed(() => page.value?.categories?.[0] ?? '技术随笔')

// 优先从轻量 API 返回的预计算值取，fallback 再从 body AST 计算
const postReadingTime = computed(() => {
  const meta = (navPosts.value ?? []).find(p => p.path === path.value)
  if (meta?.readingTime) return meta.readingTime
  if (page.value?.readingTime) return page.value.readingTime
  if (!page.value) return 1
  return readingTime(JSON.stringify(page.value.body ?? ''))
})

// TOC active heading tracking
const activeId = ref('')
const articleRef = ref<HTMLElement | null>(null)
const readingProgress = ref(0)
let progressFrame = 0
let headingObserver: IntersectionObserver | null = null
let navResizeObserver: ResizeObserver | null = null

function headingOffset() {
  return (document.querySelector('.site-nav')?.getBoundingClientRect().height ?? 0) + 16
}

function observeHeadings() {
  headingObserver?.disconnect()
  const headings = articleRef.value?.querySelectorAll('h2, h3')
  if (!headings?.length) return

  headingObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) activeId.value = entry.target.id
      }
    },
    { rootMargin: `-${headingOffset()}px 0px -60% 0px`, threshold: 0 }
  )
  headings.forEach(heading => headingObserver?.observe(heading))
}

function updateReadingProgress() {
  const article = articleRef.value
  if (!article) return

  const articleTop = window.scrollY + article.getBoundingClientRect().top
  const start = articleTop - window.innerHeight * 0.22
  const end = articleTop + article.offsetHeight - window.innerHeight * 0.72
  const progress = ((window.scrollY - start) / Math.max(1, end - start)) * 100
  readingProgress.value = Math.min(100, Math.max(0, Math.round(progress)))
}

function scheduleReadingProgress() {
  if (progressFrame) return
  progressFrame = window.requestAnimationFrame(() => {
    updateReadingProgress()
    progressFrame = 0
  })
}

function scrollToHeading(id: string) {
  const el = document.getElementById(id)
  if (el) {
    const top = el.getBoundingClientRect().top + window.scrollY - headingOffset()
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    window.scrollTo({ top, behavior: reducedMotion ? 'auto' : 'smooth' })
  }
}

onMounted(() => {
  scheduleReadingProgress()
  window.addEventListener('scroll', scheduleReadingProgress, { passive: true })
  window.addEventListener('resize', scheduleReadingProgress)
})

onBeforeUnmount(() => {
  window.removeEventListener('scroll', scheduleReadingProgress)
  window.removeEventListener('resize', scheduleReadingProgress)
  if (progressFrame) window.cancelAnimationFrame(progressFrame)
  headingObserver?.disconnect()
  navResizeObserver?.disconnect()
})

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

  observeHeadings()
  const nav = document.querySelector('.site-nav')
  if (nav) {
    navResizeObserver = new ResizeObserver(observeHeadings)
    navResizeObserver.observe(nav)
  }
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

<style scoped>
.post-inline-toc {
  margin-bottom: 2rem;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  padding: 0.85rem 0;
  color: var(--muted);
}

.post-inline-toc summary {
  cursor: pointer;
  color: var(--text);
  font-weight: 600;
}

.post-inline-toc-links {
  display: grid;
  gap: 0.5rem;
  padding-top: 1rem;
  font-size: 0.875rem;
  line-height: 1.6;
}

.post-inline-toc-links a {
  width: fit-content;
}

.post-inline-toc-links a[aria-current="location"] {
  color: var(--accent);
}

.post-inline-toc-child {
  margin-left: 1rem;
}

@media (min-width: 1280px) {
  .post-inline-toc {
    display: none;
  }
}
</style>
