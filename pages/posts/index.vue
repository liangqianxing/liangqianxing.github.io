<template>
  <main class="page-frame">
    <header class="page-hero">
      <p class="eyebrow">Archive</p>
      <h1>文章库</h1>
      <p>共 {{ posts.length }} 篇公开文章。支持关键词搜索和主题筛选，适合当资料库使用。</p>
    </header>

    <section class="archive-panel">
      <label class="search-box">
        <span aria-hidden="true">⌕</span>
        <input v-model.trim="query" type="search" placeholder="搜索标题、摘要或标签" aria-label="搜索文章" />
      </label>
      <span class="archive-count">{{ filteredPosts.length }} / {{ posts.length }} 篇</span>
    </section>

    <div class="filter-row" aria-label="主题筛选">
      <button class="topic-button" :class="{ active: activeTag === '' }" type="button" @click="activeTag = ''">
        全部
      </button>
      <button
        v-for="[tag, count] in topTags"
        :key="tag"
        class="topic-button"
        :class="{ active: activeTag === tag }"
        type="button"
        @click="activeTag = activeTag === tag ? '' : tag"
      >
        #{{ tag }}
        <span>{{ count }}</span>
      </button>
    </div>

    <template v-if="postsByYear.length">
      <section v-for="[year, yearPosts] in postsByYear" :key="year" class="year-section">
        <h2>
          {{ year }}
          <span>{{ yearPosts.length }} 篇</span>
        </h2>
        <ArticleStream :posts="yearPosts" />
      </section>
    </template>

    <p v-else class="empty-state">没有匹配的文章，换个关键词试试。</p>
  </main>
</template>

<script setup lang="ts">
import type { PostMeta } from '~/server/api/posts.get'

const { data } = await useAsyncData<PostMeta[]>('posts-archive', () =>
  $fetch('/api/posts')
)

const query = ref('')
const activeTag = ref('')
const posts = computed(() => data.value ?? [])

const topTags = computed(() => {
  const counts = new Map<string, number>()
  for (const post of posts.value) {
    for (const tag of post.tags ?? []) counts.set(tag, (counts.get(tag) ?? 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 14)
})

const filteredPosts = computed(() => {
  const q = query.value.toLowerCase()
  return posts.value.filter((post) => {
    const matchesTag = activeTag.value ? post.tags?.includes(activeTag.value) : true
    if (!q) return matchesTag
    const haystack = [post.title, post.description, post.excerpt, ...(post.tags ?? [])].join(' ').toLowerCase()
    return matchesTag && haystack.includes(q)
  })
})

const postsByYear = computed(() => {
  const map = new Map<string, PostMeta[]>()
  for (const post of filteredPosts.value) {
    const year = new Date(post.date).getFullYear().toString()
    if (!map.has(year)) map.set(year, [])
    map.get(year)!.push(post)
  }
  return [...map.entries()].sort((a, b) => Number(b[0]) - Number(a[0]))
})

useHead({
  title: '文章库',
  meta: [{ name: 'description', content: '所有文章按年份归档，并支持关键词与标签筛选' }],
})
</script>
