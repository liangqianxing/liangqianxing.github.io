<template>
  <main class="page-frame">
    <NuxtLink to="/tags" class="back-link">← 所有主题</NuxtLink>
    <header class="page-hero">
      <p class="eyebrow">Topic</p>
      <h1>#{{ displayTag }}</h1>
      <p>{{ tagPosts.length }} 篇文章收录在这个主题下。</p>
    </header>

    <template v-if="postsByYear.length">
      <section v-for="[year, yearPosts] in postsByYear" :key="year" class="year-section">
        <h2>
          {{ year }}
          <span>{{ yearPosts.length }} 篇</span>
        </h2>
        <ArticleStream :posts="yearPosts" />
      </section>
    </template>

    <p v-else class="empty-state">该主题下暂无文章。</p>
  </main>
</template>

<script setup lang="ts">
import { tagSlug } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

const route = useRoute()
const tagParam = computed(() => route.params.tag as string)
const decodedTag = computed(() => decodeURIComponent(tagParam.value))

const { data: allPosts } = await useAsyncData<PostMeta[]>(`tag-${tagParam.value}`, () =>
  $fetch('/api/posts')
)

const tagPosts = computed(() =>
  (allPosts.value ?? []).filter(post =>
    (post.tags ?? []).some(t => tagSlug(t) === tagParam.value)
  )
)

const displayTag = computed(() => {
  for (const post of allPosts.value ?? []) {
    const match = (post.tags ?? []).find(t => tagSlug(t) === tagParam.value)
    if (match) return match
  }
  return decodedTag.value
})

const postsByYear = computed(() => {
  const map = new Map<string, PostMeta[]>()
  for (const post of tagPosts.value) {
    const year = new Date(post.date).getFullYear().toString()
    if (!map.has(year)) map.set(year, [])
    map.get(year)!.push(post)
  }
  return [...map.entries()].sort((a, b) => Number(b[0]) - Number(a[0]))
})

useHead(() => ({
  title: `#${displayTag.value}`,
  meta: [{ name: 'description', content: `主题 ${displayTag.value} 下的所有文章` }],
}))
</script>
