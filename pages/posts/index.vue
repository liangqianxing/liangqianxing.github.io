<template>
  <div class="page-wrapper">
    <h1 class="page-title">文章归档</h1>
    <p class="page-desc">共 {{ posts?.length ?? 0 }} 篇文章</p>

    <div v-for="[year, posts] in postsByYear" :key="year" class="year-block">
      <div class="year-label">
        {{ year }}
        <span class="year-count">{{ posts.length }} 篇</span>
      </div>
      <NuxtLink
        v-for="post in posts"
        :key="post.path"
        :to="post.path"
        class="post-row"
      >
        <span class="post-row-date">{{ formatMonthDay(post.date) }}</span>
        <span class="post-row-title">{{ post.title }}</span>
        <span class="post-row-tag">{{ post.tags?.[0] ?? '' }}</span>
      </NuxtLink>
    </div>
  </div>
</template>

<script setup lang="ts">
import { formatMonthDay } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

const { data: posts } = await useAsyncData<PostMeta[]>('posts-archive', () =>
  $fetch('/api/posts')
)

const postsByYear = computed(() => {
  const map = new Map<string, PostMeta[]>()
  for (const post of posts.value ?? []) {
    const year = new Date(post.date).getFullYear().toString()
    if (!map.has(year)) map.set(year, [])
    map.get(year)!.push(post)
  }
  // Sort years descending
  return [...map.entries()].sort((a, b) => Number(b[0]) - Number(a[0]))
})

useHead({
  title: '文章归档',
  meta: [{ name: 'description', content: '所有文章按年份归档' }],
})
</script>
