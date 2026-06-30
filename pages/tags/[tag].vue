<template>
  <div class="page-wrapper">
    <NuxtLink to="/tags" class="post-header-back" style="display: inline-flex; align-items: center; gap: 0.4rem; margin-bottom: 1.5rem">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <polyline points="15 18 9 12 15 6"/>
      </svg>
      所有标签
    </NuxtLink>

    <h1 class="page-title">
      <span style="color: var(--text-very-muted); font-size: 0.8em">#</span>
      {{ decodedTag }}
    </h1>
    <p class="page-desc">{{ tagPosts.length }} 篇文章</p>

    <div v-if="tagPosts.length">
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

    <div v-else style="text-align: center; padding: 4rem 0; color: var(--text-muted)">
      <p>该标签下暂无文章</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { tagSlug, formatMonthDay } from '~/utils/blog'
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
  title: `#${decodedTag.value}`,
  meta: [{ name: 'description', content: `标签 ${decodedTag.value} 下的所有文章` }],
}))
</script>
