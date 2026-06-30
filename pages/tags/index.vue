<template>
  <div class="page-wrapper">
    <h1 class="page-title">标签</h1>
    <p class="page-desc">共 {{ allTags.length }} 个标签</p>

    <div class="tags-grid">
      <NuxtLink
        v-for="[tag, count] in sortedTags"
        :key="tag"
        :to="`/tags/${tagSlug(tag)}`"
        class="tag-item"
      >
        {{ tag }}
        <span class="tag-item-count">{{ count }}</span>
      </NuxtLink>
    </div>
  </div>
</template>

<script setup lang="ts">
import { tagSlug } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

const { data: allPosts } = await useAsyncData<PostMeta[]>('tags-all', () =>
  $fetch('/api/posts')
)

const sortedTags = computed(() => {
  const counts = new Map<string, number>()
  for (const post of allPosts.value ?? []) {
    for (const tag of post.tags ?? []) {
      counts.set(tag, (counts.get(tag) ?? 0) + 1)
    }
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])
})

const allTags = computed(() => sortedTags.value)

useHead({
  title: '标签',
  meta: [{ name: 'description', content: '所有文章标签' }],
})
</script>
