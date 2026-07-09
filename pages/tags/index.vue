<template>
  <main class="page-frame">
    <header class="page-hero">
      <p class="eyebrow">Topics</p>
      <h1>主题索引</h1>
      <p>共 {{ sortedTags.length }} 个主题。每个标签都是一条进入文章库的路径。</p>
    </header>

    <div class="topic-cloud topic-cloud-large">
      <TopicChip v-for="[tag, count] in sortedTags" :key="tag" :tag="tag" :count="count" />
    </div>
  </main>
</template>

<script setup lang="ts">
import type { PostMeta } from '~/server/api/posts.get'

const { data: allPosts } = await useAsyncData<PostMeta[]>('tags-all', () =>
  $fetch('/api/posts')
)

const sortedTags = computed(() => {
  const counts = new Map<string, number>()
  for (const post of allPosts.value ?? []) {
    for (const tag of post.tags ?? []) counts.set(tag, (counts.get(tag) ?? 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])
})

useHead({
  title: '主题索引',
  meta: [{ name: 'description', content: '所有文章主题标签' }],
})
</script>
