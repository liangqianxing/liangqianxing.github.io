<template>
  <NuxtLink :to="post.path" class="article-card" :class="{ 'article-card-large': large }">
    <div class="article-meta">
      <span>{{ formatDate(post.date) }}</span>
      <span>{{ post.readingTime ?? 1 }} min</span>
    </div>
    <h3 class="article-title">{{ post.title }}</h3>
    <p class="article-desc">{{ post.description || post.excerpt || '继续阅读这篇技术笔记。' }}</p>
    <div class="article-foot">
      <span v-for="tag in (post.tags ?? []).slice(0, 3)" :key="tag" class="mini-tag">{{ tag }}</span>
    </div>
  </NuxtLink>
</template>

<script setup lang="ts">
import { formatDate } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

withDefaults(defineProps<{
  post: PostMeta
  large?: boolean
}>(), {
  large: false,
})
</script>
