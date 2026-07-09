<template>
  <NuxtLink :to="post.path" class="article-line">
    <time class="article-line-date">{{ formatMonthDay(post.date) }}</time>
    <span class="article-line-copy">
      <span class="article-line-title">{{ post.title }}</span>
      <span v-if="post.description || post.excerpt" class="article-line-desc">
        {{ post.description || post.excerpt }}
      </span>
    </span>
    <span class="article-line-tag">{{ post.tags?.[0] ?? `${post.readingTime ?? 1} min` }}</span>
  </NuxtLink>
</template>

<script setup lang="ts">
import { formatMonthDay } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

defineProps<{ post: PostMeta }>()
</script>
