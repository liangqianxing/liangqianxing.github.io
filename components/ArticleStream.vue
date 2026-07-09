<template>
  <div class="article-stream">
    <ArticleLine v-for="post in posts" :key="post.path" :post="post" />
  </div>
</template>

<script setup lang="ts">
import type { PostMeta } from '~/server/api/posts.get'

defineProps<{ posts: PostMeta[] }>()
</script>
