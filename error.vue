<template>
  <div class="error-page">
    <div class="error-code">404</div>
    <p class="error-msg">页面不见了 · Page not found</p>
    <p style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-very-muted); margin-bottom: 2rem">
      {{ error?.url ?? '' }}
    </p>
    <NuxtLink to="/" class="error-link">← 回到首页</NuxtLink>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  error: {
    statusCode: number
    message: string
    url?: string
  } | null
}>()

useHead({
  title: '404 · 页面不存在',
})
</script>
