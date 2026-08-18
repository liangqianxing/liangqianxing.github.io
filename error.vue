<template>
  <div class="error-page">
    <div class="error-scene">
      <div class="error-orbits" aria-hidden="true">
        <span class="error-orbit" />
        <span class="error-orbit" />
      </div>
      <div class="error-code">{{ error?.statusCode ?? 404 }}</div>
      <p class="error-msg">页面不见了 · Page not found</p>
      <p v-if="error?.url" class="error-path">
        <span aria-hidden="true">⌁</span>
        <span>{{ error.url }}</span>
      </p>
      <div class="error-actions">
        <NuxtLink to="/" class="primary-action">← 回到首页</NuxtLink>
        <NuxtLink to="/tags" class="secondary-action">逛逛标签</NuxtLink>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
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
