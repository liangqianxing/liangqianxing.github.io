<template>
  <div class="knowledge-grid">
    <NuxtLink
      v-for="item in items"
      :key="item.key"
      :to="item.to"
      class="knowledge-card"
    >
      <span class="knowledge-index">{{ item.key }}</span>
      <h3>{{ item.title }}</h3>
      <p>{{ item.desc }}</p>
      <span class="knowledge-link">{{ item.label }}</span>
    </NuxtLink>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  items: Array<{
    key: string
    title: string
    desc: string
    label: string
    to: string
  }>
}>()
</script>
