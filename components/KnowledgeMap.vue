<template>
  <div class="knowledge-grid">
    <NuxtLink
      v-for="item in items"
      :key="item.key"
      :to="item.to"
      class="knowledge-card"
      :data-key="item.key"
    >
      <div class="knowledge-heading">
        <div class="knowledge-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <template v-if="item.key === 'AI'"><rect x="7" y="7" width="10" height="10" rx="3" /><path d="M9 3v4m6-4v4M9 17v4m6-4v4M3 9h4m-4 6h4m10-6h4m-4 6h4" /></template>
            <template v-else-if="item.key === 'SYS'"><rect x="4" y="3" width="16" height="7" rx="2" /><rect x="4" y="14" width="16" height="7" rx="2" /><path d="M8 6.5h.01M8 17.5h.01M12 10v4" /></template>
            <template v-else-if="item.key === 'SRC'"><path d="m8 6-5 6 5 6m8-12 5 6-5 6M14 4l-4 16" /></template>
            <template v-else><rect x="4" y="5" width="16" height="16" rx="2" /><path d="M9 5V3h6v2M8 11h8m-8 5h5" /></template>
          </svg>
        </div>
        <span class="knowledge-index">{{ item.eyebrow ?? item.key }}</span>
      </div>
      <h3>{{ item.title }} <span class="knowledge-arrow" aria-hidden="true">→</span></h3>
      <p>{{ item.desc }}</p>
    </NuxtLink>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  items: Array<{
    key: string
    eyebrow?: string
    title: string
    desc: string
    label: string
    to: string
  }>
}>()
</script>
