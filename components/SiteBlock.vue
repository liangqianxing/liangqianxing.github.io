<template>
  <section class="site-block" :class="tone ? `site-block-${tone}` : ''">
    <header v-if="eyebrow || title || description || actionTo" class="block-head">
      <div>
        <p v-if="eyebrow" class="block-eyebrow">{{ eyebrow }}</p>
        <h2 v-if="title" class="block-title">{{ title }}</h2>
        <p v-if="description" class="block-desc">{{ description }}</p>
      </div>
      <NuxtLink v-if="actionTo && actionLabel" :to="actionTo" class="text-link">
        {{ actionLabel }}
        <span aria-hidden="true">→</span>
      </NuxtLink>
    </header>
    <slot />
  </section>
</template>

<script setup lang="ts">
defineProps<{
  eyebrow?: string
  title?: string
  description?: string
  actionTo?: string
  actionLabel?: string
  tone?: 'compact' | 'panel'
}>()
</script>
