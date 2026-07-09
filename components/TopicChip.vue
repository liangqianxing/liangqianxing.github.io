<template>
  <NuxtLink :to="`/tags/${tagSlug(tag)}`" class="topic-chip" :class="{ 'topic-chip-active': active }">
    <span>#{{ tag }}</span>
    <span v-if="typeof count === 'number'" class="topic-count">{{ count }}</span>
  </NuxtLink>
</template>

<script setup lang="ts">
import { tagSlug } from '~/utils/blog'

defineProps<{
  tag: string
  count?: number
  active?: boolean
}>()
</script>
