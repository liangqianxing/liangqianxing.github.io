<template>
  <NuxtLink :to="tagPath(tag)" class="topic-chip" :class="{ 'topic-chip-active': active }">
    <span>#{{ tag }}</span>
    <span v-if="typeof count === 'number'" class="topic-count">{{ count }}</span>
  </NuxtLink>
</template>

<script setup lang="ts">
import { tagPath } from '~/utils/archive'

defineProps<{
  tag: string
  count?: number
  active?: boolean
}>()
</script>
