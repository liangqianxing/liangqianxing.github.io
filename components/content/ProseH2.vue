<template>
  <h2 :id="id" class="prose-heading">
    <slot />
    <a
      v-if="id"
      :href="`#${id}`"
      class="prose-heading-anchor"
      aria-label="定位到此小节"
    >#</a>
  </h2>
</template>

<script setup lang="ts">
defineProps<{ id?: string }>()
</script>
