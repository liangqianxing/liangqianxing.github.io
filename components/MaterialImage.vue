<template>
  <span class="material-figure">
    <a :href="src" target="_blank" rel="noopener noreferrer" :aria-label="`${alt || '原文插图'}，打开大图`">
      <img :src="src" :alt="alt" loading="lazy" decoding="async" :width="width" :height="height" />
    </a>
    <span v-if="alt" class="material-figure-caption">{{ alt }} · 点击查看大图</span>
  </span>
</template>

<script setup lang="ts">
withDefaults(defineProps<{ src: string; alt?: string; width?: string | number; height?: string | number }>(), { alt: '' })
</script>

<style scoped>
.material-figure { display: block; margin: 1.8em 0; }
.material-figure a { display: block; }
.material-figure img { display: block; width: 100%; max-width: 100%; height: auto; margin: 0; border: 1px solid var(--line); border-radius: 8px; background: white; }
.material-figure-caption { display: block; margin-top: 8px; text-align: center; color: var(--muted); font-size: 0.74rem; line-height: 1.7; }
</style>
