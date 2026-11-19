<template>
  <img
    class="site-brand-symbol"
    src="/logo.svg?v=5"
    alt=""
    aria-hidden="true"
    :width="size"
    :height="size"
    decoding="async"
  />
</template>

<script setup lang="ts">
withDefaults(defineProps<{ size?: number }>(), { size: 32 })
</script>

<style scoped>
.site-brand-symbol {
  display: block;
  flex: 0 0 auto;
  border-radius: 25%;
}
</style>
