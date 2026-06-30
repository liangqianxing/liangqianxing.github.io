<template>
  <button
    class="back-to-top"
    :class="{ visible: isVisible }"
    aria-label="回到顶部"
    @click="scrollToTop"
  >
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <polyline points="18 15 12 9 6 15"/>
    </svg>
  </button>
</template>

<script setup lang="ts">
const isVisible = ref(false)

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

onMounted(() => {
  const handler = () => {
    isVisible.value = window.scrollY > 400
  }
  window.addEventListener('scroll', handler, { passive: true })
  onUnmounted(() => window.removeEventListener('scroll', handler))
})
</script>
