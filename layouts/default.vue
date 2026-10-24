<template>
  <div id="app-root" class="blog-site">
    <a class="blog-skip" href="#blog-main">跳到主要内容</a>
    <div id="reading-bar" aria-hidden="true" />
    <SiteEffects />
    <div class="theme-transition-layer" aria-hidden="true">
      <span class="theme-transition-orb" />
      <span class="theme-transition-sheen" />
    </div>
    <AppNav @open-search="openCommandPalette" />
    <main id="blog-main" tabindex="-1">
      <slot />
    </main>
    <AppFooter />
    <BackToTop />
    <CommandPalette v-model:open="commandOpen" />
  </div>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
const route = useRoute()
const {
  themeMode,
  isDark,
  isThemeReady,
  setTheme,
  toggleTheme,
  initThemeFromStorage,
  cleanupThemeTimers,
} = useTheme()
const {
  commandOpen,
  openCommandPalette,
  toggleCommandPalette,
} = useCommandPalette()

let progressFrame = 0

function updateProgress() {
  progressFrame = 0
  const bar = document.getElementById('reading-bar')
  const height = document.documentElement.scrollHeight - window.innerHeight
  if (bar) bar.style.width = `${height > 0 ? Math.min(100, window.scrollY / height * 100) : 0}%`
}

function onScroll() {
  if (!progressFrame) progressFrame = requestAnimationFrame(updateProgress)
}

watch(() => route.fullPath, () => {
  nextTick(updateProgress)
})

function onKeydown(event: KeyboardEvent) {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault()
    toggleCommandPalette()
    return
  }
  const target = event.target as HTMLElement | null
  if (target?.closest('input, textarea, select, [contenteditable="true"]')) return
  if (event.key === '/' && !event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey) {
    event.preventDefault()
    openCommandPalette()
    return
  }
  if (event.key === 't' && !event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey) {
    toggleTheme()
  }
}

onMounted(() => {
  initThemeFromStorage()
  updateProgress()
  window.addEventListener('scroll', onScroll, { passive: true })
  window.addEventListener('resize', onScroll, { passive: true })
  window.addEventListener('keydown', onKeydown)
})

onUnmounted(() => {
  window.removeEventListener('scroll', onScroll)
  window.removeEventListener('resize', onScroll)
  window.removeEventListener('keydown', onKeydown)
  cancelAnimationFrame(progressFrame)
  cleanupThemeTimers()
})

provide('isDark', isDark)
provide('isThemeReady', isThemeReady)
provide('themeMode', themeMode)
provide('setTheme', setTheme)
provide('toggleTheme', toggleTheme)
provide('openCommandPalette', openCommandPalette)

useHead(() => ({
  titleTemplate: title => title ? `${title} · ${appConfig.title}` : appConfig.title,
  meta: [
    { name: 'theme-color', content: themeMode.value === 'light' ? '#fffdf8' : themeMode.value === 'cyber' ? '#07110f' : '#10151f' },
    { name: 'description', content: appConfig.description },
    { property: 'og:site_name', content: appConfig.title },
    { property: 'og:type', content: 'website' },
    { name: 'twitter:card', content: 'summary' },
  ],
  htmlAttrs: { lang: 'zh-CN' },
  bodyAttrs: { class: 'blog-body' },
}))
</script>
