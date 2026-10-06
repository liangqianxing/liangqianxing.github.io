<template>
  <div id="app-root" class="blog-site">
    <a class="blog-skip" href="#blog-main">跳到主要内容</a>
    <div id="reading-bar" aria-hidden="true" />
    <AppNav />
    <main id="blog-main" tabindex="-1">
      <slot />
    </main>
    <AppFooter />
    <BackToTop />
  </div>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
type ThemeMode = 'dark' | 'light' | 'cyber'
const isDark = ref(false)
const isThemeReady = ref(false)
const themeMode = ref<ThemeMode>('light')
const themeModes: ThemeMode[] = ['light', 'dark', 'cyber']
let progressFrame = 0

function normalizeTheme(value: string | null): ThemeMode {
  return value === 'dark' || value === 'cyber' ? value : 'light'
}

function setTheme(mode: ThemeMode, persist = true) {
  const root = document.documentElement
  themeMode.value = mode
  isDark.value = mode !== 'light'
  root.classList.toggle('light', mode === 'light')
  root.classList.toggle('dark', mode !== 'light')
  root.classList.toggle('cyber', mode === 'cyber')
  root.dataset.theme = mode
  document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')
    ?.setAttribute('content', mode === 'light' ? '#fffdf8' : mode === 'cyber' ? '#07110f' : '#10151f')
  if (persist) {
    try { localStorage.setItem('theme', mode) } catch { /* Storage may be unavailable. */ }
  }
}

function toggleTheme() {
  setTheme(themeModes[(themeModes.indexOf(themeMode.value) + 1) % themeModes.length])
}

function updateProgress() {
  progressFrame = 0
  const bar = document.getElementById('reading-bar')
  const height = document.documentElement.scrollHeight - window.innerHeight
  if (bar) bar.style.width = `${height > 0 ? Math.min(100, window.scrollY / height * 100) : 0}%`
}

function onScroll() {
  if (!progressFrame) progressFrame = requestAnimationFrame(updateProgress)
}

function onKeydown(event: KeyboardEvent) {
  const target = event.target as HTMLElement | null
  if (target?.closest('input, textarea, select, [contenteditable="true"]')) return
  if (event.key === 't' && !event.ctrlKey && !event.metaKey && !event.altKey) toggleTheme()
}

onMounted(() => {
  let stored: string | null = null
  try { stored = localStorage.getItem('theme') } catch { /* Use the default theme. */ }
  setTheme(normalizeTheme(stored), false)
  isThemeReady.value = true
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
})

provide('isDark', isDark)
provide('isThemeReady', isThemeReady)
provide('themeMode', themeMode)
provide('setTheme', setTheme)
provide('toggleTheme', toggleTheme)

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
