<template>
  <div id="app-root">
    <div id="reading-bar" />
    <AppNav />
    <main>
      <slot />
    </main>
    <AppFooter />
    <BackToTop />
  </div>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()

// Theme state
const isDark = ref(true)

// Initialize theme from localStorage on client
onMounted(() => {
  const stored = localStorage.getItem('theme')
  if (stored === 'light') {
    isDark.value = false
    document.documentElement.classList.remove('dark')
    document.documentElement.classList.add('light')
  } else {
    isDark.value = true
    document.documentElement.classList.add('dark')
    document.documentElement.classList.remove('light')
  }

  // Reading progress bar
  const bar = document.getElementById('reading-bar')
  if (bar) {
    const updateBar = () => {
      const scrollTop = window.scrollY
      const docHeight = document.documentElement.scrollHeight - window.innerHeight
      const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0
      bar.style.width = `${pct}%`
    }
    window.addEventListener('scroll', updateBar, { passive: true })
  }
})

// Toggle theme
function toggleTheme() {
  document.documentElement.classList.add('transitioning')
  isDark.value = !isDark.value
  if (isDark.value) {
    document.documentElement.classList.add('dark')
    document.documentElement.classList.remove('light')
    localStorage.setItem('theme', 'dark')
  } else {
    document.documentElement.classList.remove('dark')
    document.documentElement.classList.add('light')
    localStorage.setItem('theme', 'light')
  }
  setTimeout(() => {
    document.documentElement.classList.remove('transitioning')
  }, 350)
}

// Keyboard shortcut: T to toggle theme
onMounted(() => {
  const handler = (e: KeyboardEvent) => {
    const tag = (e.target as HTMLElement)?.tagName
    if (tag === 'INPUT' || tag === 'TEXTAREA') return
    if (e.key === 't' && !e.ctrlKey && !e.metaKey && !e.altKey) {
      toggleTheme()
    }
  }
  window.addEventListener('keydown', handler)
  onUnmounted(() => window.removeEventListener('keydown', handler))
})

// Provide theme state to children
provide('isDark', isDark)
provide('toggleTheme', toggleTheme)

useHead({
  titleTemplate: (title) => title ? `${title} · ${appConfig.title}` : appConfig.title,
  meta: [
    { name: 'description', content: appConfig.description },
    { property: 'og:site_name', content: appConfig.title },
    { property: 'og:type', content: 'website' },
    { name: 'twitter:card', content: 'summary' },
  ],
  htmlAttrs: { lang: 'zh-CN' },
})
</script>
