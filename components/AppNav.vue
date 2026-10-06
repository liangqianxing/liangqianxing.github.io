<template>
  <header ref="headerRef" class="site-nav">
    <div class="blog-header-row">
      <NuxtLink to="/blog" class="blog-identity" aria-label="gu.log 博客首页">
        <img src="/logo.svg?v=3" alt="" width="32" height="32" />
        <span class="blog-identity-copy">
          <span class="blog-identity-name">{{ appConfig.authorEN }}</span>
          <span class="blog-identity-label">{{ appConfig.title }}</span>
        </span>
      </NuxtLink>
      <nav class="blog-site-links" aria-label="个人主页导航">
        <NuxtLink to="/">主页</NuxtLink>
        <NuxtLink to="/academic">学术</NuxtLink>
        <NuxtLink to="/blog" aria-current="page">博客</NuxtLink>
      </nav>
    </div>
    <div class="blog-toolbar">
      <nav class="blog-section-links" aria-label="博客导航">
        <NuxtLink
          v-for="item in appConfig.nav.filter(item => item.path !== '/')"
          :key="item.path"
          :to="item.path"
          :aria-current="isActive(item.path) ? 'page' : undefined"
        >{{ item.label }}</NuxtLink>
      </nav>
      <div class="blog-theme-control" :class="{ 'is-ready': isThemeReady }" role="group" aria-label="阅读主题" :data-theme="themeMode">
        <span class="blog-theme-thumb" aria-hidden="true" />
        <button
          v-for="option in themeOptions"
          :key="option.mode"
          class="blog-theme-option"
          type="button"
          :data-mode="option.mode"
          :aria-label="option.label"
          :title="option.label"
          :aria-pressed="themeMode === option.mode"
          @click="setTheme(option.mode)"
        >
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <template v-if="option.mode === 'light'">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
            </template>
            <path v-else-if="option.mode === 'dark'" d="M20.5 13.1A8.5 8.5 0 0 1 10.9 3.5a8.5 8.5 0 1 0 9.6 9.6Z" />
            <path v-else d="m5 6 6 6-6 6m9 0h5" />
          </svg>
        </button>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
type ThemeMode = 'dark' | 'light' | 'cyber'
const appConfig = useAppConfig()
const route = useRoute()
const headerRef = ref<HTMLElement | null>(null)
const isThemeReady = inject<Ref<boolean>>('isThemeReady', ref(false))
const themeMode = inject<Ref<ThemeMode>>('themeMode', ref('light'))
const setTheme = inject<(mode: ThemeMode) => void>('setTheme', () => {})
const themeOptions: { mode: ThemeMode; label: string }[] = [
  { mode: 'light', label: '浅色主题' },
  { mode: 'dark', label: '深色主题' },
  { mode: 'cyber', label: '终端主题' },
]
let headerObserver: ResizeObserver | undefined

function isActive(path: string): boolean {
  const currentPath = route.path.replace(/\/$/, '') || '/'
  if (path === '/blog') return currentPath === path
  return currentPath === path || currentPath.startsWith(`${path}/`)
}

onMounted(() => {
  const updateHeaderHeight = () => {
    if (headerRef.value) {
      document.documentElement.style.setProperty('--nav-h', `${headerRef.value.offsetHeight}px`)
    }
  }
  updateHeaderHeight()
  headerObserver = new ResizeObserver(updateHeaderHeight)
  if (headerRef.value) headerObserver.observe(headerRef.value)
})

onUnmounted(() => {
  headerObserver?.disconnect()
  document.documentElement.style.removeProperty('--nav-h')
})
</script>
