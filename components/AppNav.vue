<template>
  <header ref="headerRef" class="site-nav">
    <div class="blog-header-row">
      <NuxtLink to="/blog" class="blog-identity" aria-label="gu.log 博客首页">
        <img src="/logo.svg?v=3" alt="" width="32" height="32" />
        <span class="blog-identity-name">{{ appConfig.authorEN }}</span>
        <span class="blog-identity-label">{{ appConfig.title }}</span>
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
      <label class="blog-theme-control">
        <span class="sr-only">阅读主题</span>
        <select :value="themeMode" aria-label="阅读主题" @change="onThemeChange">
          <option value="light">浅色</option>
          <option value="dark">深色</option>
          <option value="cyber">终端</option>
        </select>
      </label>
    </div>
  </header>
</template>

<script setup lang="ts">
type ThemeMode = 'dark' | 'light' | 'cyber'
const appConfig = useAppConfig()
const route = useRoute()
const headerRef = ref<HTMLElement | null>(null)
const themeMode = inject<Ref<ThemeMode>>('themeMode', ref('light'))
const setTheme = inject<(mode: ThemeMode) => void>('setTheme', () => {})
let headerObserver: ResizeObserver | undefined

function onThemeChange(event: Event) {
  setTheme((event.target as HTMLSelectElement).value as ThemeMode)
}

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
