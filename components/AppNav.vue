<template>
  <header ref="headerRef" class="site-nav">
    <div class="blog-nav-inner">
      <div class="blog-nav-left">
        <NuxtLink to="/blog" class="blog-identity" aria-label="gu.log 博客首页">
          <span class="blog-identity-logo">
            <img src="/logo.svg?v=4" alt="" width="30" height="30" />
          </span>
          <span class="blog-identity-copy">
            <span class="blog-identity-name">{{ appConfig.authorEN }}</span>
            <span class="blog-identity-label">{{ appConfig.title }}</span>
          </span>
        </NuxtLink>

        <span class="blog-nav-divider" aria-hidden="true" />

        <nav
          ref="sectionNavRef"
          class="blog-section-links"
          :class="{ 'has-pill': pillReady }"
          aria-label="博客导航"
          @pointerleave="onNavLeave"
        >
          <span
            v-if="pillReady"
            class="blog-nav-pill"
            :class="{ 'is-hover-preview': hoveredPath !== null && !isActive(hoveredPath) }"
            :style="pillStyle"
            aria-hidden="true"
          />
          <NuxtLink
            v-for="item in navItems"
            :key="item.path"
            :to="item.path"
            :data-nav-path="item.path"
            :aria-current="isActive(item.path) ? 'page' : undefined"
            @pointerenter="onLinkEnter(item.path)"
            @focus="onLinkEnter(item.path)"
            @blur="onNavLeave"
          >{{ item.label }}</NuxtLink>
        </nav>
      </div>

      <div class="blog-toolbar-actions">
        <nav class="blog-site-links" aria-label="全站模式切换">
          <NuxtLink to="/academic" class="blog-portal-pill">
            <span class="blog-portal-dot" aria-hidden="true" />
            学术主页
          </NuxtLink>
          <NuxtLink to="/" class="blog-portal-link" title="3D 交互个人门户">门户</NuxtLink>
        </nav>

        <button
          type="button"
          class="blog-search-trigger"
          aria-label="打开全局搜索与快捷面板 (⌘K)"
          title="全局搜索与快捷面板 (⌘K / /)"
          @click="emit('open-search')"
        >
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="11" cy="11" r="6.5" />
            <path d="m16 16 4.2 4.2" />
          </svg>
          <span class="blog-search-trigger-text">搜索</span>
          <kbd class="blog-search-trigger-kbd">⌘K</kbd>
        </button>

        <div
          ref="themeControlRef"
          class="blog-theme-control"
          :class="{ 'is-ready': isThemeReady }"
          role="group"
          aria-label="阅读主题"
          :data-theme="themeMode"
          @keydown="onThemeKeydown"
        >
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
            @click="setTheme(option.mode, true, $event)"
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
    </div>
  </header>
</template>

<script setup lang="ts">
import type { ThemeMode } from '~/composables/useTheme'

const emit = defineEmits<{
  'open-search': []
}>()
const appConfig = useAppConfig()
const route = useRoute()
const headerRef = ref<HTMLElement | null>(null)
const sectionNavRef = ref<HTMLElement | null>(null)
const themeControlRef = ref<HTMLElement | null>(null)
const hoveredPath = ref<string | null>(null)
const pillReady = ref(false)
const pillStyle = ref<{ transform: string; width: string; height: string }>({
  transform: 'translate3d(0, 0, 0)',
  width: '0px',
  height: '0px',
})
const { isThemeReady, themeMode, setTheme } = useTheme()
const navItems = computed(() => appConfig.nav.filter(item => item.path !== '/'))
const themeOptions: { mode: ThemeMode; label: string }[] = [
  { mode: 'light', label: '浅色主题' },
  { mode: 'dark', label: '深色主题' },
  { mode: 'cyber', label: '终端主题' },
]
let headerObserver: ResizeObserver | undefined
let isMounted = false

function isActive(path: string): boolean {
  const currentPath = route.path.replace(/\/$/, '') || '/'
  if (path === '/blog') return currentPath === path
  return currentPath === path || currentPath.startsWith(`${path}/`)
}

function syncNavPill() {
  const container = sectionNavRef.value
  if (!container) return
  const targetPath = hoveredPath.value ?? navItems.value.find(item => isActive(item.path))?.path
  if (!targetPath) {
    pillReady.value = false
    return
  }
  const link = container.querySelector<HTMLElement>(`[data-nav-path="${targetPath}"]`)
  if (!link) {
    pillReady.value = false
    return
  }
  pillStyle.value = {
    transform: `translate3d(${link.offsetLeft}px, ${link.offsetTop}px, 0)`,
    width: `${link.offsetWidth}px`,
    height: `${link.offsetHeight}px`,
  }
  pillReady.value = true
}

function onLinkEnter(path: string) {
  hoveredPath.value = path
  syncNavPill()
}

function onNavLeave() {
  if (hoveredPath.value === null) return
  hoveredPath.value = null
  syncNavPill()
}

function onThemeKeydown(event: KeyboardEvent) {
  const keys = ['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End']
  if (!keys.includes(event.key)) return
  event.preventDefault()

  const currentIdx = themeOptions.findIndex(o => o.mode === themeMode.value)
  const len = themeOptions.length
  let nextIdx = currentIdx >= 0 ? currentIdx : 0

  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
    nextIdx = (nextIdx + 1) % len
  } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
    nextIdx = (nextIdx - 1 + len) % len
  } else if (event.key === 'Home') {
    nextIdx = 0
  } else if (event.key === 'End') {
    nextIdx = len - 1
  }

  const nextOption = themeOptions[nextIdx]
  if (!nextOption) return
  const btn = themeControlRef.value?.querySelector<HTMLElement>(`.blog-theme-option[data-mode="${nextOption.mode}"]`)
  btn?.focus()
  const rect = btn?.getBoundingClientRect()
  const origin = rect && rect.width > 0
    ? { clientX: Math.round(rect.left + rect.width / 2), clientY: Math.round(rect.top + rect.height / 2) }
    : undefined
  setTheme(nextOption.mode, true, origin)
}

watch(() => route.path, () => {
  hoveredPath.value = null
  nextTick(syncNavPill)
})

onMounted(() => {
  isMounted = true
  const updateHeaderHeight = () => {
    if (!isMounted) return
    if (headerRef.value) {
      document.documentElement.style.setProperty('--nav-h', `${headerRef.value.offsetHeight}px`)
    }
    syncNavPill()
  }
  updateHeaderHeight()
  headerObserver = new ResizeObserver(updateHeaderHeight)
  if (headerRef.value) headerObserver.observe(headerRef.value)
  if (sectionNavRef.value) headerObserver.observe(sectionNavRef.value)
  document.fonts?.ready?.then(() => {
    if (isMounted) updateHeaderHeight()
  }).catch(() => {})
})

onUnmounted(() => {
  isMounted = false
  headerObserver?.disconnect()
  document.documentElement.style.removeProperty('--nav-h')
})
</script>
