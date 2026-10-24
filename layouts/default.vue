<template>
  <div id="app-root" class="blog-site">
    <a class="blog-skip" href="#blog-main">跳到主要内容</a>
    <div id="reading-bar" aria-hidden="true" />
    <SiteEffects />
    <div class="theme-transition-layer" aria-hidden="true">
      <span class="theme-transition-orb" />
      <span class="theme-transition-sheen" />
    </div>
    <AppNav @open-search="commandOpen = true" />
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
type ThemeMode = 'dark' | 'light' | 'cyber'
type ThemeOriginEvent = MouseEvent | { clientX?: number; clientY?: number } | undefined

const isDark = ref(false)
const isThemeReady = ref(false)
const themeMode = ref<ThemeMode>('light')
const themeModes: ThemeMode[] = ['light', 'dark', 'cyber']
const commandOpen = ref(false)
let progressFrame = 0
let themeSwitchTimer: ReturnType<typeof setTimeout> | undefined
let fallbackSwitchTimer: ReturnType<typeof setTimeout> | undefined

function normalizeTheme(value: string | null): ThemeMode {
  return value === 'dark' || value === 'cyber' ? value : 'light'
}

function resolveThemeOrigin(mode: ThemeMode, event?: ThemeOriginEvent): { x: number; y: number } {
  if (
    event &&
    typeof event.clientX === 'number' &&
    typeof event.clientY === 'number' &&
    (event.clientX > 0 || event.clientY > 0)
  ) {
    return { x: event.clientX, y: event.clientY }
  }
  const target =
    document.querySelector<HTMLElement>(`.blog-theme-control .blog-theme-option[data-mode="${mode}"]`) ??
    document.querySelector<HTMLElement>(`.blog-theme-control .blog-theme-option[data-mode="${themeMode.value}"]`) ??
    document.querySelector<HTMLElement>('.blog-theme-control')
  if (target) {
    const rect = target.getBoundingClientRect()
    if (rect.width > 0 && rect.height > 0) {
      return {
        x: Math.round(rect.left + rect.width / 2),
        y: Math.round(rect.top + rect.height / 2),
      }
    }
  }
  return { x: Math.max(window.innerWidth - 74, 24), y: 42 }
}

function applyThemeDom(mode: ThemeMode, persist: boolean) {
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

function setTheme(mode: ThemeMode, persist = true, event?: ThemeOriginEvent) {
  if (persist && themeMode.value === mode) return
  if (!persist || typeof window === 'undefined') {
    applyThemeDom(mode, persist)
    return
  }

  const root = document.documentElement
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  if (reduced || document.visibilityState !== 'visible') {
    applyThemeDom(mode, persist)
    return
  }

  const { x, y } = resolveThemeOrigin(mode, event)
  const endRadius = Math.ceil(
    Math.hypot(
      Math.max(x, window.innerWidth - x),
      Math.max(y, window.innerHeight - y),
    ),
  )
  root.style.setProperty('--theme-x', `${x}px`)
  root.style.setProperty('--theme-y', `${y}px`)
  root.style.setProperty('--theme-r', `${endRadius}px`)

  const triggerOverlayWave = () => {
    root.classList.remove('theme-switching')
    void root.offsetWidth
    root.classList.add('theme-switching')
    if (themeSwitchTimer) clearTimeout(themeSwitchTimer)
    themeSwitchTimer = setTimeout(() => {
      root.classList.remove('theme-switching')
    }, 540)
  }

  const docWithVT = document as Document & {
    startViewTransition?: (cb: () => void) => { ready: Promise<void>; finished: Promise<void> }
  }

  if (typeof docWithVT.startViewTransition === 'function') {
    try {
      root.classList.add('theme-vt-active')
      const transition = docWithVT.startViewTransition(() => {
        applyThemeDom(mode, persist)
      })
      triggerOverlayWave()
      transition.ready
        .then(() => {
          document.documentElement.animate(
            {
              clipPath: [
                `circle(0px at ${x}px ${y}px)`,
                `circle(${endRadius}px at ${x}px ${y}px)`,
              ],
            },
            {
              duration: 520,
              easing: 'cubic-bezier(0.22, 1, 0.36, 1)',
              pseudoElement: '::view-transition-new(root)',
            },
          )
        })
        .catch(() => {})
      transition.finished.finally(() => {
        root.classList.remove('theme-vt-active')
      })
      return
    } catch {
      root.classList.remove('theme-vt-active')
    }
  }

  root.classList.add('theme-fallback-switching')
  if (fallbackSwitchTimer) clearTimeout(fallbackSwitchTimer)
  fallbackSwitchTimer = setTimeout(() => {
    root.classList.remove('theme-fallback-switching')
  }, 420)
  triggerOverlayWave()
  applyThemeDom(mode, persist)
}

function toggleTheme(event?: ThemeOriginEvent) {
  setTheme(themeModes[(themeModes.indexOf(themeMode.value) + 1) % themeModes.length], true, event)
}

function openCommandPalette() {
  commandOpen.value = true
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
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault()
    commandOpen.value = !commandOpen.value
    return
  }
  const target = event.target as HTMLElement | null
  if (target?.closest('input, textarea, select, [contenteditable="true"]')) return
  if (event.key === '/' && !event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey) {
    event.preventDefault()
    commandOpen.value = true
    return
  }
  if (event.key === 't' && !event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey) toggleTheme()
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
  if (themeSwitchTimer) clearTimeout(themeSwitchTimer)
  if (fallbackSwitchTimer) clearTimeout(fallbackSwitchTimer)
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
