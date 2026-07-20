<template>
  <div id="app-root">
    <div id="reading-bar" />
    <div class="theme-transition-layer" aria-hidden="true">
      <span class="theme-transition-orb" />
      <span class="theme-transition-sheen" />
    </div>
    <SiteEffects />
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

type ThemeMode = 'dark' | 'light' | 'cyber'
type ThemeViewTransition = {
  ready: Promise<void>
  finished: Promise<void>
}
type ThemeTransitionDocument = Document & {
  startViewTransition?: (callback: () => void) => ThemeViewTransition
}

// Theme state
const isDark = ref(true)
const isThemeReady = ref(false)
const themeMode = ref<ThemeMode>('dark')
const themeModes: ThemeMode[] = ['dark', 'light', 'cyber']
const themeTransitionDuration = 440
let themeTimer: ReturnType<typeof window.setTimeout> | undefined

function normalizeTheme(value: string | null): ThemeMode {
  return value === 'light' || value === 'cyber' || value === 'dark' ? value : 'dark'
}

function applyTheme(mode: ThemeMode, persist = true) {
  const h = document.documentElement
  themeMode.value = mode
  isDark.value = mode !== 'light'

  h.classList.toggle('light', mode === 'light')
  h.classList.toggle('dark', mode !== 'light')
  h.classList.toggle('cyber', mode === 'cyber')
  h.dataset.theme = mode
  document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')
    ?.setAttribute('content', mode === 'light' ? '#f5f1e8' : mode === 'cyber' ? '#07110f' : '#101214')

  if (persist) {
    localStorage.setItem('theme', mode)
  }
}

// Initialize theme from localStorage on client
onMounted(() => {
  const stored = normalizeTheme(localStorage.getItem('theme'))
  applyTheme(stored, false)
  isThemeReady.value = true

  // Reading progress bar
  const bar = document.getElementById('reading-bar')
  if (bar) {
    let ticking = false
    const updateBar = () => {
      const scrollTop = window.scrollY
      const docHeight = document.documentElement.scrollHeight - window.innerHeight
      const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0
      bar.style.width = `${pct}%`
      ticking = false
    }
    const onScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(updateBar)
        ticking = true
      }
    }
    updateBar()
    window.addEventListener('scroll', onScroll, { passive: true })
    onUnmounted(() => window.removeEventListener('scroll', onScroll))
  }
})

// Apply a selected theme with a soft bloom from the interaction point.
function setTheme(nextTheme: ThemeMode, event?: MouseEvent) {
  const h = document.documentElement
  if (nextTheme === themeMode.value || h.classList.contains('transitioning')) return

  const transitionDocument = document as ThemeTransitionDocument
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const contentHeavy = document.getElementsByTagName('*').length > 1400
  const x = event?.clientX ?? window.innerWidth - 74
  const y = event?.clientY ?? 42

  h.style.setProperty('--theme-x', x + 'px')
  h.style.setProperty('--theme-y', y + 'px')
  h.dataset.themeNext = nextTheme
  h.classList.add('transitioning')

  if (themeTimer) window.clearTimeout(themeTimer)
  const finishTransition = () => {
    if (themeTimer) {
      window.clearTimeout(themeTimer)
      themeTimer = undefined
    }
    h.classList.remove('transitioning', 'theme-switching')
    delete h.dataset.themeNext
  }

  if (reduceMotion || contentHeavy) {
    applyTheme(nextTheme)
    finishTransition()
    return
  }

  if (transitionDocument.startViewTransition) {
    const transition = transitionDocument.startViewTransition(() => applyTheme(nextTheme))

    transition.ready.then(() => {
      const radius = Math.hypot(
        Math.max(x, window.innerWidth - x),
        Math.max(y, window.innerHeight - y),
      )

      h.animate(
        {
          clipPath: [
            'circle(0px at ' + x + 'px ' + y + 'px)',
            'circle(' + radius + 'px at ' + x + 'px ' + y + 'px)',
          ],
        },
        {
          duration: 360,
          easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
          pseudoElement: '::view-transition-new(root)',
        } as KeyframeAnimationOptions & { pseudoElement: string },
      )
    }).catch(() => {})

    transition.finished.then(finishTransition, finishTransition)
    themeTimer = window.setTimeout(finishTransition, themeTransitionDuration)
    return
  }

  h.classList.add('theme-switching')
  window.requestAnimationFrame(() => applyTheme(nextTheme))
  themeTimer = window.setTimeout(finishTransition, themeTransitionDuration)
}

function toggleTheme(event?: MouseEvent) {
  const currentIndex = themeModes.indexOf(themeMode.value)
  const nextTheme = themeModes[(currentIndex + 1) % themeModes.length]
  setTheme(nextTheme, event)
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
provide('isThemeReady', isThemeReady)
provide('themeMode', themeMode)
provide('setTheme', setTheme)
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
