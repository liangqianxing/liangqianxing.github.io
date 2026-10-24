export type ThemeMode = 'dark' | 'light' | 'cyber'
export type ThemeOriginEvent = MouseEvent | { clientX?: number; clientY?: number } | undefined

const THEME_MODES: ThemeMode[] = ['light', 'dark', 'cyber']

let themeSwitchTimer: ReturnType<typeof setTimeout> | undefined
let fallbackSwitchTimer: ReturnType<typeof setTimeout> | undefined

export function normalizeTheme(value: string | null | undefined): ThemeMode {
  return value === 'dark' || value === 'cyber' ? value : 'light'
}

export function useTheme() {
  const themeMode = useState<ThemeMode>('blog-theme-mode', () => 'light')
  const isThemeReady = useState<boolean>('blog-theme-ready', () => false)
  const isDark = computed(() => themeMode.value !== 'light')

  function resolveThemeOrigin(mode: ThemeMode, event?: ThemeOriginEvent): { x: number; y: number } {
    if (
      event &&
      typeof event.clientX === 'number' &&
      typeof event.clientY === 'number' &&
      (event.clientX > 0 || event.clientY > 0)
    ) {
      return { x: event.clientX, y: event.clientY }
    }
    if (typeof document !== 'undefined') {
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
    }
    return {
      x: typeof window !== 'undefined' ? Math.max(window.innerWidth - 74, 24) : 24,
      y: 42,
    }
  }

  function applyThemeDom(mode: ThemeMode, persist: boolean) {
    themeMode.value = mode
    if (typeof document === 'undefined') return
    const root = document.documentElement
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
    if (!persist || typeof window === 'undefined' || typeof document === 'undefined') {
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
    const next = THEME_MODES[(THEME_MODES.indexOf(themeMode.value) + 1) % THEME_MODES.length]
    setTheme(next, true, event)
  }

  function initThemeFromStorage() {
    let stored: string | null = null
    try { stored = localStorage.getItem('theme') } catch { /* Use the default theme. */ }
    setTheme(normalizeTheme(stored), false)
    isThemeReady.value = true
  }

  function cleanupThemeTimers() {
    if (themeSwitchTimer) clearTimeout(themeSwitchTimer)
    if (fallbackSwitchTimer) clearTimeout(fallbackSwitchTimer)
    if (typeof document !== 'undefined') {
      document.documentElement.classList.remove('theme-switching', 'theme-vt-active', 'theme-fallback-switching')
    }
  }

  return {
    themeMode,
    isDark,
    isThemeReady,
    themeModes: THEME_MODES,
    setTheme,
    toggleTheme,
    initThemeFromStorage,
    cleanupThemeTimers,
  }
}

export function useCommandPalette() {
  const commandOpen = useState<boolean>('blog-command-open', () => false)
  function openCommandPalette() {
    commandOpen.value = true
  }
  function closeCommandPalette() {
    commandOpen.value = false
  }
  function toggleCommandPalette() {
    commandOpen.value = !commandOpen.value
  }
  return {
    commandOpen,
    openCommandPalette,
    closeCommandPalette,
    toggleCommandPalette,
  }
}
