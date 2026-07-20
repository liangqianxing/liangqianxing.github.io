<template>
  <nav class="site-nav" aria-label="主导航">
    <div class="nav-shell">
      <NuxtLink to="/" class="brand" aria-label="回到首页">
        <span class="brand-mark" aria-hidden="true">
          <img src="/logo.svg" alt="" width="42" height="42" />
        </span>
        <span class="brand-copy">
          <strong>{{ appConfig.title }}</strong>
          <em>{{ appConfig.authorEN }}</em>
        </span>
      </NuxtLink>

      <div class="nav-center" role="list">
        <NuxtLink
          v-for="item in appConfig.nav"
          :key="item.path"
          :to="item.path"
          class="nav-link"
          :class="{ 'nav-link-active': isActive(item.path) }"
          role="listitem"
        >
          {{ item.label }}
        </NuxtLink>
      </div>

      <div class="nav-actions">
        <a
          href="https://www.travellings.cn/go.html"
          target="_blank"
          rel="noreferrer noopener"
          class="nav-chip"
          aria-label="开往 - 友链接力"
        >
          <span aria-hidden="true">↗</span>
          <span class="optional-label">开往</span>
        </a>
        <a
          :href="appConfig.github"
          target="_blank"
          rel="noopener noreferrer"
          class="icon-button optional-github"
          aria-label="GitHub"
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
          </svg>
        </a>
        <div
          class="theme-toggle"
          role="group"
          aria-label="主题"
          :data-theme="isThemeReady ? themeMode : 'dark'"
          data-allow-mismatch="attribute"
        >
          <span class="theme-toggle-track">
            <span class="theme-toggle-glow" />
            <span class="theme-toggle-indicator" />
            <button
              class="theme-toggle-option"
              :class="{ active: themeMode === 'light' }"
              type="button"
              aria-label="纸感浅色"
              title="纸感浅色"
              :aria-pressed="themeMode === 'light'"
              @click="setTheme('light', $event)"
            >
              <span aria-hidden="true">☼</span>
            </button>
            <button
              class="theme-toggle-option"
              :class="{ active: themeMode === 'dark' }"
              type="button"
              aria-label="暖黑深色"
              title="暖黑深色"
              :aria-pressed="themeMode === 'dark'"
              @click="setTheme('dark', $event)"
            >
              <span aria-hidden="true">☾</span>
            </button>
            <button
              class="theme-toggle-option theme-toggle-terminal"
              :class="{ active: themeMode === 'cyber' }"
              type="button"
              aria-label="荧光终端"
              title="荧光终端"
              :aria-pressed="themeMode === 'cyber'"
              @click="setTheme('cyber', $event)"
            >
              <span aria-hidden="true">&gt;_</span>
            </button>
          </span>
          <span class="sr-only" data-allow-mismatch="text">{{ themeLabel }}</span>
        </div>
      </div>
    </div>
  </nav>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
const route = useRoute()

const isThemeReady = inject<Ref<boolean>>('isThemeReady', ref(false))
const themeMode = inject<Ref<'dark' | 'light' | 'cyber'>>('themeMode', ref('dark'))
const setTheme = inject<(mode: 'dark' | 'light' | 'cyber', event?: MouseEvent) => void>('setTheme', () => {})

const themeLabel = computed(() => {
  if (themeMode.value === 'light') return '纸感浅色'
  if (themeMode.value === 'cyber') return '荧光终端'
  return '暖黑深色'
})

function isActive(path: string): boolean {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}
</script>
