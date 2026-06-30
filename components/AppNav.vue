<template>
  <nav class="nav" aria-label="主导航">
    <div class="nav-inner">
      <!-- Brand -->
      <NuxtLink to="/" class="nav-brand" aria-label="回到首页">
        <span class="nav-brand-prefix">&gt;_</span>gu.log
      </NuxtLink>

      <!-- Nav links -->
      <div class="nav-links" role="list">
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

      <!-- Actions -->
      <div class="nav-actions">
        <!-- GitHub link (hidden on very small screens) -->
        <a
          :href="appConfig.github"
          target="_blank"
          rel="noopener noreferrer"
          class="nav-icon-btn"
          aria-label="GitHub"
          style="display: none"
          ref="githubBtn"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
          </svg>
        </a>

        <!-- Theme toggle -->
        <button
          class="nav-icon-btn"
          :aria-label="isDark ? '切换到亮色模式' : '切换到暗色模式'"
          @click="toggleTheme()"
        >
          <!-- Sun icon (shown in dark mode) -->
          <svg v-if="isDark" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="5"/>
            <line x1="12" y1="1" x2="12" y2="3"/>
            <line x1="12" y1="21" x2="12" y2="23"/>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
            <line x1="1" y1="12" x2="3" y2="12"/>
            <line x1="21" y1="12" x2="23" y2="12"/>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
          </svg>
          <!-- Moon icon (shown in light mode) -->
          <svg v-else width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
          </svg>
        </button>
      </div>
    </div>
  </nav>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
const route = useRoute()

const isDark = inject<Ref<boolean>>('isDark', ref(true))
const toggleTheme = inject<() => void>('toggleTheme', () => {})

const githubBtn = ref<HTMLElement | null>(null)

onMounted(() => {
  // Show github button on screens wider than 640px
  const mq = window.matchMedia('(min-width: 640px)')
  const update = (e: MediaQueryListEvent | MediaQueryList) => {
    if (githubBtn.value) {
      githubBtn.value.style.display = e.matches ? 'flex' : 'none'
    }
  }
  update(mq)
  mq.addEventListener('change', update)
})

function isActive(path: string): boolean {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}
</script>
