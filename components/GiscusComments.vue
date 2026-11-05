<template>
  <section v-if="enabled" class="post-comments" aria-labelledby="post-comments-title">
    <h2 id="post-comments-title" class="post-comments-title">评论</h2>
    <p class="post-comments-hint">评论由 GitHub Discussions 驱动，使用 GitHub 账号登录即可留言。</p>
    <div ref="hostRef" class="post-comments-host" />
  </section>
</template>

<script setup lang="ts">
import type { ThemeMode } from '~/composables/useTheme'

const props = defineProps<{ term: string }>()

const { giscus } = useAppConfig()
const { themeMode } = useTheme()
const hostRef = ref<HTMLElement | null>(null)
const enabled = computed(() => Boolean(giscus?.repo && giscus.repoId && giscus.categoryId))

const GISCUS_ORIGIN = 'https://giscus.app'
const giscusThemes: Record<ThemeMode, string> = {
  light: 'light',
  dark: 'dark_dimmed',
  cyber: 'transparent_dark',
}

// Strip trailing slashes so /posts/a and /posts/a/ share one discussion thread.
function normalizedTerm() {
  return props.term.replace(/\/+$/, '') || '/'
}

function mount() {
  const host = hostRef.value
  if (!host || !enabled.value) return
  host.replaceChildren()
  const script = document.createElement('script')
  script.src = `${GISCUS_ORIGIN}/client.js`
  script.async = true
  script.crossOrigin = 'anonymous'
  const attrs: Record<string, string> = {
    'data-repo': giscus.repo,
    'data-repo-id': giscus.repoId,
    'data-category': giscus.category,
    'data-category-id': giscus.categoryId,
    'data-mapping': 'specific',
    'data-term': normalizedTerm(),
    'data-strict': '1',
    'data-reactions-enabled': '1',
    'data-emit-metadata': '0',
    'data-input-position': 'top',
    'data-theme': giscusThemes[themeMode.value],
    'data-lang': 'zh-CN',
    'data-loading': 'lazy',
  }
  for (const [key, value] of Object.entries(attrs)) script.setAttribute(key, value)
  host.appendChild(script)
}

function syncTheme(mode: ThemeMode) {
  const frame = hostRef.value?.querySelector<HTMLIFrameElement>('iframe.giscus-frame')
  frame?.contentWindow?.postMessage({ giscus: { setConfig: { theme: giscusThemes[mode] } } }, GISCUS_ORIGIN)
}

onMounted(mount)
watch(() => props.term, () => nextTick(mount))
watch(themeMode, syncTheme)
</script>

<style scoped>
.post-comments {
  margin-top: 56px;
  padding-top: 28px;
  border-top: 1px solid var(--line);
}

.post-comments-title {
  margin: 0;
  color: var(--ink);
  font-size: 1.15rem;
}

.post-comments-hint {
  margin: 6px 0 20px;
  color: var(--muted);
  font-size: 0.8rem;
}

.post-comments-host {
  min-height: 160px;
}
</style>
