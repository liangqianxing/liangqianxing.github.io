<template>
  <Teleport to="body">
    <Transition name="cmdk">
      <div
        v-if="open"
        class="cmdk-overlay"
        role="dialog"
        aria-modal="true"
        aria-label="全局搜索与快捷面板"
        @mousedown.self="close"
      >
        <div ref="modalRef" class="cmdk-modal" @keydown="onModalKeydown">
          <div class="cmdk-header">
            <svg class="cmdk-search-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle cx="11" cy="11" r="7" />
              <path d="m16.2 16.2 4.3 4.3" />
            </svg>
            <input
              ref="inputRef"
              v-model="query"
              type="text"
              class="cmdk-input"
              placeholder="搜索文章、课程、标签，或切换主题…"
              autocomplete="off"
              spellcheck="false"
            />
            <button v-if="query" type="button" class="cmdk-clear" aria-label="清空输入" @click="query = ''">×</button>
            <kbd class="cmdk-esc" @click="close">ESC</kbd>
          </div>

          <div ref="listRef" class="cmdk-body" role="listbox" aria-label="快捷指令与搜索结果">
            <template v-if="groupedItems.length">
              <div v-for="group in groupedItems" :key="group.category" class="cmdk-group">
                <div class="cmdk-group-label">{{ group.category }}</div>
                <button
                  v-for="item in group.items"
                  :key="item.id"
                  type="button"
                  role="option"
                  class="cmdk-item"
                  :class="{ 'is-active': item.flatIndex === activeIndex }"
                  :aria-selected="item.flatIndex === activeIndex"
                  :data-index="item.flatIndex"
                  @mouseenter="activeIndex = item.flatIndex"
                  @click="execute(item, $event)"
                >
                  <span class="cmdk-item-icon" :data-kind="item.kind" aria-hidden="true">{{ item.icon }}</span>
                  <span class="cmdk-item-main">
                    <span class="cmdk-item-title">{{ item.title }}</span>
                    <span v-if="item.subtitle" class="cmdk-item-sub">{{ item.subtitle }}</span>
                  </span>
                  <span v-if="item.badge" class="cmdk-item-badge">{{ item.badge }}</span>
                  <span class="cmdk-item-arrow" aria-hidden="true">↵</span>
                </button>
              </div>
            </template>
            <div v-else class="cmdk-empty">
              <span class="cmdk-empty-icon" aria-hidden="true">✦</span>
              <p>未找到与 “{{ query }}” 匹配的笔记或指令</p>
              <small>试试搜索 LLM、Agent、CS336、CMU 或输入主题名称</small>
            </div>
          </div>

          <footer class="cmdk-footer">
            <div class="cmdk-hints">
              <span><kbd>↑</kbd><kbd>↓</kbd> 导航</span>
              <span><kbd>↵</kbd> 跳转 / 执行</span>
              <span><kbd>⌃T</kbd> 切换主题</span>
            </div>
            <span class="cmdk-brand">{{ appConfig.title }} · Command</span>
          </footer>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import type { ThemeOriginEvent } from '~/composables/useTheme'
import { learningCourses } from '~/data/learning/courses'
import { collectArchiveTags, tagPath } from '~/utils/archive'
import { formatDate } from '~/utils/blog'

interface CommandItem {
  id: string
  category: string
  kind: 'nav' | 'post' | 'course' | 'tag' | 'action'
  icon: string
  title: string
  subtitle?: string
  badge?: string
  keywords?: string
  action: (origin?: ThemeOriginEvent) => void
}

const props = defineProps<{
  open: boolean
}>()

const emit = defineEmits<{
  'update:open': [value: boolean]
}>()

const appConfig = useAppConfig()
const router = useRouter()
const { data: postsData, status: postsStatus, refresh: refreshPosts } = usePublicArchive({ lazy: true, immediate: false })
const { themeMode, setTheme, toggleTheme } = useTheme()

const query = ref('')
const activeIndex = ref(0)
const modalRef = ref<HTMLElement | null>(null)
const inputRef = ref<HTMLInputElement | null>(null)
const listRef = ref<HTMLElement | null>(null)

function resolveOrigin(event?: MouseEvent): ThemeOriginEvent {
  if (event && (event.clientX !== 0 || event.clientY !== 0)) {
    return event
  }
  const activeEl = listRef.value?.querySelector<HTMLElement>(`[data-index="${activeIndex.value}"]`)
  const targetEl = activeEl ?? modalRef.value
  if (targetEl) {
    const rect = targetEl.getBoundingClientRect()
    return {
      clientX: Math.round(rect.left + rect.width / 2),
      clientY: Math.round(rect.top + rect.height / 2),
    }
  }
  return undefined
}

function close() {
  emit('update:open', false)
}

function go(path: string) {
  close()
  router.push(path)
}

const allItems = computed<CommandItem[]>(() => {
  const posts = postsData.value ?? []
  const tags = collectArchiveTags(posts)

  const navItems: CommandItem[] = [
    {
      id: 'nav-blog',
      category: '快速导航 / Navigation',
      kind: 'nav',
      icon: '⌂',
      title: '博客首页 · Notebook',
      subtitle: '回到 gu.log 个人笔记首页',
      badge: '/blog',
      keywords: 'home blog 首页 博客 笔记',
      action: () => go('/blog'),
    },
    {
      id: 'nav-posts',
      category: '快速导航 / Navigation',
      kind: 'nav',
      icon: '✎',
      title: '文章手记 · All Posts',
      subtitle: `浏览全部 ${posts.length} 篇公开技术文章`,
      badge: '/posts',
      keywords: 'posts articles 文章 手记 归档',
      action: () => go('/posts'),
    },
    {
      id: 'nav-learning',
      category: '快速导航 / Navigation',
      kind: 'nav',
      icon: '◈',
      title: '课程学习 · Learning',
      subtitle: `${learningCourses.length} 门系统与大模型课程专题`,
      badge: '/learning',
      keywords: 'learning courses 课程 学习 专题 cs336 cmu',
      action: () => go('/learning'),
    },
    {
      id: 'nav-academic',
      category: '快速导航 / Navigation',
      kind: 'nav',
      icon: '🎓',
      title: '学术主页 · Academic Profile',
      subtitle: '研究方向、发表论文与学术经历',
      badge: '/academic',
      keywords: 'academic research paper 学术 论文 研究',
      action: () => go('/academic'),
    },
    {
      id: 'nav-tags',
      category: '快速导航 / Navigation',
      kind: 'nav',
      icon: '#',
      title: '标签索引 · Topic Index',
      subtitle: `按 ${tags.length} 个主题标签探索技术脉络`,
      badge: '/tags',
      keywords: 'tags topics 标签 分类 索引',
      action: () => go('/tags'),
    },
    {
      id: 'nav-about',
      category: '快速导航 / Navigation',
      kind: 'nav',
      icon: '✳',
      title: '关于作者 · About Enhao Gu',
      subtitle: '古恩豪的个人介绍、经历与工具箱',
      badge: '/about',
      keywords: 'about me 关于 作者 古恩豪 简历',
      action: () => go('/about'),
    },
    {
      id: 'nav-gateway',
      category: '快速导航 / Navigation',
      kind: 'nav',
      icon: '◐',
      title: '个人门户 · Gateway Poster',
      subtitle: '3D 交互海报式个人主页入口',
      badge: '/',
      keywords: 'gateway index 主页 门户 海报',
      action: () => go('/'),
    },
  ]

  const courseItems: CommandItem[] = learningCourses.map(course => ({
    id: `course-${course.slug}`,
    category: '课程专题 / Courses',
    kind: 'course',
    icon: '❖',
    title: `${course.title} · ${course.code}`,
    subtitle: `${course.institution} — ${course.courseTitle}`,
    badge: '专题',
    keywords: `${course.title} ${course.code} ${course.institution} ${course.courseTitle} ${course.topics.join(' ')}`,
    action: () => go(`/learning/${encodeURIComponent(course.slug)}`),
  }))

  const postItems: CommandItem[] = posts.map(post => ({
    id: `post-${post.slug}`,
    category: '技术手记 / Articles',
    kind: 'post',
    icon: '↗',
    title: post.title,
    subtitle: post.description || post.excerpt,
    badge: formatDate(post.date).slice(5) || `${post.readingTime}m`,
    keywords: `${post.title} ${post.description} ${post.series ?? ''} ${(post.tags ?? []).join(' ')}`,
    action: () => go(post.path),
  }))

  const tagItems: CommandItem[] = tags.slice(0, 20).map(entry => ({
    id: `tag-${entry.slug}`,
    category: '热门标签 / Tags',
    kind: 'tag',
    icon: '#',
    title: `#${entry.tag}`,
    subtitle: `最新：${entry.latestPost.title}`,
    badge: `${entry.count} 篇`,
    keywords: `tag ${entry.tag} 标签`,
    action: () => go(tagPath(entry.tag)),
  }))

  const actionItems: CommandItem[] = [
    {
      id: 'theme-light',
      category: '外观与操作 / Appearance',
      kind: 'action',
      icon: '☀',
      title: '切换为浅色纸感主题 · Warm Paper',
      subtitle: '柔和米白手账风格',
      badge: themeMode.value === 'light' ? '当前' : 'Light',
      keywords: 'theme light 浅色 亮色 主题 日间',
      action: (origin) => { close(); setTheme('light', true, origin) },
    },
    {
      id: 'theme-dark',
      category: '外观与操作 / Appearance',
      kind: 'action',
      icon: '☾',
      title: '切换为深色夜读主题 · Midnight Ink',
      subtitle: '静谧深蓝低眩光暗色模式',
      badge: themeMode.value === 'dark' ? '当前' : 'Dark',
      keywords: 'theme dark 深色 暗黑 夜间 主题',
      action: (origin) => { close(); setTheme('dark', true, origin) },
    },
    {
      id: 'theme-cyber',
      category: '外观与操作 / Appearance',
      kind: 'action',
      icon: '>_',
      title: '切换为终端极客主题 · Cyber Terminal',
      subtitle: '荧光绿矩阵极客风格',
      badge: themeMode.value === 'cyber' ? '当前' : 'Cyber',
      keywords: 'theme cyber terminal 终端 极客 赛博 主题',
      action: (origin) => { close(); setTheme('cyber', true, origin) },
    },
  ]

  return [...navItems, ...courseItems, ...postItems, ...tagItems, ...actionItems]
})

const filteredItems = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) {
    // When empty, show a curated set: nav, courses, latest 6 posts, actions
    const navs = allItems.value.filter(i => i.kind === 'nav')
    const courses = allItems.value.filter(i => i.kind === 'course')
    const posts = allItems.value.filter(i => i.kind === 'post').slice(0, 6)
    const actions = allItems.value.filter(i => i.kind === 'action')
    return [...navs, ...courses, ...posts, ...actions]
  }
  const terms = q.split(/\s+/).filter(Boolean)
  return allItems.value.filter(item => {
    const haystack = `${item.title} ${item.subtitle ?? ''} ${item.keywords ?? ''} ${item.badge ?? ''}`.toLowerCase()
    return terms.every(term => haystack.includes(term))
  }).slice(0, 28)
})

const groupedItems = computed(() => {
  const groups: { category: string; items: (CommandItem & { flatIndex: number })[] }[] = []
  const map = new Map<string, (CommandItem & { flatIndex: number })[]>()
  filteredItems.value.forEach((item, idx) => {
    let list = map.get(item.category)
    if (!list) {
      list = []
      map.set(item.category, list)
      groups.push({ category: item.category, items: list })
    }
    list.push({ ...item, flatIndex: idx })
  })
  return groups
})

watch(() => props.open, async (isOpen) => {
  if (isOpen) {
    query.value = ''
    activeIndex.value = 0
    if (!postsData.value?.length && postsStatus.value !== 'pending') {
      void refreshPosts()
    }
    await nextTick()
    inputRef.value?.focus()
  }
})

watch(query, () => {
  activeIndex.value = 0
})

function scrollActiveIntoView() {
  nextTick(() => {
    const el = listRef.value?.querySelector<HTMLElement>(`[data-index="${activeIndex.value}"]`)
    el?.scrollIntoView({ block: 'nearest' })
  })
}

function execute(item: CommandItem, event?: MouseEvent) {
  item.action(resolveOrigin(event))
}

function onModalKeydown(event: KeyboardEvent) {
  const total = filteredItems.value.length
  if (event.key === 'Escape') {
    event.preventDefault()
    close()
    return
  }
  if ((event.ctrlKey || event.altKey) && !event.metaKey && event.key.toLowerCase() === 't') {
    event.preventDefault()
    toggleTheme(resolveOrigin())
    return
  }
  if (!total) return
  if (event.key === 'ArrowDown') {
    event.preventDefault()
    activeIndex.value = (activeIndex.value + 1) % total
    scrollActiveIntoView()
  } else if (event.key === 'ArrowUp') {
    event.preventDefault()
    activeIndex.value = (activeIndex.value - 1 + total) % total
    scrollActiveIntoView()
  } else if (event.key === 'Enter') {
    event.preventDefault()
    const item = filteredItems.value[activeIndex.value]
    if (item) execute(item)
  }
}
</script>

<style scoped>
.cmdk-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: min(12vh, 96px) 16px 24px;
  background: rgba(15, 14, 22, 0.48);
  backdrop-filter: blur(12px) saturate(1.2);
  -webkit-backdrop-filter: blur(12px) saturate(1.2);
}

.cmdk-modal {
  width: min(640px, 100%);
  max-height: min(74vh, 580px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--accent) 26%, var(--line));
  border-radius: 22px;
  background: color-mix(in srgb, var(--surface) 94%, transparent);
  box-shadow:
    0 28px 80px rgba(10, 8, 20, 0.34),
    0 0 0 1px color-mix(in srgb, white 14%, transparent) inset,
    0 0 48px color-mix(in srgb, var(--accent) 12%, transparent);
  color: var(--text);
}

.cmdk-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 18px;
  border-bottom: 1px solid var(--line);
  background: color-mix(in srgb, var(--bg) 55%, transparent);
}

.cmdk-search-icon {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
  stroke: var(--accent);
  stroke-width: 1.8;
  stroke-linecap: round;
}

.cmdk-input {
  flex: 1;
  min-width: 0;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--ink);
  font-size: 0.96rem;
  line-height: 1.5;
}

.cmdk-input::placeholder {
  color: var(--muted);
}

.cmdk-clear {
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 50%;
  background: var(--surface-2);
  color: var(--muted);
  font-size: 1rem;
  cursor: pointer;
}

.cmdk-clear:hover {
  color: var(--ink);
}

.cmdk-esc {
  padding: 4px 8px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: var(--surface);
  color: var(--muted);
  font: 0.68rem/1.2 var(--font-mono);
  cursor: pointer;
  user-select: none;
}

.cmdk-body {
  flex: 1;
  overflow-y: auto;
  padding: 10px 10px 12px;
  scrollbar-width: thin;
}

.cmdk-group + .cmdk-group {
  margin-top: 12px;
}

.cmdk-group-label {
  padding: 6px 12px 5px;
  color: var(--muted);
  font: 600 0.66rem/1.4 var(--font-mono);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.cmdk-item {
  width: 100%;
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) auto 20px;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border: 0;
  border-radius: 13px;
  background: transparent;
  color: var(--text);
  text-align: left;
  cursor: pointer;
  transition: background-color 140ms ease, transform 140ms ease;
}

.cmdk-item.is-active {
  background: color-mix(in srgb, var(--accent) 13%, var(--surface-2));
}

.cmdk-item-icon {
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: var(--bg-2);
  color: var(--accent);
  font: 500 0.85rem/1 var(--font-mono);
  border: 1px solid color-mix(in srgb, var(--line) 80%, transparent);
  transition: transform 180ms ease, background-color 180ms ease;
}

.cmdk-item.is-active .cmdk-item-icon {
  background: var(--surface);
  border-color: color-mix(in srgb, var(--accent) 40%, var(--line));
  transform: scale(1.04);
}

.cmdk-item-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.cmdk-item-title {
  color: var(--ink);
  font-size: 0.86rem;
  font-weight: 500;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cmdk-item-sub {
  color: var(--muted);
  font-size: 0.74rem;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cmdk-item-badge {
  padding: 3px 8px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--surface) 75%, var(--bg-2));
  border: 1px solid var(--line);
  color: var(--muted);
  font: 0.65rem/1.3 var(--font-mono);
  white-space: nowrap;
}

.cmdk-item-arrow {
  color: var(--accent);
  font-size: 0.8rem;
  opacity: 0;
  transform: translateX(-4px);
  transition: opacity 150ms ease, transform 150ms ease;
}

.cmdk-item.is-active .cmdk-item-arrow {
  opacity: 1;
  transform: translateX(0);
}

.cmdk-empty {
  padding: 44px 20px;
  text-align: center;
  color: var(--muted);
}

.cmdk-empty-icon {
  display: inline-block;
  font-size: 1.8rem;
  color: var(--accent);
  margin-bottom: 8px;
}

.cmdk-empty p {
  margin: 0;
  color: var(--ink);
  font-size: 0.9rem;
}

.cmdk-empty small {
  display: block;
  margin-top: 6px;
  font-size: 0.75rem;
}

.cmdk-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 18px;
  border-top: 1px solid var(--line);
  background: color-mix(in srgb, var(--bg-2) 70%, transparent);
  color: var(--muted);
  font-size: 0.7rem;
}

.cmdk-hints {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}

.cmdk-hints span {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.cmdk-hints kbd {
  padding: 2px 5px;
  border: 1px solid var(--line);
  border-radius: 5px;
  background: var(--surface);
  font: 0.64rem/1.1 var(--font-mono);
}

.cmdk-brand {
  font: 0.66rem var(--font-mono);
  opacity: 0.75;
}

.cmdk-enter-active,
.cmdk-leave-active {
  transition: opacity 200ms cubic-bezier(0.16, 1, 0.3, 1);
}

.cmdk-enter-active .cmdk-modal,
.cmdk-leave-active .cmdk-modal {
  transition: transform 220ms cubic-bezier(0.16, 1, 0.3, 1), opacity 200ms ease;
}

.cmdk-enter-from,
.cmdk-leave-to {
  opacity: 0;
}

.cmdk-enter-from .cmdk-modal,
.cmdk-leave-to .cmdk-modal {
  opacity: 0;
  transform: scale(0.96) translateY(-10px);
}

@media (max-width: 560px) {
  .cmdk-overlay {
    padding: 16px 12px;
  }
  .cmdk-hints {
    display: none;
  }
}
</style>
