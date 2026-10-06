<template>
  <div class="page-frame writing-archive">
    <header class="writing-welcome">
      <div>
        <p class="writing-kicker"><span aria-hidden="true">✳</span> Notes & explorations</p>
        <h1>文章<span>手记<svg viewBox="0 0 160 12" fill="none" aria-hidden="true"><path d="M3 8q71-9 154-3M15 11q61-6 128-3" /></svg></span></h1>
        <p class="writing-intro">把问题拆开，把理解留下。<br />这里收着一路折腾的笔记，也欢迎你随意翻翻。</p>
        <p v-if="!error && status === 'success'" class="writing-stats"><span>{{ posts.length }} 篇文章</span><span>{{ tags.length }} 个标签</span><NuxtLink to="/tags">按标签逛逛 <span aria-hidden="true">↗</span></NuxtLink></p>
      </div>
      <div class="writing-doodle" aria-hidden="true">
        <svg viewBox="0 0 210 170" fill="none">
          <path class="doodle-shadow" d="m37 60 106-10 20 102-110 9Z" />
          <path class="doodle-paper" d="m51 27 108 10-12 114-109-9Z" />
          <path class="doodle-spine" d="m65 28-12 115" />
          <path class="doodle-lines" d="m82 61 52 5m-55 17 48 5m-51 17 29 3" />
          <path class="doodle-tape" d="m86 18 45 4-3 21-44-4Z" />
          <path class="doodle-pencil" d="m169 62 10 4-28 63-11 8 1-14Z" />
          <path class="doodle-spark" d="M25 42q2-9 10-11-8-2-10-11-2 9-10 11 8 2 10 11ZM180 145q1-6 7-8-6-1-7-7-2 6-8 7 6 2 8 8Z" />
          <path class="doodle-loop" d="M164 24q17-21 28-8c9 14-7 28-15 20-7-7 13-11 20-1" />
        </svg>
        <span>a little notebook</span>
      </div>
    </header>

    <section class="writing-find" aria-label="查找文章">
      <div class="writing-find-top">
        <label class="writing-search">
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" /><path d="m15.5 15.5 4 4" /></svg>
          <input v-model="query" type="search" placeholder="搜索文章或关键词…" aria-label="搜索文章" />
          <button v-if="query" type="button" aria-label="清空搜索" @click="query = ''">×</button>
        </label>
        <NuxtLink to="/tags" class="writing-tag-link">标签索引 <span aria-hidden="true">↗</span></NuxtLink>
      </div>
      <div v-if="tags.length" class="writing-filters" role="group" aria-label="按标签筛选文章">
        <button type="button" :aria-pressed="!activeTag" @click="activeTag = ''">全部 <span>{{ posts.length }}</span></button>
        <button v-for="entry in tags" :key="entry.slug" type="button" :aria-pressed="activeTag === entry.slug" :data-tone="archiveTone(entry.slug)" @click="activeTag = activeTag === entry.slug ? '' : entry.slug">{{ entry.tag }} <span>{{ entry.count }}</span></button>
      </div>
    </section>

    <div class="writing-results">
      <p role="status" aria-live="polite" aria-atomic="true">{{ hasFilters ? `找到 ${filteredPosts.length} 篇文章` : '按时间，慢慢翻。' }}<span v-if="activeTagName"> · #{{ activeTagName }}</span></p>
      <button v-if="hasFilters" type="button" @click="resetFilters">重置筛选 <span aria-hidden="true">↺</span></button>
    </div>
    <div v-if="error" class="writing-empty" role="alert"><p>文章暂时没能加载，刷新页面再试试吧。</p><a href="/posts/">刷新页面 <span aria-hidden="true">↻</span></a></div>
    <div v-else-if="status === 'pending'" class="writing-empty" role="status"><p>正在整理笔记…</p></div>
    <ArchiveEntries v-else-if="filteredPosts.length" :posts="filteredPosts" />
    <div v-else class="writing-empty">
      <span class="writing-empty-mark" aria-hidden="true">⌕</span>
      <h2>{{ posts.length ? '这次没找到。' : '手记正在慢慢积累。' }}</h2>
      <p>{{ posts.length ? '换个关键词，或放开标签筛选，再翻翻看。' : '有新的公开文章时，就会出现在这里。' }}</p>
      <button v-if="hasFilters" type="button" @click="resetFilters">查看全部文章 <span aria-hidden="true">→</span></button>
      <NuxtLink v-else to="/blog">回到博客 <span aria-hidden="true">→</span></NuxtLink>
    </div>

    <p class="writing-signoff"><span aria-hidden="true">✧</span> 写下来，就是思考留下的脚印。<NuxtLink to="/blog">回到博客 <span aria-hidden="true">→</span></NuxtLink></p>
  </div>
</template>

<script setup lang="ts">
import { archiveTone, collectArchiveTags } from '~/utils/archive'
import { tagSlug } from '~/utils/blog'

const { data, error, status } = await usePublicArchive()
const posts = computed(() => data.value ?? [])
const tags = computed(() => collectArchiveTags(posts.value))
const query = ref('')
const activeTag = ref('')
const activeTagName = computed(() => tags.value.find(tag => tag.slug === activeTag.value)?.tag ?? '')
const hasFilters = computed(() => Boolean(query.value.trim() || activeTag.value))
const filteredPosts = computed(() => {
  const keyword = query.value.trim().toLocaleLowerCase('zh-CN')
  return posts.value.filter(post => {
    if (activeTag.value && !post.tags.some(tag => tagSlug(tag) === activeTag.value)) return false
    if (!keyword) return true
    return [post.title, post.description, post.excerpt, post.series, ...post.tags].join(' ').toLocaleLowerCase('zh-CN').includes(keyword)
  })
})
function resetFilters() {
  query.value = ''
  activeTag.value = ''
}

useHead({
  title: '文章库',
  meta: [{ name: 'description', content: '古恩豪的技术笔记与工程手记，按时间阅读，也可以搜索关键词或按标签浏览。' }],
})
</script>

<style scoped>
.writing-archive :where(h1, h2, p) { margin: 0; }
.writing-welcome { display: grid; grid-template-columns: minmax(0, 1fr) 240px; align-items: center; gap: 40px; padding: 0 28px 26px; }
.writing-kicker { display: flex; align-items: center; gap: 9px; font-family: Georgia, serif; font-style: italic; font-size: 0.85rem; color: var(--muted); }
.writing-kicker > span { font-size: 1.4rem; color: var(--accent-2); }
.writing-welcome h1 { margin-block: 10px 12px; color: var(--ink); font-size: clamp(2.5rem, 4.2vw, 2.7rem); font-weight: 500; letter-spacing: -0.05em; line-height: 1.5; }
.writing-welcome h1 > span { position: relative; display: inline-block; color: var(--accent); }
.writing-welcome h1 svg { position: absolute; width: 106%; height: 12px; left: -3%; bottom: 0; stroke: var(--accent-2); stroke-width: 1.6; stroke-linecap: round; opacity: 0.6; }
.writing-intro { color: var(--text); font-size: 0.88rem; line-height: 1.95; }
.writing-stats { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; margin-top: 18px; color: var(--muted); font-size: 0.74rem; }
.writing-stats > span + span::before { content: '·'; margin-right: 18px; color: var(--line-strong); }
.writing-stats a { color: var(--accent); }
.writing-doodle { position: relative; width: 180px; justify-self: center; transform: rotate(3deg); }
.writing-doodle svg { width: 100%; height: auto; }
.doodle-shadow { fill: var(--topic-purple); }
.doodle-paper { fill: var(--surface); stroke: var(--line-strong); stroke-width: 1.5; stroke-linejoin: round; }
.doodle-spine, .doodle-lines { stroke: var(--accent-2); stroke-width: 1.5; stroke-linecap: round; opacity: 0.45; }
.doodle-tape { fill: var(--topic-green); stroke: color-mix(in srgb, var(--accent) 14%, transparent); }
.doodle-pencil { fill: var(--topic-orange); stroke: var(--accent-2); stroke-width: 1.2; stroke-linejoin: round; }
.doodle-spark { fill: var(--accent-2); opacity: 0.55; }
.doodle-loop { stroke: var(--accent-2); stroke-width: 1.5; stroke-linecap: round; opacity: 0.5; }
.writing-doodle > span { display: block; text-align: center; font-family: Georgia, serif; font-style: italic; font-size: 0.85rem; color: var(--muted); }
.writing-find { padding: 20px 26px; border-radius: 22px; background: var(--bg-2); }
.writing-find-top { display: flex; align-items: center; gap: 24px; }
.writing-search { flex: 1; display: flex; align-items: center; gap: 10px; min-width: 0; padding: 0 14px; border: 1px solid var(--line); border-radius: 100px; background: var(--surface); }
.writing-search > svg { width: 18px; height: 18px; flex: 0 0 auto; stroke: var(--muted); stroke-width: 1.5; stroke-linecap: round; }
.writing-search input { width: 100%; min-width: 0; height: 44px; padding: 0; border: 0; outline: 0; background: transparent; color: var(--ink); font-size: 0.8rem; }
.writing-search:focus-within { outline: 2px solid var(--accent); outline-offset: 3px; }
.writing-search input:focus-visible { outline: 0; }
.writing-search input::placeholder { color: var(--muted); }
.writing-search input::-webkit-search-cancel-button { display: none; }
.writing-search button { flex: 0 0 auto; width: 30px; height: 32px; border: 0; background: transparent; color: var(--muted); font-size: 1.3rem; cursor: pointer; }
.writing-tag-link { flex: 0 0 auto; font-size: 0.75rem; color: var(--accent); white-space: nowrap; }
.writing-tag-link > span { margin-left: 6px; }
.writing-filters { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 16px; }
.writing-filters button { display: inline-flex; align-items: center; gap: 7px; min-height: 34px; padding: 5px 11px; border: 1px solid transparent; border-radius: 100px; background: transparent; color: var(--text); font-size: 0.72rem; cursor: pointer; overflow-wrap: anywhere; }
.writing-filters button > span { color: var(--muted); font-size: 0.64rem; }
.writing-filters button:hover { background: var(--surface); }
.writing-filters button[aria-pressed="true"] { background: var(--surface); border-color: var(--accent); color: var(--accent); }
.writing-filters button[data-tone="purple"][aria-pressed="true"] { background: var(--topic-purple); }
.writing-filters button[data-tone="green"][aria-pressed="true"] { background: var(--topic-green); }
.writing-filters button[data-tone="blue"][aria-pressed="true"] { background: var(--topic-blue); }
.writing-filters button[data-tone="orange"][aria-pressed="true"] { background: var(--topic-orange); }
.writing-results { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-block: 22px 16px; color: var(--muted); font-size: 0.74rem; }
.writing-results p { min-width: 0; overflow-wrap: anywhere; }
.writing-results button { flex: 0 0 auto; padding: 6px 0; border: 0; background: transparent; color: var(--accent); font-size: 0.73rem; cursor: pointer; }
.writing-empty { padding: 50px 20px 60px; text-align: center; }
.writing-empty-mark { display: block; color: var(--accent-2); font-size: 2.6rem; line-height: 1.2; }
.writing-empty h2 { margin-top: 15px; color: var(--ink); font-size: 1.15rem; font-weight: 500; }
.writing-empty p { margin-top: 10px; color: var(--muted); font-size: 0.85rem; }
.writing-empty button, .writing-empty a { display: inline-flex; gap: 16px; align-items: center; min-height: 40px; margin-top: 20px; padding: 5px 18px; border: 0; border-radius: 100px; background: var(--surface-2); color: var(--accent); font-size: 0.8rem; cursor: pointer; }
.writing-signoff { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin-top: 36px; padding-inline: 4px; color: var(--muted); font-size: 0.74rem; }
.writing-signoff > span { font-size: 1.3rem; color: var(--accent-2); }
.writing-signoff a { margin-left: auto; color: var(--accent); }
@media (hover: hover) and (pointer: fine) {
  .writing-doodle { transition: transform 220ms ease; }
  .writing-doodle:hover { transform: rotate(0deg) translateY(-2px); }
}
@media (max-width: 700px) {
  .writing-welcome { grid-template-columns: minmax(0, 1fr) 160px; gap: 20px; padding-inline: 4px; }
  .writing-doodle { width: 155px; }
  .writing-find { padding: 20px; }
  .writing-find-top { gap: 16px; }
}
@media (max-width: 480px) {
  .writing-welcome { grid-template-columns: 1fr; padding: 0 2px 28px; }
  .writing-doodle { display: none; }
  .writing-welcome h1 { font-size: 2.5rem; margin-bottom: 14px; }
  .writing-intro { font-size: 0.83rem; }
  .writing-stats { gap: 9px 14px; font-size: 0.7rem; margin-top: 18px; }
  .writing-stats > span + span::before { margin-right: 14px; }
  .writing-find { padding: 16px; border-radius: 18px; }
  .writing-find-top { flex-wrap: wrap; gap: 10px; }
  .writing-search { flex-basis: 100%; }
  .writing-search input { font-size: 1rem; }
  .writing-tag-link { margin-left: auto; font-size: 0.7rem; }
  .writing-filters { gap: 4px; margin-top: 10px; }
  .writing-filters button { padding-inline: 9px; font-size: 0.68rem; }
  .writing-results { font-size: 0.7rem; margin-block: 20px; }
  .writing-signoff { font-size: 0.7rem; gap: 9px; }
}
@media (prefers-reduced-motion: reduce) {
  .writing-doodle { transition: none; }
  .writing-doodle:hover { transform: rotate(3deg); }
}
</style>
