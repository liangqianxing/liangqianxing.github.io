<template>
  <div class="page-frame tag-index-journal">
    <header class="tag-index-intro">
      <div class="tag-index-copy">
        <p class="tag-index-kicker">A LITTLE MAP OF MY NOTES</p>
        <h1>标签索引<span aria-hidden="true">.</span></h1>
        <p class="tag-index-signature">Tags &amp; threads</p>
        <p class="tag-index-description">顺着一根线头，找到感兴趣的笔记。</p>
        <p v-if="tags.length" class="tag-index-count">{{ tags.length }} 个标签，串起 {{ posts.length }} 篇公开笔记。</p>
      </div>
      <svg class="tag-index-doodle" viewBox="0 0 260 180" fill="none" aria-hidden="true">
        <path class="tag-index-thread" d="M19 151c33 12 28-33 54-26s-8 35 22 33c25-2 13-43 39-53s25 31 54 14c19-11 20-33 35-32" />
        <g transform="rotate(-10 97 64)"><path class="tag-index-purple" d="M50 31h83l23 32-23 32H50V31Z" /><circle cx="133" cy="63" r="5" /><path d="M66 52h41M66 68h29" /></g>
        <g transform="rotate(13 188 97)"><path class="tag-index-green" d="M142 77h65l19 25-19 25h-65V77Z" /><circle cx="207" cy="102" r="4" /><path d="M153 94h27M153 108h36" /></g>
        <path class="tag-index-star" d="M205 21q2 13 15 15-13 2-15 15-2-13-15-15 13-2 15-15ZM28 67q1 8 9 9-8 1-9 9-1-8-9-9 8-1 9-9Z" />
      </svg>
    </header>
    <section class="tag-index-body" aria-label="浏览标签">
      <div v-if="tags.length" class="tag-index-controls">
        <div class="tag-index-search-field"><label for="tag-index-search">找一个标签</label><div class="tag-index-search">
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" /><path d="m15.5 15.5 5 5" /></svg>
          <input id="tag-index-search" v-model="query" type="search" autocomplete="off" placeholder="输入标签名称…" />
          <button v-if="query" type="button" aria-label="清空标签搜索" @click="query = ''">清空</button>
        </div></div>
        <div class="tag-index-sort-field"><label for="tag-index-sort">排列方式</label><select id="tag-index-sort" v-model="sortBy"><option value="count">按文章篇数</option><option value="name">按标签名称</option></select></div>
      </div>
      <p v-if="tags.length" class="tag-index-result" role="status" aria-live="polite">{{ query.trim() ? `找到 ${visibleTags.length} 个标签` : `${visibleTags.length} 个标签，都在这里` }}</p>
      <div v-if="error" class="tag-index-empty" role="alert"><h2>标签暂时没能加载</h2><p>稍后刷新页面，再来翻翻这些笔记。</p><NuxtLink to="/blog">回到博客 <span aria-hidden="true">→</span></NuxtLink></div>
      <div v-else-if="status === 'pending'" class="tag-index-empty" role="status"><p>正在整理标签…</p></div>
      <div v-else-if="visibleTags.length" class="tag-index-grid">
        <NuxtLink v-for="(entry, index) in visibleTags" :key="entry.slug" :to="tagPath(entry.tag)" class="tag-index-note" :data-tone="archiveTone(entry.slug)">
          <div class="tag-index-note-top"><span>{{ String(index + 1).padStart(2, '0') }} / TOPIC</span><span class="tag-index-note-count">{{ entry.count }} 篇</span></div>
          <h2><span class="tag-index-hash" aria-hidden="true">#</span>{{ entry.tag }}</h2>
          <div class="tag-index-latest"><span>最近写下的</span><p>{{ entry.latestPost.title }}</p></div>
          <span class="tag-index-note-link">顺着标签读下去 <span aria-hidden="true">↗</span></span>
        </NuxtLink>
      </div>
      <div v-else-if="tags.length" class="tag-index-empty"><h2>暂时没有找到这个标签</h2><p>换个关键词，或回到完整的标签索引。</p><button type="button" @click="query = ''">清空搜索 <span aria-hidden="true">→</span></button></div>
      <div v-else class="tag-index-empty"><h2>标签还在慢慢生长</h2><p>这里暂时没有公开文章的标签。</p><NuxtLink to="/posts">去文章库看看 <span aria-hidden="true">→</span></NuxtLink></div>
    </section>
    <footer class="tag-index-footer"><p>每一个标签，都是继续探索的线索。</p><NuxtLink to="/posts">所有文章 <span aria-hidden="true">→</span></NuxtLink></footer>
  </div>
</template>

<script setup lang="ts">
import { archiveTone, collectArchiveTags, tagPath } from '~/utils/archive'
const { data, error, status } = await usePublicArchive()
const posts = computed(() => data.value ?? [])
const tags = computed(() => collectArchiveTags(posts.value))
const query = ref('')
const sortBy = ref<'count' | 'name'>('count')
const visibleTags = computed(() => {
  const search = query.value.trim().toLocaleLowerCase('zh-CN')
  const result = tags.value.filter(entry => entry.tag.toLocaleLowerCase('zh-CN').includes(search))
  return result.sort((a, b) => sortBy.value === 'name' ? a.tag.localeCompare(b.tag, 'zh-CN') : b.count - a.count || a.tag.localeCompare(b.tag, 'zh-CN'))
})
useHead({ title: '标签索引', meta: [{ name: 'description', content: '顺着标签浏览古恩豪的技术笔记与学习记录。' }] })
</script>

<style scoped>
.tag-index-journal :where(h1, h2, p) { margin: 0; }
.tag-index-journal a { text-decoration: none; }
.tag-index-intro { display: grid; grid-template-columns: minmax(0, 1fr) 250px; gap: 40px; align-items: center; padding: 20px 12px 42px; }
.tag-index-copy { min-width: 0; }
.tag-index-kicker { color: var(--muted); font-size: .64rem; letter-spacing: .14em; }
.tag-index-intro h1 { margin-top: 14px; color: var(--ink); font-size: clamp(2rem, 4vw, 2.7rem); line-height: 1.5; font-weight: 500; letter-spacing: -.05em; }
.tag-index-intro h1 > span { color: var(--accent-2); margin-left: 4px; }
.tag-index-signature { margin-top: 2px; color: var(--accent); font: italic 1.25rem/1.6 Georgia, serif; }
.tag-index-description { margin-top: 16px; color: var(--text); font-size: .88rem; line-height: 1.9; }
.tag-index-count { margin-top: 7px; color: var(--muted); font-size: .75rem; line-height: 1.8; }
.tag-index-doodle { display: block; width: 100%; height: auto; stroke: var(--accent-2); stroke-width: 1.3; stroke-linecap: round; stroke-linejoin: round; pointer-events: none; }
.tag-index-thread { stroke-dasharray: 3 5; }
.tag-index-purple { fill: var(--topic-purple); }
.tag-index-green { fill: var(--topic-green); }
.tag-index-star { fill: var(--topic-orange); }
.tag-index-controls { display: flex; align-items: end; gap: 20px; padding: 22px 24px; background: var(--bg-2); border-radius: 20px; }
.tag-index-controls label { display: block; margin-bottom: 8px; color: var(--muted); font-size: .72rem; }
.tag-index-search-field { flex: 1; min-width: 0; }
.tag-index-search { display: flex; align-items: center; gap: 10px; min-height: 44px; padding: 0 14px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
.tag-index-search svg { width: 18px; height: 18px; flex-shrink: 0; stroke: var(--muted); stroke-width: 1.5; stroke-linecap: round; }
.tag-index-search input { flex: 1; min-width: 0; width: 100%; padding: 10px 0; border: 0; background: transparent; color: var(--ink); font: inherit; font-size: .82rem; }
.tag-index-search input::placeholder { color: var(--muted); opacity: 1; }
.tag-index-search input::-webkit-search-cancel-button { display: none; }
.tag-index-search:focus-within { outline: 2px solid var(--accent); outline-offset: 3px; }
.tag-index-search input:focus-visible { outline: 0; }
.tag-index-search button { border: 0; background: transparent; color: var(--accent); padding: 10px 2px; font: inherit; font-size: .72rem; cursor: pointer; flex-shrink: 0; }
.tag-index-sort-field { flex: 0 0 156px; }
.tag-index-sort-field select { min-height: 44px; width: 100%; padding: 8px 10px; color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 12px; font: inherit; font-size: .78rem; }
.tag-index-result { margin: 22px 4px 18px; color: var(--muted); font-size: .75rem; line-height: 1.8; }
.tag-index-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; }
.tag-index-note { display: flex; flex-direction: column; min-width: 0; padding: 25px 24px 22px; background: var(--topic-purple); border: 1px solid transparent; border-radius: 17px 23px 20px 23px; color: var(--ink); transition: transform 200ms ease, border-color 200ms ease; }
.tag-index-note[data-tone="green"] { background: var(--topic-green); }
.tag-index-note[data-tone="blue"] { background: var(--topic-blue); }
.tag-index-note[data-tone="orange"] { background: var(--topic-orange); }
.tag-index-note-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; color: var(--text); font: .62rem/1.6 var(--font-mono); letter-spacing: .04em; }
.tag-index-note-count { padding: 3px 8px; background: color-mix(in srgb, var(--surface) 62%, transparent); border-radius: 7px; flex-shrink: 0; font-family: var(--font-sans); letter-spacing: 0; }
.tag-index-note h2 { margin-top: 18px; color: var(--ink); font-size: 1.25rem; font-weight: 500; line-height: 1.6; overflow-wrap: anywhere; }
.tag-index-hash { margin-right: 4px; color: var(--accent); font: italic 1.2em Georgia, serif; }
.tag-index-latest { margin-top: 16px; margin-bottom: 22px; flex: 1; }
.tag-index-latest > span { color: var(--muted); font-size: .65rem; }
.tag-index-latest p { margin-top: 5px; color: var(--text); font-size: .85rem; line-height: 1.85; overflow-wrap: anywhere; }
.tag-index-note-link { display: flex; align-items: center; justify-content: space-between; gap: 12px; color: var(--ink); font-size: .72rem; }
.tag-index-note-link > span { transition: transform 200ms ease; }
.tag-index-empty { padding: 52px 24px; border: 1px dashed var(--line-strong); border-radius: 20px; text-align: center; color: var(--text); }
.tag-index-empty h2 { font-size: 1.2rem; font-weight: 500; color: var(--ink); }
.tag-index-empty p { margin-top: 10px; font-size: .86rem; line-height: 1.9; }
.tag-index-empty a, .tag-index-empty button { display: inline-block; margin-top: 18px; padding: 8px 0; border: 0; background: transparent; color: var(--accent); font: inherit; font-size: .82rem; cursor: pointer; }
.tag-index-footer { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-top: 36px; padding: 22px 4px 0; border-top: 1px solid var(--line); color: var(--muted); font-size: .75rem; line-height: 1.8; }
.tag-index-footer a { color: var(--accent); flex-shrink: 0; padding-block: 6px; }
@media (hover: hover) and (pointer: fine) { .tag-index-note:hover { transform: translateY(-3px); border-color: color-mix(in srgb, var(--accent) 26%, transparent); } .tag-index-note:hover .tag-index-note-link > span { transform: translate(2px, -2px); } }
@media (max-width: 700px) { .tag-index-intro { grid-template-columns: minmax(0, 1fr) 180px; gap: 20px; padding-inline: 4px; } .tag-index-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; } .tag-index-controls { padding: 20px; gap: 14px; } .tag-index-sort-field { flex-basis: 140px; } }
@media (max-width: 480px) { .tag-index-intro { grid-template-columns: minmax(0, 1fr); padding-top: 4px; gap: 12px; padding-bottom: 28px; } .tag-index-doodle { width: 160px; justify-self: end; } .tag-index-description { max-width: 270px; } .tag-index-controls { flex-wrap: wrap; padding: 18px; } .tag-index-search-field { flex-basis: 100%; } .tag-index-sort-field { flex-basis: 100%; } .tag-index-grid { grid-template-columns: minmax(0, 1fr); } .tag-index-note { padding: 24px; } .tag-index-note h2 { margin-top: 14px; } .tag-index-latest { margin-bottom: 18px; } .tag-index-footer { flex-wrap: wrap; gap: 8px; } }
@media (max-width: 480px) {
  .tag-index-intro { position: relative; }
  .tag-index-intro h1 { padding-right: 85px; }
  .tag-index-doodle { position: absolute; top: 22px; right: 0; width: 90px; opacity: .75; }
  .tag-index-search input, .tag-index-sort-field select { font-size: 1rem; }
}
@media (prefers-reduced-motion: reduce) { .tag-index-note, .tag-index-note-link > span { transition: none; } .tag-index-note:hover { transform: none; } .tag-index-note:hover .tag-index-note-link > span { transform: none; } }
</style>
