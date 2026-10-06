<template>
  <div class="page-frame tag-detail-journal">
    <nav class="tag-detail-back" aria-label="标签页导航"><NuxtLink to="/tags"><span aria-hidden="true">←</span> 标签索引</NuxtLink><NuxtLink to="/posts">所有文章 <span aria-hidden="true">↗</span></NuxtLink></nav>
    <header class="tag-detail-intro" :data-tone="archiveTone(tagParam)">
      <div class="tag-detail-copy"><p class="tag-detail-kicker">FOLLOW THIS THREAD</p><h1><span aria-hidden="true">#</span>{{ displayTag }}</h1><p class="tag-detail-signature">Notes on this topic</p><p class="tag-detail-description">{{ tagPosts.length }} 篇公开笔记，沿着这个标签慢慢读。</p></div>
      <svg class="tag-detail-doodle" viewBox="0 0 190 150" fill="none" aria-hidden="true"><path class="tag-detail-thread" d="M10 132c24 9 38-10 34-25s-31-3-21 14c13 22 40 9 43-15s22-9 31-17" /><g transform="rotate(12 107 60)"><path class="tag-detail-label" d="M51 32h77l24 31-24 31H51V32Z" /><circle cx="130" cy="63" r="5" /><path d="M66 53h39M66 69h26" /></g><path class="tag-detail-star" d="M160 9q2 11 13 13-11 2-13 13-2-11-13-13 11-2 13-13Z" /></svg>
    </header>
    <section v-if="relatedTags.length" class="tag-detail-related" aria-labelledby="tag-related-heading"><h2 id="tag-related-heading">这些笔记也带着</h2><div><NuxtLink v-for="entry in relatedTags" :key="entry.slug" :to="tagPath(entry.tag)" :data-tone="archiveTone(entry.slug)">#{{ entry.tag }} <span>{{ entry.count }} 篇</span></NuxtLink></div></section>
    <div v-if="error" class="tag-detail-empty" role="alert"><h2>笔记暂时没能加载</h2><p>稍后刷新页面，或先看看其他标签。</p><NuxtLink to="/tags">返回标签索引 <span aria-hidden="true">→</span></NuxtLink></div>
    <div v-else-if="status === 'pending'" class="tag-detail-empty" role="status"><p>正在整理这个标签的笔记…</p></div>
    <ArchiveEntries v-else-if="tagPosts.length" :posts="tagPosts" />
    <div v-else class="tag-detail-empty"><h2>这根线头还没有笔记</h2><p>这个标签下暂无公开文章，去其他标签看看吧。</p><NuxtLink to="/tags">返回标签索引 <span aria-hidden="true">→</span></NuxtLink></div>
    <footer class="tag-detail-footer"><p>记录从这里继续。</p><NuxtLink to="/tags">继续找标签 <span aria-hidden="true">→</span></NuxtLink></footer>
  </div>
</template>

<script setup lang="ts">
import { archiveTone, collectArchiveTags, tagPath } from '~/utils/archive'
import { tagSlug } from '~/utils/blog'
const route = useRoute()
// Nuxt already decodes the parameter; a second decode would damage literal percent signs.
const tagParam = computed(() => String(route.params.tag ?? ''))
const { data, error, status } = await usePublicArchive()
const posts = computed(() => data.value ?? [])
const allTags = computed(() => collectArchiveTags(posts.value))
const activeTag = computed(() => allTags.value.find(entry => entry.slug === tagParam.value))
const displayTag = computed(() => activeTag.value?.tag ?? tagParam.value)
const tagPosts = computed(() => posts.value.filter(post => (post.tags ?? []).some(tag => tagSlug(tag) === tagParam.value)))
const relatedTags = computed(() => collectArchiveTags(tagPosts.value).filter(entry => entry.slug !== tagParam.value))
useHead(() => ({ title: `#${displayTag.value}`, meta: [{ name: 'description', content: `标签 ${displayTag.value} 下的技术笔记与学习记录。` }] }))
</script>

<style scoped>
.tag-detail-journal :where(h1, h2, p) { margin: 0; }
.tag-detail-journal a { text-decoration: none; }
.tag-detail-back { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 24px; color: var(--muted); font-size: .78rem; }
.tag-detail-back a { display: inline-flex; align-items: center; gap: 8px; padding-block: 8px; }
.tag-detail-back a:hover { color: var(--accent); }
.tag-detail-intro { display: grid; grid-template-columns: minmax(0, 1fr) 190px; gap: 32px; align-items: center; padding: 30px 36px; margin-bottom: 32px; background: var(--topic-purple); border-radius: 20px 26px 22px 26px; }
.tag-detail-intro[data-tone="green"] { background: var(--topic-green); }
.tag-detail-intro[data-tone="blue"] { background: var(--topic-blue); }
.tag-detail-intro[data-tone="orange"] { background: var(--topic-orange); }
.tag-detail-copy { min-width: 0; }
.tag-detail-kicker { color: var(--text); font-size: .64rem; letter-spacing: .14em; }
.tag-detail-intro h1 { margin-top: 10px; color: var(--ink); font-size: clamp(2rem, 4vw, 2.7rem); line-height: 1.5; font-weight: 500; letter-spacing: -.03em; overflow-wrap: anywhere; }
.tag-detail-intro h1 > span { color: var(--accent); margin-right: 5px; font-family: Georgia, serif; font-style: italic; }
.tag-detail-signature { margin-top: 2px; color: var(--accent); font: italic 1.1rem/1.7 Georgia, serif; }
.tag-detail-description { margin-top: 16px; color: var(--text); font-size: .83rem; line-height: 1.9; }
.tag-detail-doodle { display: block; width: 100%; height: auto; stroke: var(--accent-2); stroke-width: 1.3; stroke-linecap: round; stroke-linejoin: round; pointer-events: none; }
.tag-detail-thread { stroke-dasharray: 3 5; }
.tag-detail-label { fill: color-mix(in srgb, var(--surface) 65%, transparent); }
.tag-detail-star { fill: var(--topic-orange); }
.tag-detail-related { margin-bottom: 34px; }
.tag-detail-related h2 { color: var(--muted); font-size: .75rem; font-weight: 400; margin-bottom: 12px; }
.tag-detail-related > div { display: flex; flex-wrap: wrap; gap: 8px; }
.tag-detail-related a { display: inline-flex; align-items: center; gap: 8px; min-height: 34px; padding: 5px 12px; background: var(--topic-purple); color: var(--ink); border: 1px solid transparent; border-radius: 8px; font-size: .75rem; overflow-wrap: anywhere; max-width: 100%; }
.tag-detail-related a[data-tone="green"] { background: var(--topic-green); }
.tag-detail-related a[data-tone="blue"] { background: var(--topic-blue); }
.tag-detail-related a[data-tone="orange"] { background: var(--topic-orange); }
.tag-detail-related a span { color: var(--muted); font-size: .65rem; flex-shrink: 0; }
.tag-detail-related a:hover { border-color: var(--line-strong); }
.tag-detail-empty { padding: 44px 24px; background: var(--bg-2); border: 1px dashed var(--line-strong); border-radius: 20px; color: var(--text); text-align: center; }
.tag-detail-empty h2 { color: var(--ink); font-size: 1.2rem; font-weight: 500; }
.tag-detail-empty p { margin-top: 10px; font-size: .86rem; line-height: 1.9; }
.tag-detail-empty a { display: inline-block; margin-top: 16px; padding-block: 8px; color: var(--accent); font-size: .8rem; }
.tag-detail-footer { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-top: 32px; padding: 22px 4px 0; border-top: 1px solid var(--line); color: var(--muted); font-size: .75rem; line-height: 1.8; }
.tag-detail-footer a { flex-shrink: 0; color: var(--accent); padding-block: 6px; }
@media (max-width: 700px) { .tag-detail-intro { grid-template-columns: minmax(0, 1fr) 135px; gap: 20px; padding: 26px; } }
@media (max-width: 480px) { .tag-detail-back { gap: 12px; font-size: .74rem; } .tag-detail-intro { grid-template-columns: minmax(0, 1fr); padding: 24px; gap: 8px; } .tag-detail-intro h1 { font-size: 2rem; } .tag-detail-doodle { width: 130px; justify-self: end; } .tag-detail-related { margin-bottom: 26px; } .tag-detail-footer { flex-wrap: wrap; gap: 8px; } }
@media (max-width: 480px) { .tag-detail-doodle { display: none; } }
</style>
