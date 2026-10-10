<template>
  <div class="archive-entries">
    <section v-for="[year, yearPosts] in groupedPosts" :key="year" class="archive-year" :aria-label="`${year} 年的文章`">
      <h2 class="archive-year-heading">
        <span>{{ year }}</span>
        <small>{{ yearPosts.length }} 篇手记</small>
        <svg viewBox="0 0 72 16" fill="none" aria-hidden="true"><path d="M2 10q21-10 43-2t25-3" /></svg>
      </h2>
      <div class="archive-entry-list">
        <article v-for="post in yearPosts" :key="post.path" class="archive-entry" :class="{ 'archive-entry-illustrated': post.cover }" :data-tone="archiveTone(post.series || post.tags[0] || post.slug)">
          <div class="archive-entry-date">
            <time :datetime="formatDate(post.date) || undefined">{{ formatMonthDay(post.date).replace('-', '.') || '—' }}</time>
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 6q4-2 7 0v14q-3-2-7 0V6Zm7 0q3-2 7 0v14q-4-2-7 0M8 10h1m6 0h1M8 13h1m6 0h1" /></svg>
          </div>
          <div class="archive-entry-copy">
            <div class="archive-entry-meta">
              <span v-if="post.series" class="archive-entry-series">{{ post.series }}<span v-if="typeof post.seriesOrder === 'number'"> · {{ post.seriesOrder }}</span></span>
              <span class="archive-entry-time">
                <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><circle cx="10" cy="10" r="7" /><path d="M10 6v4l3 2" /></svg>
                约 {{ post.readingTime }} 分钟
              </span>
            </div>
            <h3>
              <NuxtLink :to="post.path" class="archive-entry-title">
                <span>{{ post.title }}</span>
                <span class="archive-entry-arrow" aria-hidden="true">↗</span>
              </NuxtLink>
            </h3>
            <p v-if="post.description || post.excerpt" class="archive-entry-summary">{{ post.description || post.excerpt }}</p>
            <nav v-if="post.tags.length" class="archive-entry-tags" :aria-label="`${post.title} 的标签`">
              <NuxtLink v-for="entry in collectArchiveTags([post])" :key="entry.slug" :to="tagPath(entry.tag)">#{{ entry.tag }}</NuxtLink>
            </nav>
          </div>
          <NuxtLink v-if="post.cover" :to="post.path" class="archive-entry-cover" :aria-label="`阅读：${post.title}`">
            <SiteImage :src="post.cover" :alt="post.coverAlt || ''" sizes="(max-width: 700px) calc(100vw - 80px), 192px" />
          </NuxtLink>
        </article>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { archiveTone, collectArchiveTags, groupArchiveByYear, tagPath } from '~/utils/archive'
import { formatDate, formatMonthDay } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

const props = defineProps<{ posts: PostMeta[] }>()
const groupedPosts = computed(() => groupArchiveByYear(props.posts))
</script>

<style scoped>
.archive-entries :where(h2, h3, p) { margin: 0; }
.archive-year + .archive-year { margin-top: 38px; }
.archive-year-heading { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; color: var(--ink); font-weight: 500; font-size: 1.25rem; }
.archive-year-heading > span { font-family: Georgia, serif; font-style: italic; font-size: 1.55rem; }
.archive-year-heading small { font-size: 0.75rem; font-weight: 400; color: var(--muted); }
.archive-year-heading svg { width: 64px; height: 16px; margin-left: 3px; stroke: var(--accent-2); stroke-width: 1.5; stroke-linecap: round; opacity: 0.5; }
.archive-entry-list { display: grid; gap: 18px; }
.archive-entry { --entry-paper: var(--topic-purple); display: grid; grid-template-columns: 70px minmax(0, 1fr); gap: 24px; padding: 24px 26px; border: 1px solid transparent; border-radius: 22px; background: transparent; }
.archive-entry[data-tone="blue"] { --entry-paper: var(--topic-blue); }
.archive-entry[data-tone="green"] { --entry-paper: var(--topic-green); }
.archive-entry[data-tone="orange"] { --entry-paper: var(--topic-orange); }
.archive-entry-date { display: flex; flex-direction: column; align-items: flex-start; gap: 20px; padding-top: 4px; color: var(--muted); }
.archive-entry-date time { font-family: var(--font-mono); font-size: 0.8rem; font-weight: 400; }
.archive-entry-date svg { width: 26px; height: 26px; stroke: var(--accent-2); stroke-width: 1.2; stroke-linecap: round; stroke-linejoin: round; opacity: 0.6; transform: rotate(-8deg); }
.archive-entry-copy { min-width: 0; }
.archive-entry-illustrated { grid-template-columns: 70px minmax(0, 1fr) 192px; align-items: center; }
.archive-entry-cover { display: block; overflow: hidden; aspect-ratio: 16 / 10; border: 1px solid var(--line); border-radius: 14px; background: var(--surface); }
.archive-entry-cover img { display: block; width: 100%; height: 100%; object-fit: cover; }
.archive-entry-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin-bottom: 12px; font-size: 0.7rem; color: var(--muted); }
.archive-entry-series { padding: 3px 10px; border-radius: 6px; background: var(--entry-paper); color: var(--text); }
.archive-entry-time { display: inline-flex; align-items: center; gap: 5px; }
.archive-entry-time svg { width: 13px; height: 13px; stroke: currentColor; stroke-width: 1.2; stroke-linecap: round; stroke-linejoin: round; }
.archive-entry-title { display: flex; align-items: flex-start; gap: 16px; color: var(--ink); font-size: 1.22rem; font-weight: 500; line-height: 1.65; letter-spacing: -0.025em; text-decoration: none; overflow-wrap: anywhere; }
.archive-entry-title > span:first-child { flex: 1; min-width: 0; }
.archive-entry-arrow { flex: 0 0 auto; font-size: 1.3rem; color: var(--accent); transition: transform 180ms ease; }
.archive-entry-title:hover { color: var(--accent); }
.archive-entry-summary { margin-top: 10px; font-size: 0.875rem; line-height: 1.95; color: var(--muted); overflow-wrap: anywhere; text-wrap: pretty; }
.archive-entry-tags { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 16px; }
.archive-entry-tags a { display: inline-block; padding-block: 3px; color: var(--muted); font-size: 0.72rem; text-decoration: none; overflow-wrap: anywhere; }
.archive-entry-tags a:hover { color: var(--accent); text-decoration: underline; text-underline-offset: 4px; }
.archive-entry:focus-within { border-color: var(--line-strong); background: var(--surface); }
@media (hover: hover) and (pointer: fine) {
  .archive-entry { transition: transform 180ms ease, box-shadow 180ms ease; }
  .archive-entry:hover { transform: translateY(-2px); border-color: var(--line); background: var(--surface); box-shadow: 0 8px 24px #33264004; }
  .archive-entry-title:hover .archive-entry-arrow { transform: translate(2px, -2px); }
}
@media (max-width: 700px) {
  .archive-entry { grid-template-columns: 52px minmax(0, 1fr); gap: 16px; padding: 24px 22px; }
  .archive-entry-illustrated { grid-template-columns: 52px minmax(0, 1fr); }
  .archive-entry-illustrated .archive-entry-date { grid-column: 1; grid-row: 1; }
  .archive-entry-illustrated .archive-entry-cover { grid-column: 2; grid-row: 1; aspect-ratio: 16 / 9; }
  .archive-entry-illustrated .archive-entry-copy { grid-column: 2; grid-row: 2; }
  .archive-entry-title { font-size: 1.1rem; }
}
@media (max-width: 480px) {
  .archive-entry { grid-template-columns: 1fr; gap: 11px; padding: 22px 20px; border-radius: 18px 18px 18px 6px; }
  .archive-entry-illustrated .archive-entry-date,
  .archive-entry-illustrated .archive-entry-cover,
  .archive-entry-illustrated .archive-entry-copy { grid-column: 1; grid-row: auto; }
  .archive-entry-illustrated .archive-entry-cover { order: 1; margin-block: 4px 8px; }
  .archive-entry-illustrated .archive-entry-copy { order: 2; }
  .archive-entry-date { flex-direction: row; justify-content: space-between; align-items: center; gap: 12px; padding-top: 0; }
  .archive-entry-date time { font-size: 0.7rem; }
  .archive-entry-date svg { width: 20px; height: 20px; }
  .archive-entry-meta { gap: 8px 12px; margin-bottom: 10px; }
  .archive-entry-title { font-size: 1.08rem; gap: 10px; }
  .archive-entry-summary { font-size: 0.85rem; }
  .archive-entry-tags { gap: 5px 13px; margin-top: 12px; }
}
@media (prefers-reduced-motion: reduce) {
  .archive-entry, .archive-entry-arrow { transition: none; }
  .archive-entry:hover, .archive-entry-title:hover .archive-entry-arrow { transform: none; }
}
</style>
