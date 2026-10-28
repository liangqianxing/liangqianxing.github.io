<template>
  <div class="archive-entries">
    <section
      v-for="[year, yearPosts] in groupedPosts"
      :key="year"
      class="archive-year"
      :aria-label="`${year} 年的文章`"
    >
      <header class="archive-year-heading">
        <span class="archive-year-num">{{ year }}</span>
        <span class="archive-year-count">{{ yearPosts.length }} 篇深度报告</span>
        <span class="archive-year-rule" aria-hidden="true" />
      </header>

      <div class="archive-entry-grid">
        <article
          v-for="post in yearPosts"
          :key="post.path"
          class="archive-card"
          :class="{ 'archive-card-illustrated': Boolean(post.cover) }"
          :data-tone="archiveTone(post.series || post.tags[0] || post.slug)"
        >
          <NuxtLink
            v-if="post.cover"
            :to="post.path"
            class="archive-card-media"
            :aria-label="`阅读：${post.title}`"
          >
            <SiteImage
              :src="post.cover"
              :alt="post.coverAlt || post.title"
              sizes="(max-width: 720px) calc(100vw - 48px), 296px"
            />
          </NuxtLink>

          <div class="archive-card-body">
            <div class="archive-card-meta">
              <time class="archive-card-date" :datetime="formatDate(post.date) || undefined">
                {{ formatDate(post.date) || '—' }}
              </time>
              <span v-if="post.series" class="archive-card-series">
                {{ post.series }}<template v-if="typeof post.seriesOrder === 'number'"> #{{ post.seriesOrder }}</template>
              </span>
              <span v-else-if="post.categories?.length" class="archive-card-category">
                {{ post.categories[0] }}
              </span>
              <span class="archive-card-reading">约 {{ post.readingTime }} 分钟</span>
            </div>

            <h3 class="archive-card-heading">
              <NuxtLink :to="post.path" class="archive-card-title">
                <span>{{ post.title }}</span>
                <span class="archive-card-arrow" aria-hidden="true">↗</span>
              </NuxtLink>
            </h3>

            <p v-if="post.description || post.excerpt" class="archive-card-summary">
              {{ post.description || post.excerpt }}
            </p>

            <div class="archive-card-footer">
              <nav v-if="post.tags.length" class="archive-card-tags" :aria-label="`${post.title} 的标签`">
                <NuxtLink
                  v-for="entry in collectArchiveTags([post]).slice(0, 5)"
                  :key="entry.slug"
                  :to="tagPath(entry.tag)"
                >
                  #{{ entry.tag }}
                </NuxtLink>
              </nav>
              <NuxtLink :to="post.path" class="archive-card-cta">
                阅读全文 <span aria-hidden="true">→</span>
              </NuxtLink>
            </div>
          </div>
        </article>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { archiveTone, collectArchiveTags, groupArchiveByYear, tagPath } from '~/utils/archive'
import { formatDate } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

const props = defineProps<{ posts: PostMeta[] }>()
const groupedPosts = computed(() => groupArchiveByYear(props.posts))
</script>

<style scoped>
.archive-entries :where(h2, h3, p) {
  margin: 0;
}

.archive-year + .archive-year {
  margin-top: 44px;
}

.archive-year-heading {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 20px;
}

.archive-year-num {
  font-family: var(--font-mono);
  font-size: 1.15rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--ink);
}

.archive-year-count {
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.7rem;
}

.archive-year-rule {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--line-strong), transparent);
}

.archive-entry-grid {
  display: grid;
  gap: 18px;
}

.archive-card {
  --card-accent: var(--topic-purple);
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 24px;
  padding: 24px 26px;
  border: 1px solid var(--line);
  border-radius: 20px;
  background: var(--surface);
  box-shadow: 0 2px 10px color-mix(in srgb, var(--ink) 3%, transparent);
  transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
}

.archive-card[data-tone="blue"] {
  --card-accent: var(--topic-blue);
}

.archive-card[data-tone="green"] {
  --card-accent: var(--topic-green);
}

.archive-card[data-tone="orange"] {
  --card-accent: var(--topic-orange);
}

.archive-card-illustrated {
  grid-template-columns: 280px minmax(0, 1fr);
  align-items: stretch;
}

.archive-card-media {
  position: relative;
  display: block;
  overflow: hidden;
  aspect-ratio: 16 / 9;
  border-radius: 14px;
  border: 1px solid color-mix(in srgb, var(--line-strong) 75%, transparent);
  background: var(--bg-2);
  align-self: center;
}

.archive-card-media :deep(img) {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 280ms cubic-bezier(0.22, 1, 0.36, 1);
}

.archive-card-body {
  min-width: 0;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.archive-card-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 12px;
  margin-bottom: 10px;
  font-size: 0.72rem;
  color: var(--muted);
}

.archive-card-date {
  font-family: var(--font-mono);
  color: var(--muted);
}

.archive-card-series,
.archive-card-category {
  padding: 2px 9px;
  border-radius: 6px;
  background: var(--card-accent);
  border: 1px solid color-mix(in srgb, var(--accent) 18%, transparent);
  color: var(--ink);
  font-weight: 500;
}

.archive-card-reading {
  font-family: var(--font-mono);
  margin-left: auto;
}

.archive-card-heading {
  margin: 0;
}

.archive-card-title {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  color: var(--ink);
  font-size: 1.18rem;
  font-weight: 600;
  line-height: 1.52;
  letter-spacing: -0.02em;
  text-decoration: none;
  overflow-wrap: anywhere;
}

.archive-card-arrow {
  flex-shrink: 0;
  color: var(--accent);
  font-size: 1.1rem;
  transition: transform 180ms ease;
}

.archive-card-summary {
  margin-top: 10px;
  color: var(--text);
  font-size: 0.85rem;
  line-height: 1.82;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-wrap: pretty;
}

.archive-card-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px 16px;
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px dashed color-mix(in srgb, var(--line) 85%, transparent);
}

.archive-card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 10px;
}

.archive-card-tags a {
  padding: 2px 8px;
  border-radius: 6px;
  background: var(--bg-2);
  color: var(--muted);
  font-size: 0.7rem;
  text-decoration: none;
  transition: color 150ms ease, background-color 150ms ease;
}

.archive-card-tags a:hover {
  color: var(--accent);
  background: color-mix(in srgb, var(--accent) 10%, var(--bg-2));
}

.archive-card-cta {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--accent);
  font-size: 0.75rem;
  font-weight: 500;
  text-decoration: none;
  margin-left: auto;
}

@media (hover: hover) and (pointer: fine) {
  .archive-card:hover {
    transform: translateY(-2px);
    border-color: color-mix(in srgb, var(--accent) 38%, var(--line-strong));
    box-shadow: 0 12px 32px color-mix(in srgb, var(--ink) 6%, transparent);
  }
  .archive-card:hover .archive-card-media :deep(img) {
    transform: scale(1.03);
  }
  .archive-card-title:hover {
    color: var(--accent);
  }
  .archive-card-title:hover .archive-card-arrow {
    transform: translate(2px, -2px);
  }
}

@media (max-width: 760px) {
  .archive-card-illustrated {
    grid-template-columns: 1fr;
    gap: 16px;
  }
  .archive-card {
    padding: 20px;
  }
  .archive-card-reading {
    margin-left: 0;
  }
}

@media (prefers-reduced-motion: reduce) {
  .archive-card,
  .archive-card-media :deep(img),
  .archive-card-arrow {
    transition: none;
  }
}
</style>
