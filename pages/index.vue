<template>
  <div>
    <!-- Hero -->
    <section class="hero">
      <!-- Status -->
      <div class="hero-status">
        <span class="hero-status-dot" aria-hidden="true" />
        <span>{{ appConfig.status }}</span>
      </div>

      <!-- Name -->
      <h1>
        <span class="hero-name-cn">{{ appConfig.authorCN }}</span>
        <span class="hero-cursor" aria-hidden="true">_</span>
        <span class="hero-name-en">{{ appConfig.authorEN }}</span>
      </h1>

      <!-- Bio -->
      <p class="hero-bio">{{ appConfig.bio }}</p>

      <!-- Pills -->
      <div class="hero-pills">
        <NuxtLink to="/posts" class="pill">
          <span class="pill-count">{{ visiblePosts.length }}</span>
          篇文章
        </NuxtLink>
        <a :href="appConfig.github" target="_blank" rel="noopener noreferrer" class="pill">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
          </svg>
          GitHub
        </a>
        <a href="/rss.xml" class="pill">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M6.18 15.64a2.18 2.18 0 010 4.36 2.18 2.18 0 010-4.36M4 4.44A15.56 15.56 0 0119.56 20h-2.83A12.73 12.73 0 004 7.27V4.44m0 5.66a9.9 9.9 0 019.9 9.9h-2.83A7.07 7.07 0 004 12.93V10.1z"/>
          </svg>
          RSS
        </a>
        <NuxtLink to="/about" class="pill">关于我</NuxtLink>
      </div>
    </section>

    <!-- Content sections -->
    <div class="page-wrapper" style="padding-top: 0">

      <!-- Latest post (featured) -->
      <template v-if="featured">
        <div class="section-label">
          <span class="section-label-accent">LATEST</span>
          <span>/</span>
          <span>POST</span>
        </div>
        <NuxtLink :to="featured.path" class="post-card" style="margin-bottom: 2.5rem; display: block">
          <div class="post-card-label">最新文章</div>
          <h2 class="post-card-title">{{ featured.title }}</h2>
          <p v-if="featured.description" class="post-card-desc">{{ featured.description }}</p>
          <div class="post-card-meta">
            <span>{{ formatDate(featured.date) }}</span>
            <span>·</span>
            <span>{{ readingTime(featured.body ?? '') }} min read</span>
          </div>
          <div v-if="featured.tags?.length" class="post-card-tags">
            <span v-for="tag in featured.tags.slice(0, 3)" :key="tag" class="tag-chip">{{ tag }}</span>
          </div>
        </NuxtLink>
      </template>

      <!-- Showcase grid (posts 2-4) -->
      <template v-if="showcase.length">
        <div class="section-label">
          <span class="section-label-accent">RECENT</span>
          <span>/</span>
          <span>POSTS</span>
        </div>
        <div class="showcase-grid" style="margin-bottom: 2.5rem">
          <NuxtLink
            v-for="post in showcase"
            :key="post.path"
            :to="post.path"
            class="showcase-card"
          >
            <h3 class="showcase-card-title">{{ post.title }}</h3>
            <div class="showcase-card-meta">{{ formatDate(post.date) }}</div>
          </NuxtLink>
        </div>
      </template>

      <!-- More posts list (posts 5-8) -->
      <template v-if="morePosts.length">
        <div class="section-label">
          <span class="section-label-accent">MORE</span>
          <span>/</span>
          <span>POSTS</span>
        </div>
        <div style="margin-bottom: 2.5rem">
          <NuxtLink
            v-for="post in morePosts"
            :key="post.path"
            :to="post.path"
            class="post-row"
          >
            <span class="post-row-date">{{ formatMonthDay(post.date) }}</span>
            <span class="post-row-title">{{ post.title }}</span>
            <span class="post-row-tag">{{ post.tags?.[0] ?? '' }}</span>
          </NuxtLink>
        </div>
        <div style="text-align: center; margin-top: 1rem">
          <NuxtLink to="/posts" class="pill" style="display: inline-flex">
            查看全部文章 →
          </NuxtLink>
        </div>
      </template>

      <!-- Top tags -->
      <template v-if="topTags.length">
        <div class="section-label" style="margin-top: 3rem">
          <span class="section-label-accent">TOP</span>
          <span>/</span>
          <span>TAGS</span>
        </div>
        <div class="tags-grid">
          <NuxtLink
            v-for="[tag, count] in topTags"
            :key="tag"
            :to="`/tags/${tagSlug(tag)}`"
            class="tag-item"
          >
            {{ tag }}
            <span class="tag-item-count">{{ count }}</span>
          </NuxtLink>
        </div>
      </template>

    </div>
  </div>
</template>

<script setup lang="ts">
import { readingTime, formatDate, formatMonthDay, tagSlug, isVisible } from '~/utils/blog'

const appConfig = useAppConfig()

// Fetch all posts
const { data: allPosts } = await useAsyncData('index-posts', () =>
  queryCollection('posts').order('date', 'DESC').all()
)

const visiblePosts = computed(() =>
  (allPosts.value ?? []).filter(isVisible)
)

const featured = computed(() => visiblePosts.value[0] ?? null)
const showcase = computed(() => visiblePosts.value.slice(1, 4))
const morePosts = computed(() => visiblePosts.value.slice(4, 8))

// Compute top 8 tags by frequency
const topTags = computed(() => {
  const counts = new Map<string, number>()
  for (const post of visiblePosts.value) {
    for (const tag of post.tags ?? []) {
      counts.set(tag, (counts.get(tag) ?? 0) + 1)
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
})

useHead({
  title: appConfig.title,
  meta: [
    { name: 'description', content: appConfig.description },
    { property: 'og:title', content: appConfig.title },
    { property: 'og:description', content: appConfig.description },
    { property: 'og:url', content: appConfig.url },
  ],
})
</script>
