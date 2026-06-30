<template>
  <div class="about-page">
    <!-- Hero -->
    <div class="about-hero">
      <div class="about-role">
        <span class="hero-status-dot" aria-hidden="true" />
        {{ appConfig.role }}
      </div>
      <h1 class="about-name">{{ appConfig.authorCN }}</h1>
      <p style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-very-muted); letter-spacing: 0.04em; margin-bottom: 0.25rem">
        {{ appConfig.authorEN }}
      </p>
      <p class="about-bio">{{ appConfig.bio }}</p>
      <div class="hero-pills" style="margin-top: 1.5rem">
        <a :href="appConfig.github" target="_blank" rel="noopener noreferrer" class="pill">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
          </svg>
          GitHub
        </a>
        <a href="/rss.xml" class="pill">RSS</a>
      </div>
    </div>

    <!-- Tech Stack -->
    <div class="about-section">
      <div class="about-section-title">技术栈</div>
      <div class="tech-grid">
        <span v-for="tech in appConfig.techStack" :key="tech.name" class="tech-item">
          {{ tech.name }}
        </span>
      </div>
    </div>

    <!-- Experience -->
    <div class="about-section">
      <div class="about-section-title">工作经历</div>
      <div class="timeline">
        <div v-for="exp in appConfig.experience" :key="exp.company" class="timeline-item">
          <div class="timeline-logo">
            <img v-if="exp.logo" :src="exp.logo" :alt="exp.company" class="timeline-logo-img" />
            <span v-else class="timeline-logo-fallback">{{ exp.company[0] }}</span>
          </div>
          <div class="timeline-period">{{ exp.period }}</div>
          <div>
            <div class="timeline-content-title">
              {{ exp.company }}
              <span v-if="exp.companyEN" class="timeline-company-en">{{ exp.companyEN }}</span>
            </div>
            <div class="timeline-content-sub">{{ exp.role }}</div>
            <div v-if="exp.desc" class="timeline-content-desc">{{ exp.desc }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent Posts -->
    <div class="about-section">
      <div class="about-section-title">近期文章</div>
      <NuxtLink
        v-for="post in recentPosts"
        :key="post.path"
        :to="post.path"
        class="post-row"
      >
        <span class="post-row-date">{{ formatMonthDay(post.date) }}</span>
        <span class="post-row-title">{{ post.title }}</span>
        <span class="post-row-tag">{{ post.tags?.[0] ?? '' }}</span>
      </NuxtLink>
    </div>
  </div>
</template>

<script setup lang="ts">
import { formatMonthDay } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

const appConfig = useAppConfig()

const { data: allPosts } = await useAsyncData<PostMeta[]>('about-posts', () =>
  $fetch('/api/posts')
)

const recentPosts = computed(() => (allPosts.value ?? []).slice(0, 5))

useHead({
  title: '关于',
  meta: [
    { name: 'description', content: `关于 ${appConfig.authorCN} — ${appConfig.role}` },
  ],
})
</script>
