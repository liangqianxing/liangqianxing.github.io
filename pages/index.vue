<template>
  <div>
    <section class="hero-frame">
      <div class="hero-main">
        <p class="eyebrow">{{ appConfig.status }}</p>
        <h1>
          {{ appConfig.authorCN }}
          <span>工程笔记库</span>
        </h1>
        <p class="hero-summary">{{ appConfig.description }}</p>
        <div class="hero-actions">
          <NuxtLink to="/posts" class="primary-action">进入文章库</NuxtLink>
          <NuxtLink to="/tags" class="secondary-action">按主题浏览</NuxtLink>
        </div>
      </div>

      <aside class="hero-dossier" aria-label="站点概览">
        <img src="/avatar.jpg" :alt="appConfig.authorCN" width="84" height="84" />
        <div>
          <strong>{{ appConfig.authorEN }}</strong>
          <span>{{ appConfig.role }}</span>
        </div>
        <dl>
          <div>
            <dt>Posts</dt>
            <dd>{{ posts.length }}</dd>
          </div>
          <div>
            <dt>Topics</dt>
            <dd>{{ topicCounts.length }}</dd>
          </div>
          <div>
            <dt>Latest</dt>
            <dd>{{ latestYear }}</dd>
          </div>
        </dl>
      </aside>
    </section>

    <main class="site-main">
      <SiteBlock
        eyebrow="Map"
        title="知识地图"
        description="把文章按学习路径组织起来，而不是只按发布时间堆叠。"
      >
        <KnowledgeMap :items="knowledgeMap" />
      </SiteBlock>

      <SiteBlock
        v-if="featured"
        eyebrow="Latest"
        title="最近更新"
        description="最新文章放在首页核心位置，其他文章进入连续阅读流。"
        action-to="/posts"
        action-label="查看全部"
      >
        <div class="home-latest">
          <ArticleCard :post="featured" large />
          <ArticleStream :posts="recentPosts" />
        </div>
      </SiteBlock>

      <SiteBlock
        v-if="selectedPosts.length"
        eyebrow="Selected"
        title="精选入口"
        description="适合作为新读者进入站点的三篇文章。"
      >
        <div class="card-grid">
          <ArticleCard v-for="post in selectedPosts" :key="post.path" :post="post" />
        </div>
      </SiteBlock>

      <SiteBlock
        v-if="topicCounts.length"
        eyebrow="Topics"
        title="热门主题"
        description="从主题进入，比从时间线翻找更快。"
        action-to="/tags"
        action-label="完整标签云"
      >
        <div class="topic-cloud">
          <TopicChip v-for="[tag, count] in topicCounts.slice(0, 16)" :key="tag" :tag="tag" :count="count" />
        </div>
      </SiteBlock>
    </main>
  </div>
</template>

<script setup lang="ts">
import type { PostMeta } from '~/server/api/posts.get'

const appConfig = useAppConfig()

const { data } = await useAsyncData<PostMeta[]>('home-posts', () =>
  $fetch('/api/posts')
)

const posts = computed(() => data.value ?? [])
const featured = computed(() => posts.value[0] ?? null)
const recentPosts = computed(() => posts.value.slice(1, 7))
const selectedPosts = computed(() => posts.value.slice(7, 10))
const latestYear = computed(() => featured.value ? new Date(featured.value.date).getFullYear() : new Date().getFullYear())

const topicCounts = computed(() => {
  const counts = new Map<string, number>()
  for (const post of posts.value) {
    for (const tag of post.tags ?? []) counts.set(tag, (counts.get(tag) ?? 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])
})

const knowledgeMap = [
  {
    key: 'AI',
    title: 'AI Infra / Agent',
    desc: 'RAG、上下文压缩、记忆系统、推理链路和 Agent 工程化。',
    label: '看 AI Infra',
    to: '/tags/ai-infra',
  },
  {
    key: 'SYS',
    title: 'Backend / Distributed',
    desc: '后端框架、数据库、缓存、高并发、分布式系统和系统设计。',
    label: '看后端系统',
    to: '/tags/后端',
  },
  {
    key: 'SRC',
    title: 'Source Reading',
    desc: '源码导读、框架拆解、项目复盘和可落地的工程细节。',
    label: '看源码分析',
    to: '/tags/源码分析',
  },
  {
    key: 'INT',
    title: 'Interview Kit',
    desc: '面试准备、项目包装、CS 基础和岗位方向速查。',
    label: '看面试',
    to: '/tags/面试',
  },
]

useHead({
  title: appConfig.title,
  titleTemplate: () => appConfig.title,
  meta: [
    { name: 'description', content: appConfig.description },
    { property: 'og:title', content: appConfig.title },
    { property: 'og:description', content: appConfig.description },
    { property: 'og:url', content: appConfig.url },
  ],
})
</script>
