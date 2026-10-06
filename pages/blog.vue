<template>
  <div class="journal-home">
    <NotebookMotion />
    <div class="notebook-hero-wrap">
      <section class="notebook-hero" aria-labelledby="journal-title">
        <div class="notebook-intro">
          <p class="notebook-greeting">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
            </svg>
            你好，我是{{ appConfig.authorCN }}
          </p>
          <h1 id="journal-title" class="notebook-title">认真折腾，<br /><span class="notebook-highlight">随手记录<svg viewBox="0 0 100 20" preserveAspectRatio="none" fill="none" aria-hidden="true"><path d="M5 15C20 12 50 12 95 18" stroke="currentColor" stroke-width="6" stroke-linecap="round" /></svg></span>。</h1>
          <p class="notebook-summary">这里是{{ appConfig.authorCN }}的个人笔记。记录 LLM、Agent 与工程实践，也留下一些比赛、实习和旅行的片段。</p>
          <NotebookTypewriter />
          <div class="notebook-actions">
            <NuxtLink to="#home-content" class="notebook-primary">翻翻我的笔记 <span aria-hidden="true">↗</span></NuxtLink>
            <NuxtLink to="/about" class="notebook-secondary">关于我 <span aria-hidden="true">→</span></NuxtLink>
          </div>
          <p class="notebook-signoff" aria-hidden="true">a little curiosity, a little progress.</p>
        </div>
        <NotebookCode :name="appConfig.authorCN" :notebook="appConfig.title" />
      </section>
    </div>

    <div id="home-content" class="site-main journal-content">
      <SiteBlock
        v-if="featured"
        eyebrow="01 / Fresh notes"
        title="最近写下的"
        description="新的实验、刚读懂的源码，以及值得记下的细节。"
        action-to="/posts"
        action-label="查看全部"
      >
        <div class="home-latest">
          <ArticleCard :post="featured" large />
          <ArticleStream :posts="recentPosts" />
        </div>
      </SiteBlock>
      <section class="site-block notebook-topics" aria-labelledby="notebook-topics-title">
        <div class="notebook-topic-head">
          <span class="notebook-topic-symbol" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2-5.6-3-5.6 3 1.1-6.2L3 9.6l6.2-.9L12 3Z" /></svg>
          </span>
          <div>
            <p class="notebook-topic-kicker" aria-hidden="true">a few things I’m curious about</p>
            <h2 id="notebook-topics-title">好奇心，落在这些地方</h2>
          </div>
        </div>
        <KnowledgeMap :items="knowledgeMap" />
      </section>
      <SiteBlock
        v-if="selectedPosts.length"
        eyebrow="Selected / Start here"
        title="值得从这里读起"
        description="几篇整理得比较完整的笔记。"
      >
        <div class="card-grid">
          <ArticleCard v-for="post in selectedPosts" :key="post.path" :post="post" />
        </div>
      </SiteBlock>
      <SiteBlock
        v-if="topicCounts.length"
        eyebrow="Index / Topics"
        title="笔记的关键词"
        description="顺着一个关键词，继续往下读。"
        action-to="/tags"
        action-label="完整标签云"
      >
        <div class="topic-cloud">
          <TopicChip v-for="[tag, count] in topicCounts.slice(0, 16)" :key="tag" :tag="tag" :count="count" />
        </div>
      </SiteBlock>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { PostMeta } from '~/server/api/posts.get'
import NotebookMotion from '~/components/NotebookMotion.vue'
import NotebookTypewriter from '~/components/NotebookTypewriter.vue'

const appConfig = useAppConfig()

const { data } = await useAsyncData<PostMeta[]>('home-posts', () =>
  $fetch('/api/posts')
)

const posts = computed(() => data.value ?? [])
const featured = computed(() => posts.value[0] ?? null)
const selectedSlugs = [
  'ai-infra-roadmap',
  'mini-llm-engine-from-scratch',
  'multimodal-rag-from-scratch',
]
const selectedPosts = computed(() =>
  selectedSlugs
    .map(slug => posts.value.find(post => post.slug === slug))
    .filter((post): post is PostMeta => Boolean(post))
)
const recentPosts = computed(() => {
  const excluded = new Set([featured.value?.slug, ...selectedSlugs])
  return posts.value.filter(post => !excluded.has(post.slug)).slice(0, 6)
})
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
    eyebrow: '和智能一起折腾',
    title: 'AI Infra / Agent',
    desc: '从模型基础到 RAG、上下文工程、推理服务和 Agent 平台。',
    label: '看 AI Infra',
    to: '/tags/ai-infra',
  },
  {
    key: 'SYS',
    eyebrow: '把系统慢慢搭好',
    title: '后端与分布式',
    desc: '数据库、缓存、高并发、分布式系统与 Go 后端项目。',
    label: '看后端系统',
    to: '/tags/分布式系统',
  },
  {
    key: 'SRC',
    eyebrow: '沿着代码找答案',
    title: '源码阅读',
    desc: '从入口、数据流和关键抽象读懂开源项目。',
    label: '看源码分析',
    to: '/tags/源码分析',
  },
  {
    key: 'INT',
    eyebrow: '一路走，一路积累',
    title: '面试与实践',
    desc: '按岗位组织的准备清单、项目表达和高频追问。',
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
    { property: 'og:url', content: `${appConfig.url}/blog` },
  ],
})
</script>
