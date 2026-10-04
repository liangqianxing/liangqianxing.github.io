<template>
  <div class="journal-home">
    <section class="journal-hero" aria-labelledby="journal-title">
      <div class="journal-intro">
        <p class="journal-eyebrow">
          <span class="journal-mark" aria-hidden="true" />
          THE PERSONAL NOTEBOOK
        </p>
        <h1 id="journal-title">认真折腾，<br /><span>随手记录。</span></h1>
        <p class="journal-summary">这里是{{ appConfig.authorCN }}的个人笔记。记录 LLM、Agent 与工程实践，也留下一些比赛、实习和旅行的片段。</p>
        <div class="journal-actions">
          <NuxtLink to="/posts" class="journal-primary">
            开始阅读 <span aria-hidden="true">→</span>
          </NuxtLink>
          <NuxtLink to="/about" class="journal-secondary">
            认识一下我 <span aria-hidden="true">↗</span>
          </NuxtLink>
        </div>
        <div class="journal-caption">
          <span v-if="posts.length">{{ posts.length }} 篇笔记 · {{ topicCounts.length }} 个主题</span>
          <span v-else>CODE · RESEARCH · LIFE</span>
          <span class="journal-caption-line" aria-hidden="true" />
          <a href="#home-content">往下看看 <span aria-hidden="true">↓</span></a>
        </div>
      </div>
      <aside class="workbench" aria-label="关于作者与近期关注">
        <div class="workbench-topline">
          <span>FROM MY WORKBENCH</span>
          <span class="workbench-symbol" aria-hidden="true">✳</span>
        </div>
        <NuxtLink to="/about" class="workbench-identity">
          <img src="/avatar.jpg" :alt="appConfig.authorCN" width="68" height="68" />
          <div>
            <strong>{{ appConfig.authorEN }}</strong>
            <span>{{ appConfig.authorCN }} · Research & Engineering</span>
          </div>
        </NuxtLink>
        <p class="workbench-label">近期关注 / CURRENT FOCUS</p>
        <NuxtLink
          v-for="(focus, index) in focusLinks"
          :key="focus.to"
          :to="focus.to"
          class="workbench-link"
        >
          <span class="workbench-index">0{{ index + 1 }}</span>
          <span>{{ focus.title }}</span>
          <span class="workbench-arrow" aria-hidden="true">↗</span>
        </NuxtLink>
        <div class="workbench-note">
          <span class="workbench-note-dot" aria-hidden="true" />
          <p>{{ appConfig.status }}</p>
        </div>
      </aside>
    </section>

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
      <SiteBlock
        :eyebrow="featured ? '02 / Explore' : '01 / Explore'"
        title="从一个方向开始"
        description="不必从头翻起，挑一个感兴趣的主题。"
        action-to="/tags"
        action-label="所有主题"
      >
        <KnowledgeMap :items="knowledgeMap" />
      </SiteBlock>
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
const focusLinks = [
  { title: 'AI Infrastructure', to: '/tags/ai-infra' },
  { title: 'LLM Agents', to: '/tags/agent' },
  { title: 'Systems & Source Reading', to: '/tags/源码分析' },
]

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
    desc: '从模型基础到 RAG、上下文工程、推理服务和 Agent 平台。',
    label: '看 AI Infra',
    to: '/tags/ai-infra',
  },
  {
    key: 'SYS',
    title: '后端与分布式',
    desc: '数据库、缓存、高并发、分布式系统与 Go 后端项目。',
    label: '看后端系统',
    to: '/tags/分布式系统',
  },
  {
    key: 'SRC',
    title: '源码阅读',
    desc: '从入口、数据流和关键抽象读懂开源项目。',
    label: '看源码分析',
    to: '/tags/源码分析',
  },
  {
    key: 'INT',
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
