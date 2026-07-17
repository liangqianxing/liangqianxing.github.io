<template>
  <div>
    <section class="hero-frame">
      <div class="hero-main">
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
    desc: '从模型基础到 RAG、上下文工程、推理服务和 Agent 平台。',
    label: '看 AI Infra',
    to: '/tags/ai-infra',
  },
  {
    key: 'SYS',
    title: 'Backend / Distributed',
    desc: '数据库、缓存、高并发、分布式系统与 Go 后端项目。',
    label: '看后端系统',
    to: '/tags/分布式系统',
  },
  {
    key: 'SRC',
    title: 'Source Reading',
    desc: '从入口、数据流和关键抽象读懂开源项目，而非罗列目录。',
    label: '看源码分析',
    to: '/tags/源码分析',
  },
  {
    key: 'INT',
    title: 'Interview Kit',
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
    { property: 'og:url', content: appConfig.url },
  ],
})
</script>
