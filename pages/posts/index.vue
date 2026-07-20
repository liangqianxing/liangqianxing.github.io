<template>
  <main class="page-frame">
    <header class="page-hero">
      <p class="eyebrow">Archive</p>
      <h1>文章库</h1>
      <p>从主题进入，再按关键词或标签缩小范围。共 {{ posts.length }} 篇公开文章。</p>
    </header>

    <section class="archive-directory" aria-labelledby="archive-directory-title">
      <div class="archive-directory-head">
        <div>
          <p class="eyebrow">Topics</p>
          <h2 id="archive-directory-title">从一个方向开始</h2>
        </div>
        <p>文章可以同时属于多个主题。</p>
      </div>

      <div class="topic-directory">
        <button
          v-for="topic in topicSummaries"
          :key="topic.id"
          class="archive-topic"
          :class="{ active: activeTopic === topic.id }"
          type="button"
          :aria-pressed="activeTopic === topic.id"
          @click="setActiveTopic(topic.id)"
        >
          <span class="archive-topic-index">{{ topic.index }}</span>
          <span class="archive-topic-copy">
            <strong>{{ topic.name }}</strong>
            <small>{{ topic.description }}</small>
          </span>
          <span class="archive-topic-count">{{ topic.count }}</span>
        </button>
      </div>
    </section>

    <div class="archive-results-head">
      <div>
        <p class="eyebrow">Selection</p>
        <h2>{{ selectedTopic.name }}</h2>
      </div>
      <span class="archive-count">{{ filteredPosts.length }} / {{ topicPosts.length }} 篇</span>
    </div>

    <section class="archive-panel">
      <label class="search-box">
        <span aria-hidden="true">⌕</span>
        <input
          v-model.trim="query"
          type="search"
          :placeholder="`在「${selectedTopic.name}」中搜索`"
          :aria-label="`在${selectedTopic.name}中搜索文章`"
        />
      </label>
    </section>

    <div v-if="topTags.length" class="filter-row" aria-label="标签筛选">
      <button class="topic-button" :class="{ active: activeTag === '' }" type="button" @click="activeTag = ''">
        全部标签
      </button>
      <button
        v-for="[tag, count] in topTags"
        :key="tag"
        class="topic-button"
        :class="{ active: activeTag === tag }"
        type="button"
        @click="activeTag = activeTag === tag ? '' : tag"
      >
        #{{ tag }}
        <span>{{ count }}</span>
      </button>
    </div>

    <template v-if="postsByYear.length">
      <section v-for="[year, yearPosts] in postsByYear" :key="year" class="year-section">
        <h2>
          {{ year }}
          <span>{{ yearPosts.length }} 篇</span>
        </h2>
        <ArticleStream :posts="yearPosts" />
      </section>
    </template>

    <p v-else class="empty-state">没有匹配的文章，换个关键词试试。</p>
  </main>
</template>

<script setup lang="ts">
import type { PostMeta } from '~/server/api/posts.get'

const { data } = await useAsyncData<PostMeta[]>('posts-archive', () =>
  $fetch('/api/posts')
)

const query = ref('')
const activeTag = ref('')
const activeTopic = ref<TopicId>('all')
const posts = computed(() => data.value ?? [])

type TopicId = 'all' | 'interview' | 'llm-systems' | 'foundations' | 'training' | 'multimodal' | 'backend' | 'low-latency' | 'algorithms' | 'other'

interface ArchiveTopic {
  id: TopicId
  index: string
  name: string
  description: string
  tags?: string[]
  titleTerms?: string[]
  seriesTerms?: string[]
}

const topicDefinitions: ArchiveTopic[] = [
  {
    id: 'all',
    index: '00',
    name: '全部文章',
    description: '按时间浏览完整归档',
  },
  {
    id: 'interview',
    index: '01',
    name: '求职与面试',
    description: '八股、项目包装与岗位准备',
    tags: ['面试', '实习求职'],
    titleTerms: ['面经', '面试'],
  },
  {
    id: 'llm-systems',
    index: '02',
    name: 'LLM 与 Agent',
    description: 'RAG、智能体与 AI 系统工程',
    tags: ['AI Infra', 'Agent', 'RAG', 'vLLM', 'OpenClaw', 'LLM Memory', 'Vector DB', '向量检索', '混合检索'],
    titleTerms: ['AI Infra', 'Agent', 'RAG', '推理引擎', 'AI 编程助手', 'AI 科研'],
  },
  {
    id: 'foundations',
    index: '03',
    name: '模型原理',
    description: '基模、架构与生成模型',
    tags: ['Transformer', 'ViT', 'CLIP', '深度学习', 'Autograd', '生成模型', 'Diffusion', 'Flow Matching', '基础模型'],
    titleTerms: ['LLM 技术路线', 'Transformer', '基模'],
    seriesTerms: ['LLM 核心原理'],
  },
  {
    id: 'training',
    index: '04',
    name: '训练与推理',
    description: '预训练、后训练与推理优化',
    tags: ['大模型训练', '预训练', '分布式训练', '推理优化', 'GPU', 'TensorRT', 'Nanotron', 'PagedAttention', '投机解码'],
    titleTerms: ['训练', '推理', 'LLM Serving'],
  },
  {
    id: 'multimodal',
    index: '05',
    name: '多模态',
    description: '视觉语言、OCR 与多模态检索',
    tags: ['多模态', '多模态大模型', 'CLIP', 'ViT', 'OCR', '医疗 AI', 'VLA'],
    titleTerms: ['多模态', 'Vision Transformer', 'CLIP', 'OCR'],
  },
  {
    id: 'backend',
    index: '06',
    name: '后端与系统',
    description: 'Go、数据库与分布式工程',
    tags: ['Go', '后端', '后端架构', '后端框架', '数据库', '分布式系统', 'Kafka', '消息队列', 'Git', 'GitHub Actions', '全栈', '前端'],
    titleTerms: ['后端', '数据库', '分布式系统', 'Kafka', 'CloudVault', 'GoFoundry'],
  },
  {
    id: 'low-latency',
    index: '07',
    name: '量化与低延迟',
    description: 'C++、并发与性能优化',
    tags: ['量化开发', '低延迟', '并发', '无锁编程', '性能优化'],
    titleTerms: ['C++ 并发', '无锁并发'],
  },
  {
    id: 'algorithms',
    index: '08',
    name: '算法与竞赛',
    description: 'LeetCode、数据结构与 XCPC',
    tags: ['算法', 'LeetCode', 'XCPC', '比赛', '链表', 'LRU'],
    titleTerms: ['算法', '竞赛', 'XCPC'],
  },
  {
    id: 'other',
    index: '09',
    name: '随笔与其他',
    description: '语言学习、旅行与零散记录',
  },
]

const normalize = (value: string | undefined) => value?.toLocaleLowerCase('zh-CN') ?? ''

function matchesTopic(post: PostMeta, topic: ArchiveTopic) {
  if (topic.id === 'all') return true

  if (topic.id === 'other') {
    return !topicDefinitions.some((candidate) =>
      candidate.id !== 'all' && candidate.id !== 'other' && matchesTopic(post, candidate)
    )
  }

  const tags = new Set((post.tags ?? []).map(normalize))
  const title = normalize(post.title)
  const series = normalize(post.series)
  const matchesTag = topic.tags?.some((tag) => tags.has(normalize(tag))) ?? false
  const matchesTitle = topic.titleTerms?.some((term) => title.includes(normalize(term))) ?? false
  const matchesSeries = topic.seriesTerms?.some((term) => series.includes(normalize(term))) ?? false
  return matchesTag || matchesTitle || matchesSeries
}

const selectedTopic = computed(() =>
  topicDefinitions.find((topic) => topic.id === activeTopic.value) ?? topicDefinitions[0]
)

const topicPosts = computed(() =>
  posts.value.filter((post) => matchesTopic(post, selectedTopic.value))
)

const topicSummaries = computed(() =>
  topicDefinitions.map((topic) => ({
    ...topic,
    count: posts.value.filter((post) => matchesTopic(post, topic)).length,
  }))
)

const topTags = computed(() => {
  const counts = new Map<string, number>()
  for (const post of topicPosts.value) {
    for (const tag of post.tags ?? []) counts.set(tag, (counts.get(tag) ?? 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)
})

const filteredPosts = computed(() => {
  const q = query.value.toLowerCase()
  return topicPosts.value.filter((post) => {
    const matchesTag = activeTag.value ? post.tags?.includes(activeTag.value) : true
    if (!q) return matchesTag
    const haystack = [post.title, post.description, post.excerpt, post.series, ...(post.tags ?? [])].join(' ').toLowerCase()
    return matchesTag && haystack.includes(q)
  })
})

function setActiveTopic(topic: TopicId) {
  activeTopic.value = topic
  activeTag.value = ''
}

const postsByYear = computed(() => {
  const map = new Map<string, PostMeta[]>()
  for (const post of filteredPosts.value) {
    const year = new Date(post.date).getFullYear().toString()
    if (!map.has(year)) map.set(year, [])
    map.get(year)!.push(post)
  }
  return [...map.entries()].sort((a, b) => Number(b[0]) - Number(a[0]))
})

useHead({
  title: '文章库',
  meta: [{ name: 'description', content: '按求职、大模型、多模态、后端、量化和算法等主题浏览文章归档' }],
})
</script>
