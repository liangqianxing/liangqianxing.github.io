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
          <h1 id="journal-title" class="notebook-title">构建系统，<br /><span class="notebook-highlight">用数据说话<svg viewBox="0 0 100 20" preserveAspectRatio="none" fill="none" aria-hidden="true"><path d="M5 15C20 12 50 12 95 18" stroke="currentColor" stroke-width="6" stroke-linecap="round" /></svg></span>。</h1>
          <p class="notebook-summary">这里是{{ appConfig.authorCN }}的技术手记。聚焦 LLM Agent 架构、后训练对齐（GRPO / DPO）、多模态推理蒸馏与 AI Infra（SGLang / CUDA 算子 / C++ 并发系统）的实验与源码剖析。</p>
          <NotebookTypewriter />
          <div class="notebook-actions">
            <NuxtLink to="#home-content" class="notebook-primary">浏览技术长文 <span aria-hidden="true">↗</span></NuxtLink>
            <NuxtLink to="/academic" class="notebook-secondary">学术与项目主页 <span aria-hidden="true">→</span></NuxtLink>
          </div>
          <p class="notebook-signoff" aria-hidden="true">from mathematical formulation to systems execution.</p>
        </div>
        <NotebookCode :name="appConfig.authorCN" :notebook="appConfig.title" />
      </section>
    </div>

    <div id="home-content" class="site-main journal-content">
      <LearningCoursesPreview />
      <SiteBlock
        v-if="featured"
        eyebrow="01 / Latest Research & Engineering"
        title="最新深度长文"
        description="近期完成的算法实验、系统性能剖析与开源底层修复记录。"
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
            <p class="notebook-topic-kicker" aria-hidden="true">core research &amp; engineering tracks</p>
            <h2 id="notebook-topics-title">核心研究与系统工程脉络</h2>
          </div>
        </div>
        <KnowledgeMap :items="knowledgeMap" />
      </section>
      <SiteBlock
        v-if="selectedPosts.length"
        eyebrow="Selected / Flagship Deep Dives"
        title="代表性研究与系统实录"
        description="涵盖开源 C++ 并发修复、工业级多约束 Agent 规划、多模态蒸馏与可审计评审、单卡 FSDP2 多模态 GRPO 及 CUDA 融合算子。"
      >
        <div class="card-grid">
          <ArticleCard v-for="post in selectedPosts" :key="post.path" :post="post" />
        </div>
      </SiteBlock>
      <SiteBlock
        v-if="topicCounts.length"
        eyebrow="Index / Topics"
        title="技术专题索引"
        description="顺着一个技术方向或系统组件，检索全部相关实验与源码分析。"
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

const { data } = await usePublicArchive()

const posts = computed(() => data.value ?? [])
const COURSE_SERIES = new Set(['CMU 15-779', 'Stanford CS336'])
const flagshipPosts = computed(() =>
  posts.value.filter(post => !(post.series && COURSE_SERIES.has(post.series))),
)
const featured = computed(() => flagshipPosts.value[0] ?? posts.value[0] ?? null)
const selectedSlugs = [
  'xgrammar-concurrent-cache-generation-fix',
  'multi-constraint-agent-routing-dag-eval',
  'llm-mrd-distillation-auditable-review-agents',
  'in-situ-autoguidance-editable-scientific-illustration',
  'qwen25-sft-dpo-alignment-tax-ablation',
  'llm-gateway-semantic-cache-dynamic-batching',
  'qwen25-vl-sglang-verl-single-gpu-lab',
  'cuda-fused-rmsnorm-kernel',
  'deepscientist-tech-stack',
]
const selectedPosts = computed(() =>
  selectedSlugs
    .map(slug => flagshipPosts.value.find(post => post.slug === slug))
    .filter((post): post is PostMeta => Boolean(post)),
)
const recentPosts = computed(() => {
  const excluded = new Set([featured.value?.slug])
  return flagshipPosts.value.filter(post => !excluded.has(post.slug)).slice(0, 6)
})
const topicCounts = computed(() => {
  const counts = new Map<string, number>()
  for (const post of posts.value) {
    for (const tag of post.tags ?? []) counts.set(tag, (counts.get(tag) ?? 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])
})

function countByTagSlug(slug: string): number {
  const normalized = slug.toLowerCase()
  return posts.value.filter(post =>
    (post.tags ?? []).some(t =>
      t.trim().replace(/\s*\/\s*/g, '-').replace(/\s+/g, '-').toLowerCase() === normalized
    )
  ).length
}

const knowledgeMap = computed(() => [
  {
    key: 'AI',
    eyebrow: '推理引擎与智能体架构',
    title: 'AI Infra & Agent',
    desc: 'SGLang / vLLM 推理服务、CUDA 融合算子、DAG 规划与 MCP 沙箱平台。',
    label: '看 AI Infra',
    count: countByTagSlug('ai-infra'),
    to: '/tags/ai-infra',
  },
  {
    key: 'RL',
    eyebrow: '强化学习、偏好对齐与蒸馏',
    title: 'Post-Training & RL',
    desc: 'Verifiable GRPO（verl）、SFT → DPO 偏好对齐、对齐税评测与跨模态知识蒸馏。',
    label: '看后训练与对齐',
    count: countByTagSlug('post-training'),
    to: '/tags/post-training',
  },
  {
    key: 'SYS',
    eyebrow: '高并发与分布式基础设施',
    title: '后端与分布式系统',
    desc: 'Go 微服务、Redis / PostgreSQL、分布式锁与高吞吐数据链路设计。',
    label: '看分布式系统',
    count: countByTagSlug('分布式系统'),
    to: '/tags/分布式系统',
  },
  {
    key: 'SRC',
    eyebrow: '开源项目与底层实现拆解',
    title: '源码剖析与架构复盘',
    desc: 'XGrammar C++ 并发修复、DeepScientist、MiniCode 与推理引擎核心源码。',
    label: '看源码分析',
    count: countByTagSlug('源码分析'),
    to: '/tags/源码分析',
  },
])

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
