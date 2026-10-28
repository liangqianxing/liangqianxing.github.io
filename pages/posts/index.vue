<template>
  <div class="page-frame writing-archive">
    <header class="writing-hero">
      <div class="writing-hero-copy">
        <p class="writing-kicker"><span aria-hidden="true">◆</span> RESEARCH &amp; SYSTEMS ARCHIVE</p>
        <h1>代表性技术长文与<span>实验报告</span></h1>
        <p class="writing-intro">
          收录围绕 <strong>LLM Agents、后训练与偏好对齐（GRPO / DPO / SimPO）、多模态推理蒸馏、视频扩散模型（DiT）加速</strong>与<strong>底层 AI 推理系统（XGrammar / SGLang / CUDA Kernel）</strong>撰写的源码拆解与可复核实验记录。
        </p>
        <div v-if="!error && status === 'success'" class="writing-stats">
          <span><strong>{{ flagshipPosts.length }}</strong> 篇旗舰研究与系统长文</span>
          <span><strong>{{ coursePosts.length }}</strong> 篇 CMU / Stanford 课程实验手记</span>
          <NuxtLink to="/academic">查看学术论文主页 <span aria-hidden="true">↗</span></NuxtLink>
          <NuxtLink to="/learning">进入课程专题 <span aria-hidden="true">↗</span></NuxtLink>
        </div>
      </div>

      <aside class="writing-metrics-panel" aria-label="核心研究与系统工程方向">
        <div class="metric-card">
          <span class="metric-code">01 / AGENT &amp; MULTIMODAL</span>
          <strong>智能体与多模态系统</strong>
          <p>多约束 DAG 规划、DeepScientist 100K+ 沙箱、MiniCode、AutoFigure-Edit 与具身鲁棒性</p>
        </div>
        <div class="metric-card">
          <span class="metric-code">02 / POST-TRAINING &amp; DISTILL</span>
          <strong>后训练对齐与推理蒸馏</strong>
          <p>Qwen2.5 SFT/DPO 对齐税、单卡 verl GRPO、CycleResearcher 与 LLM-MRD 17.2ms 蒸馏</p>
        </div>
        <div class="metric-card">
          <span class="metric-code">03 / AI INFRA &amp; ACCELERATION</span>
          <strong>推理系统与算子加速</strong>
          <p>XGrammar PR #1011 并发修复、EditCache H100 2.46× 加速、CUDA 融合 RMSNorm 4.27× 与推理网关</p>
        </div>
      </aside>
    </header>

    <section class="writing-controls" aria-label="专题切换与文章检索">
      <div class="writing-tracks" role="tablist" aria-label="技术领域分类">
        <button
          v-for="tab in trackTabs"
          :key="tab.key"
          type="button"
          role="tab"
          class="track-tab"
          :class="{ 'is-active': activeTrack === tab.key }"
          :aria-selected="activeTrack === tab.key"
          @click="selectTrack(tab.key)"
        >
          <span class="track-tab-label">{{ tab.label }}</span>
          <span class="track-tab-count">{{ tab.count }}</span>
        </button>
      </div>

      <div class="writing-find-top">
        <label class="writing-search">
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="10.5" cy="10.5" r="6.5" />
            <path d="m15.5 15.5 4 4" />
          </svg>
          <input
            v-model="query"
            type="search"
            placeholder="搜索论文、系统、算子、算法或实验指标（如 XGrammar、DPO、EditCache、CUDA）…"
            aria-label="搜索文章"
          />
          <button v-if="query" type="button" aria-label="清空搜索" @click="query = ''">×</button>
        </label>
        <NuxtLink to="/tags" class="writing-tag-link">标签索引 <span aria-hidden="true">↗</span></NuxtLink>
      </div>

      <div v-if="tags.length" class="writing-filters" role="group" aria-label="按标签筛选文章">
        <button
          type="button"
          :aria-pressed="!activeTag"
          @click="activeTag = ''"
        >
          全部标签 <span>{{ trackScopedPosts.length }}</span>
        </button>
        <button
          v-for="entry in visibleFilterTags"
          :key="entry.slug"
          type="button"
          :aria-pressed="activeTag === entry.slug"
          :data-tone="archiveTone(entry.slug)"
          @click="activeTag = activeTag === entry.slug ? '' : entry.slug"
        >
          #{{ entry.tag }} <span>{{ entry.count }}</span>
        </button>
        <button
          v-if="tags.length > FILTER_TAG_LIMIT"
          type="button"
          class="writing-filters-more"
          :aria-expanded="showAllTags"
          @click="showAllTags = !showAllTags"
        >
          {{ showAllTags ? '收起标签 ↑' : `展开全部 ${tags.length} 个标签 ↓` }}
        </button>
      </div>
    </section>

    <div class="writing-results">
      <p role="status" aria-live="polite" aria-atomic="true">
        <strong>{{ currentTrackMeta.label }}</strong> · 共 {{ filteredPosts.length }} 篇
        <span v-if="currentTrackMeta.desc" class="writing-results-desc">— {{ currentTrackMeta.desc }}</span>
        <span v-if="activeTagName"> · 标签 #{{ activeTagName }}</span>
      </p>
      <button v-if="hasFilters" type="button" @click="resetFilters">重置筛选 <span aria-hidden="true">↺</span></button>
    </div>

    <div v-if="error" class="writing-empty" role="alert">
      <p>文章暂时没能加载，刷新页面再试试吧。</p>
      <a href="/posts/">刷新页面 <span aria-hidden="true">↻</span></a>
    </div>
    <div v-else-if="status === 'pending'" class="writing-empty" role="status">
      <p>正在加载技术档案…</p>
    </div>
    <ArchiveEntries v-else-if="filteredPosts.length" :posts="filteredPosts" />
    <div v-else class="writing-empty">
      <span class="writing-empty-mark" aria-hidden="true">⌕</span>
      <h2>未找到匹配的技术文章</h2>
      <p>换个关键词，或切换上方研究方向标签再试试。</p>
      <button v-if="hasFilters" type="button" @click="resetFilters">查看全部旗舰长文 <span aria-hidden="true">→</span></button>
      <NuxtLink v-else to="/blog">回到博客首页 <span aria-hidden="true">→</span></NuxtLink>
    </div>

    <footer class="writing-signoff">
      <span><span aria-hidden="true">◆</span> 所有实验数据、代码与模型散列均保留可复核记录。</span>
      <div class="writing-signoff-links">
        <NuxtLink to="/academic">学术主页 <span aria-hidden="true">↗</span></NuxtLink>
        <NuxtLink to="/blog">博客首页 <span aria-hidden="true">→</span></NuxtLink>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { archiveTone, collectArchiveTags } from '~/utils/archive'
import { tagSlug } from '~/utils/blog'
import type { PostMeta } from '~/server/api/posts.get'

type TrackKey = 'flagship' | 'agent' | 'post-training' | 'infra' | 'courses'

const COURSE_SERIES = new Set(['CMU 15-779', 'Stanford CS336'])

const AGENT_SLUGS = new Set([
  'multi-constraint-agent-routing-dag-eval',
  'in-situ-autoguidance-editable-scientific-illustration',
  'deepscientist-tech-stack',
  'minicode-source-analysis',
  'agent-memory',
  'embodied-pickplace-lab-robustness',
])

const POST_TRAINING_SLUGS = new Set([
  'llm-mrd-distillation-auditable-review-agents',
  'qwen25-sft-dpo-alignment-tax-ablation',
  'qwen25-vl-sglang-verl-single-gpu-lab',
  'cycleresearcher-post-training',
  'qwen3-mid-post-train-mlx',
])

const INFRA_SLUGS = new Set([
  'editcache-video-dit-inference-acceleration',
  'xgrammar-concurrent-cache-generation-fix',
  'llm-gateway-semantic-cache-dynamic-batching',
  'cuda-fused-rmsnorm-kernel',
  'backend-stack-deep-dive',
])

function isCourseLabPost(post: PostMeta): boolean {
  return Boolean(post.series && COURSE_SERIES.has(post.series))
}

const route = useRoute()
const router = useRouter()
const { data, error, status } = await usePublicArchive()
const allPosts = computed(() => data.value ?? [])

const flagshipPosts = computed(() => allPosts.value.filter(post => !isCourseLabPost(post)))
const coursePosts = computed(() => allPosts.value.filter(post => isCourseLabPost(post)))

const query = ref(typeof route.query.q === 'string' ? route.query.q : '')
const activeTag = ref(typeof route.query.tag === 'string' ? route.query.tag : '')
const activeTrack = ref<TrackKey>(
  typeof route.query.track === 'string' && ['flagship', 'agent', 'post-training', 'infra', 'courses'].includes(route.query.track)
    ? (route.query.track as TrackKey)
    : 'flagship',
)

function selectTrack(track: TrackKey) {
  activeTrack.value = track
  activeTag.value = ''
}

const trackTabs = computed(() => [
  {
    key: 'flagship' as TrackKey,
    label: '全部旗舰长文',
    desc: '代表性学术研究、开源基础设施与生产级系统拆解',
    count: flagshipPosts.value.length,
  },
  {
    key: 'agent' as TrackKey,
    label: 'Agent 与多模态系统',
    desc: 'DAG 规划、MCP 沙箱、AutoFigure-Edit、原位自引导与具身控制',
    count: flagshipPosts.value.filter(p => AGENT_SLUGS.has(p.slug)).length,
  },
  {
    key: 'post-training' as TrackKey,
    label: '后训练与推理蒸馏',
    desc: 'GRPO、SFT/DPO 对齐税、SimPO、生成式奖励模型与多视角蒸馏',
    count: flagshipPosts.value.filter(p => POST_TRAINING_SLUGS.has(p.slug)).length,
  },
  {
    key: 'infra' as TrackKey,
    label: 'AI Infra 与算子加速',
    desc: 'XGrammar 并发缓存、视频 DiT 加速、CUDA 融合算子与高并发推理网关',
    count: flagshipPosts.value.filter(p => INFRA_SLUGS.has(p.slug)).length,
  },
  {
    key: 'courses' as TrackKey,
    label: '课程实验手记 (CMU / CS336)',
    desc: 'CMU 15-779 与 Stanford CS336 算子、量化、MoE 与分页 KV 逐课实现记录',
    count: coursePosts.value.length,
  },
])

const currentTrackMeta = computed(() => trackTabs.value.find(t => t.key === activeTrack.value) ?? trackTabs.value[0])

const trackScopedPosts = computed(() => {
  if (activeTrack.value === 'courses') return coursePosts.value
  if (activeTrack.value === 'agent') return flagshipPosts.value.filter(p => AGENT_SLUGS.has(p.slug))
  if (activeTrack.value === 'post-training') return flagshipPosts.value.filter(p => POST_TRAINING_SLUGS.has(p.slug))
  if (activeTrack.value === 'infra') return flagshipPosts.value.filter(p => INFRA_SLUGS.has(p.slug))
  return flagshipPosts.value
})

const tags = computed(() => collectArchiveTags(trackScopedPosts.value))

watch(
  () => [route.query.q, route.query.tag, route.query.track],
  ([q, tag, track]) => {
    const nextQ = typeof q === 'string' ? q : ''
    const nextTag = typeof tag === 'string' ? tag : ''
    const nextTrack = typeof track === 'string' && ['flagship', 'agent', 'post-training', 'infra', 'courses'].includes(track)
      ? (track as TrackKey)
      : 'flagship'
    if (query.value !== nextQ) query.value = nextQ
    if (activeTag.value !== nextTag) activeTag.value = nextTag
    if (activeTrack.value !== nextTrack) activeTrack.value = nextTrack
  },
)

watch([query, activeTag, activeTrack], ([q, tag, track]) => {
  if (import.meta.server) return
  const nextQuery: Record<string, string> = {}
  const trimmedQ = q.trim()
  if (trimmedQ) nextQuery.q = trimmedQ
  if (tag) nextQuery.tag = tag
  if (track !== 'flagship') nextQuery.track = track
  const currQ = typeof route.query.q === 'string' ? route.query.q : ''
  const currTag = typeof route.query.tag === 'string' ? route.query.tag : ''
  const currTrack = typeof route.query.track === 'string' ? route.query.track : 'flagship'
  if (trimmedQ === currQ && tag === currTag && track === currTrack) return
  router.replace({ query: nextQuery })
})

const activeTagName = computed(() => tags.value.find(tag => tag.slug === activeTag.value)?.tag ?? '')
const FILTER_TAG_LIMIT = 18
const showAllTags = ref(false)
const visibleFilterTags = computed(() => {
  if (showAllTags.value || tags.value.length <= FILTER_TAG_LIMIT) return tags.value
  const top = tags.value.slice(0, FILTER_TAG_LIMIT)
  if (activeTag.value && !top.some(entry => entry.slug === activeTag.value)) {
    const selected = tags.value.find(entry => entry.slug === activeTag.value)
    if (selected) return [...top, selected]
  }
  return top
})

const hasFilters = computed(() => Boolean(query.value.trim() || activeTag.value || activeTrack.value !== 'flagship'))

const filteredPosts = computed(() => {
  const keyword = query.value.trim().toLocaleLowerCase('zh-CN')
  // When user types a search query while on default tab, search across all posts if no flagship hits
  const pool = trackScopedPosts.value
  return pool.filter((post) => {
    if (activeTag.value && !post.tags.some(tag => tagSlug(tag) === activeTag.value)) return false
    if (!keyword) return true
    return [post.title, post.description, post.excerpt, post.series, ...post.tags]
      .join(' ')
      .toLocaleLowerCase('zh-CN')
      .includes(keyword)
  })
})

function resetFilters() {
  query.value = ''
  activeTag.value = ''
  activeTrack.value = 'flagship'
}

useHead({
  title: '技术长文与研究档案',
  meta: [
    {
      name: 'description',
      content: '古恩豪（Enhao Gu）的代表性技术长文与实验报告：涵盖 LLM Agent 架构、后训练对齐（GRPO / DPO）、多模态推理蒸馏、视频扩散模型加速与 AI 推理系统源码分析。',
    },
  ],
})
</script>

<style scoped>
.writing-archive :where(h1, h2, p) {
  margin: 0;
}

.writing-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.95fr);
  align-items: stretch;
  gap: 28px;
  padding: 6px 4px 28px;
}

.writing-hero-copy {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.writing-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  color: var(--accent);
}

.writing-hero h1 {
  margin-block: 10px 14px;
  color: var(--ink);
  font-size: clamp(2rem, 3.6vw, 2.55rem);
  font-weight: 600;
  letter-spacing: -0.035em;
  line-height: 1.35;
}

.writing-hero h1 > span {
  color: var(--accent);
}

.writing-intro {
  color: var(--text);
  font-size: 0.88rem;
  line-height: 1.9;
}

.writing-intro strong {
  color: var(--ink);
  font-weight: 500;
}

.writing-stats {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px 18px;
  margin-top: 18px;
  color: var(--muted);
  font-size: 0.75rem;
}

.writing-stats strong {
  color: var(--ink);
  font-family: var(--font-mono);
}

.writing-stats > span + span::before {
  content: '·';
  margin-right: 18px;
  color: var(--line-strong);
}

.writing-stats a {
  color: var(--accent);
  font-weight: 500;
  text-decoration: none;
}

.writing-stats a:hover {
  text-decoration: underline;
  text-underline-offset: 4px;
}

.writing-metrics-panel {
  display: grid;
  gap: 10px;
  padding: 16px;
  border-radius: 20px;
  border: 1px solid var(--line);
  background: color-mix(in srgb, var(--surface) 82%, var(--bg-2));
}

.metric-card {
  padding: 11px 14px;
  border-radius: 13px;
  border: 1px solid color-mix(in srgb, var(--line) 85%, transparent);
  background: var(--surface);
}

.metric-code {
  display: block;
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 0.08em;
  color: var(--accent);
}

.metric-card strong {
  display: block;
  margin-top: 2px;
  color: var(--ink);
  font-size: 0.82rem;
  font-weight: 600;
}

.metric-card p {
  margin-top: 3px;
  color: var(--muted);
  font-size: 0.72rem;
  line-height: 1.55;
}

.writing-controls {
  padding: 20px 24px;
  border-radius: 22px;
  border: 1px solid var(--line);
  background: var(--bg-2);
}

.writing-tracks {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding-bottom: 16px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--line);
}

.track-tab {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 38px;
  padding: 6px 14px;
  border-radius: 11px;
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--text);
  font-size: 0.78rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 160ms ease;
}

.track-tab:hover {
  border-color: var(--line-strong);
  color: var(--ink);
}

.track-tab.is-active {
  background: var(--ink);
  border-color: var(--ink);
  color: var(--bg);
}

.track-tab-count {
  padding: 1px 7px;
  border-radius: 999px;
  background: var(--bg-2);
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.66rem;
}

.track-tab.is-active .track-tab-count {
  background: color-mix(in srgb, var(--bg) 18%, transparent);
  color: var(--bg);
}

.writing-find-top {
  display: flex;
  align-items: center;
  gap: 20px;
}

.writing-search {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  padding: 0 14px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--surface);
}

.writing-search > svg {
  width: 18px;
  height: 18px;
  flex: 0 0 auto;
  stroke: var(--muted);
  stroke-width: 1.6;
  stroke-linecap: round;
}

.writing-search input {
  width: 100%;
  min-width: 0;
  height: 42px;
  padding: 0;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--ink);
  font-size: 0.82rem;
}

.writing-search:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 14%, transparent);
}

.writing-search input::placeholder {
  color: var(--muted);
}

.writing-search input::-webkit-search-cancel-button {
  display: none;
}

.writing-search button {
  flex: 0 0 auto;
  width: 28px;
  height: 28px;
  border: 0;
  background: transparent;
  color: var(--muted);
  font-size: 1.2rem;
  cursor: pointer;
}

.writing-tag-link {
  flex: 0 0 auto;
  font-size: 0.75rem;
  color: var(--accent);
  font-weight: 500;
  white-space: nowrap;
  text-decoration: none;
}

.writing-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 14px;
}

.writing-filters button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 32px;
  padding: 4px 11px;
  border: 1px solid transparent;
  border-radius: 999px;
  background: transparent;
  color: var(--text);
  font-size: 0.72rem;
  cursor: pointer;
  transition: background-color 150ms ease, border-color 150ms ease, color 150ms ease;
}

.writing-filters button > span {
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.64rem;
}

.writing-filters button:hover {
  background: var(--surface);
  border-color: var(--line);
}

.writing-filters button[aria-pressed="true"] {
  background: var(--surface);
  border-color: color-mix(in srgb, var(--accent) 55%, var(--line-strong));
  color: var(--ink);
  font-weight: 600;
}

.writing-filters button[aria-pressed="true"] > span {
  color: var(--accent);
}

.writing-filters button.writing-filters-more {
  border-style: dashed;
  border-color: var(--line-strong);
  color: var(--accent);
  font-weight: 500;
}

.writing-results {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  margin-block: 22px 18px;
  padding-inline: 4px;
  color: var(--muted);
  font-size: 0.76rem;
}

.writing-results strong {
  color: var(--ink);
}

.writing-results-desc {
  color: var(--muted);
}

.writing-results button {
  flex: 0 0 auto;
  padding: 4px 0;
  border: 0;
  background: transparent;
  color: var(--accent);
  font-size: 0.74rem;
  cursor: pointer;
}

.writing-empty {
  padding: 52px 20px 60px;
  border: 1px dashed var(--line-strong);
  border-radius: 20px;
  text-align: center;
}

.writing-empty-mark {
  display: block;
  color: var(--accent);
  font-size: 2.2rem;
}

.writing-empty h2 {
  margin-top: 12px;
  color: var(--ink);
  font-size: 1.12rem;
  font-weight: 600;
}

.writing-empty p {
  margin-top: 8px;
  color: var(--muted);
  font-size: 0.84rem;
}

.writing-empty button,
.writing-empty a {
  display: inline-flex;
  gap: 10px;
  align-items: center;
  min-height: 38px;
  margin-top: 18px;
  padding: 5px 16px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--surface);
  color: var(--accent);
  font-size: 0.78rem;
  text-decoration: none;
  cursor: pointer;
}

.writing-signoff {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 40px;
  padding: 20px 4px 0;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.75rem;
}

.writing-signoff-links {
  display: flex;
  gap: 18px;
}

.writing-signoff a {
  color: var(--accent);
  text-decoration: none;
}

@media (max-width: 860px) {
  .writing-hero {
    grid-template-columns: 1fr;
    gap: 20px;
  }
}

@media (max-width: 520px) {
  .writing-controls {
    padding: 16px;
  }
  .writing-find-top {
    flex-wrap: wrap;
    gap: 10px;
  }
  .writing-search {
    flex-basis: 100%;
  }
  .writing-results-desc {
    display: none;
  }
}
</style>
