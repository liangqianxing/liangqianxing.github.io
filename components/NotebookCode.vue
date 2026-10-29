<template>
  <aside
    class="notebook-code-canvas"
    :class="{ 'notebook-code-paused': isPageHidden, 'notebook-code-reduced': prefersReducedMotion }"
    aria-label="作者的三页手记"
  >
    <div class="notebook-code-shell">
      <div class="notebook-code-tabbar">
        <span class="notebook-code-dots" aria-hidden="true"><i /><i /><i /></span>
        <span class="notebook-code-file">{{ cards[activeIndex].file }}</span>
        <span class="notebook-code-badge">dossier</span>
      </div>

      <div class="notebook-code-stage">
        <div class="notebook-code-stack">
          <section
            v-for="(card, index) in cards"
            :key="card.file"
            class="notebook-code-window"
            :class="{ 'notebook-code-active': index === activeIndex, 'notebook-code-leaving': index === leavingIndex }"
            :style="cardStyle(index)"
            :aria-label="card.label"
            :aria-hidden="index !== activeIndex"
            :inert="index !== activeIndex"
          >
            <template v-if="index === 0">
              <div class="notebook-paper-intro">
                <div class="notebook-paper-heading">
                  <span class="notebook-paper-kicker">profile.ts</span>
                  <h2 class="notebook-paper-name">我是{{ name }}</h2>
                  <p class="notebook-paper-role">{{ notebook }} · 算法研究 / 系统构建</p>
                </div>
                <figure class="notebook-paper-photo">
                  <SiteImage :src="academicProfile.avatar" :alt="`${name}的相片`" width="82" height="90" priority />
                </figure>
              </div>
              <p class="notebook-paper-thought">算法建模与底层系统，<br />最终都要落到可复现的代码里。</p>
              <div class="notebook-paper-chips" aria-label="关注的方向">
                <span>Agent &amp; RL</span><span>Multimodal</span><span>AI Infra</span>
              </div>
            </template>

            <template v-else-if="index === 1">
              <div class="notebook-paper-heading">
                <span class="notebook-paper-kicker">interests.ts</span>
                <h2 class="notebook-paper-name">核心方向</h2>
              </div>
              <dl class="notebook-paper-interests">
                <div v-for="(interest, i) in interests" :key="interest.title">
                  <dt><span aria-hidden="true">0{{ i + 1 }}</span>{{ interest.title }}</dt>
                  <dd>{{ interest.note }}</dd>
                </div>
              </dl>
            </template>

            <template v-else>
              <div class="notebook-paper-heading">
                <span class="notebook-paper-kicker">now.ts</span>
                <h2 class="notebook-paper-name">近期动态与下一站</h2>
              </div>
              <div class="notebook-paper-updates">
                <p><span class="notebook-paper-mark">学术与升学</span>已保研录取华东师范大学软件工程（<strong>2027 入学</strong>）；近期工作发表于 <strong>ACL 2026 Demo</strong> 与 <strong>DASFAA 2026 Oral</strong>。</p>
                <p><span class="notebook-paper-mark">工业与开源</span>现于美团担任 Agent 算法工程实习生；曾任西湖大学 NLP 实验室访问学生与北大 IFlab 科研实习生。</p>
              </div>
            </template>

            <p class="notebook-paper-signature">
              <span>{{ notebook }}</span>
              <i aria-hidden="true" />
            </p>
          </section>
        </div>
      </div>

      <nav class="notebook-code-controls" aria-label="翻阅个人手记">
        <div class="notebook-code-pages">
          <button
            v-for="(card, index) in cards"
            :key="card.file"
            type="button"
            class="notebook-code-page"
            :aria-label="`查看${card.label}，第 ${index + 1} 页，共 ${cards.length} 页`"
            :aria-current="index === activeIndex ? 'true' : undefined"
            :disabled="isSwitching"
            @click="showCard(index)"
          >{{ card.tab }}</button>
        </div>
        <button type="button" class="notebook-code-next" :disabled="isSwitching" @click="showCard((activeIndex + 1) % cards.length)">
          翻一页 <span aria-hidden="true">→</span>
        </button>
      </nav>
    </div>
    <span class="notebook-code-status" role="status">{{ cards[activeIndex].label }}，第 {{ activeIndex + 1 }} 页，共 {{ cards.length }} 页</span>
  </aside>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { academicProfile } from '~/data/academic'

defineProps<{ name: string; notebook: string }>()

const cards = [
  { file: 'profile.ts', tab: '手记', label: '个人手记' },
  { file: 'interests.ts', tab: '方向', label: '核心研究与工程方向' },
  { file: 'now.ts', tab: '近况', label: '近期经历与入学计划' },
]
const interests = [
  { title: '智能体与长程规划', note: 'DAG 拓扑调度、两阶段工具路由、100K+ 上下文与 MCP 沙箱。' },
  { title: '后训练对齐与蒸馏', note: 'Verifiable GRPO（verl）、SFT → DPO 偏好对齐与多视角推理蒸馏。' },
  { title: 'AI 推理系统与算子', note: 'SGLang 服务调优、CUDA 融合算子、XGrammar 并发解码与 DiT 加速。' },
]
const activeIndex = ref(0)
const leavingIndex = ref<number | null>(null)
const isSwitching = ref(false)
const prefersReducedMotion = ref(false)
const isPageHidden = ref(false)
let switchTimer: ReturnType<typeof setTimeout> | undefined
let motionQuery: MediaQueryList | undefined

function cardStyle(index: number) {
  const depth = (index - activeIndex.value + cards.length) % cards.length
  return {
    '--card-depth': depth,
    '--card-scale': 1 - depth * 0.02,
    zIndex: index === leavingIndex.value ? cards.length + 1 : cards.length - depth,
  }
}

function finishSwitch() {
  if (switchTimer !== undefined) clearTimeout(switchTimer)
  switchTimer = undefined
  leavingIndex.value = null
  isSwitching.value = false
}

function showCard(index: number) {
  if (isSwitching.value || index === activeIndex.value) return
  if (prefersReducedMotion.value) {
    activeIndex.value = index
    return
  }
  leavingIndex.value = activeIndex.value
  activeIndex.value = index
  isSwitching.value = true
  switchTimer = setTimeout(finishSwitch, 480)
}

function updateMotionPreference() {
  prefersReducedMotion.value = motionQuery?.matches ?? false
  if (prefersReducedMotion.value) finishSwitch()
}

function updateVisibility() {
  isPageHidden.value = document.hidden
}

onMounted(() => {
  motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
  updateMotionPreference()
  updateVisibility()
  motionQuery.addEventListener('change', updateMotionPreference)
  document.addEventListener('visibilitychange', updateVisibility)
})

onUnmounted(() => {
  finishSwitch()
  motionQuery?.removeEventListener('change', updateMotionPreference)
  document.removeEventListener('visibilitychange', updateVisibility)
})
</script>

<style scoped>
.notebook-code-canvas {
  position: relative;
  width: 100%;
  max-width: 436px;
  min-width: 0;
  margin-inline: auto;
}

.notebook-code-shell {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  background: var(--surface);
  box-shadow:
    0 1px 2px rgba(15, 23, 42, 0.04),
    0 24px 60px -32px rgba(15, 23, 42, 0.28);
}

:global(html.dark .notebook-code-shell) {
  box-shadow: 0 1px 2px rgba(2, 6, 23, 0.5), 0 28px 64px -34px rgba(2, 6, 23, 0.95);
}

.notebook-code-tabbar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
  background: var(--bg-2);
}

.notebook-code-dots {
  display: inline-flex;
  gap: 5px;
}

.notebook-code-dots i {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: color-mix(in srgb, var(--line-strong) 85%, transparent);
}

.notebook-code-dots i:first-child { background: color-mix(in srgb, var(--accent) 55%, var(--line-strong)); }

.notebook-code-file {
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.01em;
}

.notebook-code-badge {
  margin-left: auto;
  padding: 1px 7px;
  border: 1px solid var(--line);
  border-radius: 5px;
  background: var(--surface);
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 0.6rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.notebook-code-stage {
  position: relative;
  padding: 18px;
  background: var(--bg-2);
  overflow: clip;
}

.notebook-code-stack {
  position: relative;
  display: grid;
}

.notebook-code-window {
  display: flex;
  flex-direction: column;
  grid-area: 1 / 1;
  position: relative;
  min-width: 0;
  min-height: 330px;
  padding: 24px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  color: var(--text);
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
  transform: translateY(calc(var(--card-depth) * -7px)) scale(var(--card-scale));
  transform-origin: 50% 0%;
  transition:
    transform 460ms cubic-bezier(0.22, 1, 0.36, 1),
    opacity 320ms ease,
    box-shadow 320ms ease;
  pointer-events: none;
}

.notebook-code-window:not(.notebook-code-active) {
  opacity: 0.55;
}

.notebook-code-active {
  pointer-events: auto;
  opacity: 1;
  box-shadow: 0 8px 24px -18px rgba(15, 23, 42, 0.4);
}

.notebook-code-leaving {
  animation: notebook-code-deal 460ms cubic-bezier(0.4, 0, 0.2, 1) both;
}

.notebook-paper-intro {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.notebook-paper-heading { min-width: 0; }

.notebook-paper-kicker {
  display: inline-block;
  margin-bottom: 8px;
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: 0.64rem;
  letter-spacing: 0.04em;
}

.notebook-paper-name {
  margin: 0;
  color: var(--ink);
  font-size: 1.18rem;
  font-weight: 600;
  line-height: 1.4;
  letter-spacing: -0.018em;
}

.notebook-paper-role {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 0.74rem;
  line-height: 1.6;
}

.notebook-paper-photo {
  flex: 0 0 auto;
  width: 82px;
  margin: 0;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--bg-2);
}

.notebook-paper-photo img {
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 1;
  object-fit: cover;
}

.notebook-paper-thought {
  margin: 22px 0 18px;
  color: var(--ink);
  font-size: 1.16rem;
  font-weight: 500;
  line-height: 1.65;
  letter-spacing: -0.012em;
}

.notebook-paper-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.notebook-paper-chips span {
  padding: 4px 10px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--bg-2);
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.66rem;
  line-height: 1.5;
}

.notebook-paper-interests {
  display: grid;
  gap: 14px;
  margin: 20px 0 0;
}

.notebook-paper-interests div {
  padding-bottom: 14px;
  border-bottom: 1px solid var(--line);
}

.notebook-paper-interests div:last-child { border-bottom: 0; padding-bottom: 0; }

.notebook-paper-interests dt {
  display: flex;
  align-items: baseline;
  gap: 8px;
  color: var(--ink);
  font-size: 0.88rem;
  font-weight: 500;
  line-height: 1.6;
}

.notebook-paper-interests dt span {
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 0.64rem;
}

.notebook-paper-interests dd {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.7;
  overflow-wrap: anywhere;
}

.notebook-paper-updates {
  display: grid;
  gap: 16px;
  margin: 20px 0 0;
}

.notebook-paper-updates p {
  margin: 0;
  color: var(--text);
  font-size: 0.82rem;
  line-height: 1.85;
}

.notebook-paper-updates strong { color: var(--ink); font-weight: 600; }

.notebook-paper-mark {
  display: block;
  width: fit-content;
  margin-bottom: 4px;
  padding: 1px 7px;
  border: 1px solid color-mix(in srgb, var(--accent) 22%, var(--line));
  border-radius: 5px;
  background: color-mix(in srgb, var(--accent) 8%, var(--surface));
  color: var(--accent);
  font-size: 0.65rem;
  line-height: 1.7;
}

.notebook-paper-signature {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: auto 0 0;
  padding-top: 20px;
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 0.68rem;
}

.notebook-paper-signature i {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--line-strong), transparent);
}

.notebook-code-controls {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 14px;
  border-top: 1px solid var(--line);
  background: var(--surface);
}

.notebook-code-pages {
  display: flex;
  align-items: center;
  gap: 2px;
}

.notebook-code-page,
.notebook-code-next {
  appearance: none;
  min-height: 30px;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  font: 500 0.74rem/1.4 var(--font-sans);
  transition: color 160ms ease, background-color 160ms ease;
}

.notebook-code-page { min-width: 44px; padding: 5px 10px; }

.notebook-code-page[aria-current="true"] {
  background: var(--surface-2);
  color: var(--ink);
}

.notebook-code-next {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 8px;
  color: var(--accent);
}

.notebook-code-next span { transition: translate 160ms ease; }

.notebook-code-page:hover,
.notebook-code-next:hover { background: var(--surface-2); color: var(--ink); }
.notebook-code-next:hover { color: var(--accent); }
.notebook-code-next:hover span { translate: 2px 0; }

.notebook-code-page:focus-visible,
.notebook-code-next:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.notebook-code-page:disabled,
.notebook-code-next:disabled { cursor: default; }

.notebook-code-status {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
  border: 0;
}

.notebook-code-paused :is(.notebook-code-page, .notebook-code-next) { animation-play-state: paused; }
.notebook-code-reduced :is(.notebook-code-window, .notebook-code-next span) { animation: none; transition: none; }

@keyframes notebook-code-deal {
  0% { transform: translateY(0) scale(1); opacity: 1; }
  100% { transform: translate3d(0, 18px, 0) scale(0.985); opacity: 0; }
}

@media (max-width: 980px) {
  .notebook-code-canvas { max-width: 400px; }
  .notebook-code-window { padding: 20px; min-height: 320px; }
  .notebook-paper-thought { font-size: 1.06rem; }
}

@media (max-width: 420px) {
  .notebook-code-stage { padding: 12px; }
  .notebook-code-window { padding: 18px; min-height: 316px; }
  .notebook-paper-photo { width: 68px; }
  .notebook-paper-thought { font-size: 1rem; }
}

@media (prefers-reduced-motion: reduce) {
  .notebook-code-window,
  .notebook-code-next span { animation: none; transition: none; }
}
</style>
