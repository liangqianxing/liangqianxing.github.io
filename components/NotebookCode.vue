<template>
  <aside
    class="notebook-code-canvas"
    :class="{ 'notebook-code-paused': isPageHidden, 'notebook-code-reduced': prefersReducedMotion }"
    aria-label="作者的三页手记"
  >
    <div class="notebook-code-glow" aria-hidden="true" />
    <svg class="notebook-code-doodle" width="150" height="92" viewBox="0 0 150 92" fill="none" aria-hidden="true">
      <path d="M5 72C25 88 60 74 64 47C68 20 35 25 44 43C53 61 112 63 142 16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" />
    </svg>
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
              <div>
                <p class="notebook-paper-heading">hello, world.</p>
                <h2 class="notebook-paper-name">我是{{ name }}</h2>
              </div>
              <figure class="notebook-paper-photo">
                <span class="notebook-paper-tape" aria-hidden="true" />
                <SiteImage :src="academicProfile.avatar" :alt="`${name}的相片`" width="82" height="90" priority />
              </figure>
            </div>
            <p class="notebook-paper-thought">喜欢研究，<br />也喜欢把想法做出来。</p>
            <div class="notebook-paper-chips" aria-label="关注的方向">
              <span>LLM</span><span>Agent</span><span>Systems</span>
            </div>
          </template>
          <template v-else-if="index === 1">
            <p class="notebook-paper-heading">things I explore.</p>
            <h2 class="notebook-paper-name">好奇的方向</h2>
            <dl class="notebook-paper-interests">
              <div v-for="interest in interests" :key="interest.title">
                <dt>{{ interest.title }}</dt>
                <dd>{{ interest.note }}</dd>
              </div>
            </dl>
          </template>
          <template v-else>
            <p class="notebook-paper-heading">a little update.</p>
            <h2 class="notebook-paper-name">最近，慢慢向前</h2>
            <div class="notebook-paper-updates">
              <p><span class="notebook-paper-mark notebook-paper-mark-lilac">下一站</span>已保研录取华东师范大学软件工程，<strong>2027 年入学</strong>。</p>
              <p><span class="notebook-paper-mark notebook-paper-mark-peach">现在</span>在美团做开发实习，把想法放进实际的工程里。</p>
            </div>
          </template>
          <p class="notebook-paper-signature"><span>{{ notebook }}</span><svg width="36" height="13" viewBox="0 0 36 13" fill="none" aria-hidden="true"><path d="M1 8C8 0 14 2 13 8C12 13 20 10 22 4C24 0 20 2 22 7C24 12 30 7 35 8" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" /></svg></p>
        </section>
      </div>
    </div>
    <span class="notebook-code-logo" aria-hidden="true"><img src="/logo.svg?v=3" alt="" width="48" height="48" /></span>
    <svg class="notebook-code-star" width="38" height="38" viewBox="0 0 38 38" fill="none" aria-hidden="true">
      <path d="M19 4C22 4 24 8 22 12C26 9 30 10 31 13C32 16 30 19 26 19C30 21 31 25 29 27C27 30 23 29 21 25C21 30 18 33 15 31C12 30 12 26 15 23C10 25 6 23 7 20C7 17 11 15 15 17C12 13 13 8 16 9C18 9 19 11 19 14C17 10 16 5 19 4Z" fill="currentColor" />
      <circle cx="19" cy="19" r="3.2" fill="var(--paper)" />
    </svg>
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
        翻一页 <span aria-hidden="true">↗</span>
      </button>
    </nav>
    <span class="notebook-code-status" role="status">{{ cards[activeIndex].label }}，第 {{ activeIndex + 1 }} 页，共 {{ cards.length }} 页</span>
  </aside>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { academicProfile } from '~/data/academic'

defineProps<{ name: string; notebook: string }>()

const cards = [
  { file: 'profile.ts', tab: '手记', label: '个人手记' },
  { file: 'interests.ts', tab: '探索', label: '好奇的方向' },
  { file: 'now.ts', tab: '近况', label: '近期经历与入学计划' },
]
const interests = [
  { title: '大语言模型', note: '理解模型，也试着让它解决实际问题。' },
  { title: '智能体', note: '探索工具、记忆与上下文如何配合。' },
  { title: 'AI 系统', note: '把模型与服务连接起来，跑得更稳。' },
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
    '--card-scale': 1 - depth * 0.035,
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
  switchTimer = setTimeout(finishSwitch, 620)
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
  --paper: #fffdf8;
  --paper-behind: #f1ecfa;
  --paper-ink: #4e485d;
  --paper-muted: #766d86;
  --paper-line: #e6dccb44;
  --paper-lilac: #eee5fa;
  --paper-peach: #fce7dc;
  --paper-sky: #e6f2fb;
  --paper-shadow: #6d598814;
  --paper-control: #785997;
  position: relative;
  width: 100%;
  max-width: 420px;
  min-width: 0;
  margin-inline: auto;
}
.notebook-code-glow,
.notebook-code-doodle,
.notebook-code-logo,
.notebook-code-star { pointer-events: none; }
.notebook-code-glow {
  position: absolute;
  inset: 10% -3% 18%;
  border-radius: 50%;
  background: radial-gradient(ellipse at 65% 35%, #dbc6f84a, transparent 70%), radial-gradient(ellipse at 22% 78%, #ffe5d856, transparent 72%);
  filter: blur(24px);
}
.notebook-code-doodle { position: absolute; top: -12px; left: -20px; z-index: 2; color: var(--paper-control); opacity: .32; transform: rotate(-9deg); }
.notebook-code-stage { position: relative; padding: 28px 12px 20px; overflow: clip; isolation: isolate; }
.notebook-code-stack { display: grid; position: relative; perspective: 1000px; }
.notebook-code-window {
  display: flex;
  flex-direction: column;
  grid-area: 1 / 1;
  position: relative;
  min-width: 0;
  min-height: 316px;
  padding: 28px 28px 22px;
  border: 1px solid color-mix(in srgb, var(--paper-muted) 11%, var(--paper));
  border-radius: 24px 38px 30px 20px;
  background-color: color-mix(in srgb, var(--paper-behind) calc(var(--card-depth) * 15%), var(--paper));
  background-image: repeating-linear-gradient(transparent 0 31px, var(--paper-line) 31px 32px);
  background-position: 0 10px;
  color: var(--paper-ink);
  box-shadow: 0 8px 24px var(--paper-shadow);
  font-family: var(--font-sans);
  transform: translateY(calc(var(--card-depth) * -10px)) rotate(calc(var(--card-depth) * 2deg - 1deg)) scale(var(--card-scale));
  transform-origin: 50% 22%;
  transition: transform 600ms cubic-bezier(.22, 1, .36, 1), background-color 300ms, box-shadow 600ms;
  pointer-events: none;
}
.notebook-code-active { pointer-events: auto; box-shadow: 0 14px 40px var(--paper-shadow); }
.notebook-code-leaving { animation: notebook-code-deal 600ms cubic-bezier(.45, 0, .55, 1) both; }
.notebook-paper-intro { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; }
.notebook-paper-heading { margin: 0 0 10px; color: var(--paper-control); font: italic 28px / 1.2 Georgia, 'Times New Roman', serif; letter-spacing: -.025em; }
.notebook-paper-name { margin: 0; color: var(--paper-ink); font-size: 14px; line-height: 1.8; font-weight: 500; }
.notebook-paper-photo { position: relative; flex: 0 0 auto; width: 90px; margin: 0 0 0 auto; padding: 5px 5px 14px; border-radius: 2px; background: var(--paper); box-shadow: 0 3px 13px var(--paper-shadow); transform: rotate(6deg); }
.notebook-paper-photo img { display: block; width: 80px; height: 86px; border-radius: 1px; object-fit: cover; }
.notebook-paper-tape { position: absolute; top: -10px; left: 21px; width: 47px; height: 20px; background: color-mix(in srgb, var(--paper-lilac) 80%, transparent); clip-path: polygon(3% 0, 96% 0, 100% 15%, 96% 28%, 100% 43%, 96% 62%, 100% 78%, 96% 100%, 3% 100%, 0 86%, 3% 72%, 0 56%, 3% 39%, 0 22%); transform: rotate(-9deg); }
.notebook-paper-thought { margin: 22px 0 18px; color: var(--paper-ink); font-size: 21px; line-height: 1.7; letter-spacing: .025em; }
.notebook-paper-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.notebook-paper-chips span { padding: 5px 12px; border-radius: 50px; background: var(--paper-lilac); color: var(--paper-control); font-size: 11px; line-height: 1.4; }
.notebook-paper-chips span:nth-child(2) { background: var(--paper-peach); }
.notebook-paper-chips span:nth-child(3) { background: var(--paper-sky); }
.notebook-paper-interests { display: grid; gap: 14px; margin: 21px 0 20px; }
.notebook-paper-interests div { position: relative; padding-left: 15px; }
.notebook-paper-interests div::before { position: absolute; top: 7px; left: 0; width: 6px; height: 6px; border-radius: 50%; background: var(--paper-control); opacity: .5; content: ''; }
.notebook-paper-interests dt { color: var(--paper-ink); font-size: 13px; line-height: 1.6; font-weight: 600; }
.notebook-paper-interests dd { margin: 3px 0 0; color: var(--paper-muted); font-size: 12px; line-height: 1.7; overflow-wrap: anywhere; }
.notebook-paper-updates { display: grid; gap: 18px; margin: 24px 0 22px; }
.notebook-paper-updates p { margin: 0; color: var(--paper-ink); font-size: 14px; line-height: 1.95; }
.notebook-paper-updates strong { font-weight: 500; }
.notebook-paper-mark { display: table; margin-bottom: 5px; padding: 1px 8px; border-radius: 5px 10px 4px 8px; color: var(--paper-control); font-size: 11px; line-height: 1.8; }
.notebook-paper-mark-lilac { background: var(--paper-lilac); }
.notebook-paper-mark-peach { background: var(--paper-peach); }
.notebook-paper-signature { display: flex; align-items: center; gap: 10px; margin: auto 0 0; padding-top: 18px; color: var(--paper-muted); font: italic 13px / 1.4 Georgia, 'Times New Roman', serif; }
.notebook-code-logo { position: absolute; right: 1px; bottom: 52px; z-index: 7; width: 54px; height: 54px; padding: 7px; border: 1px solid color-mix(in srgb, var(--paper-muted) 12%, var(--paper)); border-radius: 16px 20px 15px 18px; background: var(--paper); box-shadow: 0 5px 16px var(--paper-shadow); transform: rotate(7deg); animation: notebook-code-float 8s ease-in-out infinite reverse; }
.notebook-code-logo img { width: 100%; height: 100%; }
.notebook-code-star { position: absolute; top: 2px; right: -2px; z-index: 6; color: #eab89f; transform: rotate(8deg); animation: notebook-code-float 7s ease-in-out infinite; }
.notebook-code-controls { position: relative; z-index: 8; display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 0 12px; }
.notebook-code-pages { display: flex; align-items: center; gap: 4px; padding: 3px; border-radius: 50px; background: color-mix(in srgb, var(--paper) 65%, transparent); }
.notebook-code-page,
.notebook-code-next { appearance: none; min-height: 36px; border: 0; border-radius: 50px; background: transparent; color: var(--paper-muted); cursor: pointer; font: 12px / 1.4 var(--font-sans); transition: color 180ms, background-color 180ms; }
.notebook-code-page { min-width: 49px; padding: 7px 11px; }
.notebook-code-page[aria-current="true"] { background: var(--paper-lilac); color: var(--paper-control); }
.notebook-code-next { display: inline-flex; align-items: center; gap: 8px; padding: 7px 8px; color: var(--paper-control); }
.notebook-code-next span { font-size: 17px; transition: translate 180ms; }
.notebook-code-page:hover,
.notebook-code-next:hover { background: color-mix(in srgb, var(--paper-lilac) 60%, transparent); color: var(--paper-control); }
.notebook-code-next:hover span { translate: 2px -2px; }
.notebook-code-page:focus-visible,
.notebook-code-next:focus-visible { outline: 2px solid var(--paper-control); outline-offset: 3px; }
.notebook-code-page:disabled,
.notebook-code-next:disabled { cursor: default; }
.notebook-code-status { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0; }
.notebook-code-paused :is(.notebook-code-logo, .notebook-code-star) { animation-play-state: paused; }
.notebook-code-reduced .notebook-code-window { animation: none; transition: none; }
.notebook-code-reduced :is(.notebook-code-logo, .notebook-code-star) { animation: none; }
@keyframes notebook-code-deal {
  0% { transform: translateY(0) rotate(-1deg) scale(1); opacity: 1; }
  45% { opacity: 1; }
  100% { transform: translate3d(36px, 42px, 0) rotate(13deg) scale(.95); opacity: 0; }
}
@keyframes notebook-code-float {
  0%, 100% { translate: 0 0; }
  50% { translate: 0 -4px; }
}
:global(html.dark .notebook-code-canvas) { --paper: #2b2935; --paper-behind: #373042; --paper-ink: #ede4f1; --paper-muted: #b9abc8; --paper-line: #a28aae12; --paper-lilac: #443550; --paper-peach: #4b3736; --paper-sky: #2f4251; --paper-shadow: #07030f2b; --paper-control: #d5bde8; }
:global(html.cyber .notebook-code-canvas) { --paper: #20362f; --paper-behind: #2e423a; --paper-ink: #e6eee5; --paper-muted: #adbfad; --paper-line: #abbe9f10; --paper-lilac: #3d4b42; --paper-peach: #49433a; --paper-sky: #2c4649; --paper-shadow: #03170e2b; --paper-control: #c1d9b8; }
@media (min-width: 769px) and (max-width: 1020px) {
  .notebook-code-window { padding: 25px 22px 20px; }
  .notebook-paper-heading { font-size: 25px; }
  .notebook-paper-photo { width: 77px; padding: 4px 4px 12px; }
  .notebook-paper-photo img { width: 69px; height: 75px; }
  .notebook-paper-tape { left: 15px; }
  .notebook-paper-thought { font-size: 20px; }
}
@media (max-width: 768px) {
  .notebook-code-canvas { max-width: 390px; }
  .notebook-code-stage { padding-inline: 10px; }
  .notebook-code-window { min-height: 314px; padding: 26px 24px 22px; }
  .notebook-code-glow { filter: blur(18px); }
  .notebook-code-doodle { left: -7px; width: 115px; }
  .notebook-code-star { right: 2px; width: 32px; height: 32px; }
  .notebook-code-logo { right: 3px; bottom: 54px; width: 48px; height: 48px; padding: 6px; }
}
@media (max-width: 380px) {
  .notebook-code-window { padding: 24px 19px 20px; }
  .notebook-paper-heading { font-size: 25px; }
  .notebook-paper-photo { width: 68px; padding: 4px 4px 12px; }
  .notebook-paper-photo img { width: 60px; height: 66px; }
  .notebook-paper-tape { left: 12px; width: 40px; }
  .notebook-paper-thought { font-size: 19px; }
  .notebook-paper-chips { gap: 6px; }
  .notebook-paper-chips span { padding-inline: 10px; }
  .notebook-code-controls { padding-inline: 9px; gap: 4px; }
  .notebook-code-page { min-width: 44px; padding-inline: 9px; }
  .notebook-code-next { gap: 5px; padding-inline: 5px; }
}
@media (prefers-reduced-motion: reduce) {
  .notebook-code-window,
  .notebook-code-next span { animation: none; transition: none; }
  .notebook-code-logo,
  .notebook-code-star { animation: none; }
}
</style>
