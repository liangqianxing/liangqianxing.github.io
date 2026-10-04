<template>
  <aside
    class="notebook-code-canvas"
    :class="{ 'notebook-code-paused': isPageHidden, 'notebook-code-reduced': prefersReducedMotion }"
    aria-label="作者的代码名片"
  >
    <div class="notebook-code-glow" aria-hidden="true" />
    <svg class="notebook-code-doodle" width="200" height="100" viewBox="0 0 200 100" fill="none" aria-hidden="true"><path d="M10 80C40 70 80 90 120 70C160 50 180 20 190 10" stroke="currentColor" stroke-width="2" stroke-dasharray="6 6" /></svg>
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
          <div class="notebook-code-header">
            <span class="notebook-code-dots" aria-hidden="true"><i /><i /><i /></span>
            <span>{{ card.file }}</span>
          </div>
          <pre v-if="index === 0" class="notebook-code-profile"><code><span class="notebook-code-keyword">const</span> <span class="notebook-code-variable">enhao</span> = {
  name: <span class="notebook-code-string">"{{ name }}"</span>,
  notebook: <span class="notebook-code-string">"{{ notebook }}"</span>,
  focus: [<span class="notebook-code-string">"LLM"</span>, <span class="notebook-code-string">"Agent"</span>, <span class="notebook-code-string">"Systems"</span>],
  next: <span class="notebook-code-string">"ECNU · 2027 incoming"</span>
};</code></pre>
          <pre v-else-if="index === 1" class="notebook-code-profile"><code><span class="notebook-code-keyword">const</span> <span class="notebook-code-variable">interests</span> = {
  research: [
    <span class="notebook-code-string">"LLM"</span>,
    <span class="notebook-code-string">"Agent Systems"</span>
  ],
  journal: <span class="notebook-code-string">"{{ notebook }}"</span>
};</code></pre>
          <pre v-else class="notebook-code-profile"><code><span class="notebook-code-keyword">const</span> <span class="notebook-code-variable">now</span> = {
  next: <span class="notebook-code-string">"ECNU"</span>,
  entry: <span class="notebook-code-string">"2027 incoming"</span>,
  internship: <span class="notebook-code-string">"Meituan"</span>,
  role: <span class="notebook-code-string">"Full-stack Engineer"</span>
};</code></pre>
        </section>
      </div>
    </div>
    <span class="notebook-code-note" aria-hidden="true">&lt;/&gt;</span>
    <span class="notebook-code-logo" aria-hidden="true"><img src="/logo.svg?v=3" alt="" width="48" height="48" /></span>
    <svg class="notebook-code-star" width="44" height="44" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2 14.5 9.5 22 12 14.5 14.5 12 22 9.5 14.5 2 12 9.5 9.5Z" /></svg>
    <span class="notebook-code-dot" aria-hidden="true" />
    <span class="notebook-code-sticker" aria-hidden="true">LEARN · BUILD · REPEAT</span>
    <nav class="notebook-code-controls" aria-label="切换代码名片">
      <div class="notebook-code-pages">
        <button
          v-for="(card, index) in cards"
          :key="card.file"
          type="button"
          class="notebook-code-page"
          :aria-label="`查看 ${card.file}，第 ${index + 1} 张，共 ${cards.length} 张`"
          :aria-current="index === activeIndex ? 'true' : undefined"
          :disabled="isSwitching"
          @click="showCard(index)"
        ><span aria-hidden="true">{{ String(index + 1).padStart(2, '0') }}</span></button>
      </div>
      <button type="button" class="notebook-code-next" :disabled="isSwitching" @click="showCard((activeIndex + 1) % cards.length)">
        下一张 <span aria-hidden="true">↗</span>
      </button>
    </nav>
    <span class="notebook-code-status" role="status">{{ cards[activeIndex].file }}，第 {{ activeIndex + 1 }} 张，共 {{ cards.length }} 张</span>
  </aside>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

defineProps<{ name: string; notebook: string }>()

const cards = [
  { file: 'profile.ts', label: '个人介绍' },
  { file: 'interests.ts', label: '研究兴趣' },
  { file: 'now.ts', label: '近期经历与入学计划' },
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
  --syntax-keyword: #7550c0;
  --syntax-variable: #c53c66;
  --syntax-string: #08785b;
  --note-bg: #ffe898;
  --note-ink: #b45309;
  --sticker-bg: #f5ebff;
  --sticker-ink: #7550aa;
  --code-height: 276px;
  position: relative;
  min-width: 0;
  margin-inline: 2px 8px;
}
.notebook-code-glow,
.notebook-code-doodle,
.notebook-code-note,
.notebook-code-logo,
.notebook-code-star,
.notebook-code-dot,
.notebook-code-sticker { pointer-events: none; }
.notebook-code-glow {
  position: absolute;
  inset: 0 0 32px;
  background: radial-gradient(ellipse at 60% 30%, #b6a4ff80, transparent 68%), radial-gradient(ellipse at 20% 80%, #9ccfff90, transparent 70%);
  filter: blur(22px);
}
.notebook-code-doodle { position: absolute; top: -8px; left: -26px; z-index: 2; color: var(--accent); opacity: .6; transform: rotate(-12deg); }
.notebook-code-stage { position: relative; padding: 34px 10px 28px; overflow: clip; isolation: isolate; }
.notebook-code-stack { position: relative; height: var(--code-height); perspective: 1000px; }
.notebook-code-window {
  position: absolute;
  inset: 0;
  padding: 22px;
  border: 1px solid color-mix(in srgb, var(--accent) 16%, var(--surface));
  border-radius: 20px;
  background: color-mix(in srgb, var(--accent) calc(var(--card-depth) * 6%), var(--surface));
  color: var(--text);
  box-shadow: 0 12px 24px #0000000b;
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.85;
  transform: translateY(calc(var(--card-depth) * -14px)) rotate(calc(var(--card-depth) * 1.25deg)) scale(var(--card-scale));
  transform-origin: 50% 22%;
  transition: transform 600ms cubic-bezier(.22, 1, .36, 1), background-color 300ms, box-shadow 600ms;
  pointer-events: none;
}
.notebook-code-active { pointer-events: auto; box-shadow: 0 15px 35px #0000000f; }
.notebook-code-leaving { animation: notebook-code-deal 600ms cubic-bezier(.45, 0, .55, 1) both; }
.notebook-code-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 11px; }
.notebook-code-dots { display: flex; gap: 6px; }
.notebook-code-dots i { width: 10px; height: 10px; border-radius: 50%; background: #ff5f56; }
.notebook-code-dots i:nth-child(2) { background: #ffbd2e; }
.notebook-code-dots i:nth-child(3) { background: #27c93f; }
.notebook-code-profile { margin: 0; padding: 0; border: 0; background: transparent; color: var(--text); font: inherit; white-space: pre-wrap; overflow-wrap: anywhere; }
.notebook-code-profile code { display: block; padding: 0; background: transparent; color: inherit; font: inherit; }
.notebook-code-keyword { color: var(--syntax-keyword); }
.notebook-code-variable { color: var(--syntax-variable); }
.notebook-code-string { color: var(--syntax-string); }
.notebook-code-note { position: absolute; top: 0; right: 22px; z-index: 7; display: grid; place-items: center; width: 58px; height: 58px; border-radius: 4px; background: var(--note-bg); color: var(--note-ink); box-shadow: 2px 4px 10px #00000014; transform: rotate(8deg); font-size: 22px; animation: notebook-code-float 7s ease-in-out infinite; }
.notebook-code-logo { position: absolute; right: 22px; bottom: 56px; z-index: 7; width: 64px; height: 64px; padding: 8px; border: 1px solid color-mix(in srgb, var(--note-bg) 45%, var(--line)); border-radius: 16px; background: var(--surface); box-shadow: 0 10px 25px #00000019; transform: rotate(-6deg); animation: notebook-code-float 8s ease-in-out infinite reverse; }
.notebook-code-logo img { width: 100%; height: 100%; }
.notebook-code-star { position: absolute; top: 2px; left: -6px; z-index: 6; color: #f7b827; transform: rotate(8deg); filter: drop-shadow(0 2px 4px #fbbf2433); animation: notebook-code-float 6s ease-in-out infinite; }
.notebook-code-dot { position: absolute; bottom: 36%; left: -7px; z-index: 6; width: 12px; height: 12px; border-radius: 50%; background: #ff9387; }
.notebook-code-sticker { position: absolute; bottom: 66px; left: 18px; z-index: 7; padding: 7px 11px; border: 1px solid color-mix(in srgb, var(--sticker-ink) 18%, var(--surface)); border-radius: 8px; background: var(--sticker-bg); color: var(--sticker-ink); transform: rotate(-4deg); font: 10px var(--font-mono); letter-spacing: .06em; animation: notebook-code-float 9s ease-in-out infinite reverse; }
.notebook-code-controls { position: relative; z-index: 8; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 12px; }
.notebook-code-pages { display: flex; align-items: center; gap: 3px; }
.notebook-code-page,
.notebook-code-next { appearance: none; min-height: 34px; border: 0; border-radius: 8px; background: transparent; color: var(--muted); cursor: pointer; font: 11px var(--font-mono); transition: color 180ms, background-color 180ms; }
.notebook-code-page { min-width: 34px; padding: 7px; }
.notebook-code-page[aria-current="true"] { background: color-mix(in srgb, var(--accent) 12%, var(--surface)); color: var(--accent); }
.notebook-code-next { display: inline-flex; align-items: center; gap: 10px; padding: 7px 9px; color: var(--accent); }
.notebook-code-next span { font-size: 17px; transition: translate 180ms; }
.notebook-code-page:hover,
.notebook-code-next:hover { background: color-mix(in srgb, var(--accent) 9%, var(--surface)); color: var(--accent); }
.notebook-code-next:hover span { translate: 2px -2px; }
.notebook-code-page:focus-visible,
.notebook-code-next:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.notebook-code-page:disabled,
.notebook-code-next:disabled { cursor: default; }
.notebook-code-status { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0; }
.notebook-code-paused :is(.notebook-code-note, .notebook-code-logo, .notebook-code-star, .notebook-code-sticker) { animation-play-state: paused; }
.notebook-code-reduced .notebook-code-window { animation: none; transition: none; }
.notebook-code-reduced :is(.notebook-code-note, .notebook-code-logo, .notebook-code-star, .notebook-code-sticker) { animation: none; }
@keyframes notebook-code-deal {
  0% { transform: translateY(0) rotate(0) scale(1); opacity: 1; }
  45% { opacity: 1; }
  100% { transform: translate3d(54px, 48px, 0) rotate(18deg) scale(.93); opacity: 0; }
}
@keyframes notebook-code-float {
  0%, 100% { translate: 0 0; }
  50% { translate: 0 -6px; }
}
:global(html.dark .notebook-code-canvas) { --syntax-keyword: #c9afff; --syntax-variable: #ff9bb9; --syntax-string: #86e7c5; --note-bg: #d4a847; --note-ink: #382604; --sticker-bg: #37294e; --sticker-ink: #ddc1ff; }
:global(html.cyber .notebook-code-canvas) { --syntax-keyword: #81e5f2; --syntax-variable: #ffa9c7; --syntax-string: #9affc4; --note-bg: #d4a847; --note-ink: #382604; --sticker-bg: #193b32; --sticker-ink: #9affc4; }
@media (min-width: 769px) and (max-width: 1020px) {
  .notebook-code-window { padding: 18px; font-size: 12px; line-height: 1.7; }
  .notebook-code-header { margin-bottom: 12px; padding-bottom: 10px; }
}
@media (max-width: 768px) {
  .notebook-code-canvas { --code-height: 286px; margin-inline: 2px 6px; }
  .notebook-code-stage { padding-bottom: 24px; }
  .notebook-code-window { padding: 16px; font-size: 12px; line-height: 1.7; }
  .notebook-code-header { margin-bottom: 12px; padding-bottom: 10px; }
  .notebook-code-glow { filter: blur(16px); }
  .notebook-code-star { top: 7px; left: -2px; width: 36px; height: 36px; }
  .notebook-code-note { top: 8px; right: 18px; width: 44px; height: 44px; font-size: 18px; }
  .notebook-code-logo { right: 14px; bottom: 54px; width: 48px; height: 48px; padding: 6px; }
  .notebook-code-doodle { display: none; }
  .notebook-code-sticker { bottom: 61px; left: 12px; font-size: 9px; }
}
@media (prefers-reduced-motion: reduce) {
  .notebook-code-window,
  .notebook-code-next span { animation: none; transition: none; }
  .notebook-code-note,
  .notebook-code-logo,
  .notebook-code-star,
  .notebook-code-sticker { animation: none; }
}
</style>
