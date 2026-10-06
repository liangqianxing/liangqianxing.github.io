<template>
  <div class="post-author-card author-note">
    <p class="author-note-kicker">
      写字的人
      <svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <path d="M16 15C5 17 6 5 12 7c4 1 4 5 4 8Zm1 1C15 5 27 6 25 12c-1 4-5 4-8 4Zm-1 1c11-2 10 10 4 8-4-1-4-5-4-8Zm-1-1c2 11-10 10-8 4 1-4 5-4 8-4Z" />
        <circle cx="16" cy="16" r="2" />
      </svg>
    </p>

    <NuxtLink to="/about" class="author-note-identity">
      <span class="author-note-photo" aria-hidden="true">
        <SiteImage src="/avatar.jpg" alt="" width="58" height="58" />
      </span>
      <span class="author-note-name">
        <strong>{{ appConfig.authorCN }}</strong>
        <small>{{ appConfig.authorEN }}</small>
      </span>
    </NuxtLink>

    <p class="author-note-role"><span v-for="(line, index) in roleLines" :key="index">{{ line }}</span></p>
    <div class="author-note-focus" aria-label="关注领域">
      <span>LLM</span>
      <span>Agent</span>
      <span>Backend</span>
    </div>
    <p class="author-note-signoff">随手记录，慢慢探索。</p>

    <div class="author-note-reading">
      <div class="author-note-reading-label">
        <a href="#article-start" aria-label="跳到文章正文">读到这里 <span aria-hidden="true">↓</span></a>
        <span aria-hidden="true">{{ progress }}%</span>
      </div>
      <div class="author-note-meter">
        <input
          class="author-note-slider"
          type="range"
          min="0"
          max="100"
          step="1"
          :value="progress"
          :disabled="!seekable"
          aria-label="文章阅读进度，拖动滑页"
          :aria-valuetext="`已读 ${progress}%`"
          @input="onSeek"
        />
        <svg viewBox="0 0 180 24" fill="none" aria-hidden="true">
          <path class="author-note-trail" d="M6 14Q90-1 174 14" />
          <path class="author-note-trail-fill" :d="readingTrail.path" />
          <g :transform="`translate(${readingTrail.x} ${readingTrail.y})`">
            <path class="author-note-star" d="M0-6Q1-1 6 0Q1 1 0 6Q-1 1-6 0Q-1-1 0-6Z" />
          </g>
        </svg>
      </div>
      <p class="author-note-reading-hint">拖动星星，快速翻页</p>
    </div>

    <div class="author-note-links">
      <NuxtLink to="/about">认识一下 <span aria-hidden="true">→</span></NuxtLink>
      <a :href="appConfig.github" target="_blank" rel="noopener noreferrer" :aria-label="`${appConfig.authorCN} 的 GitHub`">
        GitHub <span aria-hidden="true">↗</span>
      </a>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{ readingProgress: number; seekable: boolean }>()
const emit = defineEmits<{ seek: [progress: number] }>()
const appConfig = useAppConfig()
const roleLines = computed(() => appConfig.role.split(' · '))
const progress = computed(() => props.readingProgress)
const readingTrail = computed(() => {
  const t = progress.value / 100
  const x = 6 + 168 * t
  const y = 14 - 30 * t * (1 - t)
  return { x, y, path: `M6 14Q${6 + 84 * t} ${14 - 15 * t} ${x} ${y}` }
})

function onSeek(event: Event) {
  emit('seek', (event.target as HTMLInputElement).valueAsNumber)
}
</script>

<style scoped>
.author-note {
  --note-purple-ink: #755b92;
  --note-green-ink: #557867;
  --note-peach-ink: #986b50;
  position: relative;
  padding: 18px 18px 8px;
  border: 1px solid color-mix(in srgb, var(--line) 78%, transparent);
  border-radius: 8px 20px 18px 22px;
  background: radial-gradient(ellipse at 0 0, color-mix(in srgb, var(--topic-orange) 40%, transparent), transparent 70%), var(--surface);
  box-shadow: 0 5px 14px #33264008;
  animation: author-note-arrive 420ms ease-out both;
}
html.dark .author-note { --note-purple-ink: #d3c3ec; --note-green-ink: #afcdbf; --note-peach-ink: #e3b79c; }
html.cyber .author-note { --note-purple-ink: #bddbc8; --note-green-ink: #abe3c5; --note-peach-ink: #dfd7a8; }
.author-note-kicker {
  display: flex; align-items: center; justify-content: space-between;
  margin: 0 0 18px; color: var(--muted); font-size: 12px; letter-spacing: .08em;
}
.author-note-kicker svg { width: 23px; height: 23px; stroke: var(--note-peach-ink); stroke-width: 1.2; rotate: -12deg; transition: rotate 350ms ease; }
.author-note-identity { display: grid; grid-template-columns: 64px minmax(0, 1fr); align-items: center; gap: 12px; min-height: 68px; }
.author-note-photo {
  position: relative; display: block; width: 64px; padding: 3px 3px 7px;
  border-radius: 3px; background: var(--bg); box-shadow: 0 3px 7px #33264012;
  transform: rotate(-6deg); transition: transform 250ms ease;
}
.author-note-photo::before {
  content: ''; position: absolute; z-index: 1; width: 36px; height: 12px; top: -5px; left: 14px;
  background: color-mix(in srgb, var(--accent) 22%, transparent); rotate: 4deg;
  clip-path: polygon(0 0, 100% 6%, 96% 100%, 3% 92%);
}
.author-note-photo img { display: block; width: 58px; height: 58px; border-radius: 1px; object-fit: cover; }
.author-note-name { display: grid; min-width: 0; gap: 5px; }
.author-note-name strong { color: var(--ink); font-size: 18px; line-height: 1.4; font-weight: 500; }
.author-note-name small { color: var(--accent); font: italic 15px/1.45 Georgia, serif; }
.author-note-role { margin: 17px 0 12px; color: var(--muted); font-size: 12px; line-height: 1.85; overflow-wrap: anywhere; }
.author-note-role span { display: block; }
.author-note-focus { display: flex; flex-wrap: wrap; gap: 6px; padding: 2px 0; }
.author-note-focus span { display: block; padding: 4px 8px; border-radius: 7px 9px 6px 8px; font-size: 11px; line-height: 1.5; transition: rotate 200ms ease; }
.author-note-focus span:nth-child(1) { background: var(--topic-purple); color: var(--note-purple-ink); rotate: -4deg; }
.author-note-focus span:nth-child(2) { background: var(--topic-green); color: var(--note-green-ink); rotate: 3deg; }
.author-note-focus span:nth-child(3) { background: var(--topic-orange); color: var(--note-peach-ink); rotate: -2deg; }
.author-note-signoff { margin: 17px 0 20px; color: var(--text); font-size: 13px; line-height: 1.7; }
.author-note-reading-label { display: flex; justify-content: space-between; gap: 12px; align-items: center; font-size: 11px; color: var(--muted); }
.author-note-reading-label a { display: inline-flex; align-items: center; gap: 6px; min-height: 28px; color: inherit; }
.author-note-reading-label a:hover { color: var(--accent); }
.author-note-reading-label > span { font-variant-numeric: tabular-nums; }
.author-note-meter { position: relative; display: grid; align-items: center; height: 44px; }
.author-note-slider { position: absolute; inset: 0; z-index: 1; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: grab; }
.author-note-slider:active { cursor: grabbing; }
.author-note-slider:disabled { cursor: default; }
.author-note-meter svg { display: block; width: 100%; height: 24px; overflow: visible; pointer-events: none; }
.author-note-slider:focus-visible + svg { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 8px; }
.author-note-reading-hint { margin: -2px 0 5px; color: var(--muted); font-size: 10px; line-height: 1.6; }
.author-note-trail, .author-note-trail-fill { stroke-width: 1.4; stroke-linecap: round; }
.author-note-trail { stroke: var(--line-strong); }
.author-note-trail-fill { stroke: var(--accent); }
.author-note-star { fill: var(--accent); stroke: var(--surface); stroke-width: 1.5; }
.author-note-links { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-top: 5px; }
.author-note-links a { display: inline-flex; align-items: center; gap: 5px; min-height: 44px; color: var(--muted); font-size: 12px; font-weight: 400; }
.author-note-links a span { transition: transform 180ms ease; }
.author-note-links a:hover { color: var(--accent); }
.author-note-links a:hover span { transform: translate(2px, -1px); }
.author-note:focus-within .author-note-photo { transform: rotate(-2deg) translateY(-2px); }
@media (hover: hover) and (pointer: fine) {
  .author-note:hover .author-note-kicker svg { rotate: 16deg; }
  .author-note-identity:hover .author-note-photo { transform: rotate(-2deg) translateY(-2px); }
  .author-note-focus span:hover { rotate: 0deg; }
}
@keyframes author-note-arrive { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
@media (prefers-reduced-motion: reduce) {
  .author-note, .author-note *, .author-note *::before { animation: none; transition: none; }
  .author-note:focus-within .author-note-photo, .author-note-identity:hover .author-note-photo { transform: rotate(-6deg); }
  .author-note:hover .author-note-kicker svg { rotate: -12deg; }
}
</style>
