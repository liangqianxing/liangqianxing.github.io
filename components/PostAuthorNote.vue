<template>
  <div class="post-author-card author-note">
    <p class="author-note-kicker">
      写字的人
      <span aria-hidden="true">{{ appConfig.authorEN }}</span>
    </p>

    <NuxtLink to="/about" class="author-note-identity">
      <span class="author-note-photo" aria-hidden="true">
        <SiteImage src="/avatar.jpg" alt="" width="56" height="56" />
      </span>
      <span class="author-note-name">
        <strong>{{ appConfig.authorCN }}</strong>
        <small>{{ appConfig.authorEN }}</small>
      </span>
    </NuxtLink>

    <p class="author-note-role"><span v-for="(line, index) in roleLines" :key="index">{{ line }}</span></p>
    <div class="author-note-focus" aria-label="关注领域">
      <span>Agent &amp; RL</span>
      <span>AI Infra</span>
      <span>Multimodal</span>
    </div>
    <p class="author-note-signoff">严谨建模，用数据验证系统假设。</p>

    <div class="author-note-reading">
      <div class="author-note-reading-label">
        <a href="#article-start" aria-label="跳到文章正文">读到这里 <span aria-hidden="true">↓</span></a>
        <span aria-hidden="true">{{ progress }}%</span>
      </div>
      <div class="author-note-meter">
        <div class="author-note-meter-track" aria-hidden="true">
          <span class="author-note-meter-fill" :style="{ width: `${progress}%` }" />
        </div>
        <input
          class="author-note-slider"
          type="range"
          min="0"
          max="100"
          step="1"
          :value="progress"
          :disabled="!seekable"
          aria-label="文章阅读进度"
          :aria-valuetext="`已读 ${progress}%`"
          @input="onSeek"
        />
      </div>
      <p class="author-note-reading-hint">拖动进度条，快速翻页</p>
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

function onSeek(event: Event) {
  emit('seek', (event.target as HTMLInputElement).valueAsNumber)
}
</script>

<style scoped>
.author-note {
  position: relative;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow-soft);
  animation: author-note-arrive 380ms ease-out both;
}

.author-note-kicker {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin: 0 0 16px;
  color: var(--muted);
  font-size: 0.68rem;
  font-weight: 500;
  letter-spacing: 0.04em;
}

.author-note-kicker > span {
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 0;
  text-transform: uppercase;
}

.author-note-identity {
  display: grid;
  grid-template-columns: 56px minmax(0, 1fr);
  align-items: center;
  gap: 12px;
}

.author-note-photo {
  display: block;
  width: 56px;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--bg-2);
  transition: border-color 200ms ease;
}

.author-note-photo img {
  display: block;
  width: 100%;
  height: 100%;
  aspect-ratio: 1;
  object-fit: cover;
}

.author-note-name { display: grid; min-width: 0; gap: 2px; }

.author-note-name strong {
  color: var(--ink);
  font-size: 0.96rem;
  line-height: 1.4;
  font-weight: 600;
}

.author-note-name small {
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 0.68rem;
}

.author-note-role {
  margin: 16px 0 12px;
  color: var(--muted);
  font-size: 0.74rem;
  line-height: 1.8;
  overflow-wrap: anywhere;
}

.author-note-role span { display: block; }

.author-note-focus { display: flex; flex-wrap: wrap; gap: 5px; padding: 0; }

.author-note-focus span {
  display: block;
  padding: 3px 8px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--bg-2);
  color: var(--muted);
  font-size: 0.66rem;
  line-height: 1.6;
}

.author-note-signoff {
  margin: 16px 0 18px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
  color: var(--text);
  font-size: 0.76rem;
  line-height: 1.7;
}

.author-note-reading-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.66rem;
}

.author-note-reading-label a {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 26px;
  color: inherit;
}

.author-note-reading-label a:hover { color: var(--accent); }
.author-note-reading-label > span { font-variant-numeric: tabular-nums; }

.author-note-meter {
  position: relative;
  display: flex;
  align-items: center;
  height: 22px;
}

.author-note-meter-track {
  position: relative;
  width: 100%;
  height: 3px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--surface-2);
}

.author-note-meter-fill {
  position: absolute;
  inset: 0 auto 0 0;
  border-radius: 999px;
  background: var(--accent);
  transition: width 120ms linear;
}

.author-note-slider {
  position: absolute;
  inset: 0;
  z-index: 1;
  width: 100%;
  height: 22px;
  margin: 0;
  opacity: 0;
  cursor: grab;
}

.author-note-slider:active { cursor: grabbing; }
.author-note-slider:disabled { cursor: default; }

.author-note-slider:focus-visible + .author-note-meter-track,
.author-note-meter:focus-within .author-note-meter-track {
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent);
}

.author-note-reading-hint {
  margin: 2px 0 0;
  color: var(--faint);
  font-size: 0.64rem;
  line-height: 1.6;
}

.author-note-links {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
}

.author-note-links a {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 40px;
  color: var(--muted);
  font-size: 0.74rem;
  font-weight: 500;
}

.author-note-links a span { transition: transform 160ms ease; }
.author-note-links a:hover { color: var(--accent); }
.author-note-links a:hover span { transform: translate(2px, -1px); }
.author-note-identity:hover .author-note-photo { border-color: color-mix(in srgb, var(--accent) 40%, var(--line)); }

@keyframes author-note-arrive {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
  .author-note,
  .author-note * { animation: none; transition: none; }
}
</style>
