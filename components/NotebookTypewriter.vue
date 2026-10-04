<template>
  <p class="notebook-typewriter" :class="{ 'is-paused': paused }">
    <span class="sr-only">写代码、读论文，探索语言模型与系统。</span>
    <span class="notebook-typewriter-visual" aria-hidden="true">
      <span class="notebook-typewriter-prompt">&gt;</span>
      <span>{{ displayed }}</span><span class="notebook-typewriter-caret" />
    </span>
  </p>
</template>

<script setup lang="ts">
const phrases = ['写代码，也读论文。', '探索 LLM、Agent 与系统。', '把想法写成能运行的东西。']
const displayed = ref(phrases[0])
const paused = ref(false)
let timer: number | undefined
let query: MediaQueryList | undefined
let phraseIndex = 0
let length = 0
let phase: 'typing' | 'holding' | 'deleting' = 'typing'

function stop() {
  if (timer !== undefined) window.clearTimeout(timer)
  timer = undefined
}

function schedule(delay: number) {
  stop()
  if (!query?.matches && !document.hidden) timer = window.setTimeout(advance, delay)
}

function advance() {
  const letters = Array.from(phrases[phraseIndex])
  if (phase === 'holding') {
    phase = 'deleting'
  } else if (phase === 'typing') {
    length = Math.min(length + 1, letters.length)
    displayed.value = letters.slice(0, length).join('')
    if (length === letters.length) {
      phase = 'holding'
      schedule(2400)
      return
    }
  } else {
    length = Math.max(0, length - 1)
    displayed.value = letters.slice(0, length).join('')
    if (!length) {
      phraseIndex = (phraseIndex + 1) % phrases.length
      phase = 'typing'
      schedule(350)
      return
    }
  }
  schedule(phase === 'deleting' ? 42 : 85)
}

function onPreferenceChange() {
  stop()
  if (query?.matches) {
    displayed.value = phrases[0]
  } else {
    phraseIndex = 0
    length = Array.from(phrases[0]).length
    displayed.value = phrases[0]
    phase = 'holding'
    schedule(2400)
  }
}

function onVisibilityChange() {
  paused.value = document.hidden
  if (document.hidden) stop()
  else schedule(phase === 'holding' ? 2400 : 85)
}

onMounted(() => {
  query = window.matchMedia('(prefers-reduced-motion: reduce)')
  query.addEventListener('change', onPreferenceChange)
  document.addEventListener('visibilitychange', onVisibilityChange)
  paused.value = document.hidden
  if (!query.matches) {
    displayed.value = ''
    schedule(280)
  }
})

onBeforeUnmount(() => {
  stop()
  query?.removeEventListener('change', onPreferenceChange)
  document.removeEventListener('visibilitychange', onVisibilityChange)
})
</script>

<style scoped>
.notebook-typewriter { min-height: 26px; margin: 14px 0 0; color: var(--accent); font-size: 13px; line-height: 26px; }
.notebook-typewriter-visual { display: inline-flex; align-items: center; }
.notebook-typewriter-prompt { margin-right: 9px; color: var(--notebook-highlight); font-family: var(--font-mono); }
.notebook-typewriter-caret { display: inline-block; width: 2px; height: 14px; margin-left: 4px; background: currentColor; animation: notebook-caret-blink 1s steps(1) infinite; }
.is-paused .notebook-typewriter-caret { animation-play-state: paused; }
@keyframes notebook-caret-blink { 0%, 45% { opacity: 1; } 46%, 100% { opacity: 0; } }
@media (prefers-reduced-motion: reduce) { .notebook-typewriter-caret { animation: none; opacity: 1; } }
</style>
