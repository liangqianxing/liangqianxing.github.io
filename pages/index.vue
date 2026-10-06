<template>
  <div
    ref="scene"
    class="gateway-scene"
    :class="{
      'is-enhanced': enhanced,
      'is-revealing': phase === 'revealing',
      'is-entered': phase === 'entered',
      'is-paused': paused,
    }"
  >
    <GatewayBackdrop :revealed="phase !== 'opening'" />
    <a class="portal-skip" href="#gateway-choices" @click="enter">跳到主页入口</a>
    <header class="portal-header">
      <span class="portal-wordmark">{{ academicProfile.name }}<span class="portal-wordmark-dot" aria-hidden="true">.</span></span>
      <nav aria-label="直接访问">
        <NuxtLink to="/academic">学术</NuxtLink>
        <NuxtLink to="/blog">博客</NuxtLink>
        <a v-if="academicProfile.github" :href="academicProfile.github" target="_blank" rel="noopener noreferrer">GitHub <span aria-hidden="true">↗</span></a>
      </nav>
    </header>

    <main class="portal-stage">
      <section
        class="portal-opening"
        aria-labelledby="opening-name"
        :aria-hidden="enhanced && phase !== 'opening' ? true : undefined"
        :inert="enhanced && phase !== 'opening'"
        @animationend="onCurtainEnd"
      >
        <div class="portal-opening-content">
          <p class="portal-kicker">A LITTLE SPACE FOR CURIOSITY</p>
          <h1 id="opening-name" class="portal-neon"><span>{{ academicProfile.name }}</span></h1>
          <p class="portal-chinese-name">{{ academicProfile.nameCN }}</p>
          <p class="portal-subtitle" aria-label="LLM · Agent · Systems">
            <span aria-hidden="true">
              <span v-for="(letter, index) in subtitle" :key="index" class="portal-letter" :style="{ '--letter-delay': `${550 + index * 50}ms` }">{{ letter === ' ' ? '\u00a0' : letter }}</span>
            </span>
          </p>
          <a ref="enterLink" class="portal-enter" href="#gateway-choices" @click="enter">
            <span>进入主页</span>
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 5v14m-5-5 5 5 5-5" /></svg>
          </a>
          <p class="portal-opening-caption">研究、工程，以及沿途的记录。</p>
        </div>
        <div class="portal-scroll-cue">
          <p>向下滚动 · 上滑进入</p>
          <div class="portal-scroll-arrows" aria-hidden="true"><span /><span /></div>
        </div>
        <svg class="portal-curtain-edge" viewBox="0 0 1440 240" preserveAspectRatio="none" aria-hidden="true">
          <path d="M0 0H1440V22Q720 380 0 22Z" />
        </svg>
      </section>

      <section
        id="gateway-choices"
        class="portal-choices"
        aria-labelledby="choices-title"
        :aria-hidden="enhanced && phase !== 'entered' ? true : undefined"
        :inert="enhanced && phase !== 'entered'"
      >
        <div class="portal-choice-content">
          <header class="portal-identity">
            <img :src="academicProfile.avatar" :alt="academicProfile.nameCN" width="88" height="88" />
            <p class="portal-kicker">WELCOME TO MY SPACE</p>
            <h2 id="choices-title" ref="choicesTitle" tabindex="-1">{{ academicProfile.name }}<span>{{ academicProfile.nameCN }}</span></h2>
            <p class="portal-identity-summary">研究、工程，以及沿途的记录。</p>
          </header>
          <nav class="portal-destinations" aria-label="选择访问内容">
            <NuxtLink class="portal-card portal-card-academic" to="/academic">
              <div class="portal-card-top">
                <svg class="portal-card-icon" viewBox="0 0 48 48" fill="none" aria-hidden="true">
                  <path d="m6 18 18-9 18 9-18 9-18-9Zm7 4v11c7 7 15 7 22 0V22M42 19v13" />
                </svg>
                <span class="portal-card-number" aria-hidden="true">01 /</span>
              </div>
              <span class="portal-card-label">ACADEMIC PROFILE</span>
              <h3>学术主页</h3>
              <p>研究兴趣、学术经历，<br />以及正在探索的方向。</p>
              <span class="portal-card-action">进入学术主页 <span aria-hidden="true">↗</span></span>
            </NuxtLink>
            <NuxtLink class="portal-card portal-card-blog" to="/blog">
              <div class="portal-card-top">
                <svg class="portal-card-icon" viewBox="0 0 48 48" fill="none" aria-hidden="true">
                  <path d="M10 8h25a3 3 0 0 1 3 3v28H12a4 4 0 0 1-4-4V12a4 4 0 0 1 4-4ZM14 8v31m8-23-4 4 4 4m8-8 4 4-4 4M8 34h30" />
                </svg>
                <span class="portal-card-number" aria-hidden="true">02 /</span>
              </div>
              <span class="portal-card-label">PERSONAL BLOG</span>
              <h3>个人博客</h3>
              <p>技术实践、阅读笔记，<br />以及比赛与生活的片段。</p>
              <span class="portal-card-action">进入博客 <span aria-hidden="true">↗</span></span>
            </NuxtLink>
          </nav>
          <footer class="portal-footer">
            <span>&copy; {{ year }} {{ academicProfile.name }}</span>
            <button v-if="enhanced && !reducedMotion" type="button" @click="replay">重播开场 <span aria-hidden="true">↺</span></button>
          </footer>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { academicProfile } from '~/data/academic'

definePageMeta({ layout: 'gateway' })

const scene = ref<HTMLElement | null>(null)
const enterLink = ref<HTMLAnchorElement | null>(null)
const choicesTitle = ref<HTMLElement | null>(null)
const subtitle = [...'LLM · Agent · Systems']
const year = new Date().getFullYear()
const enhanced = ref(false)
const reducedMotion = ref(false)
const paused = ref(false)
const phase = ref<'opening' | 'revealing' | 'entered'>('opening')
let motionQuery: MediaQueryList | undefined
let revealTimer: ReturnType<typeof setTimeout> | undefined
let disposed = false
let wheelDistance = 0
let lastWheelTime = 0
let touchOrigin: { id: number; x: number; y: number } | undefined

function resetGesture() {
  wheelDistance = 0
  lastWheelTime = 0
  touchOrigin = undefined
}

async function finishReveal(focus = true) {
  if (disposed || phase.value === 'entered') return
  clearTimeout(revealTimer)
  resetGesture()
  phase.value = 'entered'
  await nextTick()
  if (!disposed && focus) choicesTitle.value?.focus({ preventScroll: true })
}

function enter(event: MouseEvent) {
  if (!enhanced.value || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
  event.preventDefault()
  if (phase.value === 'entered') {
    choicesTitle.value?.scrollIntoView({ behavior: 'instant', block: 'center' })
    choicesTitle.value?.focus({ preventScroll: true })
    return
  }
  startReveal()
}

function startReveal() {
  if (disposed || !enhanced.value || phase.value !== 'opening') return
  resetGesture()
  if (reducedMotion.value) { void finishReveal(); return }
  phase.value = 'revealing'
  revealTimer = setTimeout(() => void finishReveal(), 1250)
}

function onWheel(event: WheelEvent) {
  if (!enhanced.value || phase.value === 'entered' || event.ctrlKey || event.defaultPrevented) return
  if (event.deltaY <= 0 || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
    wheelDistance = 0
    return
  }
  if (event.cancelable) event.preventDefault()
  if (phase.value !== 'opening') return
  const now = performance.now()
  const unit = event.deltaMode === 1 ? 24 : event.deltaMode === 2 ? window.innerHeight : 1
  wheelDistance = (now - lastWheelTime > 350 ? 0 : wheelDistance) + event.deltaY * unit
  lastWheelTime = now
  if (wheelDistance >= 48) startReveal()
}

function onTouchStart(event: TouchEvent) {
  touchOrigin = undefined
  if (!enhanced.value || phase.value !== 'opening' || event.touches.length !== 1) return
  if (event.target instanceof Element && event.target.closest('.portal-header, .portal-skip, button, input, textarea, select, [contenteditable]')) return
  const touch = event.touches[0]!
  touchOrigin = { id: touch.identifier, x: touch.clientX, y: touch.clientY }
}

function onTouchMove(event: TouchEvent) {
  if (event.touches.length !== 1) { touchOrigin = undefined; return }
  if (!enhanced.value || phase.value === 'entered') return
  if (phase.value === 'revealing') {
    if (event.cancelable) event.preventDefault()
    return
  }
  const touch = event.touches[0]!
  if (!touchOrigin || touch.identifier !== touchOrigin.id) return
  const distance = touchOrigin.y - touch.clientY
  const sideways = Math.abs(touch.clientX - touchOrigin.x)
  if (distance >= 48 && distance > sideways * 1.25) {
    if (event.cancelable) event.preventDefault()
    startReveal()
  }
}

function onTouchEnd() {
  touchOrigin = undefined
}

function onKeyDown(event: KeyboardEvent) {
  if (!enhanced.value || phase.value === 'entered' || event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return
  const target = event.target instanceof Element ? event.target : null
  if (target?.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])')) return
  const space = event.key === ' '
  if (space && target?.closest('a, button, [role="button"]')) return
  if (event.key !== 'ArrowDown' && event.key !== 'PageDown' && !space) return
  event.preventDefault()
  startReveal()
}

function onCurtainEnd(event: AnimationEvent) {
  if (event.target === event.currentTarget && event.animationName === 'portal-curtain-rise' && phase.value === 'revealing') void finishReveal()
}

async function replay() {
  clearTimeout(revealTimer)
  resetGesture()
  phase.value = 'opening'
  await nextTick()
  if (!disposed) {
    scene.value?.scrollIntoView({ behavior: 'instant', block: 'start' })
    enterLink.value?.focus({ preventScroll: true })
  }
}

function onMotionChange() {
  reducedMotion.value = motionQuery?.matches ?? false
  if (reducedMotion.value) {
    const activeInOpening = !!document.activeElement?.closest('.portal-opening')
    void finishReveal(activeInOpening || phase.value === 'revealing')
  }
}

function onVisibilityChange() {
  paused.value = document.hidden
  if (paused.value && phase.value === 'revealing') void finishReveal(false)
}

onMounted(() => {
  motionQuery = matchMedia('(prefers-reduced-motion: reduce)')
  reducedMotion.value = motionQuery.matches
  paused.value = document.hidden
  if (reducedMotion.value || location.hash === '#gateway-choices') phase.value = 'entered'
  enhanced.value = true
  motionQuery.addEventListener('change', onMotionChange)
  document.addEventListener('visibilitychange', onVisibilityChange)
  document.addEventListener('keydown', onKeyDown)
  scene.value?.addEventListener('wheel', onWheel, { passive: false })
  scene.value?.addEventListener('touchstart', onTouchStart, { passive: true })
  scene.value?.addEventListener('touchmove', onTouchMove, { passive: false })
  scene.value?.addEventListener('touchend', onTouchEnd)
  scene.value?.addEventListener('touchcancel', onTouchEnd)
})

onBeforeUnmount(() => {
  disposed = true
  clearTimeout(revealTimer)
  resetGesture()
  motionQuery?.removeEventListener('change', onMotionChange)
  document.removeEventListener('visibilitychange', onVisibilityChange)
  document.removeEventListener('keydown', onKeyDown)
  scene.value?.removeEventListener('wheel', onWheel)
  scene.value?.removeEventListener('touchstart', onTouchStart)
  scene.value?.removeEventListener('touchmove', onTouchMove)
  scene.value?.removeEventListener('touchend', onTouchEnd)
  scene.value?.removeEventListener('touchcancel', onTouchEnd)
})

const appConfig = useAppConfig()
const description = `${academicProfile.nameCN}（${academicProfile.name}）的个人主页，包含学术主页与个人博客。`
useHead({
  title: '个人主页',
  meta: [
    { name: 'description', content: description },
    { property: 'og:title', content: `${academicProfile.name} · ${academicProfile.nameCN}` },
    { property: 'og:description', content: description },
    { property: 'og:url', content: `${appConfig.url}/` },
    { property: 'og:image', content: `${appConfig.url}${academicProfile.avatar}` },
  ],
})
</script>
