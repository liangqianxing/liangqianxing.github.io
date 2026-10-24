<template>
  <div class="effect-stage" aria-hidden="true">
    <span ref="auraRef" class="cursor-aura" />
    <span class="ambient-sweep ambient-sweep-a" />
    <span class="ambient-sweep ambient-sweep-b" />
    <span class="grain-layer" />
  </div>
</template>

<script setup lang="ts">
const route = useRoute()
const auraRef = ref<HTMLElement | null>(null)

const interactiveSelectors = [
  '.article-card',
  '.friend-card',
  '.post-nav-item',
  '.link-callout',
  '.learning-preview-card',
  '.learning-course-row',
  '.learning-note-card',
  '.tag-index-note',
  '.archive-entry',
].join(',')

const REVEAL_SELECTOR = [
  '.site-block',
  '.page-hero',
  '.writing-welcome',
  '.writing-route',
  '.writing-find',
  '.archive-year',
  '.learning-directory-head',
  '.learning-course-row',
  '.learning-welcome',
  '.learning-section-nav',
  '.learning-section',
  '.tag-index-intro',
  '.tag-index-controls',
  '.tag-index-note',
  '.tag-detail-intro',
  '.tag-detail-related',
  '.about-welcome',
  '.about-next-stop',
  '.about-section',
  '.about-tools',
  '.about-signoff',
  '.friend-card',
  '.link-callout',
  '.material-reader-head',
  '.material-mobile-toc',
  '.post-header',
  '.post-cover',
  '.post-inline-toc',
  '.post-series',
  '.post-nav',
].join(',')

let observePageTargets = () => {}

watch(() => route.fullPath, async () => {
  await nextTick()
  window.requestAnimationFrame(() => observePageTargets())
})

onMounted(() => {
  const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
  const pointerQuery = window.matchMedia('(hover: hover) and (pointer: fine)')
  const root = document.documentElement
  const aura = auraRef.value
  let frame = 0
  let observer: IntersectionObserver | null = null
  let activeSurface: HTMLElement | null = null
  let lastPointer: { x: number; y: number; target: EventTarget | null } | null = null

  root.classList.add('effects-ready')

  const setScrollState = () => {
    root.classList.toggle('is-scrolled', window.scrollY > 18)
  }

  const resetSurface = (surface = activeSurface) => {
    if (!surface) return
    surface.style.removeProperty('--spot-x')
    surface.style.removeProperty('--spot-y')
    if (surface === activeSurface) {
      activeSurface = null
      root.classList.remove('is-hovering-surface')
    }
  }

  observePageTargets = () => {
    observer?.disconnect()
    observer = null
    resetSurface()

    const targets = Array.from(document.querySelectorAll<HTMLElement>(REVEAL_SELECTOR))
      .filter(el => !el.closest('.journal-home'))

    if (reducedQuery.matches || !('IntersectionObserver' in window)) {
      for (const target of targets) {
        target.classList.remove('reveal-item')
        target.classList.add('is-visible')
      }
      return
    }

    observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue
        entry.target.classList.add('is-visible')
        observer?.unobserve(entry.target)
      }
    }, {
      rootMargin: '0px 0px -6% 0px',
      threshold: 0,
    })

    let staggerIndex = 0
    for (const target of targets) {
      if (target.classList.contains('is-visible')) continue
      const rect = target.getBoundingClientRect()
      if (rect.bottom <= 0 || rect.top < window.innerHeight * 0.92) {
        target.classList.remove('reveal-item')
        target.classList.add('is-visible')
        continue
      }
      target.classList.add('reveal-item')
      target.style.setProperty('--reveal-delay', `${(staggerIndex++ % 4) * 35}ms`)
      observer.observe(target)
    }
  }

  const flushPointerFrame = () => {
    frame = 0
    if (!lastPointer || !pointerQuery.matches || reducedQuery.matches || !aura) return
    const { x, y, target } = lastPointer

    aura.style.setProperty('--cursor-x', `${x}px`)
    aura.style.setProperty('--cursor-y', `${y}px`)

    const candidate = (target instanceof Element ? target : null)?.closest<HTMLElement>(interactiveSelectors) ?? null
    const nextSurface = candidate && !candidate.closest('.journal-home') ? candidate : null
    if (nextSurface !== activeSurface) {
      resetSurface(activeSurface)
      activeSurface = nextSurface
    }

    if (!activeSurface) {
      root.classList.remove('is-hovering-surface')
      return
    }

    const rect = activeSurface.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return
    const localX = x - rect.left
    const localY = y - rect.top

    root.classList.add('is-hovering-surface')
    activeSurface.style.setProperty('--spot-x', `${localX.toFixed(1)}px`)
    activeSurface.style.setProperty('--spot-y', `${localY.toFixed(1)}px`)
  }

  const onPointerMove = (event: PointerEvent) => {
    if (!pointerQuery.matches || reducedQuery.matches || !aura || event.pointerType === 'touch') return
    lastPointer = { x: event.clientX, y: event.clientY, target: event.target }
    if (!frame) {
      frame = window.requestAnimationFrame(flushPointerFrame)
    }
  }

  const onPointerOut = (event: PointerEvent) => {
    if (!activeSurface) return
    const related = event.relatedTarget as Node | null
    if (!related || !activeSurface.contains(related)) {
      resetSurface(activeSurface)
    }
  }

  const onMotionChange = () => {
    if (reducedQuery.matches) {
      if (frame) window.cancelAnimationFrame(frame)
      frame = 0
      resetSurface()
    }
    observePageTargets()
  }

  setScrollState()
  nextTick(() => observePageTargets())
  window.addEventListener('scroll', setScrollState, { passive: true })
  window.addEventListener('pointermove', onPointerMove, { passive: true })
  window.addEventListener('pointerout', onPointerOut, { passive: true })
  reducedQuery.addEventListener('change', onMotionChange)

  onUnmounted(() => {
    observePageTargets = () => {}
    resetSurface()
    root.classList.remove('effects-ready', 'is-scrolled', 'is-hovering-surface')
    window.removeEventListener('scroll', setScrollState)
    window.removeEventListener('pointermove', onPointerMove)
    window.removeEventListener('pointerout', onPointerOut)
    reducedQuery.removeEventListener('change', onMotionChange)
    if (frame) window.cancelAnimationFrame(frame)
    observer?.disconnect()
  })
})
</script>
