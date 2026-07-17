<template>
  <div class="effect-stage" aria-hidden="true">
    <span class="cursor-aura" />
    <span class="ambient-sweep ambient-sweep-a" />
    <span class="ambient-sweep ambient-sweep-b" />
    <span class="grain-layer" />
  </div>
</template>

<script setup lang="ts">
const interactiveSelectors = [
  '.article-card',
  '.knowledge-card',
  '.friend-card',
  '.timeline-card',
  '.focus-card',
  '.about-profile',
  '.about-snapshot',
  '.now-card',
  '.stack-panel',
  '.post-nav-item',
  '.link-callout',
].join(',')

onMounted(() => {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches
  const root = document.documentElement
  const aura = document.querySelector<HTMLElement>('.cursor-aura')
  let frame = 0

  root.classList.add('effects-ready')

  const setScrollState = () => {
    root.classList.toggle('is-scrolled', window.scrollY > 18)
  }

  const pageRevealTargets = Array.from(document.querySelectorAll<HTMLElement>([
    '.hero-main > *',
    '.hero-dossier',
    '.site-block',
    '.page-hero',
    '.archive-directory',
    '.archive-panel',
    '.filter-row',
    '.year-section',
    '.about-hero',
    '.about-profile',
    '.about-snapshot',
    '.focus-card',
    '.stack-panel',
    '.now-card',
    '.friend-card',
    '.timeline-card',
    '.post-header',
  ].join(',')))
  const proseRoot = document.querySelector<HTMLElement>('.prose')
  const proseChildren = proseRoot?.children.length === 1
    ? proseRoot.firstElementChild?.children
    : proseRoot?.children
  const revealTargets = [
    ...pageRevealTargets,
    ...Array.from(proseChildren ?? []),
  ] as HTMLElement[]

  let observer: IntersectionObserver | null = null
  if (!reduceMotion) {
    observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue
        entry.target.classList.add('is-visible')
        observer?.unobserve(entry.target)
      }
    }, {
      rootMargin: '0px 0px -9% 0px',
      threshold: 0.08,
    })

    revealTargets.forEach((target, index) => {
      target.classList.add('reveal-item')
      target.style.setProperty('--reveal-delay', `${Math.min(index % 8, 6) * 45}ms`)
      observer?.observe(target)
    })
  } else {
    revealTargets.forEach((target) => target.classList.add('is-visible'))
  }

  const onPointerMove = (event: PointerEvent) => {
    if (!finePointer || reduceMotion || !aura) return

    const x = event.clientX
    const y = event.clientY

    if (!frame) {
      frame = window.requestAnimationFrame(() => {
        root.style.setProperty('--cursor-x', `${x}px`)
        root.style.setProperty('--cursor-y', `${y}px`)
        frame = 0
      })
    }

    const target = (event.target as HTMLElement | null)?.closest<HTMLElement>(interactiveSelectors)
    if (!target) {
      root.classList.remove('is-hovering-surface')
      return
    }

    const rect = target.getBoundingClientRect()
    const localX = x - rect.left
    const localY = y - rect.top
    const rotateX = ((rect.height / 2 - localY) / rect.height) * 4
    const rotateY = ((localX - rect.width / 2) / rect.width) * 4

    root.classList.add('is-hovering-surface')
    target.style.setProperty('--spot-x', `${localX}px`)
    target.style.setProperty('--spot-y', `${localY}px`)
    target.style.setProperty('--tilt-x', `${rotateX.toFixed(2)}deg`)
    target.style.setProperty('--tilt-y', `${rotateY.toFixed(2)}deg`)
  }

  const onPointerOut = (event: PointerEvent) => {
    const target = (event.target as HTMLElement | null)?.closest<HTMLElement>(interactiveSelectors)
    if (!target || target.contains(event.relatedTarget as Node | null)) return
    root.classList.remove('is-hovering-surface')
    target.style.removeProperty('--tilt-x')
    target.style.removeProperty('--tilt-y')
  }

  setScrollState()
  window.addEventListener('scroll', setScrollState, { passive: true })
  window.addEventListener('pointermove', onPointerMove, { passive: true })
  window.addEventListener('pointerout', onPointerOut, { passive: true })

  onUnmounted(() => {
    root.classList.remove('effects-ready', 'is-scrolled', 'is-hovering-surface')
    window.removeEventListener('scroll', setScrollState)
    window.removeEventListener('pointermove', onPointerMove)
    window.removeEventListener('pointerout', onPointerOut)
    if (frame) window.cancelAnimationFrame(frame)
    observer?.disconnect()
  })
})
</script>
