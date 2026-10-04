<template>
  <div ref="stage" class="notebook-motion" aria-hidden="true">
    <div class="notebook-motion-sky">
      <span class="notebook-motion-cloud notebook-motion-cloud-a" />
      <span class="notebook-motion-cloud notebook-motion-cloud-b" />
      <span class="notebook-motion-aura" />
      <span
        v-for="(star, index) in stars"
        :key="index"
        class="notebook-motion-star"
        :style="{ left: `${star[0]}%`, top: `${star[1]}%`, '--star-delay': `${index * -.71}s`, '--star-size': `${index % 3 === 0 ? 7 : 4}px` }"
      >✦</span>
      <div class="notebook-motion-waves">
        <svg v-for="(wave, index) in waves" :key="index" class="notebook-motion-wave" :class="`notebook-motion-wave-${index}`" viewBox="0 0 2880 100" preserveAspectRatio="none">
          <path :d="wave" fill="currentColor" />
          <path :d="wave" fill="currentColor" transform="translate(1440 0)" />
        </svg>
      </div>
    </div>
    <div class="notebook-motion-particles">
      <span v-for="particle in particles" :key="particle.id" class="notebook-motion-particle" :style="particleStyle(particle)">✦</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { CSSProperties } from 'vue'

type Particle = { id: number; x: number; y: number; dx: number; dy: number; color: string; size: number; spin: number }
const stage = ref<HTMLElement | null>(null)
const particles = ref<Particle[]>([])
const stars = [[5, 19], [12, 66], [19, 8], [27, 87], [35, 28], [43, 72], [51, 14], [59, 91], [65, 39], [73, 9], [79, 76], [87, 32], [93, 64], [96, 17]]
const waves = [
  'M0 54C180 20 330 80 540 54S900 20 1080 54S1320 80 1440 54V100H0Z',
  'M0 60C160 84 350 25 540 60S870 84 1080 60S1300 25 1440 60V100H0Z',
  'M0 68C180 35 370 91 540 68S870 35 1080 68S1320 91 1440 68V100H0Z',
  'M0 76C190 95 330 48 540 76S900 95 1080 76S1320 48 1440 76V100H0Z',
]

function particleStyle(particle: Particle): CSSProperties {
  return {
    left: `${particle.x}px`, top: `${particle.y}px`, color: particle.color,
    fontSize: `${particle.size}px`, '--particle-dx': `${particle.dx}px`,
    '--particle-dy': `${particle.dy}px`, '--particle-spin': `${particle.spin}deg`,
  }
}

let cleanup = () => {}

onMounted(() => {
  const host = stage.value?.parentElement
  if (!host || !stage.value) return

  const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
  const pointerQuery = window.matchMedia('(hover: hover) and (pointer: fine)')
  const hero = host.querySelector<HTMLElement>('.notebook-hero-wrap')
  const cards = [...host.querySelectorAll<HTMLElement>('.knowledge-card')]
  const revealTargets = [...host.querySelectorAll<HTMLElement>(
    '.notebook-intro > *, .notebook-code-canvas, .notebook-topic-head, .knowledge-card, .site-block:not(.notebook-topics)',
  )]
  const colors = ['#a78bfa', '#60a5fa', '#fbbf24', '#fb7185', '#34d399']
  const timers = new Set<ReturnType<typeof setTimeout>>()
  let observer: IntersectionObserver | null = null
  let resizeObserver: ResizeObserver | null = null
  let frame = 0
  let particleId = 0
  let activeCard: HTMLElement | null = null
  let lastPointer: { x: number; y: number; target: Element | null } | null = null
  let disposed = false

  const motionAllowed = () => !reducedQuery.matches && !document.hidden
  const mouseAllowed = () => motionAllowed() && pointerQuery.matches

  const clearParticles = () => {
    for (const timer of timers) clearTimeout(timer)
    timers.clear()
    particles.value = []
  }

  const resetCard = () => {
    if (!activeCard) return
    activeCard.classList.remove('notebook-motion-hover')
    for (const property of ['--motion-tilt-x', '--motion-tilt-y', '--motion-spot-x', '--motion-spot-y']) activeCard.style.removeProperty(property)
    activeCard = null
  }

  const resetPointer = () => {
    if (frame) cancelAnimationFrame(frame)
    frame = 0
    lastPointer = null
    resetCard()
    host.style.removeProperty('--home-glow-opacity')
    host.style.removeProperty('--home-mouse-x')
    host.style.removeProperty('--home-mouse-y')
  }

  const revealAll = () => {
    observer?.disconnect()
    observer = null
    for (const target of revealTargets) {
      target.classList.remove('notebook-motion-pending')
      target.classList.add('notebook-motion-visible')
    }
  }

  const startReveal = () => {
    if (observer || reducedQuery.matches || !('IntersectionObserver' in window)) return
    try {
      observer = new IntersectionObserver((entries) => {
        if (document.hidden) return
        for (const entry of entries) {
          if (!entry.isIntersecting) continue
          entry.target.classList.remove('notebook-motion-pending')
          entry.target.classList.add('notebook-motion-visible')
          observer?.unobserve(entry.target)
        }
      }, { threshold: .08, rootMargin: '0px 0px -18px 0px' })
      revealTargets.forEach((target, index) => {
        if (target.classList.contains('notebook-motion-visible')) return
        const rect = target.getBoundingClientRect()
        // Restored scroll positions should never conceal content above the viewport.
        if (rect.bottom < 0) {
          target.classList.add('notebook-motion-visible')
          return
        }
        target.style.setProperty('--motion-reveal-delay', `${Math.min(index % 5, 4) * 65}ms`)
        target.classList.add('notebook-motion-reveal', 'notebook-motion-pending')
        observer?.observe(target)
      })
    } catch {
      revealAll()
    }
  }

  const updateHeroSize = () => {
    if (hero) host.style.setProperty('--motion-hero-height', `${hero.getBoundingClientRect().height}px`)
  }

  const updatePointer = () => {
    frame = 0
    if (!mouseAllowed() || !lastPointer || disposed) return
    const { x, y, target } = lastPointer
    const card = target?.closest<HTMLElement>('.knowledge-card')
    const nextCard = card && host.contains(card) ? card : null
    if (nextCard !== activeCard) {
      resetCard()
      activeCard = nextCard
    }
    if (activeCard) {
      const rect = activeCard.getBoundingClientRect()
      const localX = Math.max(0, Math.min(rect.width, x - rect.left))
      const localY = Math.max(0, Math.min(rect.height, y - rect.top))
      activeCard.classList.add('notebook-motion-hover')
      activeCard.style.setProperty('--motion-spot-x', `${localX}px`)
      activeCard.style.setProperty('--motion-spot-y', `${localY}px`)
      activeCard.style.setProperty('--motion-tilt-x', `${((.5 - localY / Math.max(rect.height, 1)) * 8).toFixed(2)}deg`)
      activeCard.style.setProperty('--motion-tilt-y', `${((localX / Math.max(rect.width, 1) - .5) * 8).toFixed(2)}deg`)
    }
    if (hero) {
      const rect = hero.getBoundingClientRect()
      const withinHero = y >= rect.top && y <= rect.bottom && x >= rect.left && x <= rect.right
      host.style.setProperty('--home-glow-opacity', withinHero ? '.7' : '0')
      if (withinHero) {
        const hostRect = host.getBoundingClientRect()
        host.style.setProperty('--home-mouse-x', `${x - hostRect.left}px`)
        host.style.setProperty('--home-mouse-y', `${y - hostRect.top}px`)
      }
    }
  }

  const onPointerMove = (event: PointerEvent) => {
    if (!mouseAllowed() || event.pointerType === 'touch') return
    lastPointer = { x: event.clientX, y: event.clientY, target: event.target instanceof Element ? event.target : null }
    if (!frame) frame = requestAnimationFrame(updatePointer)
  }

  const onPointerOut = (event: PointerEvent) => {
    const next = event.relatedTarget
    if (!(next instanceof Node) || !host.contains(next)) resetPointer()
    else if (activeCard && !activeCard.contains(next)) resetCard()
  }

  let pointerStart: { x: number; y: number; id: number } | null = null
  const onPointerDown = (event: PointerEvent) => {
    pointerStart = event.isPrimary && event.button === 0 ? { x: event.clientX, y: event.clientY, id: event.pointerId } : null
  }
  const onPointerCancel = () => { pointerStart = null }

  const onPointerUp = (event: PointerEvent) => {
    const start = pointerStart
    pointerStart = null
    if (!mouseAllowed() || event.pointerType === 'touch' || !start || start.id !== event.pointerId || !event.isPrimary || event.button !== 0) return
    if (Math.hypot(event.clientX - start.x, event.clientY - start.y) > 9) return
    const target = event.target instanceof Element ? event.target : null
    if (target?.closest('input, textarea, select, option, button, [contenteditable], [role="textbox"], pre, code')) return
    const selection = window.getSelection()
    if (selection && !selection.isCollapsed) return
    const available = Math.min(8, 40 - particles.value.length)
    if (available <= 0) return
    const rect = host.getBoundingClientRect()
    const ids: number[] = []
    for (let index = 0; index < available; index++) {
      const angle = index / available * Math.PI * 2 + Math.random() * .35
      const distance = 24 + Math.random() * 32
      const id = ++particleId
      ids.push(id)
      particles.value.push({ id, x: event.clientX - rect.left, y: event.clientY - rect.top,
        dx: Math.cos(angle) * distance, dy: Math.sin(angle) * distance - 14,
        color: colors[index % colors.length]!, size: 8 + Math.random() * 7, spin: Math.random() * 150 - 75 })
    }
    const timer = setTimeout(() => {
      const completed = new Set(ids)
      particles.value = particles.value.filter(particle => !completed.has(particle.id))
      timers.delete(timer)
    }, 850)
    timers.add(timer)
  }

  const applyPreferences = () => {
    host.classList.toggle('notebook-motion-reduced', reducedQuery.matches)
    host.classList.toggle('notebook-motion-paused', document.hidden)
    host.classList.toggle('notebook-motion-fine', pointerQuery.matches && !reducedQuery.matches)
    if (!mouseAllowed()) resetPointer()
    if (!motionAllowed()) {
      clearParticles()
      pointerStart = null
    }
    if (document.hidden) {
      observer?.disconnect()
      observer = null
    }
    if (reducedQuery.matches) revealAll()
    else if (!document.hidden) startReveal()
  }

  cards.forEach(card => card.classList.add('notebook-motion-card'))
  updateHeroSize()
  if ('ResizeObserver' in window && hero) {
    resizeObserver = new ResizeObserver(updateHeroSize)
    resizeObserver.observe(hero)
  }
  applyPreferences()
  host.addEventListener('pointermove', onPointerMove, { passive: true })
  host.addEventListener('pointerout', onPointerOut, { passive: true })
  host.addEventListener('pointerleave', resetPointer, { passive: true })
  host.addEventListener('pointerdown', onPointerDown, { passive: true })
  host.addEventListener('pointerup', onPointerUp, { passive: true })
  host.addEventListener('pointercancel', onPointerCancel, { passive: true })
  reducedQuery.addEventListener('change', applyPreferences)
  pointerQuery.addEventListener('change', applyPreferences)
  document.addEventListener('visibilitychange', applyPreferences)
  window.addEventListener('resize', updateHeroSize, { passive: true })

  cleanup = () => {
    disposed = true
    resetPointer()
    clearParticles()
    observer?.disconnect()
    resizeObserver?.disconnect()
    host.removeEventListener('pointermove', onPointerMove)
    host.removeEventListener('pointerout', onPointerOut)
    host.removeEventListener('pointerleave', resetPointer)
    host.removeEventListener('pointerdown', onPointerDown)
    host.removeEventListener('pointerup', onPointerUp)
    host.removeEventListener('pointercancel', onPointerCancel)
    reducedQuery.removeEventListener('change', applyPreferences)
    pointerQuery.removeEventListener('change', applyPreferences)
    document.removeEventListener('visibilitychange', applyPreferences)
    window.removeEventListener('resize', updateHeroSize)
    host.classList.remove('notebook-motion-reduced', 'notebook-motion-paused', 'notebook-motion-fine')
    host.style.removeProperty('--motion-hero-height')
    revealTargets.forEach(target => {
      target.classList.remove('notebook-motion-reveal', 'notebook-motion-pending', 'notebook-motion-visible')
      target.style.removeProperty('--motion-reveal-delay')
    })
    cards.forEach(card => card.classList.remove('notebook-motion-card', 'notebook-motion-hover'))
  }
})

onUnmounted(() => cleanup())
</script>

<style scoped>
:global(.journal-home) { position: relative; isolation: isolate; }
:global(.journal-home .notebook-hero) { position: relative; z-index: 1; }
:global(.journal-home .notebook-motion) { position: absolute; inset: 0; pointer-events: none; }
:global(.journal-home .notebook-motion-sky) { position: absolute; inset: 0 0 auto; z-index: 0; height: var(--motion-hero-height, 430px); overflow: hidden; pointer-events: none; }
:global(.journal-home .notebook-motion-cloud) { position: absolute; width: 380px; height: 300px; max-width: 60vw; border-radius: 50%; opacity: .5; background: radial-gradient(ellipse, color-mix(in srgb, var(--accent) 20%, transparent), transparent 70%); animation: notebook-cloud-drift 16s ease-in-out infinite alternate; }
:global(.journal-home .notebook-motion-cloud-a) { left: 1%; top: -65px; }
:global(.journal-home .notebook-motion-cloud-b) { right: -25px; bottom: -10px; width: 420px; color: var(--accent-2, var(--accent)); background: radial-gradient(ellipse, color-mix(in srgb, var(--accent-2, var(--accent)) 18%, transparent), transparent 70%); animation-delay: -8s; animation-direction: alternate-reverse; }
:global(.journal-home .notebook-motion-aura) { position: absolute; left: 0; top: 0; width: 260px; height: 260px; border-radius: 50%; opacity: var(--home-glow-opacity, 0); transform: translate3d(calc(var(--home-mouse-x, 0px) - 50%), calc(var(--home-mouse-y, 0px) - 50%), 0); background: radial-gradient(circle, color-mix(in srgb, var(--accent) 15%, transparent), transparent 65%); transition: opacity 280ms ease; }
:global(.journal-home .notebook-motion-star) { position: absolute; color: var(--accent); font-size: var(--star-size); line-height: 1; opacity: .22; animation: notebook-star-glimmer 5s ease-in-out var(--star-delay) infinite alternate; }
:global(.journal-home .notebook-motion-waves) { position: absolute; inset: auto 0 -1px; height: 50px; overflow: hidden; }
:global(.journal-home .notebook-motion-wave) { position: absolute; inset: 0 auto 0 0; width: 200%; height: 100%; animation: notebook-wave-flow 22s linear infinite; color: color-mix(in srgb, var(--accent) 11%, transparent); }
:global(.journal-home .notebook-motion-wave-1) { color: color-mix(in srgb, var(--accent-2, var(--accent)) 14%, transparent); animation-duration: 18s; animation-direction: reverse; }
:global(.journal-home .notebook-motion-wave-2) { color: color-mix(in srgb, var(--accent) 7%, var(--bg)); animation-duration: 27s; }
:global(.journal-home .notebook-motion-wave-3) { color: var(--bg); animation-duration: 16s; animation-direction: reverse; }
:global(.journal-home .notebook-motion-particles) { position: absolute; inset: 0; z-index: 20; overflow: hidden; pointer-events: none; }
:global(.journal-home .notebook-motion-particle) { position: absolute; line-height: 1; animation: notebook-particle-pop 780ms cubic-bezier(.16, 1, .3, 1) forwards; }
:global(.journal-home .notebook-motion-reveal) { transition: opacity 700ms cubic-bezier(.16, 1, .3, 1), translate 700ms cubic-bezier(.16, 1, .3, 1); transition-delay: var(--motion-reveal-delay, 0ms); }
:global(.journal-home .notebook-motion-pending) { opacity: 0; translate: 0 20px; }
:global(.journal-home .notebook-motion-visible) { opacity: 1; translate: 0 0; }
:global(.journal-home .knowledge-card.notebook-motion-card) { transform-style: preserve-3d; transition: transform 180ms ease, box-shadow 180ms ease, opacity 700ms cubic-bezier(.16, 1, .3, 1), translate 700ms cubic-bezier(.16, 1, .3, 1); transition-delay: 0ms, 0ms, var(--motion-reveal-delay, 0ms), var(--motion-reveal-delay, 0ms); }
:global(.journal-home .knowledge-card.notebook-motion-card::before) { display: block; content: ''; position: absolute; inset: 0; z-index: 0; opacity: 0; pointer-events: none; background: radial-gradient(circle 150px at var(--motion-spot-x, 50%) var(--motion-spot-y, 50%), color-mix(in srgb, var(--topic-ink) 18%, transparent), transparent 75%); transition: opacity 180ms ease; }
:global(.journal-home.notebook-motion-fine .knowledge-card.notebook-motion-card.notebook-motion-hover) { transform: perspective(800px) rotateX(var(--motion-tilt-x, 0deg)) rotateY(var(--motion-tilt-y, 0deg)) translateY(-5px); box-shadow: 0 14px 30px color-mix(in srgb, var(--topic-ink) 13%, transparent); }
:global(.journal-home.notebook-motion-fine .knowledge-card.notebook-motion-card.notebook-motion-hover::before) { opacity: 1; }
:global(.journal-home.notebook-motion-paused *),
:global(.journal-home.notebook-motion-paused *::before),
:global(.journal-home.notebook-motion-paused *::after) { animation-play-state: paused !important; }
:global(.journal-home.notebook-motion-reduced .notebook-motion-aura),
:global(.journal-home.notebook-motion-reduced .notebook-motion-particles) { display: none; }
:global(.journal-home.notebook-motion-reduced *),
:global(.journal-home.notebook-motion-reduced *::before),
:global(.journal-home.notebook-motion-reduced *::after) { animation: none !important; transition: none !important; }
:global(.journal-home.notebook-motion-reduced .notebook-motion-pending) { opacity: 1; translate: none; }
@keyframes notebook-wave-flow { to { transform: translateX(-50%); } }
@keyframes notebook-cloud-drift { to { transform: translate3d(35px, 15px, 0) scale(1.1); } }
@keyframes notebook-star-glimmer { to { opacity: .48; translate: 0 -7px; rotate: 22deg; } }
@keyframes notebook-particle-pop { 0% { opacity: 0; transform: translate(-50%, -50%) scale(.2); } 12% { opacity: .9; } 100% { opacity: 0; transform: translate(calc(-50% + var(--particle-dx)), calc(-50% + var(--particle-dy))) rotate(var(--particle-spin)) scale(.25); } }
@media (max-width: 760px) {
  :global(.journal-home .notebook-motion-star:nth-of-type(2n)) { display: none; }
  :global(.journal-home .notebook-motion-cloud) { width: 260px; height: 220px; opacity: .35; }
  :global(.journal-home .notebook-motion-waves) { height: 36px; }
}
@media (prefers-reduced-motion: reduce) {
  :global(.journal-home .notebook-motion *) { animation: none !important; transition: none !important; }
  :global(.journal-home .notebook-motion-pending) { opacity: 1; translate: none; }
  :global(.journal-home .knowledge-card.notebook-motion-card) { transform: none !important; }
}
</style>
