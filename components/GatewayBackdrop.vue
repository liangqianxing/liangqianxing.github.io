<template>
  <div ref="stage" class="gateway-backdrop" :class="{ 'is-revealed': revealed, 'is-paused': hidden, 'is-reduced': reduced }" aria-hidden="true">
    <div class="gateway-nebula gateway-nebula-cyan" />
    <div class="gateway-nebula gateway-nebula-violet" />
    <div class="gateway-nebula gateway-nebula-warm" />
    <canvas ref="surface" class="gateway-grid" />
    <div class="gateway-vignette" />
  </div>
</template>

<script setup lang="ts">
type GlowPoint = { x: number; y: number; age: number; radius: number; color: string }
type GridCell = { x: number; y: number; age: number }

const props = defineProps<{ revealed: boolean }>()
const stage = ref<HTMLElement | null>(null)
const surface = ref<HTMLCanvasElement | null>(null)
const hidden = ref(false)
const reduced = ref(false)
let cleanup = () => {}
let refresh = () => {}

watch(() => props.revealed, () => refresh())

onMounted(() => {
  const element = stage.value
  const canvas = surface.value
  const host = element?.closest<HTMLElement>('.gateway-scene') ?? element?.parentElement
  if (!element || !canvas || !host) return
  const context = canvas.getContext('2d')
  if (!context) return

  const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
  const pointerQuery = window.matchMedia('(hover: hover) and (pointer: fine)')
  const colors = ['68, 210, 255', '112, 138, 255', '171, 116, 255', '255, 151, 176']
  const stars = [[.07, .17], [.14, .61], [.22, .35], [.27, .82], [.36, .12], [.41, .68], [.48, .26], [.58, .86], [.63, .15], [.71, .59], [.78, .31], [.85, .77], [.91, .12], [.96, .49]]
  const glowPoints: GlowPoint[] = []
  const cells: GridCell[] = []
  let width = 1
  let height = 1
  let spacing = 54
  let frame = 0
  let lastFrame = 0
  let elapsed = 0
  let lastSample = 0
  let hueIndex = 0
  let disposed = false
  let observer: ResizeObserver | null = null
  let pointer: { x: number; y: number; smoothX: number; smoothY: number } | null = null

  const canAnimate = () => !disposed && !document.hidden && !motionQuery.matches
  const canFollow = () => canAnimate() && pointerQuery.matches

  const drawGlow = (x: number, y: number, radius: number, color: string, opacity: number) => {
    const glow = context.createRadialGradient(x, y, 0, x, y, radius)
    glow.addColorStop(0, `rgba(${color}, ${opacity})`)
    glow.addColorStop(.32, `rgba(${color}, ${opacity * .48})`)
    glow.addColorStop(1, `rgba(${color}, 0)`)
    context.fillStyle = glow
    context.fillRect(x - radius, y - radius, radius * 2, radius * 2)
  }

  const draw = (delta = 0) => {
    context.clearRect(0, 0, width, height)
    const offset = motionQuery.matches ? 0 : (elapsed * 2.8) % spacing
    const strength = props.revealed ? .56 : 1
    context.lineWidth = .7
    context.strokeStyle = props.revealed ? 'rgba(149, 180, 229, .075)' : 'rgba(130, 160, 224, .055)'
    context.beginPath()
    for (let x = offset - spacing; x <= width + spacing; x += spacing) {
      context.moveTo(x, 0)
      context.lineTo(x, height)
    }
    for (let y = offset - spacing; y <= height + spacing; y += spacing) {
      context.moveTo(0, y)
      context.lineTo(width, y)
    }
    context.stroke()

    for (let index = cells.length - 1; index >= 0; index--) {
      const cell = cells[index]!
      cell.age += delta
      if (cell.age > 1.45) { cells.splice(index, 1); continue }
      const fade = Math.pow(1 - cell.age / 1.45, 2)
      const x = cell.x * spacing + offset
      const y = cell.y * spacing + offset
      context.fillStyle = `rgba(102, 179, 255, ${fade * .11 * strength})`
      context.fillRect(x + 1, y + 1, spacing - 2, spacing - 2)
      context.strokeStyle = `rgba(134, 201, 255, ${fade * .24 * strength})`
      context.strokeRect(x + .5, y + .5, spacing - 1, spacing - 1)
    }

    stars.forEach((star, index) => {
      const shimmer = motionQuery.matches ? .6 : .55 + Math.sin(elapsed * .7 + index * 1.9) * .3
      context.fillStyle = `rgba(166, 207, 255, ${shimmer * .32})`
      context.beginPath()
      context.arc(star[0]! * width, star[1]! * height, index % 4 === 0 ? 1.25 : .8, 0, Math.PI * 2)
      context.fill()
    })

    context.globalCompositeOperation = 'screen'
    for (let index = glowPoints.length - 1; index >= 0; index--) {
      const point = glowPoints[index]!
      point.age += delta
      if (point.age > 1.15) { glowPoints.splice(index, 1); continue }
      const fade = Math.pow(1 - point.age / 1.15, 1.4)
      drawGlow(point.x, point.y, point.radius + point.age * 30, point.color, fade * .28 * strength)
    }
    if (pointer && canFollow()) {
      const catchUp = 1 - Math.exp(-delta * 12)
      pointer.smoothX += (pointer.x - pointer.smoothX) * catchUp
      pointer.smoothY += (pointer.y - pointer.smoothY) * catchUp
      drawGlow(pointer.smoothX, pointer.smoothY, 145, '78, 186, 255', .24 * strength)
      drawGlow(pointer.x, pointer.y, 35, '147, 222, 255', .3 * strength)
    }
    context.globalCompositeOperation = 'source-over'
  }

  const animate = (timestamp: number) => {
    frame = 0
    if (!canAnimate()) return
    if (!lastFrame) lastFrame = timestamp
    const interval = timestamp - lastFrame
    if (interval >= 1000 / 36) {
      const delta = Math.min(interval / 1000, .08)
      elapsed += delta
      draw(delta)
      lastFrame = timestamp
    }
    frame = requestAnimationFrame(animate)
  }

  const stop = () => {
    if (frame) cancelAnimationFrame(frame)
    frame = 0
    lastFrame = 0
  }

  const resume = () => {
    if (canAnimate() && !frame) frame = requestAnimationFrame(animate)
  }

  const resize = () => {
    const rect = element.getBoundingClientRect()
    width = Math.max(1, rect.width)
    height = Math.max(1, rect.height)
    spacing = width < 600 ? 48 : 54
    const ratio = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.round(width * ratio)
    canvas.height = Math.round(height * ratio)
    context.setTransform(ratio, 0, 0, ratio, 0, 0)
    glowPoints.length = 0
    cells.length = 0
    pointer = null
    draw()
  }

  const onPointerMove = (event: PointerEvent) => {
    if (!canFollow() || event.pointerType === 'touch') return
    const rect = element.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    if (x < 0 || x > width || y < 0 || y > height) { pointer = null; return }
    const distance = pointer ? Math.hypot(x - pointer.x, y - pointer.y) : Infinity
    pointer = pointer ? { ...pointer, x, y } : { x, y, smoothX: x, smoothY: y }
    const timestamp = performance.now()
    if (timestamp - lastSample < 22 || distance < 4) return
    lastSample = timestamp
    hueIndex++
    glowPoints.push({ x, y, age: 0, radius: 58 + Math.min(distance, 70) * .35, color: colors[Math.floor(hueIndex / 7) % colors.length]! })
    if (glowPoints.length > 28) glowPoints.shift()
    const offset = (elapsed * 2.8) % spacing
    const cellX = Math.floor((x - offset) / spacing)
    const cellY = Math.floor((y - offset) / spacing)
    const existing = cells.find(cell => cell.x === cellX && cell.y === cellY)
    if (existing) existing.age = 0
    else cells.push({ x: cellX, y: cellY, age: 0 })
    if (cells.length > 28) cells.shift()
  }

  const clearPointer = () => { pointer = null }
  const applyPreferences = () => {
    hidden.value = document.hidden
    reduced.value = motionQuery.matches
    if (!canFollow()) {
      pointer = null
      glowPoints.length = 0
      cells.length = 0
    }
    if (!canAnimate()) stop()
    if (!document.hidden) draw()
    resume()
  }

  refresh = () => { if (!disposed && !document.hidden) draw() }
  resize()
  applyPreferences()
  host.addEventListener('pointermove', onPointerMove, { passive: true })
  host.addEventListener('pointerleave', clearPointer, { passive: true })
  window.addEventListener('blur', clearPointer)
  window.addEventListener('resize', resize, { passive: true })
  document.addEventListener('visibilitychange', applyPreferences)
  motionQuery.addEventListener('change', applyPreferences)
  pointerQuery.addEventListener('change', applyPreferences)
  if ('ResizeObserver' in window) {
    observer = new ResizeObserver(resize)
    observer.observe(element)
  }

  cleanup = () => {
    disposed = true
    stop()
    observer?.disconnect()
    host.removeEventListener('pointermove', onPointerMove)
    host.removeEventListener('pointerleave', clearPointer)
    window.removeEventListener('blur', clearPointer)
    window.removeEventListener('resize', resize)
    document.removeEventListener('visibilitychange', applyPreferences)
    motionQuery.removeEventListener('change', applyPreferences)
    pointerQuery.removeEventListener('change', applyPreferences)
    glowPoints.length = 0
    cells.length = 0
    pointer = null
    refresh = () => {}
    canvas.width = 1
    canvas.height = 1
  }
})

onUnmounted(() => cleanup())
</script>

<style scoped>
.gateway-backdrop { position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none; background: #080b1b; }
.gateway-nebula { position: absolute; width: 76vmax; height: 76vmax; border-radius: 50%; opacity: .86; will-change: transform; transition: opacity 1100ms ease; }
.gateway-nebula-cyan { left: -35vmax; top: -23vmax; background: radial-gradient(ellipse, rgba(17, 125, 174, .38), rgba(14, 78, 119, .18) 35%, transparent 67%); animation: gateway-cyan-drift 18s ease-in-out infinite alternate; }
.gateway-nebula-violet { right: -31vmax; bottom: -34vmax; background: radial-gradient(ellipse, rgba(95, 48, 184, .34), rgba(67, 32, 126, .14) 38%, transparent 67%); animation: gateway-violet-drift 22s ease-in-out -8s infinite alternate; }
.gateway-nebula-warm { left: 17%; top: 50%; width: 50vmax; height: 50vmax; opacity: .54; background: radial-gradient(ellipse, rgba(164, 64, 111, .18), rgba(105, 37, 86, .08) 30%, transparent 65%); animation: gateway-warm-drift 24s ease-in-out -12s infinite alternate; }
.gateway-grid { position: absolute; inset: 0; width: 100%; height: 100%; }
.gateway-vignette { position: absolute; inset: 0; background: radial-gradient(ellipse at 50% 45%, transparent 15%, rgba(5, 7, 20, .26) 80%); }
.is-revealed .gateway-nebula { opacity: .42; }
.is-revealed .gateway-nebula-warm { opacity: .24; }
.is-paused .gateway-nebula { animation-play-state: paused; }
.is-reduced .gateway-nebula { animation: none; transition: none; will-change: auto; }
@keyframes gateway-cyan-drift { to { transform: translate3d(14vmax, 10vmax, 0) scale(1.16); } }
@keyframes gateway-violet-drift { to { transform: translate3d(-13vmax, -9vmax, 0) scale(1.2); } }
@keyframes gateway-warm-drift { to { transform: translate3d(-7vmax, -12vmax, 0) scale(1.12); } }
@media (max-width: 600px) {
  .gateway-nebula-cyan { left: -43vmax; top: -15vmax; }
  .gateway-nebula-violet { right: -45vmax; bottom: -12vmax; }
  .gateway-nebula-warm { left: -15%; }
}
@media (prefers-reduced-motion: reduce) {
  .gateway-nebula { animation: none; transition: none; will-change: auto; }
}
</style>
