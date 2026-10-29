<template>
  <div ref="root" class="gateway-grid-background" aria-hidden="true">
    <canvas ref="canvas" />
  </div>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<{ active?: boolean }>(), { active: true })
const root = ref<HTMLElement | null>(null)
const canvas = ref<HTMLCanvasElement | null>(null)

const cellSize = 46
const trailLifetime = 820
type TrailCell = { column: number; row: number; touched: number; strength: number }
const trail = new Map<string, TrailCell>()
let context: CanvasRenderingContext2D | null = null
let host: HTMLElement | null = null
let resizeObserver: ResizeObserver | null = null
let intersectionObserver: IntersectionObserver | null = null
let motionPreference: MediaQueryList | null = null
let frame = 0
let width = 0
let height = 0
let elapsed = 0
let lastFrame = 0
let visible = true
let reducedMotion = false
let mounted = false
let pointer: { x: number; y: number } | null = null

function canAnimate() {
  return mounted && props.active && visible && !document.hidden && !reducedMotion && width > 0 && height > 0
}

function markCell(column: number, row: number, strength: number) {
  const key = `${column}:${row}`
  trail.set(key, { column, row, touched: elapsed, strength })
  if (trail.size > 90) trail.delete(trail.keys().next().value!)
}

function paint(interactive: boolean) {
  if (!context || !width || !height) return
  const ctx = context
  const offset = reducedMotion ? 0 : (elapsed * 0.0036) % cellSize
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = '#06090d'
  ctx.fillRect(0, 0, width, height)

  const glow = ctx.createRadialGradient(width * 0.5, height * 0.44, 0, width * 0.5, height * 0.44, Math.max(width * 0.48, height * 0.66))
  glow.addColorStop(0, 'rgba(18, 73, 119, 0.22)')
  glow.addColorStop(0.5, 'rgba(10, 42, 69, 0.1)')
  glow.addColorStop(1, 'rgba(6, 9, 13, 0)')
  ctx.fillStyle = glow
  ctx.fillRect(0, 0, width, height)

  if (interactive && pointer) {
    const column = Math.floor((pointer.x - offset) / cellSize)
    const row = Math.floor((pointer.y - offset) / cellSize)
    markCell(column, row, 1)
    markCell(column - 1, row, 0.24)
    markCell(column + 1, row, 0.24)
    markCell(column, row - 1, 0.24)
    markCell(column, row + 1, 0.24)
  }

  if (interactive) {
    for (const [key, cell] of trail) {
      const age = elapsed - cell.touched
      if (age >= trailLifetime) {
        trail.delete(key)
        continue
      }
      const opacity = Math.pow(1 - age / trailLifetime, 2) * cell.strength
      const x = cell.column * cellSize + offset
      const y = cell.row * cellSize + offset
      ctx.fillStyle = `rgba(38, 137, 205, ${opacity * 0.2})`
      ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2)
      ctx.strokeStyle = `rgba(70, 178, 245, ${opacity * 0.38})`
      ctx.strokeRect(x + 0.5, y + 0.5, cellSize, cellSize)
    }

    // A quiet warm tile appears once during each long cycle.
    const cycle = Math.floor(elapsed / 15000)
    const pulse = (elapsed % 15000) / 15000
    if (pulse > 0.55 && pulse < 0.76) {
      const amount = Math.sin(((pulse - 0.55) / 0.21) * Math.PI)
      const column = Math.floor((width / cellSize) * (0.18 + ((cycle * 37 + 19) % 67) / 100))
      const row = Math.floor((height / cellSize) * (0.18 + ((cycle * 23 + 11) % 61) / 100))
      ctx.fillStyle = `rgba(172, 138, 65, ${amount * 0.1})`
      ctx.fillRect(column * cellSize + offset + 1, row * cellSize + offset + 1, cellSize - 2, cellSize - 2)
    }
  }

  ctx.beginPath()
  for (let x = offset; x <= width; x += cellSize) {
    ctx.moveTo(Math.floor(x) + 0.5, 0)
    ctx.lineTo(Math.floor(x) + 0.5, height)
  }
  for (let y = offset; y <= height; y += cellSize) {
    ctx.moveTo(0, Math.floor(y) + 0.5)
    ctx.lineTo(width, Math.floor(y) + 0.5)
  }
  ctx.strokeStyle = 'rgba(74, 140, 181, 0.12)'
  ctx.lineWidth = 1
  ctx.stroke()
}

function tick(now: number) {
  frame = 0
  if (!canAnimate()) return
  if (lastFrame) elapsed += Math.min(now - lastFrame, 64)
  lastFrame = now
  paint(true)
  frame = requestAnimationFrame(tick)
}

function syncAnimation() {
  if (!mounted) return
  if (canAnimate()) {
    if (!frame) {
      lastFrame = 0
      frame = requestAnimationFrame(tick)
    }
  } else {
    cancelAnimationFrame(frame)
    frame = 0
    lastFrame = 0
    pointer = null
    trail.clear()
    paint(false)
  }
}

function resize() {
  if (!root.value || !canvas.value || !context) return
  width = root.value.clientWidth
  height = root.value.clientHeight
  const ratio = Math.min(window.devicePixelRatio || 1, 1.5)
  canvas.value.width = Math.round(width * ratio)
  canvas.value.height = Math.round(height * ratio)
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  paint(canAnimate())
  syncAnimation()
}

function onPointerMove(event: PointerEvent) {
  if (!canAnimate() || event.pointerType === 'touch' || !root.value) return
  const rect = root.value.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top
  pointer = x >= 0 && x <= width && y >= 0 && y <= height ? { x, y } : null
}

function onPointerLeave() {
  pointer = null
}

function onMotionChange() {
  reducedMotion = motionPreference?.matches ?? false
  syncAnimation()
}

watch(() => props.active, syncAnimation)

onMounted(() => {
  if (!root.value || !canvas.value) return
  context = canvas.value.getContext('2d', { alpha: false })
  if (!context) return
  mounted = true
  motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)')
  reducedMotion = motionPreference.matches
  motionPreference.addEventListener('change', onMotionChange)
  host = root.value.closest('section') ?? root.value.parentElement
  host?.addEventListener('pointermove', onPointerMove, { passive: true })
  host?.addEventListener('pointerleave', onPointerLeave, { passive: true })
  document.addEventListener('visibilitychange', syncAnimation)
  window.addEventListener('resize', resize, { passive: true })
  resizeObserver = new ResizeObserver(resize)
  resizeObserver.observe(root.value)
  intersectionObserver = new IntersectionObserver(([entry]) => {
    visible = entry?.isIntersecting ?? false
    syncAnimation()
  }, { rootMargin: '0px', threshold: 0 })
  intersectionObserver.observe(root.value)
  resize()
})

onBeforeUnmount(() => {
  mounted = false
  cancelAnimationFrame(frame)
  resizeObserver?.disconnect()
  intersectionObserver?.disconnect()
  motionPreference?.removeEventListener('change', onMotionChange)
  host?.removeEventListener('pointermove', onPointerMove)
  host?.removeEventListener('pointerleave', onPointerLeave)
  document.removeEventListener('visibilitychange', syncAnimation)
  window.removeEventListener('resize', resize)
  trail.clear()
  pointer = null
  context = null
})
</script>

<style scoped>
.gateway-grid-background {
  position: absolute;
  inset: 0;
  overflow: hidden;
  pointer-events: none;
  background-color: #06090d;
  background-image:
    radial-gradient(ellipse at 50% 44%, rgb(18 73 119 / 22%), transparent 68%),
    linear-gradient(rgb(74 140 181 / 12%) 1px, transparent 1px),
    linear-gradient(90deg, rgb(74 140 181 / 12%) 1px, transparent 1px);
  background-size: auto, 46px 46px, 46px 46px;
}

.gateway-grid-background canvas {
  display: block;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
</style>
