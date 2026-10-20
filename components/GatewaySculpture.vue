<template>
  <div ref="stage" class="gateway-sculpture" :class="{ 'is-ready': ready }" aria-hidden="true">
    <div class="sculpture-aura" />
    <svg class="sculpture-fallback" viewBox="0 0 600 600" fill="none" focusable="false">
      <defs>
        <linearGradient :id="`${gradientId}-metal`" x1="168" y1="123" x2="411" y2="462" gradientUnits="userSpaceOnUse">
          <stop stop-color="#c7d4e5" />
          <stop offset=".19" stop-color="#6a7484" />
          <stop offset=".35" stop-color="#edf3fc" />
          <stop offset=".51" stop-color="#505b6c" />
          <stop offset=".72" stop-color="#151c29" />
          <stop offset=".85" stop-color="#a2b1c7" />
          <stop offset="1" stop-color="#4a5365" />
        </linearGradient>
        <linearGradient :id="`${gradientId}-edge`" x1="154" y1="119" x2="425" y2="450" gradientUnits="userSpaceOnUse">
          <stop stop-color="#e5edff" stop-opacity=".65" />
          <stop offset=".46" stop-color="#b1bfff" stop-opacity=".04" />
          <stop offset="1" stop-color="#8d9bab" stop-opacity=".32" />
        </linearGradient>
      </defs>
      <ellipse cx="300" cy="312" rx="223" ry="92" transform="rotate(-28 300 312)" stroke="#8e9bb4" stroke-opacity=".13" />
      <path :d="fallbackPath" stroke="#111723" stroke-width="69" stroke-linecap="round" stroke-linejoin="round" />
      <path :d="fallbackPath" :stroke="`url(#${gradientId}-metal)`" stroke-width="63" stroke-linecap="round" stroke-linejoin="round" />
      <path :d="fallbackPath" :stroke="`url(#${gradientId}-edge)`" stroke-width="55" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
    <div ref="canvasHost" class="sculpture-canvas" />
    <div class="sculpture-grain" />
  </div>
</template>

<script setup lang="ts">
const stage = ref<HTMLElement | null>(null)
const canvasHost = ref<HTMLElement | null>(null)
const ready = ref(false)
const gradientId = `sculpture-${useId().replace(/[^a-zA-Z0-9-]/g, '')}`

// The fallback follows the same continuous knot, so the first rendered frame
// replaces a silhouette instead of revealing a blank loading region.
const fallbackPath = Array.from({ length: 181 }, (_, index) => {
  const t = index / 180 * Math.PI * 2
  const radius = 1 + .42 * Math.cos(3 * t)
  const x = radius * Math.cos(2 * t)
  const y = radius * Math.sin(2 * t)
  const z = .46 * Math.sin(3 * t)
  const rx = x * .91 + z * .42
  const rz = z * .91 - x * .42
  const ry = y * .87 - rz * .49
  return `${index === 0 ? 'M' : 'L'} ${(300 + rx * 112).toFixed(2)} ${(300 + ry * 112).toFixed(2)}`
}).join(' ') + ' Z'

let disposed = false
let teardown = () => {}

onMounted(async () => {
  const element = stage.value
  const host = canvasHost.value
  if (!element || !host) return

  try {
    // Only the SVG is server-rendered. Load WebGL after mounting so the rest
    // of the homepage has priority over the Three.js bundle.
    const THREE = await import('three')
    if (disposed) return

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' })
    const geometries: InstanceType<typeof THREE.BufferGeometry>[] = []
    const materials: InstanceType<typeof THREE.Material>[] = []
    let environmentTarget: InstanceType<typeof THREE.WebGLRenderTarget> | null = null
    let pmrem: InstanceType<typeof THREE.PMREMGenerator> | null = null
    let sizeObserver: ResizeObserver | null = null
    let intersectionObserver: IntersectionObserver | null = null
    let frame = 0
    let lastFrame = 0
    let elapsed = 0
    let inView = false
    let contextLost = false
    let cleaned = false
    let width = 1
    let height = 1
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
    const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)')
    const narrowScreen = window.matchMedia('(max-width: 767px)')
    const pointer = { x: 0, y: 0, smoothX: 0, smoothY: 0 }

    const stop = () => {
      if (frame) cancelAnimationFrame(frame)
      frame = 0
      lastFrame = 0
    }

    const removeListeners: (() => void)[] = []
    teardown = () => {
      if (cleaned) return
      cleaned = true
      stop()
      sizeObserver?.disconnect()
      intersectionObserver?.disconnect()
      removeListeners.forEach(remove => remove())
      geometries.forEach(geometry => geometry.dispose())
      materials.forEach(material => material.dispose())
      environmentTarget?.dispose()
      pmrem?.dispose()
      renderer.renderLists.dispose()
      renderer.dispose()
      renderer.forceContextLoss()
      renderer.domElement.remove()
      ready.value = false
    }

    renderer.setClearColor(0x000000, 0)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.12
    renderer.domElement.setAttribute('aria-hidden', 'true')
    host.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(35, 1, .1, 30)
    camera.position.set(0, .05, 6.65)

    // Large studio panels reflected in the metal are what give this form its
    // volume. The environment is made locally, without an HDR or model fetch.
    const studio = new THREE.Scene()
    studio.background = new THREE.Color('#171c25')
    const roomGeometry = new THREE.BoxGeometry(12, 12, 12)
    const roomMaterial = new THREE.MeshBasicMaterial({ color: '#363e4c', side: THREE.BackSide })
    geometries.push(roomGeometry)
    materials.push(roomMaterial)
    studio.add(new THREE.Mesh(roomGeometry, roomMaterial))

    const panel = (w: number, h: number, color: string, strength: number, x: number, y: number, z: number) => {
      const geometry = new THREE.PlaneGeometry(w, h)
      const material = new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(strength), side: THREE.DoubleSide })
      const mesh = new THREE.Mesh(geometry, material)
      geometries.push(geometry)
      materials.push(material)
      mesh.position.set(x, y, z)
      mesh.lookAt(0, 0, 0)
      studio.add(mesh)
    }
    panel(3.2, 6.6, '#f3f6ff', 5, -4, 1.3, 1.8)
    panel(1.5, 6, '#c4d9ff', 3.6, 4, .4, .5)
    panel(5.5, 2.7, '#e5ecff', 3.8, 0, 4, .4)
    panel(3, 4, '#f3f5ff', 1.9, .5, .7, 4)
    panel(1.6, 4.4, '#a6a2e3', 2, -2, -1.5, -4)
    pmrem = new THREE.PMREMGenerator(renderer)
    environmentTarget = pmrem.fromScene(studio, .035, .1, 30)
    scene.environment = environmentTarget.texture
    pmrem.dispose()
    pmrem = null

    scene.add(new THREE.HemisphereLight('#eaf1ff', '#131823', 1.2))
    const keyLight = new THREE.DirectionalLight('#f4f6ff', 3.1)
    keyLight.position.set(-3, 4, 5)
    const edgeLight = new THREE.DirectionalLight('#b0caff', 2.3)
    edgeLight.position.set(4, -.5, 2)
    const rimLight = new THREE.DirectionalLight('#aaa0e2', 1.7)
    rimLight.position.set(-3, -2, -3)
    scene.add(keyLight, edgeLight, rimLight)

    const sculpture = new THREE.Group()
    scene.add(sculpture)
    const geometry = new THREE.TorusKnotGeometry(.94, .305, narrowScreen.matches ? 192 : 288, narrowScreen.matches ? 28 : 40, 2, 3)
    const material = new THREE.MeshPhysicalMaterial({
      color: '#b3bdcc', metalness: 1, roughness: .255,
      clearcoat: .48, clearcoatRoughness: .24, envMapIntensity: 1.65,
    })
    geometries.push(geometry)
    materials.push(material)
    const knot = new THREE.Mesh(geometry, material)
    knot.rotation.set(.32, -.38, -.31)
    sculpture.add(knot)

    // A single faint orbital line gives the object a reference plane.
    const orbitPoints = Array.from({ length: 161 }, (_, i) => {
      const angle = i / 160 * Math.PI * 2
      return new THREE.Vector3(Math.cos(angle) * 1.89, Math.sin(angle) * 1.89, 0)
    })
    const orbitGeometry = new THREE.BufferGeometry().setFromPoints(orbitPoints)
    const orbitMaterial = new THREE.LineBasicMaterial({ color: '#8c9ab7', transparent: true, opacity: .19, depthWrite: false })
    geometries.push(orbitGeometry)
    materials.push(orbitMaterial)
    const orbit = new THREE.Line(orbitGeometry, orbitMaterial)
    orbit.rotation.set(1.12, -.15, -.48)
    orbit.position.z = -.25
    scene.add(orbit)

    const canAnimate = () => !disposed && !contextLost && inView && !document.hidden && !motion.matches
    const render = () => {
      if (disposed || contextLost || !width || !height) return
      renderer.render(scene, camera)
      ready.value = true
    }
    const animate = (timestamp: number) => {
      frame = 0
      if (!canAnimate()) return
      if (!lastFrame) lastFrame = timestamp
      const interval = timestamp - lastFrame
      const frameInterval = 1000 / (narrowScreen.matches ? 30 : 45)
      if (interval >= frameInterval) {
        const delta = Math.min(interval / 1000, .07)
        elapsed += delta
        const ease = 1 - Math.exp(-delta * 3)
        pointer.smoothX += (pointer.x - pointer.smoothX) * ease
        pointer.smoothY += (pointer.y - pointer.smoothY) * ease
        sculpture.rotation.y = Math.sin(elapsed * .13) * .23 + pointer.smoothX * .075
        sculpture.rotation.x = Math.sin(elapsed * .19) * .035 + pointer.smoothY * .045
        sculpture.position.y = Math.sin(elapsed * .55) * .042
        render()
        lastFrame = timestamp - (interval % frameInterval)
      }
      frame = requestAnimationFrame(animate)
    }
    const resume = () => {
      if (canAnimate() && !frame) frame = requestAnimationFrame(animate)
    }
    const updateActivity = () => {
      stop()
      if (motion.matches || !finePointer.matches) {
        pointer.x = pointer.y = pointer.smoothX = pointer.smoothY = 0
        if (motion.matches) {
          sculpture.rotation.set(0, 0, 0)
          sculpture.position.y = 0
        }
      }
      if (inView && !document.hidden) render()
      resume()
    }
    const resize = () => {
      const rect = element.getBoundingClientRect()
      width = Math.max(1, rect.width)
      height = Math.max(1, rect.height)
      camera.aspect = width / height
      // Keep the same silhouette in a slightly taller mobile container.
      camera.position.z = camera.aspect < .94 ? 6.65 / camera.aspect : 6.65
      camera.updateProjectionMatrix()
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
      renderer.setSize(width, height, false)
      if (!document.hidden && inView) render()
    }
    const onPointerMove = (event: PointerEvent) => {
      if (!canAnimate() || !finePointer.matches || event.pointerType === 'touch') return
      const rect = element.getBoundingClientRect()
      pointer.x = THREE.MathUtils.clamp((event.clientX - rect.left) / rect.width * 2 - 1, -1, 1)
      pointer.y = THREE.MathUtils.clamp((event.clientY - rect.top) / rect.height * 2 - 1, -1, 1)
    }
    const clearPointer = () => { pointer.x = pointer.y = 0 }
    const onContextLost = (event: Event) => {
      event.preventDefault()
      contextLost = true
      ready.value = false
      stop()
    }
    const listen = (target: EventTarget, event: string, handler: EventListener, options?: AddEventListenerOptions) => {
      target.addEventListener(event, handler, options)
      removeListeners.push(() => target.removeEventListener(event, handler, options))
    }
    listen(element, 'pointermove', onPointerMove as EventListener, { passive: true })
    listen(element, 'pointerleave', clearPointer, { passive: true })
    listen(window, 'blur', clearPointer)
    listen(document, 'visibilitychange', updateActivity)
    listen(motion, 'change', updateActivity)
    listen(finePointer, 'change', updateActivity)
    listen(narrowScreen, 'change', updateActivity)
    listen(renderer.domElement, 'webglcontextlost', onContextLost)

    const bounds = element.getBoundingClientRect()
    inView = bounds.bottom > 0 && bounds.top < window.innerHeight && bounds.right > 0 && bounds.left < window.innerWidth
    resize()
    if ('ResizeObserver' in window) {
      sizeObserver = new ResizeObserver(resize)
      sizeObserver.observe(element)
    } else {
      listen(window, 'resize', resize, { passive: true })
    }
    if ('IntersectionObserver' in window) {
      intersectionObserver = new IntersectionObserver(([entry]) => {
        inView = Boolean(entry?.isIntersecting)
        updateActivity()
      }, { threshold: .02 })
      intersectionObserver.observe(element)
    }
    updateActivity()
  } catch {
    // Unsupported WebGL, resource limits, or a failed lazy import all retain
    // the deliberately composed static version rather than an error panel.
    teardown()
  }
})

onBeforeUnmount(() => {
  disposed = true
  teardown()
})
</script>

<style scoped>
.gateway-sculpture { position: relative; isolation: isolate; width: 100%; height: 100%; min-width: 0; overflow: visible; }
.sculpture-aura { position: absolute; inset: 12%; z-index: -1; border-radius: 50%; background: radial-gradient(ellipse, rgba(86, 107, 156, .085), rgba(72, 67, 121, .035) 46%, transparent 71%); filter: blur(24px); pointer-events: none; }
.sculpture-fallback, .sculpture-canvas { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.sculpture-fallback { opacity: .93; transition: opacity .7s ease; }
.sculpture-canvas { opacity: 0; transition: opacity .9s ease; }
.sculpture-canvas :deep(canvas) { display: block; width: 100%; height: 100%; }
.is-ready .sculpture-fallback { opacity: 0; }
.is-ready .sculpture-canvas { opacity: 1; }
.sculpture-grain { position: absolute; inset: 5%; z-index: 2; opacity: .12; pointer-events: none; mix-blend-mode: soft-light; background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.92' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' opacity='.75' filter='url(%23n)'/%3E%3C/svg%3E"); mask-image: radial-gradient(ellipse, #000 28%, transparent 70%); }
@media (prefers-reduced-motion: reduce) {
  .sculpture-fallback, .sculpture-canvas { transition: none; }
}
</style>
