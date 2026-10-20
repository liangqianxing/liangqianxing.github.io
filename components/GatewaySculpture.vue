<template>
  <div ref="stage" class="gateway-sculpture" :class="{ 'is-ready': ready }" aria-hidden="true">
    <img class="sculpture-fallback" :src="posterSrc" width="390" height="590" alt="" decoding="async" fetchpriority="high" />
    <div ref="canvasHost" class="sculpture-canvas" />
  </div>
</template>

<script setup lang="ts">
const stage = ref<HTMLElement | null>(null)
const canvasHost = ref<HTMLElement | null>(null)
const ready = ref(false)
const baseURL = useRuntimeConfig().app.baseURL
const posterSrc = `${baseURL.replace(/\/?$/, '/')}models/flight-helmet/poster.png`

// Art direction controls. The glTF faces +Z and uses +Y as its vertical axis.
const MODEL_HEIGHT = 4.05
const CAMERA_DISTANCE = 7.65
const BASE_YAW = -.55
const SWAY_DEGREES = 7
const SWAY_PERIOD = 14
const POINTER_DEGREES = 3
const GRAIN_CSS_PIXELS = 1.05
const EXPOSURE = 1.1

let disposed = false
let teardown = () => {}

onMounted(async () => {
  const element = stage.value
  const host = canvasHost.value
  if (!element || !host) return

  try {
    // Neither Three.js nor the loader executes during server rendering.
    const [THREE, { GLTFLoader }, { DRACOLoader }] = await Promise.all([
      import('three'),
      import('three/addons/loaders/GLTFLoader.js'),
      import('three/addons/loaders/DRACOLoader.js'),
    ])
    if (disposed) return

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false, premultipliedAlpha: false, powerPreference: 'low-power' })
    const geometries = new Set<InstanceType<typeof THREE.BufferGeometry>>()
    const materials = new Set<InstanceType<typeof THREE.Material>>()
    const textures = new Set<InstanceType<typeof THREE.Texture>>()
    let environmentTarget: InstanceType<typeof THREE.WebGLRenderTarget> | null = null
    let renderTarget: InstanceType<typeof THREE.WebGLRenderTarget> | null = null
    let pmrem: InstanceType<typeof THREE.PMREMGenerator> | null = null
    let dracoLoader: InstanceType<typeof DRACOLoader> | null = null
    let sizeObserver: ResizeObserver | null = null
    let intersectionObserver: IntersectionObserver | null = null
    let model: InstanceType<typeof THREE.Group> | null = null
    let frame = 0
    let lastFrame = 0
    let elapsed = 0
    let width = 1
    let height = 1
    let ratio = 1
    let inView = false
    let cleaned = false
    let contextLost = false
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
    const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)')
    const narrowScreen = window.matchMedia('(max-width: 767px)')
    const pointer = { x: 0, y: 0, smoothX: 0, smoothY: 0 }
    const removeListeners: (() => void)[] = []

    const trackModel = (object: InstanceType<typeof THREE.Object3D>) => {
      object.traverse(child => {
        if (!(child instanceof THREE.Mesh)) return
        geometries.add(child.geometry)
        const meshMaterials = Array.isArray(child.material) ? child.material : [child.material]
        meshMaterials.forEach(material => {
          materials.add(material)
          Object.values(material).forEach(value => {
            if (value && typeof value === 'object' && 'isTexture' in value) textures.add(value as InstanceType<typeof THREE.Texture>)
          })
        })
      })
    }
    const disposeTextures = (items: Iterable<InstanceType<typeof THREE.Texture>>) => {
      const images = new Set<ImageBitmap>()
      for (const texture of items) {
        const image = texture.source?.data
        texture.dispose()
        if (image && typeof image.close === 'function') images.add(image)
      }
      images.forEach(image => image.close())
    }
    const stop = () => {
      if (frame) cancelAnimationFrame(frame)
      frame = 0
      lastFrame = 0
    }
    teardown = () => {
      if (cleaned) return
      cleaned = true
      stop()
      sizeObserver?.disconnect()
      intersectionObserver?.disconnect()
      removeListeners.forEach(remove => remove())
      model?.removeFromParent()
      geometries.forEach(geometry => geometry.dispose())
      materials.forEach(material => material.dispose())
      disposeTextures(textures)
      renderTarget?.dispose()
      environmentTarget?.dispose()
      pmrem?.dispose()
      dracoLoader?.dispose()
      renderer.renderLists.dispose()
      renderer.dispose()
      renderer.forceContextLoss()
      renderer.domElement.remove()
      ready.value = false
    }

    renderer.setClearColor(0x080808, 0)
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace
    // Tone mapping happens in our final shader, before the Bayer threshold.
    renderer.toneMapping = THREE.NoToneMapping
    renderer.domElement.setAttribute('aria-hidden', 'true')
    host.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(35, 1, .1, 35)
    camera.position.set(0, .04, CAMERA_DISTANCE)
    camera.lookAt(0, .04, 0)
    const sculpture = new THREE.Group()
    sculpture.rotation.set(.025, BASE_YAW, 0)
    scene.add(sculpture)

    // A small local studio supplies long white reflections without loading HDR.
    const studio = new THREE.Scene()
    studio.background = new THREE.Color('#171717')
    const roomGeometry = new THREE.BoxGeometry(12, 12, 12)
    const roomMaterial = new THREE.MeshBasicMaterial({ color: '#343434', side: THREE.BackSide })
    geometries.add(roomGeometry)
    materials.add(roomMaterial)
    studio.add(new THREE.Mesh(roomGeometry, roomMaterial))
    const panel = (w: number, h: number, strength: number, x: number, y: number, z: number) => {
      const geometry = new THREE.PlaneGeometry(w, h)
      const material = new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(strength, strength, strength), side: THREE.DoubleSide })
      geometries.add(geometry)
      materials.add(material)
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set(x, y, z)
      mesh.lookAt(0, 0, 0)
      studio.add(mesh)
    }
    panel(3, 6, 5, -4, 2, 2)
    panel(1, 5, 2.5, 4, .6, -.5)
    panel(5, 2.5, 4, -.5, 4, 0)
    panel(3, 4, 1.5, .5, 1, 4)
    pmrem = new THREE.PMREMGenerator(renderer)
    environmentTarget = pmrem.fromScene(studio, .04, .1, 30)
    scene.environment = environmentTarget.texture
    pmrem.dispose()
    pmrem = null

    const key = new THREE.DirectionalLight('#ffffff', 5)
    key.position.set(-3, 4, 5)
    const rim = new THREE.DirectionalLight('#ffffff', 3.2)
    rim.position.set(3, 1, -3)
    const fill = new THREE.HemisphereLight('#ffffff', '#080808', .6)
    scene.add(key, rim, fill)

    renderTarget = new THREE.WebGLRenderTarget(1, 1, {
      type: renderer.extensions.has('EXT_color_buffer_float') ? THREE.HalfFloatType : THREE.UnsignedByteType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
      stencilBuffer: false,
    })
    const postScene = new THREE.Scene()
    const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const postMaterial = new THREE.ShaderMaterial({
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
      uniforms: {
        tScene: { value: renderTarget.texture },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uPixelSize: { value: GRAIN_CSS_PIXELS },
        uExposure: { value: EXPOSURE },
      },
      vertexShader: `
        void main() {
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        precision highp float;
        uniform sampler2D tScene;
        uniform vec2 uResolution;
        uniform float uPixelSize;
        uniform float uExposure;

        float bayer4(vec2 cell) {
          vec2 p = mod(cell, 4.0);
          float value;
          if (p.y < 1.0) value = p.x < 1.0 ? 0.0 : p.x < 2.0 ? 8.0 : p.x < 3.0 ? 2.0 : 10.0;
          else if (p.y < 2.0) value = p.x < 1.0 ? 12.0 : p.x < 2.0 ? 4.0 : p.x < 3.0 ? 14.0 : 6.0;
          else if (p.y < 3.0) value = p.x < 1.0 ? 3.0 : p.x < 2.0 ? 11.0 : p.x < 3.0 ? 1.0 : 9.0;
          else value = p.x < 1.0 ? 15.0 : p.x < 2.0 ? 7.0 : p.x < 3.0 ? 13.0 : 5.0;
          return (value + 0.5) / 16.0;
        }

        void main() {
          vec2 cell = floor(gl_FragCoord.xy / uPixelSize);
          vec2 uv = (cell + 0.5) * uPixelSize / uResolution;
          vec4 source = texture2D(tScene, uv);
          if (source.a < 0.01) {
            gl_FragColor = vec4(vec3(0.03137255), 0.0);
            return;
          }
          float light = dot(source.rgb, vec3(0.2126, 0.7152, 0.0722));
          light *= uExposure;
          light = light / (1.0 + light);
          // Lift the leather's dark folds while retaining clean white highlights.
          light = smoothstep(0.015, 0.86, pow(light, 0.55));
          float ink = step(bayer4(cell), light);
          vec3 paper = mix(vec3(0.03137255), vec3(0.96), ink);
          gl_FragColor = vec4(paper, source.a);
        }
      `,
    })
    materials.add(postMaterial)
    const quadGeometry = new THREE.PlaneGeometry(2, 2)
    geometries.add(quadGeometry)
    const quad = new THREE.Mesh(quadGeometry, postMaterial)
    quad.frustumCulled = false
    postScene.add(quad)

    const canAnimate = () => !disposed && !cleaned && !contextLost && !!model && inView && !document.hidden && !motion.matches
    const render = () => {
      if (disposed || cleaned || contextLost || !model) return
      renderer.setRenderTarget(renderTarget)
      renderer.clear()
      renderer.render(scene, camera)
      renderer.setRenderTarget(null)
      renderer.clear()
      renderer.render(postScene, postCamera)
      ready.value = true
    }
    const animate = (timestamp: number) => {
      frame = 0
      if (!canAnimate()) return
      if (!lastFrame) lastFrame = timestamp
      const interval = timestamp - lastFrame
      const frameInterval = 1000 / (narrowScreen.matches ? 30 : 40)
      if (interval >= frameInterval) {
        const delta = Math.min(interval / 1000, .07)
        elapsed += delta
        const ease = 1 - Math.exp(-delta * 3.5)
        pointer.smoothX += (pointer.x - pointer.smoothX) * ease
        pointer.smoothY += (pointer.y - pointer.smoothY) * ease
        sculpture.rotation.y = BASE_YAW + Math.sin(elapsed / SWAY_PERIOD * Math.PI * 2) * THREE.MathUtils.degToRad(SWAY_DEGREES) + pointer.smoothX * THREE.MathUtils.degToRad(POINTER_DEGREES)
        sculpture.rotation.x = .025 + pointer.smoothY * THREE.MathUtils.degToRad(POINTER_DEGREES * .45)
        render()
        lastFrame = timestamp - interval % frameInterval
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
        if (motion.matches) sculpture.rotation.set(.025, BASE_YAW, 0)
      }
      if (inView && !document.hidden) render()
      resume()
    }
    const resize = () => {
      const rect = element.getBoundingClientRect()
      width = Math.max(1, rect.width)
      height = Math.max(1, rect.height)
      ratio = Math.min(window.devicePixelRatio || 1, 1.5)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setPixelRatio(ratio)
      renderer.setSize(width, height, false)
      const drawingSize = renderer.getDrawingBufferSize(new THREE.Vector2())
      renderTarget!.setSize(drawingSize.x, drawingSize.y)
      postMaterial.uniforms.uResolution.value.copy(drawingSize)
      postMaterial.uniforms.uPixelSize.value = GRAIN_CSS_PIXELS * ratio
      if (inView && !document.hidden) render()
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

    const assetBase = baseURL.replace(/\/?$/, '/')
    dracoLoader = new DRACOLoader()
    dracoLoader.setDecoderPath(`${assetBase}models/draco/`)
    dracoLoader.setDecoderConfig({ type: 'wasm' })
    dracoLoader.setWorkerLimit(1)
    const loader = new GLTFLoader().setDRACOLoader(dracoLoader)
    const gltf = await loader.loadAsync(`${assetBase}models/flight-helmet/FlightHelmet.bust.glb`)
    dracoLoader.dispose()
    dracoLoader = null
    if (disposed || cleaned) {
      // A route change can happen while the glTF and its textures are loading.
      // Dispose these late arrivals as well as the renderer already released.
      trackModel(gltf.scene)
      geometries.forEach(geometry => geometry.dispose())
      materials.forEach(material => material.dispose())
      disposeTextures(textures)
      return
    }
    model = gltf.scene
    trackModel(model)
    const box = new THREE.Box3().setFromObject(model)
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    const scale = MODEL_HEIGHT / Math.max(size.y, .001)
    model.scale.setScalar(scale)
    model.position.copy(center).multiplyScalar(-scale)
    model.traverse(child => {
      if (!(child instanceof THREE.Mesh)) return
      const meshMaterials = Array.isArray(child.material) ? child.material : [child.material]
      meshMaterials.forEach(material => {
        if (!(material instanceof THREE.MeshStandardMaterial)) return
        const part = `${child.name} ${material.name}`.toLowerCase()
        const isMetal = part.includes('metal')
        const isGlass = part.includes('glass') || part.includes('lenses')
        const isSupport = part.includes('rubberwood')
        material.roughness = isGlass ? .09 : isMetal ? .2 : isSupport ? .5 : .33
        material.metalness = isMetal ? .95 : isGlass ? .5 : isSupport ? .18 : .46
        material.envMapIntensity = isSupport ? .6 : 1.45
        // Textures retain the cap's seams and lens detail; the final pass removes hue.
        if (isSupport) material.color.multiplyScalar(.48)
        material.needsUpdate = true
      })
    })
    sculpture.add(model)
    updateActivity()
  } catch {
    // A failed model request or unavailable WebGL preserves the SSR helmet.
    teardown()
  }
})

onBeforeUnmount(() => {
  disposed = true
  teardown()
})
</script>

<style scoped>
.gateway-sculpture { position: relative; isolation: isolate; width: 100%; height: 100%; min-width: 0; overflow: hidden; }
.sculpture-fallback { position: absolute; left: 51.2%; top: 49.2%; width: auto; max-width: 90%; height: 79.3%; object-fit: contain; transform: translate(-50%, -50%); transition: opacity .4s ease; pointer-events: none; }
.sculpture-canvas { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; transition: opacity .55s ease; pointer-events: none; }
.sculpture-canvas :deep(canvas) { display: block; width: 100%; height: 100%; }
.is-ready .sculpture-fallback { opacity: 0; }
.is-ready .sculpture-canvas { opacity: 1; }
@media (prefers-reduced-motion: reduce) {
  .sculpture-fallback, .sculpture-canvas { transition: none; }
}
</style>
