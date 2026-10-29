<template>
  <div ref="stage" class="gateway-fluid" :class="{ 'is-ready': ready }" aria-hidden="true">
    <div class="gateway-fluid-fallback" />
    <canvas ref="surface" tabindex="-1" aria-hidden="true" />
    <div class="gateway-fluid-shade" />
  </div>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<{ active?: boolean }>(), { active: true })
const stage = ref<HTMLDivElement | null>(null)
const surface = ref<HTMLCanvasElement | null>(null)
const ready = ref(false)
let updateActivity = () => {}
let dispose = () => {}

watch(() => props.active, () => updateActivity())

onMounted(() => {
  const element = stage.value
  const canvas = surface.value
  if (!element || !canvas) return
  // The section receives input; the decorative canvas never intercepts scrolling.
  const host = element.closest('section') ?? element.parentElement
  const gl = canvas.getContext('webgl', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    preserveDrawingBuffer: false,
    powerPreference: 'low-power',
  })
  if (!gl) return

  const shaders: WebGLShader[] = []
  let program: WebGLProgram | null = null
  let buffer: WebGLBuffer | null = null
  let observer: ResizeObserver | null = null
  let frame = 0
  let disposed = false
  let contextLost = false
  let elapsed = 17.3
  let previous = 0
  let lastPointerSample = 0
  let trailIndex = 0
  const trails = new Float32Array(12 * 4)
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
  const mobile = window.matchMedia('(max-width: 700px)')
  const listeners: (() => void)[] = []

  const listen = (target: EventTarget, type: string, handler: EventListener, options?: AddEventListenerOptions) => {
    target.addEventListener(type, handler, options)
    listeners.push(() => target.removeEventListener(type, handler, options))
  }
  const stop = () => {
    if (frame) cancelAnimationFrame(frame)
    frame = 0
    previous = 0
  }
  dispose = () => {
    if (disposed) return
    disposed = true
    stop()
    observer?.disconnect()
    listeners.forEach(remove => remove())
    if (buffer) gl.deleteBuffer(buffer)
    if (program) gl.deleteProgram(program)
    shaders.forEach(shader => gl.deleteShader(shader))
    ready.value = false
  }

  try {
    // Noise hashing needs full precision on mobile GPUs; keep the CSS fallback
    // on older hardware whose fragment shaders cannot provide it.
    if (!gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT)?.precision) {
      dispose()
      return
    }
    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type)
      if (!shader) throw new Error('Shader unavailable')
      shaders.push(shader)
      gl.shaderSource(shader, source)
      gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error('Shader compilation failed')
      return shader
    }
    const vertex = compile(gl.VERTEX_SHADER, `
      attribute vec2 aPosition;
      varying vec2 vUv;
      void main() {
        vUv = aPosition * 0.5 + 0.5;
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `)
    const fragment = compile(gl.FRAGMENT_SHADER, `
      precision highp float;
      varying vec2 vUv;
      uniform vec2 uResolution;
      uniform float uTime;
      uniform vec4 uTrails[12];

      // Smooth value noise and domain warping form continuous, curling dye.
      float hash(vec2 p) {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
      }
      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
                   mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0)), f.x), f.y);
      }
      float fbm(vec2 p) {
        float value = 0.0;
        float amplitude = 0.55;
        mat2 turn = mat2(0.8, -0.6, 0.6, 0.8);
        for (int i = 0; i < 4; i++) {
          value += noise(p) * amplitude;
          p = turn * p * 2.03 + 3.1;
          amplitude *= 0.48;
        }
        return value;
      }
      vec3 palette(float position) {
        vec3 cyan = vec3(0.025, 0.57, 0.77);
        vec3 blue = vec3(0.12, 0.23, 0.82);
        vec3 violet = vec3(0.46, 0.12, 0.72);
        vec3 pink = vec3(0.77, 0.12, 0.43);
        float phase = fract(position) * 4.0;
        if (phase < 1.0) return mix(cyan, blue, smoothstep(0.0, 1.0, phase));
        if (phase < 2.0) return mix(blue, violet, smoothstep(1.0, 2.0, phase));
        if (phase < 3.0) return mix(violet, pink, smoothstep(2.0, 3.0, phase));
        return mix(pink, cyan, smoothstep(3.0, 4.0, phase));
      }
      void main() {
        float aspect = uResolution.x / uResolution.y;
        vec2 p = (vUv - 0.5) * vec2(aspect, 1.0);
        float t = uTime * 0.13;
        vec2 field = p * 2.2;
        vec2 warp = vec2(fbm(field + vec2(t * 0.19, -t * 0.13)),
                         fbm(field + vec2(5.7 - t * 0.14, 2.4 + t * 0.17)));
        vec2 flow = field + (warp - 0.5) * 2.6;
        float density = fbm(flow * 1.9 + vec2(t * 0.1, -t * 0.17));
        float fine = noise(flow * 9.0 + warp * 4.0 - t * 0.13);

        // Two winding currents leave a quiet center behind the name.
        float first = p.y - 0.23 * sin(p.x * 2.3 + t * 0.34) - 0.09;
        float second = p.y + 0.28 * sin(p.x * 1.7 - t * 0.23) + 0.13;
        float currents = exp(-first * first * 15.0) + exp(-second * second * 22.0) * 0.7;
        float folds = pow(smoothstep(0.2, 0.78, density), 1.4);
        float filaments = pow(1.0 - abs(density * 2.0 - 1.0), 8.0);
        float intensity = currents * (folds * 0.8 + filaments * 0.2) * (0.85 + fine * 0.15);
        float center = smoothstep(0.02, 0.62, length(p * vec2(0.8, 1.2)));
        intensity *= mix(0.28, 1.0, center);
        vec3 color = palette(p.x * 0.24 + p.y * 0.26 + warp.x * 0.3 + t * 0.018);
        vec3 result = vec3(0.018, 0.027, 0.046) + color * intensity * 0.82;

        // Recent pointer samples bloom, drift and dissolve into the same flow.
        for (int i = 0; i < 12; i++) {
          vec4 trail = uTrails[i];
          float age = uTime - trail.z;
          float life = max(0.0, 1.0 - age / 3.8) * trail.w;
          vec2 delta = (vUv - trail.xy) * vec2(aspect, 1.0);
          delta += vec2(sin(age * 1.3 + float(i)) * 0.025, -age * 0.014);
          delta += (warp - 0.5) * 0.06 * min(age, 2.0);
          float radius = 0.004 + max(age, 0.0) * 0.002;
          float glow = exp(-dot(delta, delta) / radius) * life * life;
          result += palette(float(i) / 12.0 + uTime * 0.014) * glow * 0.22;
        }

        float edge = 1.0 - smoothstep(0.5, 1.15, length((vUv - 0.5) * vec2(1.0, 0.8)));
        result *= edge;
        // A subtle grain removes banding without a texture dependency.
        result += (hash(gl_FragCoord.xy + fract(uTime) * 3.0) - 0.5) / 255.0;
        gl_FragColor = vec4(result, 1.0);
      }
    `)
    program = gl.createProgram()
    if (!program) throw new Error('Program unavailable')
    gl.attachShader(program, vertex)
    gl.attachShader(program, fragment)
    gl.linkProgram(program)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('Program link failed')
    gl.useProgram(program)

    buffer = gl.createBuffer()
    if (!buffer) throw new Error('Buffer unavailable')
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)
    const position = gl.getAttribLocation(program, 'aPosition')
    gl.enableVertexAttribArray(position)
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0)
    const resolutionLocation = gl.getUniformLocation(program, 'uResolution')
    const timeLocation = gl.getUniformLocation(program, 'uTime')
    const trailsLocation = gl.getUniformLocation(program, 'uTrails[0]')
    gl.disable(gl.DEPTH_TEST)
    gl.disable(gl.BLEND)

    const render = () => {
      if (disposed || contextLost) return
      gl.uniform2f(resolutionLocation, canvas.width, canvas.height)
      gl.uniform1f(timeLocation, elapsed)
      gl.uniform4fv(trailsLocation, trails)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      ready.value = true
    }
    const canAnimate = () => !disposed && !contextLost && props.active && !document.hidden && !reduced.matches
    const animate = (timestamp: number) => {
      frame = 0
      if (!canAnimate()) return
      if (!previous) previous = timestamp
      const interval = timestamp - previous
      const frameInterval = 1000 / (mobile.matches ? 24 : 30)
      if (interval >= frameInterval) {
        elapsed += Math.min(interval / 1000, 0.1)
        previous = timestamp - interval % frameInterval
        render()
      }
      frame = requestAnimationFrame(animate)
    }
    updateActivity = () => {
      stop()
      if (!props.active || document.hidden || disposed || contextLost) return
      render()
      if (canAnimate()) frame = requestAnimationFrame(animate)
    }
    const resize = () => {
      const bounds = element.getBoundingClientRect()
      const maxWidth = mobile.matches ? 720 : 1440
      const scale = Math.min(window.devicePixelRatio || 1, mobile.matches ? 1 : 1.25, maxWidth / Math.max(bounds.width, 1))
      canvas.width = Math.max(1, Math.round(bounds.width * scale))
      canvas.height = Math.max(1, Math.round(bounds.height * scale))
      gl.viewport(0, 0, canvas.width, canvas.height)
      if (props.active && !document.hidden) render()
    }
    const onPointer = (event: PointerEvent) => {
      if (!canAnimate() || event.isPrimary === false) return
      const now = performance.now()
      if (event.type === 'pointermove' && now - lastPointerSample < 55) return
      lastPointerSample = now
      const bounds = element.getBoundingClientRect()
      if (!bounds.width || !bounds.height) return
      const x = (event.clientX - bounds.left) / bounds.width
      const y = 1 - (event.clientY - bounds.top) / bounds.height
      if (x < 0 || x > 1 || y < 0 || y > 1) return
      trails.set([x, y, elapsed, 1], trailIndex * 4)
      trailIndex = (trailIndex + 1) % 12
    }
    const onContextLost = (event: Event) => {
      event.preventDefault()
      contextLost = true
      ready.value = false
      stop()
    }
    if (host) {
      listen(host, 'pointermove', onPointer as EventListener, { passive: true })
      listen(host, 'pointerdown', onPointer as EventListener, { passive: true })
    }
    listen(document, 'visibilitychange', updateActivity)
    listen(reduced, 'change', updateActivity)
    listen(mobile, 'change', resize)
    listen(canvas, 'webglcontextlost', onContextLost)
    if ('ResizeObserver' in window) {
      observer = new ResizeObserver(resize)
      observer.observe(element)
    } else {
      listen(window, 'resize', resize, { passive: true })
    }
    resize()
    updateActivity()
  } catch {
    // Content and the CSS atmosphere remain visible without a GPU context.
    dispose()
  }
})

onBeforeUnmount(() => dispose())
</script>

<style scoped>
.gateway-fluid { position: absolute; inset: 0; overflow: hidden; pointer-events: none; background: #05080f; }
.gateway-fluid canvas, .gateway-fluid-fallback, .gateway-fluid-shade { position: absolute; inset: 0; width: 100%; height: 100%; }
.gateway-fluid canvas { display: block; opacity: 0; transition: opacity 1.1s ease; }
.gateway-fluid.is-ready canvas { opacity: 1; }
.gateway-fluid-fallback { background: radial-gradient(ellipse at 12% 65%, #087d8859, transparent 48%), radial-gradient(ellipse at 88% 20%, #5731a55c, transparent 48%), radial-gradient(ellipse at 84% 80%, #263d9d66, transparent 44%), #05080f; }
.gateway-fluid-shade { background: radial-gradient(ellipse at 50% 48%, #03070c36, transparent 62%), linear-gradient(0deg, #06090e, transparent 24%, transparent 80%, #06090e40); }
@media (prefers-reduced-motion: reduce) { .gateway-fluid canvas { transition: none; } }
</style>
