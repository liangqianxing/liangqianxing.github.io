export interface GridAnimationOptions {
  borderColor?: string
  direction?: 'right' | 'left' | 'up' | 'down' | 'diagonal'
  hoverFillColor?: string
  hoverShadowColor?: string
  snakeColorDecay?: number
  snakeGradientStops?: number
  snakeHeadColor?: string
  snakeTailColor?: string
  specialBlockColor?: string
  specialHoverColor?: string
  speed?: number
  squareSize?: number
  touchSensitivity?: number
  trailDuration?: number
  transitionDuration?: number
  vibrationEnabled?: boolean
}

export class GridAnimation {
  canvas: HTMLCanvasElement
  ctx: CanvasRenderingContext2D
  options: Required<GridAnimationOptions>
  gridOffset = { x: 0, y: 0 }
  hoveredSquare: { x: number; y: number } | null = null
  animationFrame: number | null = null
  currentOpacity = 0
  targetOpacity = 0
  lastTimestamp = 0
  trailSquares = new Map<string, { opacity: number; x: number; y: number }>()
  specialBlock: { color: string; x: number; y: number } | null = null
  snakeBody: { x: number; y: number }[] = []
  shouldGrow = false
  isPhone = false
  private boundResize: () => void
  private boundMouseMove: (e: MouseEvent) => void
  private boundMouseLeave: () => void
  private boundVisibility: () => void
  private handleTouchStart?: (e: TouchEvent) => void
  private handleTouchMoveEvent?: (e: TouchEvent) => void
  private handleTouchEndEvent?: (e: TouchEvent) => void
  private handleTouchCancel?: (e: TouchEvent) => void
  private specialBlockTimer: ReturnType<typeof setTimeout> | null = null
  private motionPreference: MediaQueryList

  constructor(canvas: HTMLCanvasElement, opts: GridAnimationOptions = {}) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.isPhone = /Mobile|Android|iOS|iPhone|iPad|iPod|Windows Phone/i.test(navigator.userAgent)
    this.options = {
      borderColor: opts.borderColor || 'rgba(100, 181, 246, 0.08)',
      direction: opts.direction || 'diagonal',
      hoverFillColor: opts.hoverFillColor || 'rgba(100, 181, 246, 0.8)',
      hoverShadowColor: opts.hoverShadowColor || 'rgba(100, 181, 246, 0.8)',
      snakeColorDecay: opts.snakeColorDecay ?? 0.85,
      snakeGradientStops: opts.snakeGradientStops ?? 5,
      snakeHeadColor: opts.snakeHeadColor || 'rgba(144, 202, 249, 0.95)',
      snakeTailColor: opts.snakeTailColor || 'rgba(100, 181, 246, 0.2)',
      specialBlockColor: opts.specialBlockColor || 'rgba(255, 193, 7, 0.85)',
      specialHoverColor: opts.specialHoverColor || 'rgba(255, 235, 59, 0.9)',
      speed: opts.speed ?? (this.isPhone ? 0.03 : 0.05),
      squareSize: opts.squareSize ?? (this.isPhone ? 50 : 40),
      touchSensitivity: opts.touchSensitivity ?? (this.isPhone ? 1.2 : 1),
      trailDuration: opts.trailDuration ?? (this.isPhone ? 2000 : 1500),
      transitionDuration: opts.transitionDuration ?? (this.isPhone ? 150 : 200),
      vibrationEnabled: opts.vibrationEnabled ?? this.isPhone,
    }
    this.boundResize = () => this.resizeCanvas()
    this.boundMouseMove = (e: MouseEvent) => this.handleMouseMove(e)
    this.boundMouseLeave = () => this.handleMouseLeave()
    this.boundVisibility = () => this.handleVisibilityChange()
    this.motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)')
  }

  init() {
    if (!this.ctx) return
    this.resizeCanvas()
    this.setupEventListeners()
    if (this.isPhone) {
      this.specialBlockTimer = setTimeout(() => {
        this.createSpecialBlock()
        if (this.motionPreference.matches) this.drawGrid()
      }, 500)
    } else {
      this.createSpecialBlock()
    }
    document.addEventListener('visibilitychange', this.boundVisibility)
    this.motionPreference.addEventListener('change', this.boundVisibility)
    this.handleVisibilityChange()
  }

  resizeCanvas() {
    const dpr = window.devicePixelRatio || 1
    const bounds = this.canvas.parentElement?.getBoundingClientRect()
    const w = bounds?.width || window.innerWidth
    const h = bounds?.height || window.innerHeight
    this.canvas.width = Math.floor(w * dpr)
    this.canvas.height = Math.floor(h * dpr)
    this.canvas.style.width = `${w}px`
    this.canvas.style.height = `${h}px`
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  setupEventListeners() {
    window.addEventListener('resize', this.boundResize)
    this.canvas.addEventListener('mousemove', this.boundMouseMove)
    this.canvas.addEventListener('mouseleave', this.boundMouseLeave)
    if (this.isPhone) {
      this.setupTouchEvents()
    }
  }

  setupTouchEvents() {
    let startInfo: { time: number; x: number; y: number } | null = null
    let touching = false
    let lastTime = 0
    let lastTapTime = 0

    this.handleTouchStart = (e: TouchEvent) => {
      const now = Date.now()
      if (now - lastTime < 16) return
      lastTime = now
      if (e.touches.length === 1) {
        const t = e.touches[0]
        const rect = this.canvas.getBoundingClientRect()
        startInfo = { time: now, x: t.clientX - rect.left, y: t.clientY - rect.top }
        touching = true
        this.handleTouchMove(startInfo.x, startInfo.y)
        if (!this.hoveredSquare) {
          this.targetOpacity = 0.8 * this.options.touchSensitivity
        }
      }
    }

    this.handleTouchMoveEvent = (e: TouchEvent) => {
      if (touching && e.touches.length === 1) {
        const t = e.touches[0]
        const rect = this.canvas.getBoundingClientRect()
        this.handleTouchMove(t.clientX - rect.left, t.clientY - rect.top)
      }
    }

    this.handleTouchEndEvent = () => {
      const now = Date.now()
      const isTap = startInfo && now - startInfo.time < 300
      const isDoubleTap = isTap && lastTapTime > 0 && now - lastTapTime < 350
      touching = false
      startInfo = null
      if (isDoubleTap) {
        lastTapTime = 0
        this.resetSnake()
        return
      }
      lastTapTime = isTap ? now : 0
      this.handleTouchEnd()
    }

    this.handleTouchCancel = () => {
      touching = false
      startInfo = null
      lastTapTime = 0
    }

    this.canvas.addEventListener('touchstart', this.handleTouchStart, { passive: true })
    this.canvas.addEventListener('touchmove', this.handleTouchMoveEvent, { passive: true })
    this.canvas.addEventListener('touchend', this.handleTouchEndEvent, { passive: true })
    this.canvas.addEventListener('touchcancel', this.handleTouchCancel, { passive: true })
  }

  handleTouchMove(cx: number, cy: number) {
    const size = this.options.squareSize
    const baseX = Math.floor(this.gridOffset.x / size) * size
    const baseY = Math.floor(this.gridOffset.y / size) * size
    const gx = Math.floor((cx + this.gridOffset.x - baseX) / size)
    const gy = Math.floor((cy + this.gridOffset.y - baseY) / size)
    if (this.hoveredSquare?.x === gx && this.hoveredSquare?.y === gy) return
    if (this.hoveredSquare) {
      this.snakeBody.unshift({ x: this.hoveredSquare.x, y: this.hoveredSquare.y })
      if (!this.shouldGrow && this.snakeBody.length > 0) this.snakeBody.pop()
      this.shouldGrow = false
    }
    this.hoveredSquare = { x: gx, y: gy }
    this.targetOpacity = 0.8 * this.options.touchSensitivity
    if (this.specialBlock && gx === this.specialBlock.x && gy === this.specialBlock.y) {
      this.shouldGrow = true
      this.createSpecialBlock()
    }
  }

  handleTouchEnd() {
    if (this.hoveredSquare) {
      this.snakeBody.unshift({ x: this.hoveredSquare.x, y: this.hoveredSquare.y })
      if (!this.shouldGrow && this.snakeBody.length > 0) this.snakeBody.pop()
      this.shouldGrow = false
      this.targetOpacity = 0.4
    }
  }

  resetSnake() {
    this.snakeBody = []
    this.hoveredSquare = null
    this.targetOpacity = 0
    this.trailSquares.clear()
    this.createSpecialBlock()
  }

  handleMouseMove(e: MouseEvent) {
    const rect = this.canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top
    const size = this.options.squareSize
    const baseX = Math.floor(this.gridOffset.x / size) * size
    const baseY = Math.floor(this.gridOffset.y / size) * size
    const gx = Math.floor((cx + this.gridOffset.x - baseX) / size)
    const gy = Math.floor((cy + this.gridOffset.y - baseY) / size)
    if (this.hoveredSquare?.x === gx && this.hoveredSquare?.y === gy) return
    if (this.hoveredSquare) {
      this.snakeBody.unshift({ x: this.hoveredSquare.x, y: this.hoveredSquare.y })
      if (!this.shouldGrow && this.snakeBody.length > 0) this.snakeBody.pop()
      this.shouldGrow = false
    }
    this.hoveredSquare = { x: gx, y: gy }
    this.targetOpacity = 0.6
    if (this.specialBlock && gx === this.specialBlock.x && gy === this.specialBlock.y) {
      this.shouldGrow = true
      this.createSpecialBlock()
    }
  }

  handleMouseLeave() {
    this.hoveredSquare = null
    this.targetOpacity = 0
  }

  createSpecialBlock() {
    const dpr = window.devicePixelRatio || 1
    const size = this.options.squareSize
    const cols = Math.max(4, Math.ceil(this.canvas.width / dpr / size))
    const rows = Math.max(4, Math.ceil(this.canvas.height / dpr / size))
    let x = 1
    let y = 1
    for (let tries = 0; tries < 50; tries++) {
      x = 1 + Math.floor(Math.random() * Math.max(1, cols - 2))
      y = 1 + Math.floor(Math.random() * Math.max(1, rows - 2))
      if (!this.snakeBody.some(b => b.x === x && b.y === y)) break
    }
    this.specialBlock = { color: this.options.specialBlockColor, x, y }
  }

  drawGrid() {
    const dpr = window.devicePixelRatio || 1
    const size = this.options.squareSize
    this.ctx.setTransform(1, 0, 0, 1, 0, 0)
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const baseX = Math.floor(this.gridOffset.x / size) * size
    const baseY = Math.floor(this.gridOffset.y / size) * size
    this.ctx.lineWidth = this.isPhone ? 1 : 0.5

    this.snakeBody.forEach((seg, idx) => {
      const px = Math.round(seg.x * size + baseX - (this.gridOffset.x % size))
      const py = Math.round(seg.y * size + baseY - (this.gridOffset.y % size))
      this.ctx.shadowColor = this.options.hoverShadowColor
      this.ctx.shadowBlur = 15
      if (idx === 0) {
        this.ctx.fillStyle = this.options.snakeHeadColor
      } else {
        const factor = Math.pow(this.options.snakeColorDecay, idx)
        this.ctx.fillStyle = `rgba(100, 181, 246, ${Math.max(0.18, factor * 0.85)})`
      }
      this.ctx.fillRect(px, py, size, size)
      this.ctx.shadowColor = 'transparent'
      this.ctx.shadowBlur = 0
    })

    const viewW = this.canvas.width / dpr
    const viewH = this.canvas.height / dpr

    for (let x = baseX; x < viewW + size; x += size) {
      for (let y = baseY; y < viewH + size; y += size) {
        const px = Math.round(x - (this.gridOffset.x % size))
        const py = Math.round(y - (this.gridOffset.y % size))
        const gx = Math.floor((x - baseX) / size)
        const gy = Math.floor((y - baseY) / size)

        if (this.specialBlock && gx === this.specialBlock.x && gy === this.specialBlock.y) {
          this.ctx.shadowColor = 'rgba(255, 255, 255, 0.5)'
          this.ctx.shadowBlur = 20
          this.ctx.fillStyle = this.specialBlock.color
          this.ctx.fillRect(px, py, size, size)
          this.ctx.shadowColor = 'transparent'
          this.ctx.shadowBlur = 0
        }

        if (this.hoveredSquare && gx === this.hoveredSquare.x && gy === this.hoveredSquare.y) {
          this.ctx.shadowColor = this.options.hoverShadowColor
          this.ctx.shadowBlur = 15
          this.ctx.fillStyle = `rgba(100, 181, 246, ${this.currentOpacity})`
          this.ctx.fillRect(px, py, size, size)
          this.ctx.shadowColor = 'transparent'
          this.ctx.shadowBlur = 0
        }

        this.ctx.strokeStyle = this.options.borderColor
        this.ctx.strokeRect(px, py, size, size)
      }
    }

    const grad = this.ctx.createRadialGradient(
      viewW / 2,
      viewH / 2,
      0,
      viewW / 2,
      viewH / 2,
      Math.sqrt(viewW * viewW + viewH * viewH) / 2,
    )
    grad.addColorStop(0, 'rgba(6, 6, 6, 0)')
    grad.addColorStop(1, '#060606')
    this.ctx.fillStyle = grad
    this.ctx.fillRect(0, 0, viewW, viewH)
  }

  updateAnimation(ts: number) {
    if (!this.lastTimestamp) this.lastTimestamp = ts
    const dt = ts - this.lastTimestamp
    this.lastTimestamp = ts

    if (this.currentOpacity !== this.targetOpacity) {
      const progress = Math.min(dt / this.options.transitionDuration, 1)
      this.currentOpacity += (this.targetOpacity - this.currentOpacity) * progress
    }

    const step = Math.max(this.isPhone ? 0.8 * this.options.speed : this.options.speed, 0)
    const size = this.options.squareSize
    this.gridOffset.x = (this.gridOffset.x - step + size) % size
    this.gridOffset.y = (this.gridOffset.y - step + size) % size

    this.drawGrid()
    this.animationFrame = requestAnimationFrame(t => this.updateAnimation(t))
  }

  animate() {
    this.animationFrame = requestAnimationFrame(t => this.updateAnimation(t))
  }

  handleVisibilityChange() {
    if (document.hidden || this.motionPreference.matches) {
      if (this.animationFrame) {
        cancelAnimationFrame(this.animationFrame)
        this.animationFrame = null
      }
      if (!document.hidden) this.drawGrid()
    } else if (!this.animationFrame) {
      this.lastTimestamp = 0
      this.animate()
    }
  }

  destroy() {
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame)
    if (this.specialBlockTimer) clearTimeout(this.specialBlockTimer)
    window.removeEventListener('resize', this.boundResize)
    this.canvas.removeEventListener('mousemove', this.boundMouseMove)
    this.canvas.removeEventListener('mouseleave', this.boundMouseLeave)
    document.removeEventListener('visibilitychange', this.boundVisibility)
    this.motionPreference.removeEventListener('change', this.boundVisibility)
    if (this.isPhone && this.handleTouchStart) {
      this.canvas.removeEventListener('touchstart', this.handleTouchStart)
      if (this.handleTouchMoveEvent) this.canvas.removeEventListener('touchmove', this.handleTouchMoveEvent)
      if (this.handleTouchEndEvent) this.canvas.removeEventListener('touchend', this.handleTouchEndEvent)
      if (this.handleTouchCancel) this.canvas.removeEventListener('touchcancel', this.handleTouchCancel)
    }
  }
}
