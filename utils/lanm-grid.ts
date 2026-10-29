interface GridCell { x: number; y: number }

export interface GridAnimationOptions {
  borderColor?: string
  direction?: 'right' | 'left' | 'up' | 'down' | 'diagonal'
  hoverFillColor?: string
  hoverShadowColor?: string
  initialSnakeLength?: number
  maxSnakeLength?: number
  snakeStepDuration?: number
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
  hoveredSquare: GridCell | null = null
  animationFrame: number | null = null
  currentOpacity = 0
  targetOpacity = 0
  lastTimestamp = 0
  trailSquares = new Map<string, GridCell & { opacity: number }>()
  specialBlock: (GridCell & { color: string }) | null = null
  snakeBody: GridCell[] = []
  isPhone = false
  private surface: HTMLElement
  private active = true
  private pointer: GridCell | null = null
  private snakeLength: number
  private stepElapsed = 0
  private heading = { x: 1, y: 0 }
  private touching = false
  private ignoreMouseUntil = 0
  private touchStart: (GridCell & { time: number }) | null = null
  private lastTap: (GridCell & { time: number }) | null = null
  private motionPreference: MediaQueryList
  private boundResize = () => this.resizeCanvas()
  private boundMouseMove = (e: MouseEvent) => this.handleMouseMove(e)
  private boundMouseLeave = () => this.handleMouseLeave()
  private boundVisibility = () => this.handleVisibilityChange()
  private boundTouchStart = (e: TouchEvent) => this.onTouchStart(e)
  private boundTouchMove = (e: TouchEvent) => this.onTouchMove(e)
  private boundTouchEnd = () => this.onTouchEnd()
  private boundTouchCancel = () => { this.touchStart = null; this.lastTap = null; this.handleTouchEnd() }

  constructor(canvas: HTMLCanvasElement, opts: GridAnimationOptions = {}) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.surface = canvas.parentElement || canvas
    this.isPhone = /Mobile|Android|iOS|iPhone|iPad|iPod|Windows Phone/i.test(navigator.userAgent)
    this.options = {
      borderColor: opts.borderColor || 'rgba(100, 181, 246, 0.08)',
      direction: opts.direction || 'diagonal',
      hoverFillColor: opts.hoverFillColor || 'rgba(100, 181, 246, 0.8)',
      hoverShadowColor: opts.hoverShadowColor || 'rgba(100, 181, 246, 0.8)',
      initialSnakeLength: Math.max(2, opts.initialSnakeLength ?? 5),
      maxSnakeLength: Math.max(2, opts.maxSnakeLength ?? 60),
      snakeStepDuration: Math.max(16, opts.snakeStepDuration ?? 65),
      snakeColorDecay: opts.snakeColorDecay ?? 0.85,
      snakeGradientStops: opts.snakeGradientStops ?? 5,
      snakeHeadColor: opts.snakeHeadColor || 'rgba(144, 202, 249, 0.95)',
      snakeTailColor: opts.snakeTailColor || 'rgba(100, 181, 246, 0.2)',
      specialBlockColor: opts.specialBlockColor || 'rgba(255, 193, 7, 0.85)',
      specialHoverColor: opts.specialHoverColor || 'rgba(255, 235, 59, 0.9)',
      speed: opts.speed ?? (this.isPhone ? 0.03 : 0.05),
      squareSize: Math.max(12, opts.squareSize ?? (this.isPhone ? 50 : 40)),
      touchSensitivity: opts.touchSensitivity ?? (this.isPhone ? 1.2 : 1),
      trailDuration: Math.max(1, opts.trailDuration ?? (this.isPhone ? 2000 : 1500)),
      transitionDuration: Math.max(1, opts.transitionDuration ?? (this.isPhone ? 150 : 200)),
      vibrationEnabled: opts.vibrationEnabled ?? this.isPhone,
    }
    this.snakeLength = Math.min(this.options.initialSnakeLength, this.options.maxSnakeLength)
    this.motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)')
  }

  init() {
    if (!this.ctx) return
    this.resizeCanvas()
    window.addEventListener('resize', this.boundResize)
    // Listen on the whole main screen so the profile and its buttons do not interrupt the chase.
    this.surface.addEventListener('mousemove', this.boundMouseMove)
    this.surface.addEventListener('mouseleave', this.boundMouseLeave)
    this.surface.addEventListener('touchstart', this.boundTouchStart, { passive: true })
    this.surface.addEventListener('touchmove', this.boundTouchMove, { passive: true })
    this.surface.addEventListener('touchend', this.boundTouchEnd, { passive: true })
    this.surface.addEventListener('touchcancel', this.boundTouchCancel, { passive: true })
    document.addEventListener('visibilitychange', this.boundVisibility)
    this.motionPreference.addEventListener('change', this.boundVisibility)
    this.handleVisibilityChange()
  }

  resizeCanvas() {
    const dpr = window.devicePixelRatio || 1
    const bounds = this.surface.getBoundingClientRect()
    const w = bounds.width || window.innerWidth
    const h = bounds.height || window.innerHeight
    this.canvas.width = Math.floor(w * dpr)
    this.canvas.height = Math.floor(h * dpr)
    this.canvas.style.width = `${w}px`
    this.canvas.style.height = `${h}px`
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    // Reflow can put an old snake or food outside the new viewport.
    this.clearSnake()
    this.createSpecialBlock()
    this.drawGrid()
  }

  setActive(active: boolean) {
    if (this.active === active) return
    this.active = active
    this.handleMouseLeave()
    this.touching = false
    this.touchStart = null
    this.lastTap = null
    this.handleVisibilityChange()
  }

  private cellAt(cx: number, cy: number): GridCell {
    return {
      x: Math.floor((cx + this.gridOffset.x) / this.options.squareSize),
      y: Math.floor((cy + this.gridOffset.y) / this.options.squareSize),
    }
  }

  private sameCell(a: GridCell | null, b: GridCell | null) {
    return !!a && !!b && a.x === b.x && a.y === b.y
  }

  private setPointer(cx: number, cy: number) {
    if (!this.active || document.hidden) return
    this.pointer = { x: cx, y: cy }
    const target = this.cellAt(cx, cy)
    this.targetOpacity = 1
    if (!this.hoveredSquare) this.seedSnake(target)
    if (this.motionPreference.matches) {
      // Reduced motion keeps a responsive, static highlight without a moving snake or trail.
      this.hoveredSquare = target
      this.snakeBody = []
      this.trailSquares.clear()
      this.currentOpacity = 1
      this.collectFood()
      this.drawGrid()
    }
  }

  private seedSnake(head: GridCell) {
    this.hoveredSquare = head
    this.snakeBody = []
    const size = this.options.squareSize
    const width = this.canvas.width / (window.devicePixelRatio || 1)
    const leftRoom = Math.floor((head.x * size - this.gridOffset.x) / size)
    const tailDirection = leftRoom >= this.snakeLength - 1 ? -1 : 1
    this.heading = { x: -tailDirection, y: 0 }
    for (let i = 1; i < this.snakeLength; i++) {
      const cell = { x: head.x + i * tailDirection, y: head.y }
      if (cell.x * size - this.gridOffset.x >= width) break
      this.snakeBody.push(cell)
    }
    this.stepElapsed = 0
    this.collectFood()
  }

  handleMouseMove(e: MouseEvent) {
    if (this.touching || Date.now() < this.ignoreMouseUntil) return
    const rect = this.canvas.getBoundingClientRect()
    this.setPointer(e.clientX - rect.left, e.clientY - rect.top)
  }

  handleMouseLeave() {
    this.pointer = null
    this.targetOpacity = 0
    this.stepElapsed = 0
    if (this.motionPreference.matches) {
      this.currentOpacity = 0
      this.drawGrid()
    }
  }

  handleTouchMove(cx: number, cy: number) {
    this.setPointer(cx, cy)
  }

  private onTouchStart(e: TouchEvent) {
    if (!this.active || e.touches.length !== 1) {
      this.boundTouchCancel()
      return
    }
    const t = e.touches[0]
    this.ignoreMouseUntil = Date.now() + 800
    this.touching = true
    this.touchStart = { x: t.clientX, y: t.clientY, time: Date.now() }
    const rect = this.canvas.getBoundingClientRect()
    this.handleTouchMove(t.clientX - rect.left, t.clientY - rect.top)
  }

  private onTouchMove(e: TouchEvent) {
    if (e.touches.length !== 1) { this.boundTouchCancel(); return }
    if (!this.touching) return
    const t = e.touches[0]
    const rect = this.canvas.getBoundingClientRect()
    this.handleTouchMove(t.clientX - rect.left, t.clientY - rect.top)
    if (this.touchStart && Math.hypot(t.clientX - this.touchStart.x, t.clientY - this.touchStart.y) > 12) {
      this.touchStart = null
    }
  }

  private onTouchEnd() {
    const now = Date.now()
    const tap = this.touchStart && now - this.touchStart.time < 300 ? this.touchStart : null
    if (tap && this.lastTap && now - this.lastTap.time < 350 && Math.hypot(tap.x - this.lastTap.x, tap.y - this.lastTap.y) < 30) {
      this.lastTap = null
      this.resetSnake()
    } else {
      this.lastTap = tap
    }
    this.touchStart = null
    this.handleTouchEnd()
  }

  handleTouchEnd() {
    if (this.touching) this.ignoreMouseUntil = Date.now() + 800
    this.touching = false
    this.handleMouseLeave()
  }

  private clearSnake() {
    this.snakeLength = Math.min(this.options.initialSnakeLength, this.options.maxSnakeLength)
    this.snakeBody = []
    this.hoveredSquare = null
    this.pointer = null
    this.currentOpacity = 0
    this.targetOpacity = 0
    this.stepElapsed = 0
    this.trailSquares.clear()
  }

  resetSnake() {
    this.clearSnake()
    this.createSpecialBlock()
    this.drawGrid()
  }

  private collectFood() {
    if (!this.sameCell(this.hoveredSquare, this.specialBlock)) return
    this.snakeLength = Math.min(this.options.maxSnakeLength, this.snakeLength + 1)
    this.createSpecialBlock()
    if (this.touching && this.options.vibrationEnabled) navigator.vibrate?.(20)
  }

  private stepSnake(target: GridCell) {
    const head = this.hoveredSquare
    if (!head || this.sameCell(head, target)) return
    const dx = target.x - head.x
    const dy = target.y - head.y
    // Each step crosses one edge, even when a mouse event jumps across the entire screen.
    const horizontal = dx !== 0 && (dy === 0 || Math.abs(dx) > Math.abs(dy) || (Math.abs(dx) === Math.abs(dy) && this.heading.y !== 0))
    let next = { x: head.x + (horizontal ? Math.sign(dx) : 0), y: head.y + (horizontal ? 0 : Math.sign(dy)) }
    const occupied = this.snakeBody.slice(0, this.sameCell(next, this.specialBlock) ? undefined : -1)
    if (occupied.some(cell => this.sameCell(cell, next))) {
      const size = this.options.squareSize
      const dpr = window.devicePixelRatio || 1
      const choices = [{ x: 1, y: 0 }, { x: 0, y: 1 }, { x: -1, y: 0 }, { x: 0, y: -1 }]
        .map(dir => ({ x: head.x + dir.x, y: head.y + dir.y }))
        .filter(cell => !occupied.some(body => this.sameCell(body, cell)))
        .filter(cell => (cell.x + 1) * size > this.gridOffset.x && cell.x * size < this.gridOffset.x + this.canvas.width / dpr
          && (cell.y + 1) * size > this.gridOffset.y && cell.y * size < this.gridOffset.y + this.canvas.height / dpr)
        .sort((a, b) => {
          const distance = (cell: GridCell) => Math.abs(cell.x - target.x) + Math.abs(cell.y - target.y)
          const straight = (cell: GridCell) => Number(cell.x - head.x === this.heading.x && cell.y - head.y === this.heading.y)
          return distance(a) - distance(b) || straight(a) - straight(b)
        })
      // A small detour prevents instant reversals from collapsing the visible body.
      next = choices[0] || next
    }
    this.heading = { x: next.x - head.x, y: next.y - head.y }
    this.snakeBody.unshift({ ...head })
    this.hoveredSquare = next
    this.collectFood()
    while (this.snakeBody.length >= this.snakeLength) {
      const tail = this.snakeBody.pop()!
      this.trailSquares.set(`${tail.x},${tail.y}`, { ...tail, opacity: 1 })
    }
  }

  createSpecialBlock() {
    const dpr = window.devicePixelRatio || 1
    const size = this.options.squareSize
    const width = this.canvas.width / dpr
    const height = this.canvas.height / dpr
    const rect = this.canvas.getBoundingClientRect()
    const obstacles = Array.from(this.surface.querySelectorAll('[data-grid-obstacle]'), el => el.getBoundingClientRect())
    const cells: GridCell[] = []
    // Prefer a margin away from the vignette, then use edge cells on short mobile screens.
    for (const margin of [size, 0]) {
      for (let x = Math.ceil((this.gridOffset.x + margin) / size); (x + 1) * size - this.gridOffset.x <= width - margin; x++) {
        for (let y = Math.ceil((this.gridOffset.y + margin) / size); (y + 1) * size - this.gridOffset.y <= height - margin; y++) {
          const cell = { x, y }
          if (this.sameCell(cell, this.hoveredSquare) || this.sameCell(cell, this.specialBlock) || this.snakeBody.some(seg => this.sameCell(seg, cell))) continue
          const px = rect.left + x * size - this.gridOffset.x
          const py = rect.top + y * size - this.gridOffset.y
          if (obstacles.some(b => px < b.right && px + size > b.left && py < b.bottom && py + size > b.top)) continue
          cells.push(cell)
        }
      }
      if (cells.length) break
    }
    const cell = cells[Math.floor(Math.random() * cells.length)]
    this.specialBlock = cell ? { ...cell, color: this.options.specialBlockColor } : null
  }

  private paintCell(cell: GridCell, color: string, opacity = 1, glow = 0) {
    const size = this.options.squareSize
    this.ctx.globalAlpha = Math.max(0, Math.min(1, opacity))
    this.ctx.fillStyle = color
    this.ctx.shadowColor = glow ? color : 'transparent'
    this.ctx.shadowBlur = glow
    this.ctx.fillRect(Math.round(cell.x * size - this.gridOffset.x) + 1, Math.round(cell.y * size - this.gridOffset.y) + 1, size - 2, size - 2)
    this.ctx.shadowBlur = 0
    this.ctx.globalAlpha = 1
  }

  drawGrid() {
    const dpr = window.devicePixelRatio || 1
    const size = this.options.squareSize
    const width = this.canvas.width / dpr
    const height = this.canvas.height / dpr
    this.ctx.setTransform(1, 0, 0, 1, 0, 0)
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    this.trailSquares.forEach(cell => this.paintCell(cell, this.options.hoverFillColor, cell.opacity * 0.2))
    // Blend the bright head colour into the dim tail colour along the body.
    for (let i = this.snakeBody.length - 1; i >= 0; i--) {
      const cell = this.snakeBody[i]
      const strength = Math.pow(this.options.snakeColorDecay, i + 1) * (1 - i / Math.max(this.snakeBody.length, this.options.snakeGradientStops))
      this.paintCell(cell, this.options.snakeTailColor, this.currentOpacity)
      this.paintCell(cell, this.options.snakeHeadColor, strength * this.currentOpacity, 10)
    }
    if (this.specialBlock) this.paintCell(this.specialBlock, this.specialBlock.color, 1, 16)
    if (this.hoveredSquare) this.paintCell(this.hoveredSquare, this.options.snakeHeadColor, this.currentOpacity, 18)

    this.ctx.lineWidth = this.isPhone ? 1 : 0.5
    this.ctx.strokeStyle = this.options.borderColor
    this.ctx.beginPath()
    for (let x = Math.floor(this.gridOffset.x / size) * size - this.gridOffset.x; x <= width; x += size) {
      this.ctx.moveTo(Math.round(x), 0)
      this.ctx.lineTo(Math.round(x), height)
    }
    for (let y = Math.floor(this.gridOffset.y / size) * size - this.gridOffset.y; y <= height; y += size) {
      this.ctx.moveTo(0, Math.round(y))
      this.ctx.lineTo(width, Math.round(y))
    }
    this.ctx.stroke()

    const grad = this.ctx.createRadialGradient(width / 2, height / 2, 0, width / 2, height / 2, Math.hypot(width, height) / 2)
    grad.addColorStop(0, 'rgba(6, 6, 6, 0)')
    grad.addColorStop(1, '#060606')
    this.ctx.fillStyle = grad
    this.ctx.fillRect(0, 0, width, height)
  }

  updateAnimation(ts: number) {
    this.animationFrame = null
    if (!this.active || document.hidden || this.motionPreference.matches) return
    const dt = this.lastTimestamp ? Math.max(0, Math.min(ts - this.lastTimestamp, 100)) : 0
    this.lastTimestamp = ts
    this.currentOpacity += (this.targetOpacity - this.currentOpacity) * (1 - Math.exp(-dt / this.options.transitionDuration))
    for (const [key, trail] of this.trailSquares) {
      trail.opacity -= dt / this.options.trailDuration
      if (trail.opacity <= 0) this.trailSquares.delete(key)
    }

    const drift = Math.max(this.options.speed, 0) * dt / (1000 / 60)
    if (this.options.direction === 'diagonal' || this.options.direction === 'left') this.gridOffset.x += drift
    if (this.options.direction === 'right') this.gridOffset.x -= drift
    if (this.options.direction === 'diagonal' || this.options.direction === 'up') this.gridOffset.y += drift
    if (this.options.direction === 'down') this.gridOffset.y -= drift

    if (this.pointer) {
      const target = this.cellAt(this.pointer.x, this.pointer.y)
      const head = this.hoveredSquare
      const distance = head ? Math.abs(target.x - head.x) + Math.abs(target.y - head.y) : 0
      // Catch up over time rather than teleporting to each incoming pointer position.
      const interval = Math.max(24, this.options.snakeStepDuration / Math.max(1, distance / 7))
      this.stepElapsed = Math.min(this.stepElapsed + dt, interval * 4)
      while (this.stepElapsed >= interval && !this.sameCell(this.hoveredSquare, target)) {
        this.stepSnake(target)
        this.stepElapsed -= interval
      }
      if (this.sameCell(this.hoveredSquare, target)) this.stepElapsed = 0
    } else if (this.currentOpacity < 0.01) {
      this.hoveredSquare = null
      this.snakeBody = []
      this.currentOpacity = 0
    }

    const size = this.options.squareSize
    const dpr = window.devicePixelRatio || 1
    if (this.specialBlock) {
      const px = this.specialBlock.x * size - this.gridOffset.x
      const py = this.specialBlock.y * size - this.gridOffset.y
      if (px < 0 || py < 0 || px + size > this.canvas.width / dpr || py + size > this.canvas.height / dpr) this.createSpecialBlock()
    }
    this.drawGrid()
    this.animationFrame = requestAnimationFrame(t => this.updateAnimation(t))
  }

  handleVisibilityChange() {
    if (this.animationFrame !== null) cancelAnimationFrame(this.animationFrame)
    this.animationFrame = null
    this.lastTimestamp = 0
    this.stepElapsed = 0
    if (document.hidden) {
      this.handleTouchEnd()
      this.touchStart = null
      this.lastTap = null
      return
    }
    if (!this.active) return
    if (this.motionPreference.matches) {
      this.trailSquares.clear()
      this.snakeBody = []
      if (this.pointer) this.hoveredSquare = this.cellAt(this.pointer.x, this.pointer.y)
      this.currentOpacity = this.targetOpacity
      this.drawGrid()
    } else {
      this.animationFrame = requestAnimationFrame(t => this.updateAnimation(t))
    }
  }

  destroy() {
    this.active = false
    if (this.animationFrame !== null) cancelAnimationFrame(this.animationFrame)
    this.animationFrame = null
    window.removeEventListener('resize', this.boundResize)
    this.surface.removeEventListener('mousemove', this.boundMouseMove)
    this.surface.removeEventListener('mouseleave', this.boundMouseLeave)
    this.surface.removeEventListener('touchstart', this.boundTouchStart)
    this.surface.removeEventListener('touchmove', this.boundTouchMove)
    this.surface.removeEventListener('touchend', this.boundTouchEnd)
    this.surface.removeEventListener('touchcancel', this.boundTouchCancel)
    document.removeEventListener('visibilitychange', this.boundVisibility)
    this.motionPreference.removeEventListener('change', this.boundVisibility)
    this.clearSnake()
  }
}
