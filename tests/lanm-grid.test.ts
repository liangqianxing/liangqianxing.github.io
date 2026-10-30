import assert from 'node:assert/strict'
import test from 'node:test'
import { GridAnimation, type GridAnimationOptions } from '../utils/lanm-grid.ts'

type Listener = (event: any) => void

class MockEvents {
  listeners = new Map<string, Set<Listener>>()
  addEventListener(name: string, listener: Listener) {
    if (!this.listeners.has(name)) this.listeners.set(name, new Set())
    this.listeners.get(name)!.add(listener)
  }
  removeEventListener(name: string, listener: Listener) { this.listeners.get(name)?.delete(listener) }
  emit(name: string, event: any = {}) { this.listeners.get(name)?.forEach(listener => listener(event)) }
  get listenerCount() { return [...this.listeners.values()].reduce((sum, listeners) => sum + listeners.size, 0) }
}

function fixture(options: GridAnimationOptions = {}, reducedMotion = false) {
  const descriptors = new Map<string, PropertyDescriptor | undefined>()
  const install = (name: string, value: unknown) => {
    descriptors.set(name, Object.getOwnPropertyDescriptor(globalThis, name))
    Object.defineProperty(globalThis, name, { value, configurable: true, writable: true })
  }
  const frames = new Map<number, FrameRequestCallback>()
  let nextFrame = 1
  let time = 1000
  const media = Object.assign(new MockEvents(), { matches: reducedMotion })
  const documentMock = Object.assign(new MockEvents(), { hidden: false })
  const windowMock = Object.assign(new MockEvents(), {
    devicePixelRatio: 1,
    innerWidth: 960,
    innerHeight: 640,
    matchMedia: () => media,
  })
  const bounds = { left: 100, top: 40, width: 960, height: 640, right: 1060, bottom: 680 }
  const obstacles: Array<{ getBoundingClientRect: () => typeof bounds }> = []
  const surface = Object.assign(new MockEvents(), {
    getBoundingClientRect: () => bounds,
    querySelectorAll: () => obstacles,
  })
  const fills: Array<{ x: number; y: number; width: number; height: number; opacity: number; color: unknown }> = []
  const context = {
    globalAlpha: 1,
    fillStyle: '' as unknown,
    shadowColor: '',
    shadowBlur: 0,
    strokeStyle: '',
    lineWidth: 0,
    setTransform() {}, clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
    createRadialGradient: () => ({ addColorStop() {} }),
    fillRect(x: number, y: number, width: number, height: number) {
      fills.push({ x, y, width, height, opacity: this.globalAlpha, color: this.fillStyle })
    },
  }
  const canvas = Object.assign(new MockEvents(), {
    parentElement: surface,
    width: 0, height: 0, style: {} as Record<string, string>,
    getContext: () => context,
    getBoundingClientRect: () => bounds,
  })
  install('window', windowMock)
  install('document', documentMock)
  install('navigator', { userAgent: 'Desktop regression test', vibrate() {} })
  install('requestAnimationFrame', (callback: FrameRequestCallback) => {
    const id = nextFrame++
    frames.set(id, callback)
    return id
  })
  install('cancelAnimationFrame', (id: number) => { frames.delete(id) })
  const grid = new GridAnimation(canvas as unknown as HTMLCanvasElement, { speed: 0, ...options })
  grid.init()
  const frame = (milliseconds = 16) => {
    time += milliseconds
    const pending = [...frames.values()]
    frames.clear()
    pending.forEach(callback => callback(time))
  }
  const advance = (milliseconds: number) => {
    for (let elapsed = 0; elapsed < milliseconds; elapsed += 16) frame()
  }
  const mouse = (x: number, y: number) => surface.emit('mousemove', { clientX: bounds.left + x, clientY: bounds.top + y })
  const dispose = () => {
    grid.destroy()
    for (const [name, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor)
      else Reflect.deleteProperty(globalThis, name)
    }
  }
  return { grid, surface, canvas, bounds, obstacles, media, documentMock, windowMock, context, fills, frames, frame, advance, mouse, dispose }
}

function assertConnected(grid: GridAnimation) {
  const cells = [grid.hoveredSquare!, ...grid.snakeBody]
  for (let index = 1; index < cells.length; index++) {
    const distance = Math.abs(cells[index].x - cells[index - 1].x) + Math.abs(cells[index].y - cells[index - 1].y)
    assert.equal(distance, 1, 'every consecutive snake segment must share an edge')
  }
}

test('a visible snake catches a fast diagonal pointer jump on the next frame', () => {
  const f = fixture()
  try {
    f.grid.specialBlock = null
    f.mouse(260, 180)
    assert.equal(f.grid.snakeBody.length, 4, 'the first pointer interaction seeds a body without collecting food')
    assertConnected(f.grid)
    f.mouse(740, 500)
    assert.deepEqual(f.grid.hoveredSquare, { x: 6, y: 4 }, 'an input event must not teleport the head')
    f.frame()
    assertConnected(f.grid)
    assert.deepEqual(f.grid.hoveredSquare, { x: 18, y: 12 })
    const head = { ...f.grid.hoveredSquare! }
    f.advance(400)
    assert.deepEqual(f.grid.hoveredSquare, head, 'the snake stops after catching up')
    assert.equal(f.grid.snakeBody.length, 4)
  } finally { f.dispose() }
})

test('continuous pointer motion stays within the current grid cell at different frame rates', () => {
  for (const frameDuration of [8, 16, 33]) {
    const f = fixture()
    try {
      f.grid.specialBlock = null
      f.mouse(100, 100)
      const points = [
        [300, 100], [540, 180], [820, 260], [820, 500],
        [580, 500], [300, 420], [100, 300], [100, 100],
      ]
      for (const [x, y] of points) {
        f.mouse(x, y)
        f.frame(frameDuration)
        assert.deepEqual(f.grid.hoveredSquare, { x: Math.floor(x / 40), y: Math.floor(y / 40) }, `the snake must not accumulate lag at ${frameDuration}ms per frame`)
        assertConnected(f.grid)
        assert.equal(f.grid.snakeBody.length, 4)
      }
    } finally { f.dispose() }
  }
})

test('food on the chased path grows the snake exactly once and respawns off the head and body', () => {
  const f = fixture()
  const originalRandom = Math.random
  Math.random = () => 0
  try {
    f.grid.specialBlock = null
    f.mouse(260, 180)
    const originalLength = f.grid.snakeBody.length + 1
    f.grid.specialBlock = { x: 7, y: 4, color: 'gold' }
    f.mouse(740, 180)
    f.frame()
    assert.deepEqual(f.grid.hoveredSquare, { x: 18, y: 4 })
    assert.equal(f.grid.snakeBody.length + 1, originalLength + 1)
    assert.ok(f.grid.specialBlock)
    const occupied = [f.grid.hoveredSquare!, ...f.grid.snakeBody]
    assert.ok(occupied.every(cell => cell.x !== f.grid.specialBlock!.x || cell.y !== f.grid.specialBlock!.y))
    f.advance(500)
    assert.equal(f.grid.snakeBody.length + 1, originalLength + 1, 'remaining on the old food cell must not grow repeatedly')
    assertConnected(f.grid)
  } finally {
    Math.random = originalRandom
    f.dispose()
  }
})

test('vacated tail cells render a fading trail that expires after movement stops', () => {
  const f = fixture({ trailDuration: 500 })
  try {
    f.grid.specialBlock = null
    f.mouse(260, 180)
    f.mouse(620, 180)
    f.frame()
    assert.deepEqual(f.grid.hoveredSquare, { x: 15, y: 4 })
    assert.ok(f.grid.trailSquares.size > 0)
    const trail = [...f.grid.trailSquares.values()][0]
    f.fills.length = 0
    f.grid.drawGrid()
    assert.ok(f.fills.some(fill => fill.x === trail.x * 40 + 1 && fill.y === trail.y * 40 + 1 && fill.opacity > 0 && fill.opacity < 1), 'a trail cell must actually be drawn')
    const opacity = trail.opacity
    f.advance(32)
    assert.ok(trail.opacity < opacity)
    f.advance(550)
    assert.equal(f.grid.trailSquares.size, 0)
    f.surface.emit('mouseleave')
    f.advance(1200)
    assert.equal(f.grid.hoveredSquare, null)
    assert.deepEqual(f.grid.snakeBody, [])
  } finally { f.dispose() }
})

test('inactive pages and hidden documents stop frames and ignore pointer input', () => {
  const f = fixture()
  try {
    f.mouse(260, 180)
    f.mouse(620, 180)
    const head = { ...f.grid.hoveredSquare! }
    f.grid.setActive(false)
    assert.equal(f.frames.size, 0)
    f.mouse(820, 420)
    f.advance(1000)
    assert.deepEqual(f.grid.hoveredSquare, head)
    f.grid.setActive(true)
    assert.equal(f.frames.size, 1)
    f.documentMock.hidden = true
    f.documentMock.emit('visibilitychange')
    assert.equal(f.frames.size, 0)
    f.mouse(900, 500)
    f.advance(1000)
    assert.deepEqual(f.grid.hoveredSquare, head)
    f.documentMock.hidden = false
    f.documentMock.emit('visibilitychange')
    assert.equal(f.frames.size, 1)
  } finally { f.dispose() }
})

test('food respects profile obstacles and remains fully visible after a resize', () => {
  const f = fixture()
  try {
    f.obstacles.push({ getBoundingClientRect: () => ({ left: 300, right: 740, top: 200, bottom: 440, width: 440, height: 240 }) })
    for (let i = 0; i < 20; i++) {
      f.grid.createSpecialBlock()
      const food = f.grid.specialBlock!
      assert.ok(food)
      const left = f.bounds.left + food.x * 40
      const top = f.bounds.top + food.y * 40
      assert.ok(left + 40 <= 300 || left >= 740 || top + 40 <= 200 || top >= 440, 'food must not overlap the profile')
    }
    f.bounds.width = 390
    f.bounds.height = 844
    f.bounds.right = f.bounds.left + 390
    f.bounds.bottom = f.bounds.top + 844
    f.windowMock.emit('resize')
    assert.equal(f.canvas.width, 390)
    assert.equal(f.canvas.height, 844)
    const food = f.grid.specialBlock!
    assert.ok(food)
    assert.ok(food.x * 40 >= 0 && (food.x + 1) * 40 <= 390)
    assert.ok(food.y * 40 >= 0 && (food.y + 1) * 40 <= 844)
  } finally { f.dispose() }
})

test('input listeners cover the parent surface and destruction removes every listener and pending frame', () => {
  const f = fixture()
  try {
    assert.equal(f.canvas.listenerCount, 0, 'the canvas must not be the only input surface')
    assert.ok(f.surface.listeners.get('mousemove')?.size)
    assert.ok(f.surface.listeners.get('touchmove')?.size)
    f.mouse(260, 180)
    assert.ok(f.grid.hoveredSquare)
    f.grid.destroy()
    assert.equal(f.surface.listenerCount, 0)
    assert.equal(f.windowMock.listenerCount, 0)
    assert.equal(f.documentMock.listenerCount, 0)
    assert.equal(f.media.listenerCount, 0)
    assert.equal(f.frames.size, 0)
    f.mouse(620, 180)
    assert.equal(f.grid.hoveredSquare, null)
  } finally { f.dispose() }
})

test('reduced motion uses a responsive static highlight without animation or trailing', () => {
  const f = fixture({}, true)
  try {
    assert.equal(f.frames.size, 0)
    f.grid.specialBlock = null
    f.fills.length = 0
    f.mouse(260, 180)
    f.mouse(620, 420)
    assert.deepEqual(f.grid.hoveredSquare, { x: 15, y: 10 })
    assert.deepEqual(f.grid.snakeBody, [])
    assert.equal(f.grid.trailSquares.size, 0)
    assert.equal(f.frames.size, 0)
    assert.ok(f.fills.some(fill => fill.x === 601 && fill.y === 401 && fill.opacity > 0), 'pointer input must repaint the static highlight')
    f.surface.emit('mouseleave')
    assert.equal(f.grid.currentOpacity, 0)
  } finally { f.dispose() }
})

test('reversing toward the tail takes a connected detour without collapsing the short snake', () => {
  const f = fixture()
  try {
    f.grid.specialBlock = null
    f.mouse(260, 180)
    const assertDistinct = () => {
      const cells = [f.grid.hoveredSquare!, ...f.grid.snakeBody]
      assert.equal(new Set(cells.map(cell => `${cell.x},${cell.y}`)).size, cells.length, 'the visible snake must not occupy the same cell twice')
      assertConnected(f.grid)
    }
    assertDistinct()
    f.mouse(100, 180)
    for (let i = 0; i < 100; i++) {
      f.frame()
      assertDistinct()
    }
    assert.deepEqual(f.grid.hoveredSquare, { x: 2, y: 4 }, 'the detour must eventually reach the pointer behind the original body')
    assert.equal(f.grid.snakeBody.length, 4)
  } finally { f.dispose() }
})

test('short landscape screens still place fully visible food outside a central profile obstacle', () => {
  const f = fixture()
  try {
    f.obstacles.push({ getBoundingClientRect: () => ({
      left: f.bounds.left, right: f.bounds.right, width: f.bounds.width,
      top: f.bounds.top + 40, bottom: f.bounds.bottom - 40, height: f.bounds.height - 80,
    }) })
    for (const height of [160, 200]) {
      f.bounds.height = height
      f.bounds.bottom = f.bounds.top + height
      f.windowMock.emit('resize')
      for (let attempt = 0; attempt < 10; attempt++) {
        f.grid.createSpecialBlock()
        const food = f.grid.specialBlock
        assert.ok(food, `food must exist with a ${height}px viewport and only edge rows available`)
        assert.ok(food.x * 40 >= 0 && (food.x + 1) * 40 <= f.bounds.width)
        assert.ok(food.y * 40 >= 0 && (food.y + 1) * 40 <= height)
        assert.ok((food.y + 1) * 40 <= 40 || food.y * 40 >= height - 40, 'food must occupy an unobstructed edge row')
      }
    }
  } finally { f.dispose() }
})

test('touch release suppresses compatibility mouse movement and interrupted touches recover after visibility changes', () => {
  const f = fixture()
  const originalNow = Date.now
  let now = 2000
  Date.now = () => now
  const touch = (name: string, x: number, y: number) => f.surface.emit(name, {
    touches: [{ clientX: f.bounds.left + x, clientY: f.bounds.top + y }],
  })
  try {
    f.grid.specialBlock = null
    touch('touchstart', 260, 180)
    assert.ok(f.grid.hoveredSquare)
    touch('touchmove', 620, 180)
    f.advance(250)
    now += 300
    f.surface.emit('touchend')
    const releasedHead = { ...f.grid.hoveredSquare! }
    assert.equal(f.grid.targetOpacity, 0)
    f.mouse(820, 420)
    assert.equal(f.grid.targetOpacity, 0, 'the compatibility mouse event must not restart the released touch')
    assert.deepEqual(f.grid.hoveredSquare, releasedHead)
    f.advance(1200)
    assert.equal(f.grid.hoveredSquare, null, 'a released touch should fade even if compatibility mouse input arrives')

    now += 1000
    f.mouse(300, 220)
    assert.equal(f.grid.targetOpacity, 1, 'a later real mouse interaction remains available')
    touch('touchstart', 340, 260)
    touch('touchmove', 740, 460)
    f.documentMock.hidden = true
    f.documentMock.emit('visibilitychange')
    assert.equal(f.frames.size, 0)
    assert.equal(f.grid.targetOpacity, 0, 'hiding the document releases an interrupted touch')
    now += 1000
    f.documentMock.hidden = false
    f.documentMock.emit('visibilitychange')
    f.mouse(540, 340)
    assert.equal(f.grid.targetOpacity, 1, 'interrupted touch state must not keep blocking mouse movement')
    f.advance(1500)
    assert.deepEqual(f.grid.hoveredSquare, { x: 13, y: 8 })
  } finally {
    Date.now = originalNow
    f.dispose()
  }
})
