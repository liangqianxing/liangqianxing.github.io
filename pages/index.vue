<template>
  <main class="lanm-gateway">
    <noscript>
      <nav class="lanm-fallback" aria-label="主页入口">
        <h1>{{ academicProfile.name }} · {{ academicProfile.nameCN }}</h1>
        <a href="/blog">个人博客</a><a href="/academic">学术科研</a><a href="/academic#projects">开源项目</a><a href="/learning">课程公开课</a>
      </nav>
    </noscript>
    <!-- 第一屏：流体仿真背景 + 个人标题 + SVG 波形遮罩过渡 -->
    <div ref="introEl" class="content content-intro" :class="{ active: currentPage === 'intro' }" :inert="currentPage !== 'intro'" :aria-hidden="currentPage !== 'intro'">
      <div class="content-inner">
        <canvas id="background" ref="bgCanvas"></canvas>
        <div class="wrap fade" :class="{ in: introLoaded }">
          <a
            class="travellings-link"
            :href="academicProfile.github"
            target="_blank"
            rel="noopener noreferrer"
            title="GitHub · LiangQianXing"
          >
            <span>GitHub · @LiangQianXing ↗</span>
          </a>
          <h2 class="content-title">{{ academicProfile.name }}</h2>
          <h3 class="content-subtitle" :aria-label="subtitleText">
            <template v-if="subtitleChars.length">
              <span
                v-for="(ch, idx) in subtitleChars"
                :key="idx"
                :style="{ animationDelay: `${(idx + 1) * 45}ms` }"
              >{{ ch === ' ' ? ' ' : ch }}</span>
            </template>
            <template v-else>&nbsp;</template>
          </h3>
          <button type="button" class="enter" @click="switchToMain">Next</button>
          <div class="arrow arrow-1" @mouseenter="switchToMain" @click="switchToMain"></div>
          <div class="arrow arrow-2" @mouseenter="switchToMain" @click="switchToMain"></div>
        </div>
      </div>
      <div class="shape-wrap">
        <svg
          ref="shapeSvg"
          class="shape"
          width="100%"
          height="100vh"
          preserveAspectRatio="none"
          viewBox="0 0 1440 800"
        >
          <path
            ref="shapePath"
            :d="PATH_INTRO"
            :data-path-to="PATH_MAIN"
          ></path>
        </svg>
      </div>
    </div>

    <!-- 第二屏：个人名片 + 分类入口 + 贪吃蛇动态网格 -->
    <div class="content content-main" :inert="currentPage !== 'main'" :aria-hidden="currentPage !== 'main'">
      <div id="card">
        <button type="button" class="arrowmain arrowmain-1" aria-label="返回首屏" @click="switchToIntro"></button>
        <div class="arrowmain arrowmain-2" aria-hidden="true"></div>
        <div class="card-inner fade" :class="{ in: mainLoaded }">
          <header>
            <img :src="academicProfile.avatar" width="104" height="104" :alt="academicProfile.name" />
            <h1>{{ academicProfile.name }} · {{ academicProfile.nameCN }}</h1>
            <h2 id="signature">LLM Agents · Post-Training · AI Systems</h2>
          </header>
          <ul class="categories-list">
            <li
              v-for="cat in categoryData"
              :key="cat.id"
              class="category-item"
            >
              <button
                type="button"
                class="category-link"
                :aria-label="cat.text"
                @click="showCategoryPage(cat.id)"
              >
                <span class="lanm-icon" v-html="getIconSvg(cat.icon)"></span>
                <span>{{ cat.text }}</span>
              </button>
            </li>
          </ul>
        </div>
      </div>
      <canvas id="gridCanvas" ref="gridCanvas" class="grid-background"></canvas>
    </div>

    <!-- 第三层：分类详情抽屉卡片 -->
    <div
      class="content content-category"
      :class="{ active: categoryActive, hiding: categoryHiding }"
      :inert="currentPage !== 'category'"
      :aria-hidden="currentPage !== 'category'"
    >
      <div id="category-card">
        <div class="category-card-inner fade" :class="{ in: categoryInnerIn, out: categoryInnerOut }">
          <div class="category-header">
            <button type="button" class="back-button" aria-label="返回" @click="hideCategoryPage">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              <span>返回</span>
            </button>
            <h2 ref="categoryHeading" class="category-title" tabindex="-1">{{ activeCategory?.text }}</h2>
          </div>
          <ul v-if="activeCategory" class="category-projects-list">
            <li
              v-for="(item, idx) in activeCategory.projects"
              :key="item.href + idx"
              :style="{ animationDelay: `${(idx + 1) * 35}ms` }"
            >
              <NuxtLink
                v-if="!item.external"
                :to="item.href"
                :aria-label="item.text"
              >
                <span class="lanm-icon" v-html="getIconSvg(item.icon)"></span>
                <span class="project-label">{{ item.text }}</span>
                <span v-if="item.sub" class="project-sub">{{ item.sub }}</span>
              </NuxtLink>
              <a
                v-else
                :href="item.href"
                :aria-label="item.text"
                target="_blank"
                rel="noopener noreferrer"
              >
                <span class="lanm-icon" v-html="getIconSvg(item.icon)"></span>
                <span class="project-label">{{ item.text }}</span>
                <span v-if="item.sub" class="project-sub">{{ item.sub }}</span>
              </a>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </main>
</template>

<script setup lang="ts">
import anime from 'animejs'
import { academicProfile } from '~/data/academic'
import { GridAnimation } from '~/utils/lanm-grid'
import { initFluidBackground } from '~/utils/lanm-fluid'

definePageMeta({ layout: 'gateway' })

const appConfig = useAppConfig()

const PATH_INTRO = 'M-44-50C-52.71 28.52 15.86 8.186 184 14.69 383.3 22.39 462.5 12.58 638 14 835.5 15.6 987 6.4 1194 13.86 1661 30.68 1652-36.74 1582-140.1 1512-243.5 15.88-589.5-44-50Z'
const PATH_MAIN = 'M -44,-50 C -137.1,117.4 67.86,445.5 236,452 435.3,459.7 500.5,242.6 676,244 873.5,245.6 957,522.4 1154,594 1593,753.7 1793,226.3 1582,-126 1371,-478.3 219.8,-524.2 -44,-50 Z'

const subtitleText = 'LLM Agents · Post-Training · AI Infra'
const subtitleChars = ref<string[]>([])

const introEl = ref<HTMLElement | null>(null)
const bgCanvas = ref<HTMLCanvasElement | null>(null)
const gridCanvas = ref<HTMLCanvasElement | null>(null)
const shapeSvg = ref<SVGSVGElement | null>(null)
const shapePath = ref<SVGPathElement | null>(null)
const categoryHeading = ref<HTMLElement | null>(null)
const pendingTimers = new Set<ReturnType<typeof setTimeout>>()
let lastCategoryButton: HTMLElement | null = null

function motionDuration(milliseconds: number) {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 1 : milliseconds
}

function later(callback: () => void, milliseconds: number) {
  const timer = setTimeout(() => { pendingTimers.delete(timer); callback() }, motionDuration(milliseconds))
  pendingTimers.add(timer)
}

const currentPage = ref<'intro' | 'main' | 'category'>('intro')
const introLoaded = ref(false)
const mainLoaded = ref(false)
const isSwitching = ref(false)

const activeCategoryId = ref<string>('blog')
const categoryActive = ref(false)
const categoryHiding = ref(false)
const categoryInnerIn = ref(false)
const categoryInnerOut = ref(false)

interface CategoryProject {
  href: string
  icon: string
  text: string
  sub?: string
  external?: boolean
}

interface CategoryGroup {
  id: string
  icon: string
  text: string
  projects: CategoryProject[]
}

const categoryData: CategoryGroup[] = [
  {
    id: 'blog',
    icon: 'personal',
    text: '个人博客',
    projects: [
      { href: '/blog', icon: 'blog', text: 'gu.log 首页', sub: '技术博客' },
      { href: '/posts', icon: 'book', text: '全部文章', sub: '深度长文库' },
      { href: '/tags', icon: 'tag', text: '标签索引', sub: '主题分类' },
      { href: '/friends', icon: 'users', text: '友情链接', sub: 'Friends' },
      { href: '/about', icon: 'user', text: '关于我', sub: '个人履历' },
      { href: '/rss.xml', icon: 'rss', text: 'RSS 订阅', sub: 'Feed', external: true },
    ],
  },
  {
    id: 'academic',
    icon: 'scholar',
    text: '学术科研',
    projects: [
      { href: '/academic', icon: 'scholar', text: '学术主页', sub: 'Publications & CV' },
      { href: 'https://aclanthology.org/2026.acl-demo.6/', icon: 'paper', text: 'AutoFigure-Edit', sub: 'ACL 2026 Demo', external: true },
      { href: 'https://arxiv.org/abs/2603.19293', icon: 'paper', text: 'LLM-MRD', sub: 'DASFAA 2026 Oral', external: true },
      { href: 'https://arxiv.org/abs/2604.09590', icon: 'paper', text: 'DeepReviewer 2.0', sub: 'arXiv 2026', external: true },
      { href: 'https://arxiv.org/abs/2510.17136', icon: 'paper', text: 'In-situ Autoguidance', sub: 'ICML 2025 W', external: true },
      { href: academicProfile.scholar, icon: 'globe', text: 'Google Scholar', sub: '学术档案', external: true },
    ],
  },
  {
    id: 'projects',
    icon: 'code',
    text: '开源项目',
    projects: [
      { href: 'https://github.com/ResearAI/DeepScientist', icon: 'cpu', text: 'DeepScientist', sub: '科研智能体平台', external: true },
      { href: 'https://github.com/LiuMengxuan04/MiniCode', icon: 'terminal', text: 'MiniCode', sub: '终端 Coding Agent', external: true },
      { href: 'https://github.com/mlc-ai/xgrammar/pull/1011', icon: 'git', text: 'XGrammar PR', sub: '并发缓存修复', external: true },
      { href: 'https://github.com/LiangQianXing/fast-llm-kernels', icon: 'zap', text: 'Fast-LLM-Kernels', sub: 'CUDA 融合算子', external: true },
      { href: 'https://github.com/LiangQianXing/llm-gateway', icon: 'server', text: 'LLM-Gateway', sub: '动态批处理网关', external: true },
      { href: academicProfile.github, icon: 'github', text: 'GitHub 主页', sub: '@LiangQianXing', external: true },
    ],
  },
  {
    id: 'learning',
    icon: 'compass',
    text: '课程公开课',
    projects: [
      { href: '/learning', icon: 'compass', text: '公开课主页', sub: 'ML Systems & LLM' },
      { href: '/learning/cmu-15-779', icon: 'layers', text: 'CMU 15-779', sub: 'ML Systems 精译与实验' },
      { href: '/learning/cs336', icon: 'spark', text: 'Stanford CS336', sub: '从零实现语言模型' },
      { href: '/posts/qwen25-vl-sglang-verl-single-gpu-lab', icon: 'zap', text: 'SGLang & GRPO', sub: '单卡多模态实训' },
      { href: '/posts/cuda-fused-rmsnorm-kernel', icon: 'cpu', text: 'CUDA RMSNorm', sub: '算子融合实战' },
    ],
  },
]

const activeCategory = computed(() =>
  categoryData.find(c => c.id === activeCategoryId.value) ?? categoryData[0],
)

const ICONS: Record<string, string> = {
  personal: '<svg viewBox="0 0 24 24"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/><path d="M8 7h6"/><path d="M8 11h8"/></svg>',
  blog: '<svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>',
  book: '<svg viewBox="0 0 24 24"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
  tag: '<svg viewBox="0 0 24 24"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/><path d="M7 7h.01"/></svg>',
  users: '<svg viewBox="0 0 24 24"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
  user: '<svg viewBox="0 0 24 24"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
  rss: '<svg viewBox="0 0 24 24"><path d="M4 11a9 9 0 0 1 9 9"/><path d="M4 4a16 16 0 0 1 16 16"/><circle cx="5" cy="19" r="1"/></svg>',
  scholar: '<svg viewBox="0 0 24 24"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>',
  paper: '<svg viewBox="0 0 24 24"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
  globe: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
  code: '<svg viewBox="0 0 24 24"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
  cpu: '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/></svg>',
  terminal: '<svg viewBox="0 0 24 24"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>',
  git: '<svg viewBox="0 0 24 24"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><line x1="6" y1="9" x2="6" y2="21"/></svg>',
  zap: '<svg viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
  server: '<svg viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>',
  github: '<svg viewBox="0 0 24 24"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>',
  compass: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
  layers: '<svg viewBox="0 0 24 24"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
  spark: '<svg viewBox="0 0 24 24"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>',
}

function getIconSvg(name: string) {
  return ICONS[name] || ICONS.personal
}

let gridAnim: GridAnimation | null = null
let destroyFluid: (() => void) | null = null

function initGridIfNeeded() {
  if (gridAnim || !gridCanvas.value) return
  gridAnim = new GridAnimation(gridCanvas.value)
  gridAnim.init()
}

function switchToMain() {
  if (currentPage.value !== 'intro' || isSwitching.value) return
  if (!introEl.value || !shapeSvg.value || !shapePath.value) return
  isSwitching.value = true
  currentPage.value = 'main'

  shapeSvg.value.style.transformOrigin = '50% 0%'

  anime({
    targets: introEl.value,
    translateY: '-200vh',
    duration: motionDuration(1100),
    easing: 'easeInOutSine',
  })

  anime({
    targets: shapeSvg.value,
    scaleY: [
      { value: [0.8, 1.8], duration: motionDuration(550), easing: 'easeInQuad' },
      { value: 1, duration: motionDuration(550), easing: 'easeOutQuad' },
    ],
  })

  anime({
    targets: shapePath.value,
    d: PATH_MAIN,
    duration: motionDuration(1100),
    easing: 'easeOutQuad',
    complete: () => {
      isSwitching.value = false
      document.querySelector<HTMLElement>('.content-main .category-link')?.focus({ preventScroll: true })
    },
  })

  later(() => {
    mainLoaded.value = true
    later(() => {
      initGridIfNeeded()
    }, 650)
  }, 400)
}

function switchToIntro() {
  if (currentPage.value === 'category') {
    hideCategoryPage()
    return
  }
  if (currentPage.value !== 'main' || isSwitching.value) return
  if (!introEl.value || !shapeSvg.value || !shapePath.value) return
  isSwitching.value = true
  currentPage.value = 'intro'

  shapeSvg.value.style.transformOrigin = '50% 0%'

  anime({
    targets: introEl.value,
    translateY: '0vh',
    duration: motionDuration(1100),
    easing: 'easeInOutSine',
  })

  anime({
    targets: shapeSvg.value,
    scaleY: [
      { value: [1, 1.8], duration: motionDuration(550), easing: 'easeInQuad' },
      { value: 0.8, duration: motionDuration(550), easing: 'easeOutQuad' },
    ],
  })

  anime({
    targets: shapePath.value,
    d: PATH_INTRO,
    duration: motionDuration(1100),
    easing: 'easeOutQuad',
    complete: () => {
      isSwitching.value = false
      document.querySelector<HTMLElement>('.content-intro .enter')?.focus({ preventScroll: true })
    },
  })
}

function showCategoryPage(id: string) {
  lastCategoryButton = document.activeElement as HTMLElement | null
  activeCategoryId.value = id
  categoryHiding.value = false
  categoryInnerOut.value = false
  categoryActive.value = true
  currentPage.value = 'category'
  later(() => {
    categoryInnerIn.value = true
    nextTick(() => categoryHeading.value?.focus({ preventScroll: true }))
  }, 120)
}

function hideCategoryPage() {
  if (currentPage.value !== 'category') return
  categoryInnerIn.value = false
  categoryInnerOut.value = true
  categoryHiding.value = true
  later(() => {
    categoryActive.value = false
    categoryHiding.value = false
    categoryInnerOut.value = false
    currentPage.value = 'main'
    nextTick(() => lastCategoryButton?.focus({ preventScroll: true }))
  }, 750)
}

function handleWheel(e: WheelEvent) {
  if (currentPage.value === 'category' || isSwitching.value) return
  const delta = e.deltaY
  if (delta > 12 && currentPage.value === 'intro') {
    switchToMain()
  } else if (delta < -12 && currentPage.value === 'main') {
    switchToIntro()
  }
}

let touchStart: { x: number; y: number; time: number } | null = null

function onTouchStart(e: TouchEvent) {
  if (e.touches.length !== 1) return
  touchStart = {
    x: e.touches[0].pageX,
    y: e.touches[0].pageY,
    time: Date.now(),
  }
}

function onTouchEnd(e: TouchEvent) {
  if (!touchStart || e.changedTouches.length === 0) return
  const dx = e.changedTouches[0].pageX - touchStart.x
  const dy = e.changedTouches[0].pageY - touchStart.y
  const dt = Date.now() - touchStart.time
  touchStart = null
  if (dt < 50 || dt > 1000) return
  if (Math.abs(dy) > 45 && Math.abs(dy) > Math.abs(dx)) {
    if (dy < 0 && currentPage.value === 'intro') {
      switchToMain()
    } else if (dy > 0 && currentPage.value === 'main') {
      switchToIntro()
    }
  }
}

onMounted(() => {
  later(() => {
    introLoaded.value = true
    later(() => {
      subtitleChars.value = Array.from(subtitleText)
    }, 270)
  }, 50)

  if (bgCanvas.value) {
    destroyFluid = initFluidBackground(bgCanvas.value, () => currentPage.value !== 'intro')
  }

  window.addEventListener('wheel', handleWheel, { passive: true })
  window.addEventListener('touchstart', onTouchStart, { passive: true })
  window.addEventListener('touchend', onTouchEnd, { passive: true })
  window.addEventListener('keydown', onEscape)
})

function onEscape(event: KeyboardEvent) {
  if (event.key === 'Escape') hideCategoryPage()
}

onUnmounted(() => {
  pendingTimers.forEach(clearTimeout)
  pendingTimers.clear()
  anime.remove([introEl.value, shapeSvg.value, shapePath.value].filter(Boolean))
  if (destroyFluid) destroyFluid()
  if (gridAnim) gridAnim.destroy()
  window.removeEventListener('wheel', handleWheel)
  window.removeEventListener('touchstart', onTouchStart)
  window.removeEventListener('touchend', onTouchEnd)
  window.removeEventListener('keydown', onEscape)
})

const description = `${academicProfile.nameCN}（${academicProfile.name}）的个人主页，聚焦大语言模型智能体（LLM Agents）、后训练对齐（GRPO / DPO）与 AI 系统基础设施。`
useHead({
  title: '个人主页',
  meta: [
    { name: 'description', content: description },
    { name: 'theme-color', content: '#222325' },
    { property: 'og:title', content: `${academicProfile.name} · ${academicProfile.nameCN}` },
    { property: 'og:description', content: description },
    { property: 'og:url', content: `${appConfig.url}/` },
    { property: 'og:image', content: `${appConfig.url}${academicProfile.avatar}` },
  ],
})
</script>
