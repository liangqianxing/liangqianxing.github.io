<template>
  <main class="portal-home">
    <a class="portal-skip" href="#explore">跳到主页入口</a>
    <section id="welcome" ref="welcome" class="portal-welcome" aria-labelledby="welcome-title">
      <GatewayFluid :active="introVisible" />
      <header class="portal-topbar">
        <a class="portal-wordmark" href="#welcome" aria-label="Enhao Gu 的个人主页"><span aria-hidden="true">✳</span> enhao.gu</a>
        <nav aria-label="快捷导航"><NuxtLink to="/blog">Blog</NuxtLink><a :href="academicProfile.github" target="_blank" rel="noopener noreferrer">GitHub <span aria-hidden="true">↗</span></a></nav>
      </header>
      <div class="portal-opening">
        <p class="portal-hello">HELLO, WORLD. I'M</p>
        <h1 id="welcome-title">{{ academicProfile.name }}</h1>
        <p class="portal-chinese-name">{{ academicProfile.nameCN }}</p>
        <p class="portal-opening-role">Researcher <span>&amp;</span> Developer</p>
        <p class="portal-opening-note">记录研究，也记录构建的过程。</p>
        <a class="portal-next" href="#explore"><span>Next</span><GatewayIcon name="arrow-down" /></a>
      </div>
      <a class="portal-scroll" href="#explore" aria-label="向下滚动，探索主页"><span>SCROLL TO EXPLORE</span><i /><i /></a>
      <div class="portal-welcome-foot" aria-hidden="true"><span>LLM AGENTS · AI SYSTEMS</span><span class="portal-pointer-hint">MOVE YOUR CURSOR. MAKE A LITTLE WAVE.</span><span>01 / 02</span></div>
    </section>
    <section id="explore" ref="explore" class="portal-explore" tabindex="-1" aria-label="个人主页入口" @keydown.esc="closeCategory">
      <GatewayGrid :active="directoryVisible" />
      <a class="portal-return-top" href="#welcome" aria-label="返回开场"><GatewayIcon name="arrow-up" /></a>
      <div id="portal-panel" class="portal-stage">
        <Transition name="portal-view" mode="out-in" @after-enter="focusPanel">
          <div v-if="!selectedCategory" key="directory" class="portal-directory">
            <header class="portal-profile">
              <div class="portal-avatar-ring"><img :src="academicProfile.avatar" :alt="academicProfile.nameCN" width="100" height="100" /></div>
              <h2>{{ academicProfile.name }}<span>{{ academicProfile.nameCN }}</span></h2>
              <p class="portal-profile-role">LLM Agents <b>·</b> Post-Training <b>·</b> AI Systems</p>
              <p class="portal-profile-note">{{ academicProfile.affiliation }}<span>2027 年入学</span></p>
            </header>
            <nav class="portal-categories" aria-label="选择访问内容">
              <a v-for="category in categories" :key="category.id" :href="category.to" :data-category="category.id" aria-controls="portal-panel" :aria-label="`${category.title}，展开入口`" @click="openCategory($event, category.id)">
                <span class="portal-category-icon"><GatewayIcon :name="category.icon" /></span><span class="portal-category-label">{{ category.title }}</span><span class="portal-category-english">{{ category.english }}</span>
              </a>
            </nav>
            <div class="portal-socials">
              <a :href="academicProfile.github" target="_blank" rel="noopener noreferrer" aria-label="GitHub（新标签页）"><GatewayIcon name="github" /><span>GitHub</span></a><span class="portal-social-divider" aria-hidden="true" /><a :href="academicProfile.scholar" target="_blank" rel="noopener noreferrer" aria-label="Google Scholar（新标签页）"><GatewayIcon name="research" /><span>Scholar</span></a>
            </div>
          </div>
          <div v-else :key="selectedCategory.id" class="portal-panel" :aria-labelledby="`panel-title-${selectedCategory.id}`">
            <header class="portal-panel-head">
              <button class="portal-back" type="button" @click="closeCategory"><GatewayIcon name="arrow-left" /><span>返回</span></button>
              <p class="portal-panel-eyebrow">{{ selectedCategory.english }}</p>
              <h2 :id="`panel-title-${selectedCategory.id}`" ref="panelHeading" tabindex="-1">{{ selectedCategory.title }}</h2>
              <p class="portal-panel-description">{{ selectedCategory.description }}</p>
            </header>
            <nav class="portal-entries" :aria-label="selectedCategory.title">
              <NuxtLink v-for="entry in selectedCategory.entries" :key="entry.to" :to="entry.to" :target="isExternal(entry.to) ? '_blank' : undefined" :rel="isExternal(entry.to) ? 'noopener noreferrer' : undefined">
                <span class="portal-entry-icon"><GatewayIcon :name="entry.icon" /></span><span class="portal-entry-copy"><strong>{{ entry.title }}</strong><span>{{ entry.description }}</span></span><span class="portal-entry-arrow" aria-hidden="true">{{ isExternal(entry.to) ? '↗' : '→' }}</span>
              </NuxtLink>
            </nav>
            <p class="portal-panel-hint">选一个入口，继续探索<span aria-hidden="true"> · </span><kbd>Esc</kbd> 返回</p>
          </div>
        </Transition>
      </div>
      <footer class="portal-footer"><span>© {{ year }} {{ academicProfile.name }}</span><p>Stay curious. Keep building.</p><a href="#welcome">Back to top <span aria-hidden="true">↑</span></a></footer>
    </section>
  </main>
</template>

<script setup lang="ts">
import { academicProfile } from '~/data/academic'
import { learningCourses } from '~/data/learning/courses'

definePageMeta({ layout: 'gateway' })
const appConfig = useAppConfig()
const year = new Date().getFullYear()
type Entry = { title: string; description: string; icon: string; to: string }
type Category = { id: string; title: string; english: string; description: string; icon: string; to: string; entries: Entry[] }
const categories: Category[] = [
  { id: 'blog', title: '个人博客', english: 'WRITING', icon: 'book', to: '/blog', description: '把想过的问题、读过的代码和做过的实验，慢慢写下来。', entries: [
    { title: 'gu.log', description: '研究笔记与工程实践', icon: 'book', to: '/blog' },
    { title: '文章归档', description: '所有长文，按时间收录', icon: 'archive', to: '/posts' },
    { title: '主题索引', description: '沿着感兴趣的方向阅读', icon: 'layers', to: '/tags' },
    { title: '我的朋友', description: '去朋友们的主页逛一逛', icon: 'friends', to: '/friends' },
  ] },
  { id: 'research', title: '学术主页', english: 'RESEARCH', icon: 'research', to: '/academic', description: '关于我的研究、论文，以及走过的一小段路。', entries: [
    { title: '关于我', description: '研究兴趣与个人简介', icon: 'person', to: '/academic' },
    { title: '学术论文', description: '论文、代码与研究记录', icon: 'paper', to: '/academic#publications' },
    { title: '科研与实习', description: '研究和工程中的经历', icon: 'research', to: '/academic#experience' },
    { title: 'Google Scholar', description: '学术成果与引用', icon: 'external', to: academicProfile.scholar },
  ] },
  { id: 'learning', title: '学习记录', english: 'LEARNING', icon: 'layers', to: '/learning', description: '从论文到代码，从理解一个概念到亲手把它实现。', entries: [
    ...learningCourses.map(course => ({ title: course.title, description: course.slug === 'cs336' ? '从零构建语言模型' : '大模型与机器学习系统', icon: 'code', to: `/learning/${course.slug}` })),
    { title: '课程目录', description: '课程材料、论文与实验', icon: 'layers', to: '/learning' },
  ] },
  { id: 'projects', title: '开源项目', english: 'BUILDING', icon: 'code', to: '/academic#projects', description: '一些亲手参与构建、还在不断打磨的东西。', entries: [
    { title: 'DeepScientist', description: '面向科研的自主智能体', icon: 'research', to: academicProfile.projects[0]!.links[0]!.url },
    { title: 'MiniCode', description: '终端里的 AI 编程伙伴', icon: 'terminal', to: academicProfile.projects[1]!.links[0]!.url },
    { title: 'XGrammar', description: '结构化解码与并发缓存修复', icon: 'code', to: academicProfile.projects[2]!.links[0]!.url },
    { title: '更多系统与实验', description: 'CUDA 算子、推理服务与后训练', icon: 'layers', to: '/academic#projects' },
  ] },
]
const welcome = ref<HTMLElement | null>(null)
const explore = ref<HTMLElement | null>(null)
const panelHeading = ref<HTMLElement | null>(null)
const introVisible = ref(true)
const directoryVisible = ref(false)
const selectedId = ref<string | null>(null)
const selectedCategory = computed(() => categories.find(category => category.id === selectedId.value) ?? null)
let lastCategoryId: string | null = null
let observer: IntersectionObserver | undefined
function isExternal(url: string) { return /^https?:\/\//.test(url) }
function openCategory(event: MouseEvent, id: string) {
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return
  event.preventDefault()
  lastCategoryId = id
  selectedId.value = id
}
function focusPanel() {
  if (selectedId.value) panelHeading.value?.focus({ preventScroll: true })
  else explore.value?.querySelector<HTMLElement>(`[data-category="${lastCategoryId}"]`)?.focus({ preventScroll: true })
}
function closeCategory() { selectedId.value = null }
onMounted(() => {
  if (!('IntersectionObserver' in window)) { directoryVisible.value = true; return }
  observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.target === welcome.value) introVisible.value = entry.isIntersecting
      if (entry.target === explore.value) directoryVisible.value = entry.isIntersecting
    }
  }, { threshold: 0 })
  if (welcome.value) observer.observe(welcome.value)
  if (explore.value) observer.observe(explore.value)
})
onBeforeUnmount(() => observer?.disconnect())
const description = `${academicProfile.nameCN}（${academicProfile.name}）的个人主页。探索 LLM Agents、后训练与 AI 系统的研究、技术博客、学习记录和开源项目。`
useHead({ title: '个人主页', meta: [
  { name: 'description', content: description },
  { property: 'og:title', content: `${academicProfile.name} · ${academicProfile.nameCN}` },
  { property: 'og:description', content: description },
  { property: 'og:url', content: `${appConfig.url}/` },
  { property: 'og:image', content: `${appConfig.url}${academicProfile.avatar}` },
], link: [{ rel: 'canonical', href: `${appConfig.url}/` }] })
</script>
