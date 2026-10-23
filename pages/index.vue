<template>
  <div id="top" class="gateway-scene">
    <GatewayBackdrop :revealed="scrolledDown" />
    <a class="poster-skip" href="#gateway-choices">跳到主页入口</a>
    <header class="poster-nav poster-container">
      <a class="poster-brand" href="#top" aria-label="Enhao Gu，返回顶部">
        <svg viewBox="0 0 42 34" fill="none" aria-hidden="true"><path d="M4 25 13 7h14L18 25H4Zm14 0L29 7h9L26 29H15" /><path d="m8 17 13 0" /></svg>
        <span>{{ academicProfile.name }}<small>{{ academicProfile.nameCN }}</small></span>
      </a>
      <nav aria-label="主导航">
        <NuxtLink to="/blog">Blog <span aria-hidden="true">↗</span></NuxtLink>
        <NuxtLink to="/academic">Research <span aria-hidden="true">↗</span></NuxtLink>
        <NuxtLink to="/learning">Learning <span aria-hidden="true">↗</span></NuxtLink>
        <a v-if="academicProfile.github" :href="academicProfile.github" target="_blank" rel="noopener noreferrer" aria-label="GitHub（在新标签页打开）">GitHub <span aria-hidden="true">↗</span></a>
      </nav>
    </header>

    <main>
      <section class="poster-hero poster-container" aria-labelledby="opening-name">
        <div class="poster-frame" aria-hidden="true"><span /><span /><span /><span /></div>
        <div class="poster-meta poster-meta-left"><span>INDEPENDENT PERSONAL SPACE</span><span>RESEARCH &amp; ENGINEERING</span></div>
        <div class="poster-meta poster-meta-right"><span>LLM · AGENT · SYSTEMS</span><span>EST. IN CURIOSITY</span></div>
        <h1 id="opening-name" class="poster-name" :aria-label="`${academicProfile.name} · ${academicProfile.nameCN}`">ENHAO GU</h1>
        <div class="poster-object" aria-hidden="true"><GatewaySculpture /></div>
        <div class="poster-intro"><span class="poster-intro-cross" aria-hidden="true">✳</span><p>研究、工程，<br />以及沿途的记录。</p></div>
        <NuxtLink to="/blog" class="poster-enter">进入博客 <span aria-hidden="true">↗</span></NuxtLink>
        <div class="poster-hero-foot"><span>{{ academicProfile.nameCN }} / {{ year }}</span><a href="#gateway-choices">SCROLL TO EXPLORE <span aria-hidden="true">↓</span></a><span>ALWAYS A WORK IN PROGRESS</span></div>
      </section>

      <section id="gateway-choices" class="poster-index poster-container" aria-labelledby="choices-title">
        <header class="poster-section-head"><p class="poster-eyebrow">01 / THE INDEX</p><h2 id="choices-title">从好奇心出发<span>。</span></h2><p class="poster-section-note">向内探索，向外记录。</p></header>
        <nav class="poster-destinations" aria-label="选择访问内容">
          <NuxtLink
            v-for="item in destinations"
            :key="item.to"
            :to="item.to"
            class="poster-destination"
            @pointermove="onCardPointerMove"
          >
            <span class="poster-row-number">{{ item.number }}</span>
            <div class="poster-row-title"><span class="poster-row-kicker">{{ item.kicker }}</span><h3>{{ item.title }}<span>.</span></h3></div>
            <p>{{ item.description }}</p>
            <span class="poster-row-arrow" aria-hidden="true">↗</span>
          </NuxtLink>
        </nav>
      </section>

      <section v-if="recentPosts.length" class="poster-writing poster-container" aria-labelledby="writing-title">
        <header class="poster-section-head"><p class="poster-eyebrow">02 / RECENT WRITING</p><h2 id="writing-title">最近写下的<span>。</span></h2><NuxtLink to="/posts" class="poster-section-note poster-text-link">全部文章 <span aria-hidden="true">↗</span></NuxtLink></header>
        <div class="poster-articles">
          <NuxtLink
            v-for="post in recentPosts"
            :key="post.path"
            :to="post.path"
            class="poster-article"
            @pointermove="onCardPointerMove"
          >
            <time :datetime="post.date.slice(0, 10)">{{ post.date.slice(0, 10).replaceAll('-', '.') }}</time>
            <h3>{{ post.title }}</h3>
            <span class="poster-article-tag">{{ post.tags?.[0] || 'NOTES' }}</span>
            <span class="poster-article-arrow" aria-hidden="true">↗</span>
          </NuxtLink>
        </div>
      </section>
    </main>

    <footer class="poster-footer poster-container">
      <div class="poster-footer-top"><p>Keep asking.<br /><em>Stay curious.</em></p><a v-if="academicProfile.github" :href="academicProfile.github" target="_blank" rel="noopener noreferrer">FIND ME ON GITHUB <span aria-hidden="true">↗</span></a></div>
      <div class="poster-footer-bottom"><span>© {{ year }} {{ academicProfile.name }}</span><span>研究、工程，以及沿途的记录。</span><a href="#top">BACK TO TOP <span aria-hidden="true">↑</span></a></div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { academicProfile } from '~/data/academic'

definePageMeta({ layout: 'gateway' })
const year = new Date().getFullYear()
const appConfig = useAppConfig()
const { data: posts } = await usePublicArchive()
const recentPosts = computed(() => (posts.value ?? []).slice(0, 4))
const scrolledDown = ref(false)
const destinations = [
  { number: '01', title: 'Notebook', kicker: 'THOUGHTS, NOTES & LIFE', description: '技术实践、源码阅读，和一些生活的片段。', to: '/blog' },
  { number: '02', title: 'Research', kicker: 'LANGUAGE, INTELLIGENCE & SYSTEMS', description: '研究兴趣、学术经历，以及正在探索的方向。', to: '/academic' },
  { number: '03', title: 'Learning', kicker: 'UNDERSTANDING BY BUILDING', description: '从课程到代码，把学过的东西慢慢搭起来。', to: '/learning' },
]

function onCardPointerMove(event: PointerEvent) {
  const el = event.currentTarget as HTMLElement | null
  if (!el) return
  const rect = el.getBoundingClientRect()
  el.style.setProperty('--row-x', `${event.clientX - rect.left}px`)
  el.style.setProperty('--row-y', `${event.clientY - rect.top}px`)
}

function onWindowScroll() {
  scrolledDown.value = window.scrollY > 140
}

onMounted(() => {
  onWindowScroll()
  window.addEventListener('scroll', onWindowScroll, { passive: true })
})

onUnmounted(() => {
  window.removeEventListener('scroll', onWindowScroll)
})

const description = `${academicProfile.nameCN}（${academicProfile.name}）的个人主页，记录大语言模型、智能体与 AI 系统的研究和工程实践。`
useHead({
  title: '个人主页',
  meta: [
    { name: 'description', content: description },
    { property: 'og:title', content: `${academicProfile.name} · ${academicProfile.nameCN}` },
    { property: 'og:description', content: description },
    { property: 'og:url', content: `${appConfig.url}/` },
    { property: 'og:image', content: `${appConfig.url}${academicProfile.avatar}` },
  ],
})
</script>
