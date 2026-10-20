<template>
  <div id="top" class="gateway-scene">
    <a class="portal-skip" href="#gateway-choices">跳到主页入口</a>
    <header class="portal-header portal-container">
      <a class="portal-wordmark" href="#top" aria-label="Enhao Gu，返回顶部">
        <svg viewBox="0 0 32 32" fill="none" aria-hidden="true"><path d="M21.5 6.5H10.5L5 16l5.5 9.5h11L27 16l-5.5-9.5Z" /><path d="m10.5 16 5.5-9.5 5.5 9.5-5.5 9.5-5.5-9.5Zm0 0h11" /></svg>
        <span>enhao<span class="portal-wordmark-dot">.</span>gu</span>
      </a>
      <nav aria-label="主导航">
        <a class="portal-nav-home" href="#top" aria-current="page"><span aria-hidden="true" />首页</a>
        <NuxtLink to="/academic">学术</NuxtLink>
        <NuxtLink to="/blog">博客</NuxtLink>
        <a v-if="academicProfile.github" class="portal-nav-github" :href="academicProfile.github" target="_blank" rel="noopener noreferrer" aria-label="GitHub（在新标签页打开）">
          <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .9a11.1 11.1 0 0 0-3.51 21.63c.56.1.76-.24.76-.54v-2.15c-3.1.68-3.75-1.32-3.75-1.32-.5-1.28-1.23-1.62-1.23-1.62-1.01-.69.08-.68.08-.68 1.12.08 1.7 1.15 1.7 1.15.99 1.7 2.6 1.21 3.23.93.1-.72.39-1.21.7-1.49-2.48-.28-5.09-1.24-5.09-5.51 0-1.22.44-2.22 1.14-3-.11-.28-.49-1.42.11-2.96 0 0 .94-.3 3.05 1.14a10.6 10.6 0 0 1 5.55 0c2.12-1.44 3.05-1.14 3.05-1.14.61 1.54.23 2.68.12 2.96.71.78 1.14 1.78 1.14 3 0 4.28-2.62 5.23-5.11 5.51.4.35.76 1.02.76 2.06v3.12c0 .3.2.65.77.54A11.1 11.1 0 0 0 12 .9Z" /></svg>
        </a>
      </nav>
    </header>

    <main>
      <section class="portal-hero portal-container" aria-labelledby="opening-name">
        <div class="portal-hero-copy">
          <p class="portal-kicker"><span aria-hidden="true" />A LITTLE SPACE FOR CURIOSITY</p>
          <h1 id="opening-name" class="portal-name"><span>{{ firstName }}</span><span>{{ lastName }}<span class="portal-name-dot">.</span><span class="portal-chinese-name">{{ academicProfile.nameCN }}</span></span></h1>
          <p class="portal-disciplines">LLM <span>/</span> AGENT <span>/</span> SYSTEMS</p>
          <p class="portal-summary">在模型与系统之间，保持好奇。<br />记录研究、工程，以及沿途的灵感。</p>
          <div class="portal-hero-actions">
            <NuxtLink class="portal-primary" to="/blog">翻翻我的笔记 <span aria-hidden="true">↗</span></NuxtLink>
            <NuxtLink class="portal-secondary" to="/academic">学术主页 <span aria-hidden="true">↗</span></NuxtLink>
          </div>
        </div>
        <div class="portal-art" aria-hidden="true">
          <div class="portal-art-grid" />
          <div class="portal-art-halo" />
          <span class="portal-art-cross portal-art-cross-top">+</span>
          <span class="portal-art-cross portal-art-cross-bottom">+</span>
          <div class="portal-sculpture"><GatewaySculpture /></div>
          <div class="portal-art-caption"><span>IDEAS IN MOTION</span><span class="portal-art-caption-line" /><span>∞</span></div>
        </div>
        <div class="portal-hero-baseline">
          <span class="portal-baseline-note"><span class="portal-status-dot" aria-hidden="true" />持续学习，持续构建</span>
          <a href="#gateway-choices" class="portal-scroll-link">向下探索 <span aria-hidden="true">↓</span></a>
          <span class="portal-baseline-index" aria-hidden="true">01 — INTRODUCTION</span>
        </div>
      </section>

      <section id="gateway-choices" class="portal-paths portal-container" aria-labelledby="choices-title">
        <div class="portal-section-heading">
          <div><p class="portal-kicker">EXPLORE MY SPACE</p><h2 id="choices-title">好奇心的两种去处<span>。</span></h2></div>
          <p class="portal-section-aside">一边深入探索，一边认真记录。</p>
        </div>
        <nav class="portal-destinations" aria-label="选择访问内容">
          <NuxtLink class="portal-card portal-card-blog" to="/blog">
            <div class="portal-card-top"><span class="portal-card-label">THE NOTEBOOK</span><span class="portal-card-number">01</span></div>
            <div class="portal-card-main">
              <div><h3>个人博客</h3><p>技术实践、源码阅读，<br />还有比赛与生活的片段。</p></div>
              <svg class="portal-card-drawing" viewBox="0 0 120 104" fill="none" aria-hidden="true"><path d="m18 20 51-8 33 18-51 9-33-19Z" /><path d="m18 20 1 55 33 19 50-10V30M51 39l1 55M29 39l13 8m-13 1 13 8m-13 1 13 8m22-12 24-5m-24 15 24-5m-24 15 17-4" /><path class="portal-drawing-accent" d="m18 20 33 19 51-9M51 39l1 55" /></svg>
            </div>
            <div class="portal-card-bottom"><span>把想法写下来</span><span class="portal-card-arrow" aria-hidden="true">↗</span></div>
          </NuxtLink>
          <NuxtLink class="portal-card portal-card-academic" to="/academic">
            <div class="portal-card-top"><span class="portal-card-label">THE RESEARCH</span><span class="portal-card-number">02</span></div>
            <div class="portal-card-main">
              <div><h3>学术主页</h3><p>研究兴趣、学术经历，<br />以及正在探索的方向。</p></div>
              <svg class="portal-card-drawing" viewBox="0 0 120 104" fill="none" aria-hidden="true"><ellipse cx="60" cy="52" rx="47" ry="18" transform="rotate(-32 60 52)" /><ellipse cx="60" cy="52" rx="47" ry="18" transform="rotate(32 60 52)" /><ellipse cx="60" cy="52" rx="47" ry="18" transform="rotate(90 60 52)" /><circle class="portal-drawing-accent" cx="60" cy="52" r="6" /><circle class="portal-drawing-node" cx="92" cy="27" r="3.5" /></svg>
            </div>
            <div class="portal-card-bottom"><span>顺着问题走下去</span><span class="portal-card-arrow" aria-hidden="true">↗</span></div>
          </NuxtLink>
        </nav>
      </section>
    </main>

    <footer class="portal-footer portal-container">
      <div class="portal-footer-message">Stay curious<span>.</span></div>
      <div class="portal-footer-bottom"><span>&copy; {{ year }} {{ academicProfile.name }}</span><span class="portal-footer-caption">研究、工程，以及沿途的记录。</span><a href="#top">回到顶部 <span aria-hidden="true">↑</span></a></div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { academicProfile } from '~/data/academic'

definePageMeta({ layout: 'gateway' })

const [firstName, ...surname] = academicProfile.name.split(' ')
const lastName = surname.join(' ')
const year = new Date().getFullYear()
const appConfig = useAppConfig()
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
