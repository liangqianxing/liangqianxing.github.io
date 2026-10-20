<template>
  <div class="home-hub">
    <section class="hub-intro" aria-labelledby="hub-name">
      <div class="hub-intro-copy">
        <p class="hub-eyebrow">研究 · 工程 · 记录</p>
        <h1 id="hub-name">{{ profile.name }}<span>{{ profile.nameCN }}</span></h1>
        <p class="hub-lead">关注大语言模型、智能体与 AI 系统，<br class="hub-desktop-break" />喜欢研究，也喜欢把想法做出来。</p>
        <p class="hub-bio">{{ profile.bio[1] }}</p>
        <div class="hub-personal-links" aria-label="个人链接">
          <a :href="profile.github" target="_blank" rel="noopener noreferrer">GitHub <span aria-hidden="true">↗</span></a>
          <a v-if="profile.scholar" :href="profile.scholar" target="_blank" rel="noopener noreferrer">Google Scholar <span aria-hidden="true">↗</span></a>
        </div>
      </div>
      <figure class="hub-portrait">
        <SiteImage :src="profile.avatar" :alt="`${profile.nameCN}的头像`" width="156" height="156" priority />
        <figcaption>Language, intelligence<br />&amp; systems.</figcaption>
      </figure>
    </section>

    <div class="hub-spaces">
      <section class="hub-space hub-research" aria-labelledby="hub-research-title">
        <header class="hub-space-head">
          <span class="hub-space-kicker">ACADEMIC</span>
          <h2 id="hub-research-title"><NuxtLink to="/academic">学术与研究<span aria-hidden="true">↗</span></NuxtLink></h2>
          <p>研究兴趣、论文，以及一路走来的经历。</p>
        </header>
        <div class="hub-papers">
          <NuxtLink v-for="paper in publications" :key="paper.title" to="/academic#publications" class="hub-paper" :aria-label="`在学术主页查看 ${paper.title}`">
            <div v-if="paper.image" class="hub-paper-image"><SiteImage :src="paper.image" :alt="paper.imageAlt || `${paper.shortTitle} 的研究框架`" width="120" height="90" sizes="120px" /></div>
            <div class="hub-paper-copy">
              <span class="hub-paper-venue">{{ paper.badge || `${paper.venue} · ${paper.year}` }}</span>
              <h3>{{ paper.shortTitle }}</h3>
              <p>{{ paper.summary }}</p>
            </div>
          </NuxtLink>
        </div>
        <NuxtLink class="hub-space-link" to="/academic">查看学术主页 <span aria-hidden="true">→</span></NuxtLink>
      </section>

      <section class="hub-space hub-notes" aria-labelledby="hub-notes-title">
        <header class="hub-space-head">
          <span class="hub-space-kicker">PERSONAL BLOG</span>
          <h2 id="hub-notes-title"><NuxtLink to="/blog">笔记与日常<span aria-hidden="true">↗</span></NuxtLink></h2>
          <p>读论文、写代码，也记下一些生活的片段。</p>
        </header>
        <div v-if="recentPosts.length" class="hub-posts">
          <NuxtLink v-for="post in recentPosts" :key="post.path" :to="post.path" class="hub-post">
            <div class="hub-post-meta"><time :datetime="post.date.slice(0, 10)">{{ post.date.slice(0, 10).replaceAll('-', '.') }}</time><span>{{ post.tags?.[0] || '随手记录' }}</span></div>
            <h3>{{ post.title }}</h3>
          </NuxtLink>
        </div>
        <p v-else class="hub-empty">课程学习、源码阅读与工程实践，慢慢整理在博客里。</p>
        <NuxtLink class="hub-space-link" to="/blog">走进我的博客 <span aria-hidden="true">→</span></NuxtLink>
      </section>
    </div>

    <aside class="hub-learning" aria-label="课程与学习">
      <div><span class="hub-learning-label">继续探索</span><p>从论文到代码，从课程到实践。</p></div>
      <NuxtLink to="/learning">课程与学习记录 <span aria-hidden="true">→</span></NuxtLink>
    </aside>
  </div>
</template>

<script setup lang="ts">
import { academicProfile as profile } from '~/data/academic'
import '~/assets/css/home.css'

definePageMeta({ layout: 'profile' })
const appConfig = useAppConfig()
const { data: posts } = await usePublicArchive()
const recentPosts = computed(() => (posts.value ?? []).slice(0, 3))
const publications = profile.publications.slice(0, 2).map(paper => ({ ...paper, shortTitle: paper.title.split(':')[0] }))
const description = `${profile.nameCN}（${profile.name}）的个人主页，连接大语言模型、智能体与 AI 系统的学术研究、工程实践和个人笔记。`

useHead({
  title: '个人主页',
  meta: [
    { name: 'description', content: description },
    { property: 'og:title', content: `${profile.name} · ${profile.nameCN}` },
    { property: 'og:description', content: description },
    { property: 'og:url', content: `${appConfig.url}/` },
    { property: 'og:image', content: `${appConfig.url}${profile.avatar}` },
  ],
  link: [{ rel: 'canonical', href: `${appConfig.url}/` }],
})
</script>
