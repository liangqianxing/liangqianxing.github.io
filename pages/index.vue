<template>
  <div class="gateway-page">
    <section class="gateway-intro" aria-labelledby="gateway-title">
      <div>
        <p class="gateway-eyebrow">PERSONAL WEBSITE</p>
        <h1 id="gateway-title">{{ academicProfile.name }}</h1>
        <span class="gateway-name-cn">{{ academicProfile.nameCN }}</span>
        <p class="gateway-summary">研究、工程，以及沿途的记录。</p>
      </div>
      <img class="gateway-portrait" :src="academicProfile.avatar" :alt="academicProfile.nameCN" width="116" height="116" />
    </section>

    <nav class="gateway-destinations" aria-label="选择访问内容">
      <NuxtLink class="gateway-card" to="/academic">
        <span class="gateway-card-label">ACADEMIC PROFILE</span>
        <h2>学术主页</h2>
        <p>研究兴趣、学术经历，以及正在探索的方向。</p>
        <span class="gateway-card-action">进入学术主页 <span aria-hidden="true">→</span></span>
      </NuxtLink>
      <NuxtLink class="gateway-card" to="/blog">
        <span class="gateway-card-label">PERSONAL BLOG</span>
        <h2>个人博客</h2>
        <p>技术实践、阅读笔记，以及比赛与生活的片段。</p>
        <span class="gateway-card-action">进入博客 <span aria-hidden="true">→</span></span>
      </NuxtLink>
    </nav>
  </div>
</template>

<script setup lang="ts">
import { academicProfile } from '~/data/academic'

definePageMeta({ layout: 'profile' })

const appConfig = useAppConfig()
const description = `${academicProfile.nameCN}（${academicProfile.name}）的个人主页，包含学术主页与个人博客。`

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
