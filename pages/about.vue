<template>
  <main class="page-frame">
    <section class="about-hero">
      <div class="about-profile">
        <img src="/avatar.jpg" :alt="appConfig.authorCN" width="148" height="148" />
        <div class="about-identity">
          <p class="eyebrow">{{ appConfig.role }}</p>
          <h1>{{ appConfig.authorCN }}</h1>
          <p class="about-en">{{ appConfig.authorEN }}</p>
          <p class="about-summary">{{ appConfig.bio }}</p>
          <div class="about-actions">
            <a :href="appConfig.github" target="_blank" rel="noopener noreferrer" class="primary-action">GitHub</a>
            <NuxtLink to="/posts" class="secondary-action">文章库</NuxtLink>
            <NuxtLink to="/tags" class="secondary-action">主题索引</NuxtLink>
          </div>
        </div>
      </div>

      <aside class="about-snapshot" aria-label="个人站点概览">
        <p class="snapshot-label">PROFILE SNAPSHOT</p>
        <div class="snapshot-grid">
          <div>
            <span>{{ posts.length }}</span>
            <em>公开文章</em>
          </div>
          <div>
            <span>{{ topicCount }}</span>
            <em>主题方向</em>
          </div>
          <div>
            <span>{{ latestYear }}</span>
            <em>最近更新</em>
          </div>
        </div>
        <p class="snapshot-note">关注 AI Infra、后端系统、工程实践和可复用的技术笔记沉淀。</p>
      </aside>
    </section>

    <div class="about-layout">
      <div>
        <SiteBlock eyebrow="Focus" title="关注方向" description="把个人介绍拆成更具体的工程兴趣，方便快速判断内容边界。">
          <div class="focus-grid">
            <article v-for="(item, index) in focusAreas" :key="item.key" class="focus-card">
              <small>0{{ index + 1 }}</small>
              <span>{{ item.key }}</span>
              <h3>{{ item.title }}</h3>
              <p>{{ item.desc }}</p>
            </article>
          </div>
        </SiteBlock>

        <SiteBlock eyebrow="Stack" title="技术栈" description="偏工程落地的全栈组合：后端、AI Infra、前端框架和基础设施。">
          <div class="stack-panel">
            <span v-for="tech in appConfig.techStack" :key="tech.name" class="skill-chip">{{ tech.name }}</span>
          </div>
        </SiteBlock>

        <SiteBlock eyebrow="Experience" title="经历">
          <div class="timeline">
            <article v-for="exp in appConfig.experience" :key="exp.company" class="timeline-card">
              <div class="timeline-logo">
                <img v-if="exp.logo" :src="exp.logo" :alt="exp.company" width="34" height="34" />
                <span v-else>{{ exp.company[0] }}</span>
              </div>
              <div>
                <time>{{ exp.period }}</time>
                <h3>
                  {{ exp.company }}
                  <span v-if="exp.companyEN">{{ exp.companyEN }}</span>
                </h3>
                <p class="timeline-role">{{ exp.role }}</p>
                <p v-if="exp.desc" class="timeline-desc">{{ exp.desc }}</p>
              </div>
            </article>
          </div>
        </SiteBlock>
      </div>

      <aside>
        <SiteBlock eyebrow="Now" title="当前状态" tone="panel">
          <div class="now-card">
            <p>持续整理后端、AI Infra 和工程实践相关的学习笔记，也会把项目拆解、源码阅读和面试准备沉淀成可检索的资料库。</p>
          </div>
        </SiteBlock>

        <SiteBlock eyebrow="Recent" title="近期文章" tone="panel">
          <ArticleStream :posts="recentPosts" />
        </SiteBlock>
      </aside>
    </div>
  </main>
</template>

<script setup lang="ts">
import type { PostMeta } from '~/server/api/posts.get'

const appConfig = useAppConfig()

const { data } = await useAsyncData<PostMeta[]>('about-posts', () =>
  $fetch('/api/posts')
)

const recentPosts = computed(() => (data.value ?? []).slice(0, 6))
const posts = computed(() => data.value ?? [])
const topicCount = computed(() => {
  const topics = new Set<string>()
  for (const post of posts.value) {
    for (const tag of post.tags ?? []) topics.add(tag)
  }
  return topics.size
})
const latestYear = computed(() => {
  const firstPost = posts.value[0]
  return firstPost ? new Date(firstPost.date).getFullYear() : new Date().getFullYear()
})

const focusAreas = [
  {
    key: 'AI',
    title: 'AI Infra / Agent',
    desc: 'RAG、上下文工程、记忆系统、推理优化和 Agent 产品化。',
  },
  {
    key: 'SYS',
    title: 'Backend Systems',
    desc: '高并发、缓存、数据库、消息队列和分布式系统设计。',
  },
  {
    key: 'SRC',
    title: 'Source Reading',
    desc: '从源码和项目结构里提炼可迁移的工程经验。',
  },
  {
    key: 'DOC',
    title: 'Knowledge Base',
    desc: '把零散学习转成可搜索、可复盘、可长期维护的笔记库。',
  },
]

useHead({
  title: '关于',
  meta: [
    { name: 'description', content: `关于 ${appConfig.authorCN} — ${appConfig.role}` },
  ],
})
</script>
