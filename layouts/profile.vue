<template>
  <div class="profile-site" :class="{ 'profile-site-academic': currentPath === '/academic' }">
    <a class="profile-skip" href="#profile-main">跳到主要内容</a>
    <header class="profile-header">
      <NuxtLink class="profile-wordmark" to="/" :aria-label="`${academicProfile.nameCN}的个人主页`">
        <span>{{ academicProfile.name }}</span>
        <span class="profile-wordmark-cn">{{ academicProfile.nameCN }}</span>
      </NuxtLink>
      <nav class="profile-nav" aria-label="个人主页导航">
        <NuxtLink
          v-for="item in navigation"
          :key="item.path"
          :to="item.path"
          :aria-current="currentPath === item.path ? 'page' : undefined"
        >{{ item.label }}</NuxtLink>
      </nav>
    </header>

    <main id="profile-main" class="profile-main" tabindex="-1">
      <slot />
    </main>

    <footer class="profile-footer">
      <span>&copy; {{ year }} {{ academicProfile.name }}</span>
      <a v-if="academicProfile.github" :href="academicProfile.github" target="_blank" rel="noopener noreferrer">GitHub <span aria-hidden="true">↗</span></a>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { academicProfile } from '~/data/academic'

const route = useRoute()
const year = new Date().getFullYear()
const currentPath = computed(() => route.path.replace(/\/$/, '') || '/')
const navigation = [
  { label: '主页', path: '/' },
  { label: '学术', path: '/academic' },
  { label: '博客', path: '/blog' },
]

useHead({
  htmlAttrs: { lang: 'zh-CN' },
  bodyAttrs: { class: 'profile-body' },
  titleTemplate: title => title ? `${title} · ${academicProfile.name}` : academicProfile.name,
  meta: [
    { name: 'theme-color', content: '#fafaf8' },
    { property: 'og:site_name', content: `${academicProfile.name} · ${academicProfile.nameCN}` },
    { property: 'og:type', content: 'website' },
    { name: 'twitter:card', content: 'summary' },
  ],
})
</script>
