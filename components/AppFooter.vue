<template>
  <footer class="blog-footer-shell" aria-label="页脚">
    <div class="blog-footer">
      <div class="blog-footer-identity">
        <NuxtLink to="/blog" class="blog-footer-brand">
          <img src="/logo.svg?v=4" alt="" width="22" height="22" />
          {{ appConfig.title }}
        </NuxtLink>
        <p class="blog-footer-tagline">随手记录，慢慢探索。</p>
        <div class="blog-footer-note">
          <span>&copy; {{ year }} {{ appConfig.authorEN }}</span>
          <a href="https://icp.gov.moe/" target="_blank" rel="noopener noreferrer">萌ICP备20262170号</a>
        </div>
      </div>
      <div class="blog-footer-links">
        <nav aria-label="博客栏目">
          <NuxtLink
            v-for="item in navItems"
            :key="item.path"
            :to="item.path"
            :aria-current="route.path === item.path ? 'page' : undefined"
          >{{ item.label }}</NuxtLink>
          <button type="button" class="blog-footer-cmd" aria-label="打开全局搜索面板 (⌘K)" @click="openCommandPalette">
            搜索 <kbd>⌘K</kbd>
          </button>
        </nav>
        <nav aria-label="站点与外部链接">
          <NuxtLink to="/">个人门户</NuxtLink>
          <NuxtLink to="/academic">学术主页</NuxtLink>
          <a :href="appConfig.github" target="_blank" rel="noopener noreferrer">GitHub ↗</a>
          <a href="/rss.xml">RSS</a>
          <a href="https://www.travellings.cn/go.html" target="_blank" rel="noopener noreferrer">开往 ↗</a>
        </nav>
      </div>
    </div>
  </footer>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
const route = useRoute()
const openCommandPalette = inject<() => void>('openCommandPalette', () => {})
const navItems = computed(() => appConfig.nav.filter(item => item.path !== '/'))
const year = new Date().getFullYear()
</script>
