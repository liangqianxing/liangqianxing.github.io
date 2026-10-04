<template>
  <footer class="blog-footer-shell" aria-label="页脚">
    <div class="blog-footer">
      <div class="blog-footer-identity">
        <NuxtLink to="/blog" class="blog-footer-brand">
          <img src="/logo.svg?v=3" alt="" width="20" height="20" />
          {{ appConfig.title }}
        </NuxtLink>
        <div class="blog-footer-note">
          <span>&copy; {{ year }} {{ appConfig.authorEN }}</span>
          <a href="https://icp.gov.moe/" target="_blank" rel="noopener noreferrer">萌ICP备20262170号</a>
        </div>
      </div>
      <nav aria-label="页脚链接">
        <a :href="appConfig.github" target="_blank" rel="noopener noreferrer">GitHub ↗</a>
        <a href="/rss.xml">RSS</a>
        <NuxtLink to="/academic">学术主页</NuxtLink>
        <a href="https://www.travellings.cn/go.html" target="_blank" rel="noopener noreferrer">开往 ↗</a>
      </nav>
    </div>
  </footer>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
const year = new Date().getFullYear()
</script>
