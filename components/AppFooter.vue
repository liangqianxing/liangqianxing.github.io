<template>
  <footer class="blog-footer" aria-label="页脚">
    <div class="blog-footer-main">
      <span>&copy; {{ year }} {{ appConfig.authorEN }} · {{ appConfig.title }}</span>
      <nav aria-label="页脚链接">
        <a :href="appConfig.github" target="_blank" rel="noopener noreferrer">GitHub ↗</a>
        <a href="/rss.xml">RSS</a>
        <NuxtLink to="/academic">学术主页</NuxtLink>
        <a href="https://www.travellings.cn/go.html" target="_blank" rel="noopener noreferrer">开往 ↗</a>
      </nav>
    </div>
    <div class="blog-footer-note">
      <span>{{ appConfig.description }}</span>
      <a href="https://icp.gov.moe/" target="_blank" rel="noopener noreferrer">萌ICP备20262170号</a>
    </div>
  </footer>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
const year = new Date().getFullYear()
</script>
