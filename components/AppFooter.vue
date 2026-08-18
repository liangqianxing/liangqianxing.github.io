<template>
  <footer class="footer" aria-label="页脚">
    <div class="footer-inner">
      <div class="footer-brand">
        <NuxtLink to="/" class="brand" aria-label="回到首页">
          <span class="brand-mark" aria-hidden="true">
            <img src="/logo.svg" alt="" width="42" height="42" />
          </span>
          <span class="brand-copy">
            <strong>{{ appConfig.title }}</strong>
            <em>{{ appConfig.authorEN }}</em>
          </span>
        </NuxtLink>
        <p class="footer-tagline">{{ appConfig.description }}</p>
        <p v-if="appConfig.status" class="footer-status">
          <span class="hero-status-dot" aria-hidden="true" />
          {{ appConfig.status }}
        </p>
      </div>

      <nav class="footer-nav" aria-label="页脚导航">
        <div class="footer-col">
          <p class="footer-col-title">导航</p>
          <NuxtLink v-for="item in appConfig.nav" :key="item.path" :to="item.path">
            {{ item.label }}
          </NuxtLink>
        </div>
        <div class="footer-col">
          <p class="footer-col-title">链接</p>
          <a :href="appConfig.github" target="_blank" rel="noopener noreferrer">GitHub</a>
          <a href="/rss.xml">RSS 订阅</a>
          <a href="https://deepscientist.cc" target="_blank" rel="noopener noreferrer">DeepScientist</a>
          <a href="https://www.travellings.cn/go.html" target="_blank" rel="noreferrer noopener">开往 · 友链接力</a>
        </div>
      </nav>
    </div>

    <div class="footer-wordmark" aria-hidden="true">{{ appConfig.title }}</div>

    <div class="footer-bottom">
      <div class="footer-bottom-inner">
        <p>&copy; {{ year }} {{ appConfig.authorCN }} · {{ appConfig.title }}</p>
        <p>
          <a href="https://icp.gov.moe/" target="_blank" rel="noopener noreferrer">萌ICP备20262170号</a>
          <span aria-hidden="true">·</span>
          <span>由 Nuxt 驱动</span>
        </p>
      </div>
    </div>
  </footer>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
const year = new Date().getFullYear()
</script>
