<template>
  <footer class="footer" aria-label="页脚">
    <div class="footer-inner">
      <p class="footer-copy">
        &copy; {{ year }} {{ appConfig.authorCN }} · {{ appConfig.title }}
      </p>
      <div class="footer-links">
        <a
          :href="appConfig.github"
          target="_blank"
          rel="noopener noreferrer"
          class="footer-link"
          aria-label="GitHub"
        >GitHub</a>
        <a href="/rss.xml" class="footer-link" aria-label="RSS 订阅">RSS</a>
        <a
          href="https://icp.gov.moe/"
          target="_blank"
          rel="noopener noreferrer"
          class="footer-link"
          aria-label="萌ICP备案"
        >萌ICP</a>
      </div>
    </div>
  </footer>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
const year = new Date().getFullYear()
</script>
