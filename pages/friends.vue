<template>
  <div class="page-wrapper">
    <h1 class="page-title">友情链接</h1>
    <p class="page-desc">认识的一些有趣的人</p>

    <div class="friends-grid">
      <a
        v-for="friend in appConfig.friends"
        :key="friend.url"
        :href="friend.url"
        target="_blank"
        rel="noopener noreferrer"
        class="friend-card"
      >
        <div class="friend-avatar">
          <img
            v-if="friend.avatar"
            :src="friend.avatar"
            :alt="friend.name"
            loading="lazy"
            width="44"
            height="44"
          />
          <span v-else class="friend-avatar-placeholder">
            {{ friend.name.charAt(0) }}
          </span>
        </div>
        <div class="friend-info">
          <div class="friend-name">{{ friend.name }}</div>
          <div class="friend-desc">{{ friend.desc }}</div>
        </div>
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          style="color: var(--text-very-muted); flex-shrink: 0"
          aria-hidden="true"
        >
          <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/>
          <polyline points="15 3 21 3 21 9"/>
          <line x1="10" y1="14" x2="21" y2="3"/>
        </svg>
      </a>
    </div>

    <!-- Apply section -->
    <div style="margin-top: 3rem; padding: 1.5rem; background: var(--bg-surface); border: 1px solid var(--border-default); border-radius: 10px">
      <div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-very-muted); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.75rem">申请友链</div>
      <p style="font-size: 0.875rem; color: var(--text-muted); line-height: 1.7; margin-bottom: 0.75rem">
        欢迎互换友链！请在 GitHub 提 Issue 或发邮件联系我，附上博客名称、URL、描述和头像链接。
      </p>
      <a :href="`${appConfig.github}/issues`" target="_blank" rel="noopener noreferrer" class="pill" style="display: inline-flex">
        提交申请 →
      </a>
    </div>
  </div>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()

useHead({
  title: '友情链接',
  meta: [{ name: 'description', content: '友情链接 — 认识的一些有趣的人' }],
})
</script>
