<template>
  <div class="page-frame">
    <header class="page-hero">
      <p class="eyebrow">Friends</p>
      <h1>友情链接</h1>
      <p>认识的一些有趣的人，也有值得常去阅读的博客。</p>
    </header>

    <section class="friends-grid">
      <a
        v-for="friend in appConfig.friends"
        :key="friend.url"
        :href="friend.url"
        target="_blank"
        rel="noopener noreferrer"
        class="friend-card"
      >
        <span class="friend-avatar">
          <SiteImage
            v-if="friend.avatar && !failedAvatars[friend.url]"
            :src="avatarSource(friend.avatar)"
            :alt="friend.name"
            loading="lazy"
            width="56"
            height="56"
            @error="failedAvatars[friend.url] = true"
          />
          <span v-else>{{ friend.name.charAt(0) }}</span>
        </span>
        <span class="friend-copy">
          <strong>{{ friend.name }}</strong>
          <em>{{ friend.desc }}</em>
        </span>
        <span class="friend-arrow" aria-hidden="true">↗</span>
      </a>
    </section>

    <section class="link-callout">
      <h2>申请友链</h2>
      <p>欢迎互换友链。请在 GitHub 提 Issue，附上博客名称、URL、描述和头像链接。</p>
      <a
        href="https://github.com/liangqianxing/liangqianxing.github.io/issues/new"
        target="_blank"
        rel="noopener noreferrer"
        class="secondary-action"
      >
        提交申请
      </a>
    </section>
  </div>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
const failedAvatars = reactive<Record<string, boolean>>({})

function avatarSource(src: string) {
  // GitHub's public avatar endpoint accepts a pixel size before redirecting.
  if (/^https:\/\/github\.com\/[^/?]+\.png$/.test(src)) return `${src}?size=112`
  return src
}

useHead({
  title: '友情链接',
  meta: [{ name: 'description', content: '友情链接 — 认识的一些有趣的人' }],
})
</script>
