<template>
  <main class="page-frame soft-admin-page">
    <header class="page-hero soft-admin-hero">
      <div>
        <p class="eyebrow">Local control</p>
        <h1>文章显示设置</h1>
        <p>管理哪些文章出现在公开页面中。</p>
      </div>
      <div class="soft-admin-hero-actions">
        <NuxtLink to="/posts" class="secondary-action">返回文章库</NuxtLink>
        <button v-if="authenticated" class="text-link" type="button" @click="logout">退出</button>
      </div>
    </header>

    <section class="soft-admin-notice" aria-label="软隐藏说明">
      <span class="soft-admin-notice-mark" aria-hidden="true">i</span>
      <p>这是前端软隐藏：设置只保存在当前浏览器，不提供真正的访问控制。文章源文件和静态内容仍然公开。</p>
    </section>

    <p v-if="!ready" class="soft-admin-loading">正在读取本机设置…</p>

    <section v-else-if="!authenticated" class="soft-admin-login" aria-labelledby="soft-login-title">
      <p class="eyebrow">{{ hasPasscode ? 'Sign in' : 'First visit' }}</p>
      <h2 id="soft-login-title">{{ hasPasscode ? '登录设置页' : '设置本机管理口令' }}</h2>
      <p>{{ hasPasscode ? '输入当前浏览器保存的管理口令。' : '口令只用于本浏览器的界面入口，不承担安全认证。' }}</p>
      <form class="soft-admin-form" @submit.prevent="submitAuth">
        <label>
          <span>{{ hasPasscode ? '管理口令' : '设置口令' }}</span>
          <input v-model="passcode" type="password" autocomplete="off" minlength="4" required />
        </label>
        <label v-if="!hasPasscode">
          <span>再次输入</span>
          <input v-model="confirmation" type="password" autocomplete="off" minlength="4" required />
        </label>
        <button class="primary-action" type="submit">
          {{ hasPasscode ? '登录' : '保存并进入' }}
          <span aria-hidden="true">→</span>
        </button>
      </form>
      <p v-if="message" class="soft-admin-message" role="alert">{{ message }}</p>
    </section>

    <section v-else class="soft-admin-workspace" aria-labelledby="soft-list-title">
      <div class="soft-admin-list-head">
        <div>
          <p class="eyebrow">Visibility map</p>
          <h2 id="soft-list-title">文章可见性</h2>
        </div>
        <span>{{ visibleCount }} / {{ posts.length }} 篇公开显示</span>
      </div>

      <label class="soft-admin-search">
        <span aria-hidden="true">⌕</span>
        <input v-model.trim="query" type="search" placeholder="搜索标题或 slug" aria-label="搜索文章" />
      </label>

      <div v-if="filteredPosts.length" class="soft-admin-list">
        <article v-for="post in filteredPosts" :key="post.slug" class="soft-admin-row">
          <div class="soft-admin-row-copy">
            <strong>{{ post.title }}</strong>
            <small>/posts/{{ post.slug }}</small>
          </div>
          <select
            :value="isHidden(post.slug) ? 'hidden' : 'public'"
            :aria-label="`${post.title} 的显示状态`"
            @change="changeVisibility(post.slug, $event)"
          >
            <option value="public">公开显示</option>
            <option value="hidden">前端隐藏</option>
          </select>
        </article>
      </div>
      <p v-else class="empty-state">没有匹配的文章。</p>

      <p v-if="message" class="soft-admin-message" role="status">{{ message }}</p>
    </section>
  </main>
</template>

<script setup lang="ts">
import type { PostMeta } from '~/server/api/posts.get'

const { data } = await useAsyncData<PostMeta[]>('soft-admin-posts', () => $fetch('/api/posts'))
const {
  authenticated,
  hasPasscode,
  initialize,
  isHidden,
  login,
  logout: clearSession,
  ready,
  setPasscode,
  setVisibility,
} = useSoftPrivacy()

onMounted(initialize)

const passcode = ref('')
const confirmation = ref('')
const query = ref('')
const message = ref('')
const posts = computed(() => data.value ?? [])
const visibleCount = computed(() => posts.value.filter(post => !isHidden(post.slug)).length)
const filteredPosts = computed(() => {
  const term = query.value.toLocaleLowerCase('zh-CN')
  return posts.value.filter(post => !term || `${post.title} ${post.slug}`.toLocaleLowerCase('zh-CN').includes(term))
})

function submitAuth() {
  message.value = ''
  if (!hasPasscode.value) {
    if (passcode.value.length < 4) {
      message.value = '口令至少需要 4 位。'
      return
    }
    if (passcode.value !== confirmation.value) {
      message.value = '两次输入的口令不一致。'
      return
    }
    setPasscode(passcode.value)
    message.value = '本机管理口令已设置。'
    return
  }

  if (!login(passcode.value)) {
    message.value = '口令不正确。'
    return
  }
  passcode.value = ''
}

function changeVisibility(slug: string, event: Event) {
  const visibility = (event.target as HTMLSelectElement).value as 'public' | 'hidden'
  setVisibility(slug, visibility)
  message.value = visibility === 'hidden' ? '文章已从前端列表隐藏。' : '文章已恢复公开显示。'
}

function logout() {
  clearSession()
  message.value = '已退出本机设置。'
}

useHead({
  title: '文章显示设置',
  meta: [{ name: 'robots', content: 'noindex, nofollow' }],
})
</script>
