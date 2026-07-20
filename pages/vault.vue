<template>
  <main class="access-page vault-page">
    <header class="access-head">
      <div>
        <p class="access-kicker">Private archive</p>
        <h1>私密文章</h1>
        <p>这里只展示当前身份有权读取的 Supabase 托管文章。</p>
      </div>
      <div class="access-actions">
        <NuxtLink v-if="isAdmin" to="/admin" class="secondary-action">权限管理</NuxtLink>
        <NuxtLink v-if="!session" :to="loginPath" class="primary-action">登录</NuxtLink>
        <button v-else class="secondary-action" type="button" @click="logout">退出登录</button>
      </div>
    </header>

    <div v-if="!configured" class="access-notice error" role="alert">
      私密阅读尚未连接 Supabase。管理员需要先完成环境变量和数据库配置。
    </div>

    <p v-else-if="loading" class="access-loading">正在读取权限…</p>

    <section v-else-if="selectedPost" class="vault-reader">
      <NuxtLink to="/vault" class="access-back">← 私密文章列表</NuxtLink>
      <header class="vault-article-head">
        <div class="vault-article-meta">
          <span>{{ formatDate(selectedPost.published_at) }}</span>
          <span>{{ visibilityLabel(selectedPost.visibility) }}</span>
        </div>
        <h1>{{ selectedPost.title }}</h1>
        <p v-if="selectedPost.description">{{ selectedPost.description }}</p>
      </header>
      <article class="prose vault-prose" v-html="renderedBody" />
    </section>

    <div v-else-if="loadError" class="access-empty">
      <h2>无法读取这篇文章</h2>
      <p>{{ loadError }}</p>
      <NuxtLink v-if="!session" :to="loginPath" class="primary-action">登录后重试</NuxtLink>
      <NuxtLink v-else to="/vault" class="secondary-action">返回列表</NuxtLink>
    </div>

    <section v-else aria-labelledby="vault-list-title">
      <div class="vault-list-head">
        <h2 id="vault-list-title">可访问文章</h2>
        <span>{{ posts.length }} 篇</span>
      </div>
      <div v-if="posts.length" class="vault-list">
        <NuxtLink
          v-for="post in posts"
          :key="post.id"
          :to="{ path: '/vault', query: { post: post.slug } }"
          class="vault-row"
        >
          <span class="vault-row-date">{{ formatDate(post.published_at) }}</span>
          <span class="vault-row-copy">
            <strong>{{ post.title }}</strong>
            <small>{{ post.description || '暂无摘要' }}</small>
          </span>
          <span class="vault-row-access">{{ visibilityLabel(post.visibility) }}</span>
          <span class="vault-row-arrow" aria-hidden="true">→</span>
        </NuxtLink>
      </div>
      <div v-else class="access-empty">
        <h2>{{ session ? '暂时没有授权文章' : '登录后查看私密内容' }}</h2>
        <p>{{ session ? '管理员尚未为当前身份开放文章。' : '公开文章仍在普通文章库中。' }}</p>
        <NuxtLink v-if="!session" :to="loginPath" class="primary-action">登录</NuxtLink>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import type { ManagedBlogPost, BlogPostVisibility } from '~/types/blog-access'
import { formatDate } from '~/utils/blog'
import { renderPrivateMarkdown } from '~/utils/privateMarkdown'

type ManagedPostSummary = Omit<ManagedBlogPost, 'body_markdown' | 'created_at'>

const route = useRoute()
const { configured, initialize, session, signOut, supabase } = useBlogAuth()
const posts = ref<ManagedPostSummary[]>([])
const selectedPost = ref<ManagedBlogPost | null>(null)
const renderedBody = ref('')
const loading = ref(true)
const loadError = ref('')
const isAdmin = ref(false)

const selectedSlug = computed(() => typeof route.query.post === 'string' ? route.query.post : '')
const loginPath = computed(() => ({
  path: '/login',
  query: { next: selectedSlug.value ? `/vault?post=${encodeURIComponent(selectedSlug.value)}` : '/vault' },
}))

function visibilityLabel(value: BlogPostVisibility) {
  if (value === 'admin') return '仅管理员'
  if (value === 'authenticated') return '登录可见'
  return '公开'
}

async function resolveAdmin() {
  if (!supabase || !session.value) {
    isAdmin.value = false
    return
  }
  const { data } = await supabase
    .from('blog_admins')
    .select('user_id')
    .eq('user_id', session.value.user.id)
    .maybeSingle()
  isAdmin.value = Boolean(data)
}

async function loadContent() {
  if (!supabase) {
    loading.value = false
    return
  }

  loading.value = true
  loadError.value = ''
  selectedPost.value = null
  renderedBody.value = ''

  try {
    if (selectedSlug.value) {
      const { data, error } = await supabase
        .from('blog_posts')
        .select('*')
        .eq('slug', selectedSlug.value)
        .maybeSingle()
      if (error) throw error
      if (!data) {
        loadError.value = session.value
          ? '文章不存在，或当前账号没有读取权限。'
          : '文章不存在，或需要登录后读取。'
        return
      }
      selectedPost.value = data as ManagedBlogPost
      renderedBody.value = await renderPrivateMarkdown(selectedPost.value.body_markdown)
      return
    }

    const { data, error } = await supabase
      .from('blog_posts')
      .select('id, slug, title, description, tags, categories, visibility, published_at, updated_at')
      .order('published_at', { ascending: false })
    if (error) throw error
    posts.value = (data ?? []) as ManagedPostSummary[]
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '文章读取失败。'
  } finally {
    loading.value = false
  }
}

async function logout() {
  await signOut()
  isAdmin.value = false
  await loadContent()
}

onMounted(async () => {
  try {
    await initialize()
    await resolveAdmin()
    await loadContent()
  } catch (error) {
    loading.value = false
    loadError.value = error instanceof Error ? error.message : '初始化失败。'
  }
})

watch(selectedSlug, () => {
  if (import.meta.client && configured) loadContent()
})

useHead({
  title: computed(() => selectedPost.value?.title ?? '私密文章'),
  meta: [{ name: 'robots', content: 'noindex, nofollow, noarchive' }],
})
</script>
