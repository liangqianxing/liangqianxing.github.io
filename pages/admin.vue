<template>
  <main class="access-page admin-page">
    <header class="access-head">
      <div>
        <p class="access-kicker">Access control</p>
        <h1>文章权限管理</h1>
        <p>管理 Supabase 中的正文和访问级别。</p>
      </div>
      <div class="access-actions">
        <NuxtLink to="/vault" class="secondary-action">查看私密文章</NuxtLink>
        <button v-if="session" class="secondary-action" type="button" @click="logout">退出</button>
      </div>
    </header>

    <div v-if="!configured" class="access-notice error" role="alert">
      Supabase 尚未配置。请按照 `docs/private-posts.md` 完成连接。
    </div>
    <p v-else-if="loading" class="access-loading">正在验证管理员权限…</p>
    <div v-else-if="!session" class="access-empty">
      <h2>需要登录</h2>
      <p>管理员面板不会向匿名访问者开放。</p>
      <NuxtLink :to="{ path: '/login', query: { next: '/admin' } }" class="primary-action">登录</NuxtLink>
    </div>
    <div v-else-if="!isAdmin" class="access-empty">
      <h2>当前账号不是管理员</h2>
      <p>请在 Supabase 的 `blog_admins` 表中登记这个账号。</p>
    </div>

    <template v-else>
      <p v-if="message" class="access-notice" :class="{ error: hasError }" role="status">
        {{ message }}
      </p>

      <section class="admin-layout">
        <form class="admin-editor" @submit.prevent="savePost">
          <div class="admin-section-head">
            <div>
              <p class="access-kicker">{{ form.id ? 'Edit' : 'New post' }}</p>
              <h2>{{ form.id ? '编辑托管文章' : '新建托管文章' }}</h2>
            </div>
            <button v-if="form.id" class="admin-text-button" type="button" @click="resetForm">取消编辑</button>
          </div>

          <label>
            <span>标题</span>
            <input v-model.trim="form.title" required />
          </label>
          <label>
            <span>Slug</span>
            <input v-model.trim="form.slug" pattern="[a-z0-9-]+" placeholder="private-note" required />
          </label>
          <label>
            <span>摘要</span>
            <textarea v-model.trim="form.description" rows="3" />
          </label>
          <div class="admin-form-row">
            <label>
              <span>权限</span>
              <select v-model="form.visibility">
                <option value="public">公开</option>
                <option value="authenticated">登录可见</option>
                <option value="admin">仅管理员</option>
              </select>
            </label>
            <label>
              <span>发布日期</span>
              <input v-model="form.publishedAt" type="datetime-local" required />
            </label>
          </div>
          <label>
            <span>标签（逗号分隔）</span>
            <input v-model.trim="form.tags" placeholder="LLM, Notes" />
          </label>
          <label>
            <span>Markdown 正文</span>
            <textarea v-model="form.body" class="admin-markdown" rows="18" required />
          </label>
          <button class="primary-action" type="submit" :disabled="saving">
            {{ saving ? '保存中' : (form.id ? '保存修改' : '创建文章') }}
          </button>
        </form>

        <section class="admin-managed" aria-labelledby="managed-title">
          <div class="admin-section-head">
            <div>
              <p class="access-kicker">Managed</p>
              <h2 id="managed-title">Supabase 托管文章</h2>
            </div>
            <span>{{ posts.length }} 篇</span>
          </div>

          <div v-if="posts.length" class="admin-post-list">
            <article v-for="post in posts" :key="post.id" class="admin-post-row">
              <div>
                <strong>{{ post.title }}</strong>
                <small>/vault?post={{ post.slug }}</small>
              </div>
              <select
                :value="post.visibility"
                :aria-label="`${post.title} 的访问权限`"
                @change="changeVisibility(post, $event)"
              >
                <option value="public">公开</option>
                <option value="authenticated">登录可见</option>
                <option value="admin">仅管理员</option>
              </select>
              <div class="admin-row-actions">
                <button type="button" @click="editPost(post)">编辑</button>
                <button class="danger" type="button" @click="deletePost(post)">删除</button>
              </div>
            </article>
          </div>
          <p v-else class="admin-empty-copy">尚未创建 Supabase 托管文章。</p>
        </section>
      </section>

      <section class="admin-static" aria-labelledby="static-title">
        <div class="admin-section-head">
          <div>
            <p class="access-kicker">Static source</p>
            <h2 id="static-title">仍然公开的 Markdown 文章</h2>
          </div>
          <span>{{ staticPosts.length }} 篇</span>
        </div>
        <p>
          这些正文已经进入 Git 仓库和静态产物，后台开关无法保护它们。需要私密化时，先在上方创建托管副本，再从 `content/posts` 删除源文件并重新部署。
        </p>
        <div class="admin-static-list">
          <span v-for="post in staticPosts" :key="post.slug">{{ post.title }}</span>
        </div>
      </section>
    </template>
  </main>
</template>

<script setup lang="ts">
import type { ManagedBlogPost, BlogPostVisibility } from '~/types/blog-access'
import type { PostMeta } from '~/server/api/posts.get'

const { configured, initialize, session, signOut, supabase } = useBlogAuth()
const loading = ref(true)
const saving = ref(false)
const isAdmin = ref(false)
const posts = ref<ManagedBlogPost[]>([])
const staticPosts = ref<PostMeta[]>([])
const message = ref('')
const hasError = ref(false)

function localDateTimeValue(value = new Date()) {
  const offset = value.getTimezoneOffset() * 60_000
  return new Date(value.getTime() - offset).toISOString().slice(0, 16)
}

const form = reactive({
  id: '',
  body: '',
  description: '',
  publishedAt: localDateTimeValue(),
  slug: '',
  tags: '',
  title: '',
  visibility: 'admin' as BlogPostVisibility,
})

function resetForm() {
  Object.assign(form, {
    id: '',
    body: '',
    description: '',
    publishedAt: localDateTimeValue(),
    slug: '',
    tags: '',
    title: '',
    visibility: 'admin' as BlogPostVisibility,
  })
}

function showMessage(text: string, error = false) {
  message.value = text
  hasError.value = error
}

async function verifyAdmin() {
  if (!supabase || !session.value) return false
  const { data, error } = await supabase
    .from('blog_admins')
    .select('user_id')
    .eq('user_id', session.value.user.id)
    .maybeSingle()
  if (error) throw error
  isAdmin.value = Boolean(data)
  return isAdmin.value
}

async function loadPosts() {
  if (!supabase || !isAdmin.value) return
  const { data, error } = await supabase
    .from('blog_posts')
    .select('*')
    .order('published_at', { ascending: false })
  if (error) throw error
  posts.value = (data ?? []) as ManagedBlogPost[]
}

async function savePost() {
  if (!supabase || !isAdmin.value) return
  saving.value = true
  showMessage('')
  const tags = form.tags.split(',').map(tag => tag.trim()).filter(Boolean)
  const payload = {
    body_markdown: form.body,
    categories: [] as string[],
    description: form.description,
    published_at: new Date(form.publishedAt).toISOString(),
    slug: form.slug.toLowerCase(),
    tags,
    title: form.title,
    visibility: form.visibility,
  }

  try {
    const query = form.id
      ? supabase.from('blog_posts').update(payload).eq('id', form.id)
      : supabase.from('blog_posts').insert(payload)
    const { error } = await query
    if (error) throw error
    showMessage(form.id ? '文章已更新。' : '文章已创建。')
    resetForm()
    await loadPosts()
  } catch (error) {
    showMessage(error instanceof Error ? error.message : '文章保存失败。', true)
  } finally {
    saving.value = false
  }
}

function editPost(post: ManagedBlogPost) {
  Object.assign(form, {
    id: post.id,
    body: post.body_markdown,
    description: post.description,
    publishedAt: localDateTimeValue(new Date(post.published_at)),
    slug: post.slug,
    tags: post.tags.join(', '),
    title: post.title,
    visibility: post.visibility,
  })
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

async function changeVisibility(post: ManagedBlogPost, event: Event) {
  if (!supabase) return
  const value = (event.target as HTMLSelectElement).value as BlogPostVisibility
  const { error } = await supabase.from('blog_posts').update({ visibility: value }).eq('id', post.id)
  if (error) {
    showMessage(error.message, true)
    ;(event.target as HTMLSelectElement).value = post.visibility
    return
  }
  post.visibility = value
  showMessage(`《${post.title}》已设为${value === 'public' ? '公开' : value === 'authenticated' ? '登录可见' : '仅管理员'}。`)
}

async function deletePost(post: ManagedBlogPost) {
  if (!supabase || !window.confirm(`确定删除《${post.title}》吗？此操作无法撤销。`)) return
  const { error } = await supabase.from('blog_posts').delete().eq('id', post.id)
  if (error) {
    showMessage(error.message, true)
    return
  }
  if (form.id === post.id) resetForm()
  showMessage('文章已删除。')
  await loadPosts()
}

async function logout() {
  await signOut()
  await navigateTo('/login?next=/admin')
}

onMounted(async () => {
  try {
    await initialize()
    if (!session.value) return
    if (!await verifyAdmin()) return
    const [publicPosts] = await Promise.all([
      $fetch<PostMeta[]>('/api/posts'),
      loadPosts(),
    ])
    staticPosts.value = publicPosts
  } catch (error) {
    showMessage(error instanceof Error ? error.message : '管理员面板加载失败。', true)
  } finally {
    loading.value = false
  }
})

useHead({
  title: '文章权限管理',
  meta: [{ name: 'robots', content: 'noindex, nofollow, noarchive' }],
})
</script>
