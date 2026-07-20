<template>
  <main class="access-page">
    <section class="access-shell" aria-labelledby="login-title">
      <p class="access-kicker">Secure access</p>
      <h1 id="login-title">登录私密阅读</h1>
      <p>登录链接会发送到邮箱，无需保存额外密码。</p>

      <div v-if="!configured" class="access-notice error" role="alert">
        私密阅读尚未连接 Supabase。请先完成项目配置。
      </div>

      <form v-else class="access-form" @submit.prevent="sendMagicLink">
        <label>
          <span>邮箱</span>
          <input
            v-model.trim="email"
            type="email"
            autocomplete="email"
            placeholder="name@example.com"
            required
          />
        </label>
        <button class="primary-action" type="submit" :disabled="submitting">
          {{ submitting ? '发送中' : '发送登录链接' }}
        </button>
      </form>

      <p v-if="message" class="access-notice" :class="{ error: hasError }" role="status">
        {{ message }}
      </p>

      <NuxtLink to="/vault" class="access-back">← 返回私密文章</NuxtLink>
    </section>
  </main>
</template>

<script setup lang="ts">
const route = useRoute()
const { configured, initialize, session, supabase } = useBlogAuth()
const email = ref('')
const submitting = ref(false)
const message = ref('')
const hasError = ref(false)

const nextPath = computed(() => {
  const value = typeof route.query.next === 'string' ? route.query.next : '/vault'
  return value.startsWith('/') && !value.startsWith('//') ? value : '/vault'
})

async function sendMagicLink() {
  if (!supabase) return
  submitting.value = true
  message.value = ''
  hasError.value = false

  try {
    const callback = new URL('/auth/callback', window.location.origin)
    callback.searchParams.set('next', nextPath.value)
    const { error } = await supabase.auth.signInWithOtp({
      email: email.value,
      options: {
        emailRedirectTo: callback.toString(),
      },
    })
    if (error) throw error
    message.value = '登录链接已发送，请在同一设备上打开邮件。'
  } catch (error) {
    hasError.value = true
    message.value = error instanceof Error ? error.message : '登录链接发送失败。'
  } finally {
    submitting.value = false
  }
}

onMounted(async () => {
  try {
    await initialize()
    if (session.value) await navigateTo(nextPath.value, { replace: true })
  } catch (error) {
    hasError.value = true
    message.value = error instanceof Error ? error.message : '读取登录状态失败。'
  }
})

useHead({
  title: '登录私密阅读',
  meta: [{ name: 'robots', content: 'noindex, nofollow' }],
})
</script>
