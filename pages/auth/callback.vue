<template>
  <main class="access-page">
    <section class="access-shell" aria-live="polite">
      <p class="access-kicker">Secure access</p>
      <h1>{{ errorMessage ? '登录未完成' : '正在确认身份' }}</h1>
      <p>{{ errorMessage || '验证完成后会自动返回文章页面。' }}</p>
      <NuxtLink v-if="errorMessage" :to="loginPath" class="primary-action">重新登录</NuxtLink>
    </section>
  </main>
</template>

<script setup lang="ts">
const route = useRoute()
const { configured, initialize, session, supabase } = useBlogAuth()
const errorMessage = ref('')

const nextPath = computed(() => {
  const value = typeof route.query.next === 'string' ? route.query.next : '/vault'
  return value.startsWith('/') && !value.startsWith('//') ? value : '/vault'
})
const loginPath = computed(() => ({ path: '/login', query: { next: nextPath.value } }))

onMounted(async () => {
  if (!configured || !supabase) {
    errorMessage.value = '私密阅读尚未连接 Supabase。'
    return
  }

  try {
    const code = typeof route.query.code === 'string' ? route.query.code : ''
    if (code) {
      const { error } = await supabase.auth.exchangeCodeForSession(code)
      if (error) throw error
    }
    await initialize()
    if (!session.value) throw new Error('登录会话不存在或已经过期。')
    await navigateTo(nextPath.value, { replace: true })
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '身份验证失败。'
  }
})

useHead({
  title: '确认登录',
  meta: [{ name: 'robots', content: 'noindex, nofollow' }],
})
</script>
