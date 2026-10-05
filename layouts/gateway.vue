<template>
  <div class="gateway-site"><slot /></div>
</template>

<script setup lang="ts">
import { academicProfile } from '~/data/academic'
import '~/assets/css/gateway.css'

onMounted(() => {
  const root = document.documentElement
  root.classList.remove('dark', 'cyber')
  root.classList.add('light')
  root.dataset.theme = 'light'
})

useHead({
  htmlAttrs: { lang: 'zh-CN' },
  bodyAttrs: { class: 'gateway-body' },
  titleTemplate: title => title ? `${title} · ${academicProfile.name}` : academicProfile.name,
  meta: [
    { name: 'theme-color', content: '#080c19' },
    { property: 'og:site_name', content: `${academicProfile.name} · ${academicProfile.nameCN}` },
    { property: 'og:type', content: 'website' },
    { name: 'twitter:card', content: 'summary' },
  ],
})
</script>
