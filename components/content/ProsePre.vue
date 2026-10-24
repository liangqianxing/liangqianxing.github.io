<template>
  <div class="code-block-wrapper">
    <pre
      ref="preRef"
      :class="$props.class"
      :style="$props.style"
      :data-language="displayLang"
    ><slot /></pre>
    <button
      type="button"
      class="code-copy-btn"
      :class="{ copied }"
      :aria-label="copied ? '已复制代码' : '复制代码'"
      @click="copyCode"
    >{{ copied ? 'copied!' : failed ? 'error' : 'copy' }}</button>
  </div>
</template>

<script setup lang="ts">
const props = defineProps({
  code: {
    type: String,
    default: '',
  },
  language: {
    type: String,
    default: null,
  },
  filename: {
    type: String,
    default: null,
  },
  highlights: {
    type: Array as () => number[],
    default: () => [],
  },
  meta: {
    type: String,
    default: null,
  },
  class: {
    type: String,
    default: null,
  },
  style: {
    type: [String, Object],
    default: null,
  },
})

const preRef = ref<HTMLElement | null>(null)
const copied = ref(false)
const failed = ref(false)
let resetTimer: ReturnType<typeof setTimeout> | undefined

const displayLang = computed(() => props.filename || props.language || 'text')

async function copyCode() {
  const text = props.code || preRef.value?.querySelector('code')?.textContent || preRef.value?.textContent || ''
  if (resetTimer) clearTimeout(resetTimer)
  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    failed.value = false
  } catch {
    copied.value = false
    failed.value = true
  }
  resetTimer = setTimeout(() => {
    copied.value = false
    failed.value = false
  }, 2000)
}

onBeforeUnmount(() => {
  if (resetTimer) clearTimeout(resetTimer)
})
</script>
