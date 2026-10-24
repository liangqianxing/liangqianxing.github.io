<template>
  <div class="code-block-wrapper">
    <div class="code-block-bar">
      <div class="code-block-meta">
        <span class="code-block-dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span class="code-block-lang">{{ displayLang }}</span>
        <span v-if="filename && language && filename !== language" class="code-block-file">{{ filename }}</span>
      </div>
      <button
        type="button"
        class="code-copy-btn"
        :class="{ copied, failed }"
        :aria-label="copied ? '已复制代码' : '复制代码'"
        @click="copyCode"
      >
        <svg v-if="copied" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="m5 12.5 4.2 4.2L19 7" />
        </svg>
        <svg v-else viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <rect x="9" y="9" width="11" height="11" rx="2.2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
        <span>{{ copied ? '已复制' : failed ? '复制失败' : '复制' }}</span>
      </button>
    </div>
    <pre
      ref="preRef"
      :class="$props.class"
      :style="$props.style"
      :data-language="displayLang"
    ><slot /></pre>
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

const displayLang = computed(() => props.language || props.filename || 'text')

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
