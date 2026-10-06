<template>
  <SiteImage
    v-if="!failed"
    :key="attempt"
    v-bind="$attrs"
    :src="src"
    :alt="alt"
    :width="width"
    :height="height"
    :style="readingLayout.style"
    :sizes="readingLayout.sizes"
    @error="failed = true"
  />
  <span v-else class="prose-image-fallback" role="group" :aria-label="alt || '文章插图'" :style="placeholderStyle">
    <span>{{ alt || '文章插图' }}</span>
    <small>图片暂时未能加载</small>
    <button type="button" @click="retry">重新加载</button>
    <a :href="originalSrc" target="_blank" rel="noopener noreferrer">查看原图 ↗</a>
  </span>
</template>

<script setup lang="ts">
import { imageDimensions, imageWithBase } from '~/utils/image-assets'

defineOptions({ inheritAttrs: false })
const props = withDefaults(defineProps<{
  src?: string
  alt?: string
  width?: string | number
  height?: string | number
}>(), { src: '', alt: '' })
const failed = ref(false)
const attempt = ref(0)
const baseURL = useRuntimeConfig().app.baseURL
const originalSrc = computed(() => imageWithBase(props.src, baseURL))
const readingLayout = computed(() => {
  const dimensions = imageDimensions(props.src, props.width, props.height)
  const width = Number(dimensions.width)
  const height = Number(dimensions.height)
  const hasWidth = Number.isFinite(width) && width > 0
  const hasRatio = hasWidth && Number.isFinite(height) && height > 0
  const limit = hasRatio && width / height < 1.15 ? 420 : 640
  const readingWidth = Math.floor(Math.min(hasWidth ? width : limit, limit, hasRatio ? 560 * width / height : limit))
  return {
    style: hasWidth ? { width: `${readingWidth}px` } : undefined,
    sizes: `(max-width: ${readingWidth + 40}px) calc(100vw - 40px), ${readingWidth}px`,
    ratio: hasRatio ? `${width} / ${height}` : undefined,
  }
})
const placeholderStyle = computed(() => ({ ...readingLayout.value.style, aspectRatio: readingLayout.value.ratio }))
function retry() {
  attempt.value++
  failed.value = false
}
watch(() => props.src, () => { failed.value = false })
</script>

<style scoped>
.prose-image-fallback {
  display: flex; flex-wrap: wrap; align-items: center; align-content: center; justify-content: center;
  gap: 10px 16px; margin: 1.7em auto; padding: 24px;
  width: 100%; max-width: min(100%, 640px);
  border: 1px dashed var(--line); border-radius: var(--radius-sm);
  background: var(--surface); color: var(--muted); font-size: 14px;
}
.prose-image-fallback > span { flex-basis: 100%; text-align: center; color: var(--text); }
.prose-image-fallback small { flex-basis: 100%; text-align: center; }
.prose-image-fallback button, .prose-image-fallback a { color: var(--accent); font: inherit; }
.prose-image-fallback button { min-height: 40px; padding: 6px 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--bg); cursor: pointer; }
.prose-image-fallback button:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>
