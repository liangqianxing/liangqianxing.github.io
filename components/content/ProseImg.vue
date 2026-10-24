<template>
  <template v-if="!failed">
    <SiteImage
      :key="attempt"
      v-bind="$attrs"
      :src="src"
      :alt="alt"
      :width="width"
      :height="height"
      :style="readingLayout.style"
      :sizes="readingLayout.sizes"
      class="prose-zoomable-img"
      role="button"
      tabindex="0"
      :aria-label="alt ? `${alt}（点击或按回车放大查看）` : '点击或按回车放大查看插图'"
      title="点击放大查看"
      @click="openZoom"
      @keydown.enter.prevent="openZoom"
      @keydown.space.prevent="openZoom"
      @error="failed = true"
    />
    <Teleport to="body">
      <Transition name="prose-lightbox">
        <div
          v-if="zoomed"
          class="prose-lightbox"
          role="dialog"
          aria-modal="true"
          :aria-label="alt || '放大查看插图'"
          @click="closeZoom"
        >
          <button
            ref="closeBtnRef"
            type="button"
            class="prose-lightbox-close"
            aria-label="关闭大图预览 (ESC)"
            @click.stop="closeZoom"
          >×</button>
          <img
            :src="originalSrc"
            :alt="alt"
            class="prose-lightbox-img"
            @click.stop="closeZoom"
          />
          <p v-if="alt" class="prose-lightbox-caption">{{ alt }}</p>
        </div>
      </Transition>
    </Teleport>
  </template>
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
const zoomed = ref(false)
const closeBtnRef = ref<HTMLButtonElement | null>(null)
let lastActiveElement: HTMLElement | null = null
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

function onEscape(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    closeZoom()
  }
}

function openZoom() {
  if (typeof document !== 'undefined') {
    lastActiveElement = document.activeElement as HTMLElement | null
  }
  zoomed.value = true
  if (typeof window !== 'undefined') {
    window.addEventListener('keydown', onEscape)
  }
  nextTick(() => {
    closeBtnRef.value?.focus()
  })
}

function closeZoom() {
  const wasZoomed = zoomed.value
  zoomed.value = false
  if (typeof window !== 'undefined') {
    window.removeEventListener('keydown', onEscape)
  }
  if (wasZoomed && lastActiveElement) {
    const target = lastActiveElement
    lastActiveElement = null
    nextTick(() => {
      target.focus()
    })
  }
}

onBeforeUnmount(() => {
  if (typeof window !== 'undefined') {
    window.removeEventListener('keydown', onEscape)
  }
})

watch(() => props.src, () => {
  failed.value = false
  closeZoom()
})
</script>

<style scoped>
.prose-zoomable-img {
  cursor: zoom-in;
  transition: transform 220ms cubic-bezier(0.22, 1, 0.36, 1), box-shadow 220ms ease;
}
@media (hover: hover) and (pointer: fine) {
  .prose-zoomable-img:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 32px rgba(20, 16, 28, 0.12);
  }
}
.prose-lightbox {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
  padding: 28px 20px;
  background: rgba(12, 11, 18, 0.84);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  cursor: zoom-out;
}
.prose-lightbox-close {
  position: absolute;
  top: 20px;
  right: 24px;
  width: 40px;
  height: 40px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.08);
  color: #f4f2ee;
  font-size: 1.35rem;
  line-height: 1;
  cursor: pointer;
  transition: background-color 160ms ease, transform 160ms ease;
}
.prose-lightbox-close:hover {
  background: rgba(255, 255, 255, 0.18);
  transform: scale(1.05);
}
.prose-lightbox-img {
  max-width: min(94vw, 1360px);
  max-height: 82vh;
  width: auto;
  height: auto;
  object-fit: contain;
  border-radius: 10px;
  background: #fff;
  box-shadow: 0 24px 72px rgba(0, 0, 0, 0.45);
}
.prose-lightbox-caption {
  margin: 0;
  max-width: 72ch;
  color: rgba(244, 242, 238, 0.86);
  font-size: 0.82rem;
  line-height: 1.6;
  text-align: center;
}
.prose-lightbox-enter-active,
.prose-lightbox-leave-active {
  transition: opacity 180ms ease;
}
.prose-lightbox-enter-from,
.prose-lightbox-leave-to {
  opacity: 0;
}
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
