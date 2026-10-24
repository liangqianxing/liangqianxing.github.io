<template>
  <span class="material-figure">
    <button
      type="button"
      class="material-figure-trigger"
      :aria-label="`${alt || '原文插图'}，放大查看`"
      @click="openZoom"
    >
      <img :src="src" :alt="alt" loading="lazy" decoding="async" :width="width" :height="height" />
    </button>
    <span v-if="alt" class="material-figure-caption">{{ alt }} · 点击放大查看</span>
    <Teleport to="body">
      <Transition name="material-lightbox">
        <div
          v-if="zoomed"
          class="material-lightbox"
          role="dialog"
          aria-modal="true"
          :aria-label="alt || '原文插图预览'"
          @click="closeZoom"
        >
          <button
            ref="closeBtnRef"
            type="button"
            class="material-lightbox-close"
            aria-label="关闭大图预览 (ESC)"
            @click.stop="closeZoom"
          >×</button>
          <img
            :src="src"
            :alt="alt"
            class="material-lightbox-img"
            @click.stop="closeZoom"
          />
          <p v-if="alt" class="material-lightbox-caption">
            {{ alt }}
            <a :href="src" target="_blank" rel="noopener noreferrer" @click.stop>打开原图 ↗</a>
          </p>
        </div>
      </Transition>
    </Teleport>
  </span>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<{ src: string; alt?: string; width?: string | number; height?: string | number }>(), { alt: '' })
const zoomed = ref(false)
const closeBtnRef = ref<HTMLButtonElement | null>(null)
let lastActiveElement: HTMLElement | null = null

function onEscape(e: KeyboardEvent) {
  if (e.key === 'Escape') closeZoom()
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

watch(() => props.src, closeZoom)
</script>

<style scoped>
.material-figure { display: block; margin: 1.8em 0; }
.material-figure-trigger {
  display: block;
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: zoom-in;
  text-align: inherit;
}
.material-figure img {
  display: block;
  width: 100%;
  max-width: 100%;
  height: auto;
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: white;
  transition: transform 200ms cubic-bezier(0.22, 1, 0.36, 1), box-shadow 200ms ease;
}
@media (hover: hover) and (pointer: fine) {
  .material-figure-trigger:hover img {
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(20, 16, 28, 0.1);
  }
}
.material-figure-caption { display: block; margin-top: 8px; text-align: center; color: var(--muted); font-size: 0.74rem; line-height: 1.7; }
.material-lightbox {
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
.material-lightbox-close {
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
}
.material-lightbox-img {
  max-width: min(94vw, 1360px);
  max-height: 82vh;
  width: auto;
  height: auto;
  object-fit: contain;
  border-radius: 10px;
  background: #fff;
  box-shadow: 0 24px 72px rgba(0, 0, 0, 0.45);
}
.material-lightbox-caption {
  margin: 0;
  display: flex;
  align-items: center;
  gap: 12px;
  color: rgba(244, 242, 238, 0.86);
  font-size: 0.82rem;
}
.material-lightbox-caption a {
  color: #a7d8ff;
  text-decoration: none;
}
.material-lightbox-enter-active,
.material-lightbox-leave-active {
  transition: opacity 180ms ease;
}
.material-lightbox-enter-from,
.material-lightbox-leave-to {
  opacity: 0;
}
</style>
