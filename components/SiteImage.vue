<template>
  <img
    ref="image"
    v-bind="$attrs"
    :src="resolvedSrc"
    :srcset="srcset"
    :sizes="srcset ? resolvedSizes : undefined"
    :alt="alt"
    :width="dimensions.width"
    :height="dimensions.height"
    :loading="priority ? 'eager' : loading"
    :fetchpriority="priority ? 'high' : 'auto'"
    decoding="async"
    :class="{ 'site-image-unavailable': unavailable }"
    @error="onError"
  />
</template>

<script setup lang="ts">
import { getImageAsset, imageDimensions, imageWithBase } from '~/utils/image-assets'

defineOptions({ inheritAttrs: false })
const props = withDefaults(defineProps<{
  src: string
  alt?: string
  width?: number | string
  height?: number | string
  sizes?: string
  priority?: boolean
  loading?: 'lazy' | 'eager'
}>(), { alt: '', priority: false, loading: 'lazy' })
const emit = defineEmits<{ error: [] }>()
const image = ref<HTMLImageElement>()
const originalOnly = ref(false)
const unavailable = ref(false)
const asset = computed(() => getImageAsset(props.src))
const baseURL = useRuntimeConfig().app.baseURL
const originalSrc = computed(() => imageWithBase(props.src, baseURL))
const variants = computed(() => asset.value?.variants ?? [])
const useVariants = computed(() => !originalOnly.value && variants.value.length > 0)
const dimensions = computed(() => imageDimensions(props.src, props.width, props.height))
const resolvedSizes = computed(() => props.sizes ?? (props.width ? `${props.width}px` : '100vw'))
const placeholder = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"%3E%3Crect width="64" height="64" rx="6" fill="%23eeece8"/%3E%3Cpath d="M18 44l9-12 7 8 6-6 6 10H18z" fill="%23b5b1ab"/%3E%3Ccircle cx="40" cy="23" r="5" fill="%23b5b1ab"/%3E%3C/svg%3E'
const resolvedSrc = computed(() => unavailable.value ? placeholder : useVariants.value
  ? imageWithBase(variants.value[0].src, baseURL)
  : originalSrc.value)
const srcset = computed(() => !unavailable.value && useVariants.value
  ? variants.value.map(v => `${imageWithBase(v.src, baseURL)} ${v.width}w`).join(', ')
  : undefined)

function onError() {
  if (unavailable.value) return
  if (useVariants.value) {
    originalOnly.value = true
    return
  }
  unavailable.value = true
  emit('error')
}

watch(() => props.src, () => {
  originalOnly.value = false
  unavailable.value = false
})

onMounted(() => {
  // A server-rendered image can fail before Vue attaches its event listener.
  if (image.value?.currentSrc && image.value.complete && !image.value.naturalWidth) onError()
})
</script>
