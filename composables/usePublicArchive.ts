import type { PostMeta } from '~/server/api/posts.get'

export interface UsePublicArchiveOptions {
  lazy?: boolean
  immediate?: boolean
}

export function usePublicArchive(options: UsePublicArchiveOptions = {}) {
  return useAsyncData<PostMeta[]>(
    'archive-public-posts',
    () => $fetch('/api/posts', { responseType: 'json' }),
    {
      lazy: options.lazy ?? false,
      immediate: options.immediate ?? true,
      getCachedData: (key, nuxtApp) => nuxtApp.payload.data[key] ?? nuxtApp.static.data[key],
    },
  )
}

