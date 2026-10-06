import type { PostMeta } from '~/server/api/posts.get'

export function usePublicArchive() {
  return useAsyncData<PostMeta[]>('archive-public-posts', () => $fetch('/api/posts'))
}
