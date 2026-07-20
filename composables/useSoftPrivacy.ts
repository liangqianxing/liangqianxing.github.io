import type { PostMeta } from '~/server/api/posts.get'

const HIDDEN_STORAGE_KEY = 'gu.log.soft-hidden-slugs'
const PASSCODE_STORAGE_KEY = 'gu.log.soft-admin-passcode'
const SESSION_STORAGE_KEY = 'gu.log.soft-admin-session'

function readStringList(): string[] {
  if (!import.meta.client) return []
  try {
    const value = JSON.parse(localStorage.getItem(HIDDEN_STORAGE_KEY) ?? '[]')
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
  } catch {
    return []
  }
}

export function useSoftPrivacy() {
  const hiddenSlugs = useState<string[]>('soft-hidden-slugs', () => [])
  const ready = useState<boolean>('soft-privacy-ready', () => false)
  const authenticated = useState<boolean>('soft-admin-authenticated', () => false)
  const hasPasscode = useState<boolean>('soft-admin-has-passcode', () => false)

  function initialize() {
    if (!import.meta.client || ready.value) return
    hiddenSlugs.value = readStringList()
    hasPasscode.value = Boolean(localStorage.getItem(PASSCODE_STORAGE_KEY))
    authenticated.value = localStorage.getItem(SESSION_STORAGE_KEY) === '1'
    ready.value = true
  }

  function persistHiddenSlugs() {
    if (!import.meta.client) return
    localStorage.setItem(HIDDEN_STORAGE_KEY, JSON.stringify(hiddenSlugs.value))
  }

  function isHidden(slug: string) {
    return hiddenSlugs.value.includes(slug)
  }

  function filterVisiblePosts(posts: PostMeta[]) {
    if (!ready.value) return posts
    return posts.filter(post => !isHidden(post.slug))
  }

  function setVisibility(slug: string, visibility: 'public' | 'hidden') {
    if (visibility === 'hidden' && !hiddenSlugs.value.includes(slug)) {
      hiddenSlugs.value = [...hiddenSlugs.value, slug]
    }
    if (visibility === 'public') {
      hiddenSlugs.value = hiddenSlugs.value.filter(item => item !== slug)
    }
    persistHiddenSlugs()
  }

  function setPasscode(passcode: string) {
    if (!import.meta.client) return
    localStorage.setItem(PASSCODE_STORAGE_KEY, passcode)
    localStorage.setItem(SESSION_STORAGE_KEY, '1')
    hasPasscode.value = true
    authenticated.value = true
  }

  function login(passcode: string) {
    if (!import.meta.client || localStorage.getItem(PASSCODE_STORAGE_KEY) !== passcode) return false
    localStorage.setItem(SESSION_STORAGE_KEY, '1')
    authenticated.value = true
    return true
  }

  function logout() {
    if (import.meta.client) localStorage.removeItem(SESSION_STORAGE_KEY)
    authenticated.value = false
  }

  return {
    authenticated,
    filterVisiblePosts,
    hasPasscode,
    hiddenSlugs,
    initialize,
    isHidden,
    login,
    logout,
    ready,
    setPasscode,
    setVisibility,
  }
}
