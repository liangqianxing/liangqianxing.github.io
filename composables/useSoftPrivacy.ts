import type { PostMeta } from '~/server/api/posts.get'

const PASSCODE_STORAGE_KEY = 'gu.log.soft-admin-passcode'
const SESSION_STORAGE_KEY = 'gu.log.soft-admin-session'
const GITHUB_TOKEN_STORAGE_KEY = 'gu.log.soft-admin-github-token'
const LEGACY_HIDDEN_STORAGE_KEY = 'gu.log.soft-hidden-slugs'
const VISIBILITY_CONFIG_URL = '/article-visibility.json'
const GITHUB_REPOSITORY = 'liangqianxing/liangqianxing.github.io'
const GITHUB_BRANCH = 'main'
const GITHUB_CONFIG_PATH = 'public/article-visibility.json'

interface VisibilityConfig {
  hiddenSlugs: string[]
}

interface GithubContentResponse {
  content: string
  encoding: string
  sha: string
}

interface VisibilityResult {
  ok: boolean
  message: string
}

function normalizeSlugs(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return [...new Set(value.filter((item): item is string => typeof item === 'string' && item.length > 0))].sort()
}

function parseVisibilityConfig(value: unknown): VisibilityConfig {
  const config = value && typeof value === 'object' ? value as Partial<VisibilityConfig> : {}
  return { hiddenSlugs: normalizeSlugs(config.hiddenSlugs) }
}

function decodeBase64(value: string) {
  const bytes = Uint8Array.from(atob(value.replace(/\s/g, '')), char => char.charCodeAt(0))
  return new TextDecoder().decode(bytes)
}

function encodeBase64(value: string) {
  const bytes = new TextEncoder().encode(value)
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return btoa(binary)
}

function githubHeaders(token: string) {
  return {
    Accept: 'application/vnd.github+json',
    Authorization: `Bearer ${token}`,
    'X-GitHub-Api-Version': '2022-11-28',
  }
}

async function readRepositoryConfig(token: string) {
  const response = await $fetch<GithubContentResponse>(
    `https://api.github.com/repos/${GITHUB_REPOSITORY}/contents/${GITHUB_CONFIG_PATH}`,
    {
      headers: githubHeaders(token),
      query: { ref: GITHUB_BRANCH, timestamp: Date.now() },
    },
  )

  if (response.encoding !== 'base64') throw new Error('GitHub 返回了无法识别的配置格式。')

  return {
    config: parseVisibilityConfig(JSON.parse(decodeBase64(response.content))),
    sha: response.sha,
  }
}

function githubErrorMessage(error: unknown) {
  const status = error && typeof error === 'object' && 'status' in error ? Number(error.status) : 0
  if (status === 401) return 'Token 无效或已过期。'
  if (status === 403) return 'Token 没有仓库 Contents 写入权限。'
  if (status === 404) return '没有找到可见性配置，请确认 Token 可访问这个仓库。'
  if (status === 409 || status === 422) return '配置刚被其他提交更新，请稍后重试。'
  return '同步失败，请检查网络和 GitHub Token 后重试。'
}

export function useSoftPrivacy() {
  const runtimeConfig = useRuntimeConfig()
  const initialHiddenSlugs = normalizeSlugs(runtimeConfig.public.softHiddenSlugs)
  const hiddenSlugs = useState<string[]>('soft-hidden-slugs', () => initialHiddenSlugs)
  const ready = useState<boolean>('soft-privacy-ready', () => false)
  const loading = useState<boolean>('soft-privacy-loading', () => false)
  const authenticated = useState<boolean>('soft-admin-authenticated', () => false)
  const hasPasscode = useState<boolean>('soft-admin-has-passcode', () => false)
  const githubToken = useState<string>('soft-admin-github-token', () => '')
  const hasGithubToken = computed(() => githubToken.value.length > 0)

  async function initialize() {
    if (!import.meta.client || ready.value || loading.value) return
    loading.value = true

    try {
      const config = await $fetch<VisibilityConfig>(VISIBILITY_CONFIG_URL, {
        query: { timestamp: Date.now() },
      })
      hiddenSlugs.value = parseVisibilityConfig(config).hiddenSlugs
    } catch {}

    githubToken.value = localStorage.getItem(GITHUB_TOKEN_STORAGE_KEY) ?? ''
    hasPasscode.value = Boolean(localStorage.getItem(PASSCODE_STORAGE_KEY))
    authenticated.value = localStorage.getItem(SESSION_STORAGE_KEY) === '1'
    localStorage.removeItem(LEGACY_HIDDEN_STORAGE_KEY)
    ready.value = true
    loading.value = false
  }

  function isHidden(slug: string) {
    return hiddenSlugs.value.includes(slug)
  }

  function filterVisiblePosts(posts: PostMeta[]) {
    return posts.filter(post => !isHidden(post.slug))
  }

  async function connectGithub(token: string): Promise<VisibilityResult> {
    if (!import.meta.client) return { ok: false, message: '只能在浏览器中连接 GitHub。' }
    const normalizedToken = token.trim()
    if (!normalizedToken) return { ok: false, message: '请输入 GitHub Token。' }

    try {
      const { config } = await readRepositoryConfig(normalizedToken)
      githubToken.value = normalizedToken
      hiddenSlugs.value = config.hiddenSlugs
      localStorage.setItem(GITHUB_TOKEN_STORAGE_KEY, normalizedToken)
      return { ok: true, message: 'GitHub 已连接，可见性设置会同步到全站。' }
    } catch (error) {
      return { ok: false, message: githubErrorMessage(error) }
    }
  }

  function disconnectGithub() {
    if (import.meta.client) localStorage.removeItem(GITHUB_TOKEN_STORAGE_KEY)
    githubToken.value = ''
  }

  async function setVisibility(slug: string, visibility: 'public' | 'hidden'): Promise<VisibilityResult> {
    if (!githubToken.value) {
      return { ok: false, message: '请先连接 GitHub，再修改文章显示状态。' }
    }

    try {
      const { config, sha } = await readRepositoryConfig(githubToken.value)
      const nextHiddenSlugs = visibility === 'hidden'
        ? normalizeSlugs([...config.hiddenSlugs, slug])
        : config.hiddenSlugs.filter(item => item !== slug)
      const nextConfig: VisibilityConfig = { hiddenSlugs: nextHiddenSlugs }

      await $fetch(
        `https://api.github.com/repos/${GITHUB_REPOSITORY}/contents/${GITHUB_CONFIG_PATH}`,
        {
          method: 'PUT',
          headers: githubHeaders(githubToken.value),
          body: {
            branch: GITHUB_BRANCH,
            content: encodeBase64(`${JSON.stringify(nextConfig, null, 2)}\n`),
            message: `${visibility === 'hidden' ? 'hide' : 'show'} post: ${slug}`,
            sha,
          },
        },
      )

      hiddenSlugs.value = nextHiddenSlugs
      return {
        ok: true,
        message: visibility === 'hidden'
          ? '隐藏设置已提交，Pages 部署完成后对所有访客生效。'
          : '公开设置已提交，Pages 部署完成后文章会重新显示。',
      }
    } catch (error) {
      return { ok: false, message: githubErrorMessage(error) }
    }
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
    connectGithub,
    disconnectGithub,
    filterVisiblePosts,
    hasGithubToken,
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
