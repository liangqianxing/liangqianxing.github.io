/**
 * Blog utility functions
 */

/**
 * Estimate reading time in minutes from markdown/HTML body string.
 * Assumes ~250 Chinese characters or ~200 English words per minute.
 */
export function readingTime(body: string): number {
  if (!body) return 1
  // Strip HTML tags
  const text = body.replace(/<[^>]*>/g, '').replace(/```[\s\S]*?```/g, '')
  // Count CJK characters
  const cjkCount = (text.match(/[一-鿿぀-ヿ가-힯]/g) || []).length
  // Count English words
  const wordCount = (text.replace(/[一-鿿぀-ヿ가-힯]/g, '').match(/\b\w+\b/g) || []).length
  const minutes = cjkCount / 250 + wordCount / 200
  return Math.max(1, Math.ceil(minutes))
}

/**
 * Format a date as YYYY-MM-DD in Asia/Shanghai timezone.
 */
export function formatDate(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date
  if (isNaN(d.getTime())) return ''
  return d.toLocaleDateString('zh-CN', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).replace(/\//g, '-')
}

/**
 * Format a date as MM-DD in Asia/Shanghai timezone.
 */
export function formatMonthDay(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date
  if (isNaN(d.getTime())) return ''
  return d.toLocaleDateString('zh-CN', {
    timeZone: 'Asia/Shanghai',
    month: '2-digit',
    day: '2-digit',
  }).replace(/\//g, '-')
}

/**
 * Convert a tag string to a URL-safe slug.
 * Replaces spaces and slashes with hyphens, lowercases ASCII.
 */
export function tagSlug(tag: string): string {
  return tag
    .trim()
    .replace(/\s*\/\s*/g, '-')
    .replace(/\s+/g, '-')
    .toLowerCase()
}

/**
 * Determine if a post should be visible in listings.
 * Returns false if draft=true, hidden=true, or published===false.
 */
export function isVisible(post: {
  draft?: boolean
  hidden?: boolean
  published?: boolean
  [key: string]: unknown
}): boolean {
  if (post.draft === true) return false
  if (post.hidden === true) return false
  if (post.published === false) return false
  return true
}
