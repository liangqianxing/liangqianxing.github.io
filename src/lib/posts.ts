import type { CollectionEntry } from 'astro:content';

export type PostEntry = CollectionEntry<'posts'>;

/** 按发布日期降序排列 */
export function sortPosts(posts: PostEntry[]) {
  return posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

/**
 * 过滤草稿：生产环境隐藏 draft:true 的文章。
 * 开发模式下全部显示，方便本地预览。
 */
export function filterDrafts(posts: PostEntry[]) {
  if (import.meta.env.DEV) return posts;
  return posts.filter((p) => !p.data.draft);
}

/** 格式化日期为 MM-DD（上海时区） */
export function formatMonthDay(date: Date) {
  return new Intl.DateTimeFormat('en-US', {
    month: '2-digit',
    day: '2-digit',
    timeZone: 'Asia/Shanghai',
  }).format(date).replace('/', '-');
}

/** 格式化完整日期为 YYYY-MM-DD */
export function formatFullDate(date: Date) {
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    timeZone: 'Asia/Shanghai',
  }).format(date).replace(/\//g, '-');
}

/** 估算阅读时间（分钟），中文 300字/分钟，英文 200词/分钟 */
export function readingTime(body: string) {
  const chineseChars = body.match(/[一-鿿]/g)?.length ?? 0;
  const englishWords = body.replace(/[一-鿿]/g, ' ').match(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g)?.length ?? 0;
  return Math.max(1, Math.ceil(chineseChars / 300 + englishWords / 200));
}

/** 将标签中的斜杠替换为连字符，用作 URL slug */
export function tagSlug(tag: string) {
  return tag.replace(/\//g, '-');
}

/**
 * 从文章 body 中提取纯文本摘要（用于无 description 时的 SEO fallback）。
 * 去掉 Markdown 语法后取前 N 个字符。
 */
export function extractExcerpt(body: string, maxLen = 120): string {
  const plain = body
    .replace(/```[\s\S]*?```/g, '')   // 移除代码块
    .replace(/`[^`]+`/g, '')          // 移除行内代码
    .replace(/#{1,6}\s+/g, '')        // 移除标题符号
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1') // 移除链接/图片
    .replace(/[*_~>|-]+/g, '')        // 移除 Markdown 符号
    .replace(/\s+/g, ' ')
    .trim();
  return plain.length > maxLen ? plain.slice(0, maxLen) + '…' : plain;
}
