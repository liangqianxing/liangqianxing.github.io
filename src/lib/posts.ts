import type { CollectionEntry } from 'astro:content';

export type PostEntry = CollectionEntry<'posts'>;

export function sortPosts(posts: PostEntry[]) {
  return posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

export function formatMonthDay(date: Date) {
  return new Intl.DateTimeFormat('en-US', {
    month: '2-digit',
    day: '2-digit',
    timeZone: 'Asia/Shanghai',
  }).format(date).replace('/', '-');
}

export function readingTime(body: string) {
  const chineseChars = body.match(/[\u4e00-\u9fff]/g)?.length ?? 0;
  const englishWords = body.replace(/[\u4e00-\u9fff]/g, ' ').match(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g)?.length ?? 0;
  return Math.max(1, Math.ceil(chineseChars / 300 + englishWords / 200));
}

export function tagSlug(tag: string) {
  return tag.replace(/\//g, '-');
}
