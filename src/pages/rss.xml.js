import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { siteConfig } from '../lib/site';
import { filterDrafts, sortPosts } from '../lib/posts';

export async function GET(context) {
  const posts = sortPosts(filterDrafts(await getCollection('posts')));
  return rss({
    title: siteConfig.title,
    description: siteConfig.description,
    site: context.site ?? siteConfig.url,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.description ?? '',
      link: `/posts/${post.slug}/`,
      categories: post.data.tags ?? [],
    })),
    customData: `<language>${siteConfig.lang}</language>`,
  });
}
