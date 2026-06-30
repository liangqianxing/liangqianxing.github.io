import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    tags: z.array(z.string()).optional(),
    categories: z.array(z.string()).optional(),
    description: z.string().optional(),
    updated: z.coerce.date().optional(),
    /** 设为 true 则该文章仅在开发环境可见，不会出现在构建产物中 */
    draft: z.boolean().optional().default(false),
  }),
});

export const collections = { posts };
