import { defineCollection, defineContentConfig, z } from '@nuxt/content'

export default defineContentConfig({
  collections: {
    posts: defineCollection({
      type: 'page',
      source: 'posts/**/*.md',
      schema: z.object({
        title: z.string(),
        date: z.string(),
        tags: z.array(z.string()).optional().default([]),
        description: z.string().optional(),
        draft: z.boolean().optional().default(false),
        hidden: z.boolean().optional().default(false),
        published: z.boolean().optional(),
        categories: z.array(z.string()).optional().default([]),
      }),
    }),
  },
})
