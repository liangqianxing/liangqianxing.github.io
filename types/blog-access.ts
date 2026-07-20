export type BlogPostVisibility = 'public' | 'authenticated' | 'admin'

export interface ManagedBlogPost {
  id: string
  slug: string
  title: string
  description: string
  body_markdown: string
  tags: string[]
  categories: string[]
  visibility: BlogPostVisibility
  published_at: string
  created_at: string
  updated_at: string
}
