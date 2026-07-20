export async function renderPrivateMarkdown(source: string): Promise<string> {
  if (!import.meta.client) return ''

  const [{ Marked }, { default: markedKatex }, { default: DOMPurify }] = await Promise.all([
    import('marked'),
    import('marked-katex-extension'),
    import('dompurify'),
  ])

  const parser = new Marked(
    markedKatex({
      nonStandard: true,
      throwOnError: false,
    }),
  )
  const html = await parser.parse(source, { gfm: true })
  return DOMPurify.sanitize(html)
}
