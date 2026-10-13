import { readFile, stat } from 'node:fs/promises'
import { join } from 'node:path'
import { parseMarkdown } from '@nuxtjs/mdc/runtime'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { MDCNode, MDCRoot } from '@nuxtjs/mdc'
import { getLearningCourse } from '../../../../../data/learning/courses'
import { getLearningMaterialArticle } from '../../../../../data/learning/materials'
import sourceImages from '../../../../../public/images/materials/cmu-15-779/manifest.json'

const imageSizes = new Map<string, { width: number; height: number }>()
for (const document of Object.values(sourceImages.documents)) {
  for (const image of 'figures' in document ? document.figures : document.slides) {
    imageSizes.set(image.url, { width: image.width, height: image.height })
  }
}

// Reserve each image's height so lazy loading does not shift directory destinations.
function sizeImages(node: MDCNode | MDCRoot) {
  if (node.type === 'element' && node.tag === 'img') {
    const size = imageSizes.get(String(node.props?.src))
    if (size) Object.assign(node.props, size)
  }
  if ('children' in node) node.children.forEach(sizeImages)
}

// CommonMark needs a space after punctuation-ending bold labels such as **作者：**.
function readingMarkdown(markdown: string): string {
  let fence = ''
  return markdown.split('\n').map(line => {
    const marker = line.match(/^\s*(`{3,}|~{3,})/)
    if (marker) {
      if (!fence) fence = marker[1][0]
      else if (marker[1][0] === fence) fence = ''
      return line
    }
    return fence ? line : line.replace(/(\*\*[^*\n]+[：:。；]\*\*)(?=\S)/g, '$1 ')
  }).join('\n')
}

export default defineEventHandler(async (event) => {
  const course = getLearningCourse(getRouterParam(event, 'course', { decode: true }) ?? '')
  const material = getLearningMaterialArticle(getRouterParam(event, 'material', { decode: true }) ?? '', course?.slug)
  if (!course || !material) {
    throw createError({ statusCode: 404, statusMessage: 'Learning material not found' })
  }

  const filePath = join(process.cwd(), 'public', 'materials', course.slug, material.fileName)
  try {
    const [markdown, fileStats] = await Promise.all([
      readFile(filePath, 'utf8'),
      stat(filePath),
    ])
    const parsed = await parseMarkdown(readingMarkdown(markdown), {
      remark: { plugins: { 'remark-mdc': false, 'remark-math': { instance: remarkMath, options: { singleDollarTextMath: true } } } },
      rehype: { plugins: { 'rehype-katex': { instance: rehypeKatex, options: { strict: false, throwOnError: false } } } },
      toc: { depth: 2, searchDepth: 3 },
    })
    sizeImages(parsed.body)
    return {
      ...material,
      body: parsed.body,
      toc: parsed.toc,
      bytes: fileStats.size,
      downloadPath: `/materials/${encodeURIComponent(course.slug)}/${encodeURIComponent(material.fileName)}`,
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      throw createError({ statusCode: 404, statusMessage: 'Learning material file not found' })
    }
    throw error
  }
})
