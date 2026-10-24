<template>
  <div class="page-frame material-reader">
    <header class="material-reader-head">
      <NuxtLink :to="`/learning/${encodeURIComponent(course.slug)}`" class="material-back">← {{ course.title }} 学习专题</NuxtLink>
      <p class="material-kicker">{{ course.title }} / 课程资料</p>
      <h1>{{ material?.title || '学习资料' }}</h1>
      <p v-if="material?.description" class="material-reader-description">{{ material.description }}</p>
      <div v-if="material" class="material-reader-actions">
        <a :href="material.downloadPath" download>下载中文译稿 ↓</a>
        <a :href="material.sourcePdf" target="_blank" rel="noopener noreferrer">原文 PDF ↗</a>
        <a href="#original-pdf" @click="pdfOpen = true">对照原文 ↓</a>
      </div>
    </header>

    <details v-if="tocLinks.length" class="material-mobile-toc">
      <summary>阅读目录</summary>
      <nav aria-label="阅读目录">
        <template v-for="heading in tocLinks" :key="heading.id">
          <a
            :href="`#${heading.id}`"
            :aria-current="activeId === heading.id ? 'location' : undefined"
            @click.prevent="scrollToHeading(heading.id)"
          >{{ heading.text }}</a>
          <a
            v-for="child in heading.children"
            :key="child.id"
            :href="`#${child.id}`"
            class="material-toc-child"
            :aria-current="activeId === child.id ? 'location' : undefined"
            @click.prevent="scrollToHeading(child.id)"
          >{{ child.text }}</a>
        </template>
      </nav>
    </details>

    <div v-if="material" class="material-reader-layout">
      <article ref="articleRef" class="prose material-reader-content" aria-label="中文译稿">
        <MDCRenderer :key="material.slug" :body="material.body" :components="readerComponents" :prose="false" />
        <details id="original-pdf" class="material-source" :open="pdfOpen" @toggle="pdfOpen = ($event.target as HTMLDetailsElement).open">
          <summary>原文 PDF 对照</summary>
          <p>正文插图来自原始 PDF，点击图片可放大查看。也可以在下方翻阅完整原文。</p>
          <object v-if="pdfOpen" :data="material.sourcePdf" type="application/pdf" class="material-source-pdf">
            <p>当前浏览器不支持内嵌 PDF，请<a :href="material.sourcePdf" target="_blank" rel="noopener noreferrer">打开原文 PDF</a>查看。</p>
          </object>
          <a :href="material.sourcePdf" target="_blank" rel="noopener noreferrer">在新窗口打开原文 PDF ↗</a>
        </details>
      </article>

      <aside v-if="tocLinks.length" ref="tocRef" class="material-toc" aria-label="阅读目录">
        <p class="material-kicker">CONTENTS / 目录</p>
        <nav>
          <template v-for="heading in tocLinks" :key="heading.id">
            <a
              :href="`#${heading.id}`"
              :class="{ 'is-active': activeId === heading.id }"
              @click.prevent="scrollToHeading(heading.id)"
            >{{ heading.text }}</a>
            <a
              v-for="child in heading.children"
              :key="child.id"
              :href="`#${child.id}`"
              class="material-toc-child"
              :class="{ 'is-active': activeId === child.id }"
              @click.prevent="scrollToHeading(child.id)"
            >{{ child.text }}</a>
          </template>
        </nav>
      </aside>
    </div>

    <div v-else class="material-reader-empty">资料不存在或暂时无法加载。</div>
  </div>
</template>

<script setup lang="ts">
import { getLearningCourse } from '~/data/learning/courses'
import { getLearningMaterialArticle } from '~/data/learning/materials'
import MaterialImage from '~/components/MaterialImage.vue'
import ProsePre from '~/components/content/ProsePre.vue'
import ProseTable from '~/components/content/ProseTable.vue'
import ProseH2 from '~/components/content/ProseH2.vue'
import ProseH3 from '~/components/content/ProseH3.vue'
import ProseH4 from '~/components/content/ProseH4.vue'

const route = useRoute()
const appConfig = useAppConfig()
const course = computed(() => {
  const registered = getLearningCourse(String(route.params.course))
  if (!registered) throw createError({ statusCode: 404, statusMessage: 'Course not found' })
  return registered
})
const materialSlug = computed(() => String(route.params.material))
const article = computed(() => getLearningMaterialArticle(materialSlug.value, course.value.slug))
const pdfOpen = ref(false)
const readerComponents = {
  img: MaterialImage,
  pre: ProsePre,
  table: ProseTable,
  h1: 'h1',
  h2: ProseH2,
  h3: ProseH3,
  h4: ProseH4,
  h5: 'h5',
  h6: 'h6',
}

definePageMeta({
  scrollToTop: true,
  key: route => route.fullPath.split('#')[0],
  validate: route => Boolean(getLearningCourse(String(route.params.course)) && getLearningMaterialArticle(String(route.params.material), String(route.params.course))),
})

const { data: material } = await useAsyncData(
  `learning-material-${course.value.slug}-${materialSlug.value}`,
  () => article.value
    ? $fetch(`/api/learning/${encodeURIComponent(course.value.slug)}/material/${encodeURIComponent(article.value.slug)}`, { responseType: 'json' })
    : Promise.resolve(null),
)
if (!material.value) throw createError({ statusCode: 404, statusMessage: 'Learning material not found' })
const tocLinks = computed(() => material.value?.toc?.links ?? [])

const activeId = ref('')
const articleRef = ref<HTMLElement | null>(null)
const tocRef = ref<HTMLElement | null>(null)
let headingObserver: IntersectionObserver | null = null
let navResizeObserver: ResizeObserver | null = null

function headingOffset() {
  return (document.querySelector('.site-nav')?.getBoundingClientRect().height ?? 0) + 18
}

function observeHeadings() {
  headingObserver?.disconnect()
  const headings = articleRef.value?.querySelectorAll('h2, h3')
  if (!headings?.length) return

  headingObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting && entry.target.id) {
          activeId.value = entry.target.id
        }
      }
    },
    { rootMargin: `-${headingOffset()}px 0px -62% 0px`, threshold: 0 },
  )
  headings.forEach(heading => headingObserver?.observe(heading))
}

function scrollToHeading(id: string) {
  const el = document.getElementById(id)
  if (!el) return
  const top = el.getBoundingClientRect().top + window.scrollY - headingOffset()
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  window.scrollTo({ top, behavior: reducedMotion ? 'auto' : 'smooth' })
  activeId.value = id
}

watch(activeId, (id) => {
  if (!id || !tocRef.value) return
  nextTick(() => {
    const activeLink = tocRef.value?.querySelector<HTMLElement>('.is-active')
    activeLink?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  })
})

watch(
  [() => material.value?.slug, articleRef],
  async () => {
    await nextTick()
    activeId.value = ''
    observeHeadings()
  },
  { flush: 'post' },
)

onMounted(() => {
  observeHeadings()
  const nav = document.querySelector('.site-nav')
  if (nav) {
    navResizeObserver = new ResizeObserver(observeHeadings)
    navResizeObserver.observe(nav)
  }
})

onBeforeUnmount(() => {
  headingObserver?.disconnect()
  navResizeObserver?.disconnect()
})

useHead(() => ({
  title: `${material.value?.title || '学习资料'} · ${course.value.title}`,
  meta: [{ name: 'description', content: material.value?.description || course.value.description }],
  link: [{ rel: 'canonical', href: `${appConfig.url}/learning/${encodeURIComponent(course.value.slug)}/materials/${encodeURIComponent(materialSlug.value)}` }],
}))
</script>

<style scoped>
.material-reader { max-width: 1180px; }
.material-reader-head :where(h1, p), .material-toc p { margin: 0; }
.material-kicker { color: var(--muted); font: 0.68rem/1.7 var(--font-mono); }
.material-reader-head { padding: 6px 4px 28px; border-bottom: 1px solid var(--line); }
.material-back { display: inline-flex; padding-block: 8px; margin-bottom: 18px; color: var(--muted); font-size: 0.74rem; text-decoration: none; }
.material-back:hover { color: var(--accent); }
.material-reader-head h1 { max-width: 900px; margin-top: 10px; color: var(--ink); font-size: 2.15rem; font-weight: 500; line-height: 1.45; }
.material-reader-description { max-width: 720px; margin-top: 10px !important; color: var(--muted); font-size: 0.84rem; line-height: 1.8; }
.material-reader-actions { display: flex; flex-wrap: wrap; gap: 10px 24px; margin-top: 16px; }
.material-reader-actions a { color: var(--accent); font-size: 0.78rem; text-decoration: none; }
.material-reader-layout { display: grid; grid-template-columns: minmax(0, 820px) minmax(180px, 240px); gap: 50px; align-items: start; padding-top: 30px; }
.material-reader-content { min-width: 0; }
.material-reader-content :deep(h1:first-child) { display: none; }
.material-reader-content :deep(:is(h2, h3, h4)) { scroll-margin-top: calc(var(--nav-h) + 24px); }
.material-reader-content :deep(.katex-display) { overflow-x: auto; overflow-y: hidden; padding-bottom: 6px; }
.material-reader-content :deep(pre) { overflow-x: auto; }
.material-toc { position: sticky; top: calc(var(--nav-h) + 22px); max-height: calc(100vh - var(--nav-h) - 44px); overflow-y: auto; padding-right: 8px; scrollbar-width: thin; }
.material-toc nav { display: flex; flex-direction: column; gap: 6px; margin-top: 16px; border-left: 1px solid var(--line); padding-left: 12px; }
.material-toc a, .material-mobile-toc a {
  position: relative;
  color: var(--muted);
  font-size: 0.75rem;
  line-height: 1.65;
  text-decoration: none;
  padding: 2px 0;
  transition: color 160ms ease, transform 160ms ease;
}
.material-toc a:hover, .material-mobile-toc a:hover { color: var(--accent); }
.material-toc a.is-active,
.material-mobile-toc a[aria-current="location"] {
  color: var(--accent);
  font-weight: 500;
}
.material-toc a.is-active::before {
  content: '';
  position: absolute;
  left: -13px;
  top: 4px;
  bottom: 4px;
  width: 2px;
  border-radius: 2px;
  background: var(--accent);
}
.material-toc-child { padding-left: 12px !important; font-size: 0.72rem !important; }
.material-mobile-toc { display: none; margin-top: 24px; padding: 14px 18px; border: 1px solid var(--line); border-radius: 10px; }
.material-mobile-toc summary { color: var(--accent); cursor: pointer; font-size: 0.82rem; }
.material-mobile-toc nav { display: flex; flex-direction: column; gap: 10px; margin-top: 14px; max-height: 50vh; overflow-y: auto; }
.material-source { margin-top: 48px; padding: 20px; border: 1px solid var(--line); border-radius: 14px; background: var(--topic-orange); scroll-margin-top: calc(var(--nav-h) + 20px); }
.material-source summary { cursor: pointer; color: var(--ink); font-size: 1rem; }
.material-source > p { margin-top: 12px; color: var(--muted); font-size: 0.78rem; line-height: 1.75; }
.material-source-pdf { display: block; width: 100%; height: 75vh; margin-block: 16px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
.material-source-pdf p { padding: 16px; color: var(--text); font-size: 0.8rem; line-height: 1.7; }
.material-source-pdf a { color: var(--accent); }
.material-reader-empty { padding: 60px 4px; color: var(--muted); }
@media (max-width: 1000px) { .material-reader { max-width: 860px; } .material-reader-layout { grid-template-columns: minmax(0, 1fr); } .material-toc { display: none; } .material-mobile-toc { display: block; } }
@media (max-width: 640px) { .material-reader-head h1 { font-size: 1.75rem; } .material-reader-layout { padding-top: 22px; } .material-source { padding: 16px; } .material-source-pdf { height: 500px; } }
</style>
