<template>
  <div class="page-frame learning-notebook">
    <header class="learning-welcome">
      <div>
        <NuxtLink to="/learning" class="learning-back">← 课程学习</NuxtLink>
        <p class="learning-kicker">LEARNING NOTEBOOK / 学习专题</p>
        <h1>{{ course.title }} <span>学习专题</span></h1>
        <p class="learning-course-title" lang="en">{{ course.courseTitle }}</p>
        <p class="learning-intro">{{ course.description }}</p>
        <div v-if="course.topics.length" class="learning-topics"><span v-for="topic in course.topics" :key="topic">{{ topic }}</span></div>
        <div class="learning-course-links">
          <a :href="course.courseUrl" class="learning-course-link" target="_blank" rel="noopener noreferrer">课程官网 <span aria-hidden="true">↗</span></a>
          <a v-if="course.codeUrl" :href="course.codeUrl" class="learning-course-link" target="_blank" rel="noopener noreferrer">学习代码 <span aria-hidden="true">↗</span></a>
        </div>
      </div>
      <div class="learning-cover" aria-hidden="true">
        <span>{{ course.institution }}</span>
        <strong>{{ course.code }}</strong>
        <p>materials / learning notes</p>
      </div>
    </header>

    <nav class="learning-section-nav" aria-label="学习专题目录">
      <a href="#notes"><span class="learning-nav-index">01</span> 学习记录 <span class="learning-nav-count">{{ notesError ? '—' : notes.length }}</span><span aria-hidden="true">↓</span></a>
      <a href="#materials"><span class="learning-nav-index">02</span> 学习资料 <span class="learning-nav-count">{{ materialsError ? '—' : materials.length }}</span><span aria-hidden="true">↓</span></a>
    </nav>

    <div class="learning-library-grid">
      <section id="notes" class="learning-section learning-notes" aria-labelledby="notes-title">
        <header class="learning-section-head">
          <div><p class="learning-kicker">01 / LEARNING LOG</p><h2 id="notes-title">学习记录 <span>{{ notesError ? '—' : notes.length }}</span></h2></div>
          <p class="learning-section-description">按学习顺序，回看每一步练习。</p>
        </header>
        <div v-if="notesError" class="learning-empty" role="alert"><p>学习记录暂时没能加载。</p><button type="button" @click="refreshNotes()">重新加载</button></div>
        <ol v-else-if="notes.length" class="learning-note-list">
          <li v-for="note in notes" :key="note.path">
            <article class="learning-note-card">
              <NuxtLink :to="note.path" class="learning-note-main" :aria-label="note.title">
                <div class="learning-note-cover" :class="{ 'learning-note-cover-empty': !note.cover }">
                  <SiteImage v-if="note.cover" :src="note.cover" :alt="note.coverAlt || ''" sizes="(max-width: 600px) calc(100vw - 48px), (max-width: 900px) calc((100vw - 70px) / 2), 320px" />
                  <span v-else class="learning-note-cover-number" aria-hidden="true">{{ String(note.seriesOrder ?? 0).padStart(2, '0') }}</span>
                </div>
                <div class="learning-note-copy">
                  <div class="learning-note-meta"><span class="learning-note-number">记录 {{ String(note.seriesOrder ?? 0).padStart(2, '0') }}</span><time :datetime="formatDate(note.date)">{{ formatDate(note.date).replaceAll('-', '.') }}</time></div>
                  <h3>{{ noteTitle(note.title) }}</h3>
                  <p>{{ note.description || note.excerpt }}</p>
                  <div class="learning-note-reading"><span>约 {{ note.readingTime }} 分钟</span><span class="learning-note-read-link">阅读手记 <span aria-hidden="true">↗</span></span></div>
                </div>
              </NuxtLink>
              <div v-if="note.codeUrl || note.attachmentUrl" class="learning-note-resources">
                <a v-if="note.codeUrl" :href="note.codeUrl" target="_blank" rel="noopener noreferrer" :aria-label="`${noteTitle(note.title)}：代码仓库`"><svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m7 6-4 4 4 4m6-8 4 4-4 4m-2-10-2 16" /></svg>代码仓库 <span aria-hidden="true">↗</span></a>
                <a v-if="note.attachmentUrl" :href="note.attachmentUrl" download :aria-label="`${noteTitle(note.title)}：下载 ZIP 附件`"><svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M10 3v9m-3-3 3 3 3-3M4 13v4h12v-4" /></svg>ZIP 附件</a>
              </div>
            </article>
          </li>
        </ol>
        <div v-else class="learning-empty"><p>还没有学习记录</p><small>从第一篇笔记开始，按学习顺序慢慢积累。</small></div>
      </section>

      <section id="materials" class="learning-section learning-materials" aria-labelledby="materials-title">
        <header class="learning-section-head">
          <div><p class="learning-kicker">02 / READING SHELF</p><h2 id="materials-title">学习资料 <span>{{ materialsError ? '—' : materials.length }}</span></h2></div>
          <p class="learning-section-description">论文、译稿与实验附件，按需查阅。</p>
        </header>
        <div v-if="materialsError" class="learning-empty" role="alert"><p>资料暂时没能加载。</p><button type="button" @click="refreshMaterials()">重新加载</button></div>
        <div v-else-if="materials.length" class="learning-material-groups">
          <section v-for="group in materialGroups" :key="group.id" class="learning-material-group" :data-kind="group.id" :aria-labelledby="`materials-${group.id}-title`">
            <header class="learning-material-group-head"><span class="learning-material-mark" aria-hidden="true">{{ materialGroupMarks[group.id] }}</span><h3 :id="`materials-${group.id}-title`">{{ group.title }}</h3><span>{{ group.files.length }} 份</span></header>
            <ul class="learning-file-list">
              <li v-for="file in group.files" :key="file.path">
                <div class="learning-file-row">
                  <NuxtLink v-if="materialArticle(file)" :to="materialReaderPath(file)" class="learning-file-main">
                    <span>{{ materialArticle(file)?.title || file.name }}</span>
                    <small>{{ file.extension.toUpperCase() }} · {{ formatBytes(file.bytes) }} · 在线阅读 ↗</small>
                  </NuxtLink>
                  <a v-else :href="file.path" :download="file.extension === 'pdf' ? undefined : file.name" :target="file.extension === 'pdf' ? '_blank' : undefined" rel="noopener noreferrer" class="learning-file-main">
                    <span>{{ sourceTitle(file) }}</span>
                    <small>{{ file.extension.toUpperCase() || 'FILE' }} · {{ formatBytes(file.bytes) }}<template v-if="file.extension === 'pdf'"> · 查看原文 ↗</template></small>
                  </a>
                  <a :href="file.path" :download="file.name" class="learning-file-download" :aria-label="`下载 ${file.name}`"><svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M10 3v9m-3-3 3 3 3-3M4 13v4h12v-4" /></svg></a>
                </div>
              </li>
            </ul>
          </section>
        </div>
        <div v-else class="learning-empty"><p>还没有学习资料</p><small>讲义、代码和实验附件，都可以收在这里。</small></div>
      </section>
    </div>

    <footer class="learning-update">
      <div class="learning-update-actions">
        <a href="/materials/templates/learning-note.md" download="learning-note.md">下载记录模板 ↓</a>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { getLearningCourse } from '~/data/learning/courses'
import { getLearningMaterialArticleByFileName, getLearningMaterialCategory, learningMaterialArticles, learningMaterialCategories } from '~/data/learning/materials'
import { formatDate } from '~/utils/blog'
import type { LearningMaterial } from '~/server/utils/learning-materials'

definePageMeta({ validate: route => Boolean(getLearningCourse(String(route.params.course))) })
const appConfig = useAppConfig()
const route = useRoute()
const course = computed(() => {
  const registered = getLearningCourse(String(route.params.course))
  if (!registered) throw createError({ statusCode: 404, statusMessage: 'Course not found' })
  return registered
})
const materialKey = computed(() => `learning-materials-${course.value.slug}`)
const [{ data: allPosts, error: notesError, refresh: refreshNotes }, { data: uploadedFiles, error: materialsError, refresh: refreshMaterials }] = await Promise.all([
  usePublicArchive(),
  useAsyncData<LearningMaterial[]>(materialKey, () => $fetch(`/api/learning/${encodeURIComponent(course.value.slug)}/materials`, { responseType: 'json' })),
])
const notes = computed(() => (allPosts.value ?? []).filter(post => post.series === course.value.series).sort((a, b) => (a.seriesOrder ?? 999) - (b.seriesOrder ?? 999)))
const materialGroupMarks = { zh: '中', en: 'EN', attachments: 'ZIP', other: 'DOC' }
function noteTitle(title: string) {
  const prefix = `${course.value.series} 学习记录`
  return title.startsWith(prefix) ? title.slice(prefix.length).replace(/^\s*\d+\s*[：:]\s*/, '') : title
}
const courseArticles = computed(() => learningMaterialArticles.filter(article => article.courseSlug === course.value.slug))
const materials = computed(() => [...(uploadedFiles.value ?? [])].sort((a, b) => {
  const aOrder = courseArticles.value.findIndex(article => article.fileName === a.name || article.sourcePdf === a.path)
  const bOrder = courseArticles.value.findIndex(article => article.fileName === b.name || article.sourcePdf === b.path)
  const articleOrder = (aOrder < 0 ? 999 : aOrder) - (bOrder < 0 ? 999 : bOrder)
  if (articleOrder) return articleOrder
  const aNoteOrder = notes.value.find(note => note.attachmentUrl === a.path)?.seriesOrder ?? 999
  const bNoteOrder = notes.value.find(note => note.attachmentUrl === b.path)?.seriesOrder ?? 999
  return aNoteOrder - bNoteOrder || a.path.localeCompare(b.path)
}))
const materialGroups = computed(() => learningMaterialCategories.map(category => ({
  ...category,
  files: materials.value.filter(file => getLearningMaterialCategory(file, course.value.slug) === category.id),
})).filter(group => group.files.length))
function sourceTitle(file: LearningMaterial) {
  const note = notes.value.find(note => note.attachmentUrl === file.path)
  if (note) return `记录 ${String(note.seriesOrder ?? 0).padStart(2, '0')} · 实验代码`
  const source = courseArticles.value.find(article => article.sourcePdf === file.path)
  return source ? `${source.title} · 原文 PDF` : file.name
}
function materialArticle(file: LearningMaterial) {
  return getLearningMaterialArticleByFileName(file.name, course.value.slug)
}
function materialReaderPath(file: LearningMaterial) {
  const article = materialArticle(file)
  return `/learning/${encodeURIComponent(course.value.slug)}/materials/${encodeURIComponent(article?.slug ?? file.name)}`
}
function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}
onNuxtReady(() => {
  const hash = window.location.hash
  if (hash === '#notes' || hash === '#materials') {
    document.getElementById(hash.slice(1))?.scrollIntoView({ block: 'start', behavior: 'instant' })
  }
})
useHead(() => ({
  title: `${course.value.title} 学习专题`,
  meta: [{ name: 'description', content: course.value.description }, { property: 'og:title', content: `${course.value.title} 学习专题` }, { property: 'og:description', content: course.value.description }],
  link: [{ rel: 'canonical', href: `${appConfig.url}/learning/${encodeURIComponent(course.value.slug)}` }],
}))
</script>

<style scoped>
.learning-notebook :where(h1, h2, h3, p) { margin: 0; }
.learning-notebook a { text-decoration: none; }
.learning-welcome { display: grid; grid-template-columns: minmax(0, 1fr) 190px; gap: 48px; align-items: center; padding: 32px 36px; border: 1px solid var(--line); border-radius: var(--radius-lg); background: var(--surface); box-shadow: var(--shadow-soft); }
.learning-back { display: inline-block; color: var(--muted); font-size: 0.74rem; padding-block: 8px; margin-bottom: 16px; }
.learning-kicker { color: var(--muted); font: 0.63rem/1.6 var(--font-mono); letter-spacing: 0.03em; }
.learning-welcome h1 { margin-top: 12px; color: var(--ink); font-size: clamp(1.9rem, 3.2vw, 2.35rem); font-weight: 600; line-height: 1.3; letter-spacing: -0.03em; }
.learning-welcome h1 span { color: var(--accent); }
.learning-course-title { max-width: 490px; margin-top: 10px !important; color: var(--muted); font-size: 0.75rem; line-height: 1.8; }
.learning-intro { margin-top: 18px !important; color: var(--text); font-size: 0.83rem; line-height: 1.85; }
.learning-topics { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.learning-topics span { padding: 4px 10px; border: 1px solid var(--line); border-radius: 6px; color: var(--accent); font-size: 0.65rem; background: var(--surface); }
.learning-course-link { display: inline-flex; align-items: center; gap: 12px; min-height: 44px; margin-top: 12px; color: var(--accent); font-size: 0.76rem; }
.learning-course-links { display: flex; flex-wrap: wrap; gap: 0 24px; }
.learning-cover { display: flex; flex-direction: column; gap: 16px; align-items: start; padding: 22px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--bg-2); color: var(--ink); box-shadow: none; }
.learning-cover > span { font: 0.65rem/1.6 var(--font-mono); color: var(--muted); }
.learning-cover strong { font: 600 1.55rem/1.3 var(--font-mono); color: var(--accent); overflow-wrap: anywhere; }
.learning-cover p { width: 100%; padding-top: 14px; border-top: 1px solid var(--line); color: var(--faint); font: 0.61rem/1.8 var(--font-mono); }
.learning-section-nav { display: flex; flex-wrap: wrap; gap: 12px; margin: 22px 0 46px; }
.learning-section-nav a { display: inline-flex; align-items: center; gap: 10px; min-height: 46px; padding: 8px 16px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); color: var(--text); font-size: 0.76rem; }
.learning-nav-index { margin-right: 3px; color: var(--accent); font: 0.65rem var(--font-mono); }
.learning-nav-count { padding: 2px 7px; border-radius: 5px; background: var(--bg-2); color: var(--muted); font: 0.62rem/1.6 var(--font-mono); }
.learning-section-nav a > span:last-child { margin-left: 6px; color: var(--accent); }
.learning-library-grid { display: grid; gap: 48px; }
.learning-section { min-width: 0; scroll-margin-top: calc(var(--nav-h) + 30px); }
.learning-section-head { display: flex; align-items: end; justify-content: space-between; flex-wrap: wrap; gap: 12px 24px; margin-bottom: 22px; }
.learning-section-head h2 { display: flex; align-items: center; gap: 12px; margin-top: 6px; font-size: 1.35rem; line-height: 1.4; font-weight: 600; color: var(--ink); letter-spacing: -0.02em; }
.learning-section-head h2 > span { min-width: 26px; padding: 2px 7px; border: 1px solid var(--line); border-radius: 7px; font: 0.68rem/1.6 var(--font-mono); text-align: center; color: var(--muted); }
.learning-section-description { padding-bottom: 4px; color: var(--muted); font-size: 0.75rem; line-height: 1.8; }
.learning-empty { padding: 30px 24px; border: 1px dashed var(--line-strong); border-radius: 16px; color: var(--text); font-size: 0.85rem; line-height: 1.8; }
.learning-empty small { display: block; color: var(--muted); font-size: 0.75rem; margin-top: 5px; }
.learning-empty button { margin-top: 10px; border: 0; background: none; color: var(--accent); cursor: pointer; }
.learning-note-list { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; list-style: none; padding: 0; margin: 0; }
.learning-note-list > li { min-width: 0; }
.learning-note-card { display: flex; flex-direction: column; height: 100%; overflow: hidden; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); box-shadow: var(--shadow-soft); transition: border-color 200ms ease, box-shadow 200ms ease; }
.learning-note-card:hover { border-color: color-mix(in srgb, var(--accent) 32%, var(--line)); box-shadow: 0 16px 36px -26px rgba(15, 23, 42, 0.35); }
.learning-note-main { display: flex; flex-direction: column; flex: 1; min-width: 0; color: var(--ink); }
.learning-note-cover { overflow: hidden; aspect-ratio: 16 / 9; border-bottom: 1px solid var(--line); background: var(--topic-purple); }
.learning-note-cover img { display: block; width: 100%; height: 100%; object-fit: cover; }
.learning-note-cover-empty { display: grid; place-items: center; }
.learning-note-cover-number { color: var(--accent); font: 600 3.2rem/1 var(--font-mono); opacity: 0.45; }
.learning-note-copy { display: flex; flex: 1; flex-direction: column; padding: 20px 22px; }
.learning-note-meta { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 12px; font: 0.62rem/1.7 var(--font-mono); }
.learning-note-number { color: var(--accent); }
.learning-note-meta time { color: var(--muted); }
.learning-note-copy h3 { color: var(--ink); font-size: 1.02rem; font-weight: 500; line-height: 1.7; overflow-wrap: anywhere; text-wrap: pretty; }
.learning-note-copy > p { flex: 1; margin-top: 12px !important; margin-bottom: 22px !important; color: var(--muted); font-size: 0.77rem; line-height: 1.9; overflow-wrap: anywhere; text-wrap: pretty; }
.learning-note-reading { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; font-size: 0.68rem; color: var(--muted); }
.learning-note-read-link { display: inline-flex; gap: 9px; color: var(--accent); }
.learning-note-resources { display: flex; flex-wrap: wrap; margin: 0 18px; padding: 8px 0; border-top: 1px solid var(--line); }
.learning-note-resources a { flex: 1; display: inline-flex; align-items: center; justify-content: center; gap: 6px; min-height: 40px; padding: 4px; color: var(--accent); font-size: 0.68rem; white-space: nowrap; }
.learning-note-resources a + a { border-left: 1px solid var(--line); }
.learning-note-resources svg, .learning-file-download svg { width: 16px; height: 16px; flex: 0 0 auto; stroke: currentColor; stroke-width: 1.35; stroke-linecap: round; stroke-linejoin: round; }
.learning-materials { padding: 26px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--bg-2); }
.learning-material-groups { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; align-items: start; }
.learning-material-group { min-width: 0; padding: 18px; border: 1px solid var(--line); border-radius: 14px; background: var(--surface); }
.learning-material-group-head { display: flex; align-items: center; gap: 10px; padding-bottom: 14px; border-bottom: 1px solid var(--line); }
.learning-material-group-head h3 { flex: 1; color: var(--ink); font-size: 0.86rem; font-weight: 500; }
.learning-material-group-head > span:last-child { color: var(--muted); font-size: 0.65rem; white-space: nowrap; }
.learning-material-mark { display: grid; place-items: center; width: 32px; height: 32px; border-radius: 9px; background: var(--topic-purple); color: var(--accent); font: 0.65rem var(--font-mono); }
.learning-material-group[data-kind="en"] .learning-material-mark { background: var(--topic-green); }
.learning-material-group[data-kind="attachments"] .learning-material-mark { background: var(--topic-orange); }
.learning-file-list { list-style: none; padding: 0; margin: 0; }
.learning-file-list li + li { border-top: 1px solid var(--line); }
.learning-file-row { display: flex; align-items: center; gap: 8px; }
.learning-file-main { display: flex; flex: 1; min-width: 0; flex-direction: column; gap: 6px; padding-block: 16px; color: var(--ink); font-size: 0.76rem; line-height: 1.8; overflow-wrap: anywhere; }
.learning-file-main small { color: var(--muted); font-size: 0.6rem; line-height: 1.7; }
.learning-file-download { display: grid; place-items: center; flex: 0 0 auto; width: 36px; min-height: 44px; border-radius: 9px; color: var(--muted); }
.learning-update { margin-top: 28px; padding: 16px 4px; border-top: 1px solid var(--line); }
.learning-update-actions { display: flex; flex-wrap: wrap; gap: 12px 24px; }
.learning-update-actions a { color: var(--accent); font-size: 0.73rem; padding-block: 7px; }
.learning-notebook a:hover { color: var(--accent); }
@media (hover: hover) and (pointer: fine) {
  .learning-note-card { transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease; }
  .learning-note-card:hover { transform: translateY(-4px); border-color: var(--line-strong); box-shadow: 0 12px 28px #33264008; }
  .learning-note-main:hover h3 { color: var(--accent); }
  .learning-note-resources a:hover, .learning-file-download:hover { border-radius: 8px; background: var(--topic-purple); }
  .learning-section-nav a:hover { border-color: var(--line-strong); background: var(--topic-purple); }
}
@media (max-width: 900px) {
  .learning-welcome { grid-template-columns: minmax(0, 1fr) 160px; gap: 30px; padding: 28px; }
  .learning-cover { padding: 24px 18px; }
  .learning-cover strong { font-size: 1.6rem; }
  .learning-note-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .learning-material-groups { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .learning-materials { padding: 24px; }
}
@media (max-width: 600px) {
  .learning-welcome { grid-template-columns: minmax(0, 1fr); padding: 22px; border-radius: var(--radius); }
  .learning-cover { display: none; }
  .learning-welcome h1 { font-size: 2rem; }
  .learning-welcome h1 span { display: block; }
  .learning-back { margin-bottom: 12px; }
  .learning-section-nav { gap: 10px; margin-bottom: 34px; }
  .learning-section-nav a { flex: 1; justify-content: center; gap: 7px; padding: 8px 10px; font-size: 0.72rem; }
  .learning-section-nav a > span:last-child { margin-left: 0; }
  .learning-section-head { gap: 10px; margin-bottom: 18px; }
  .learning-section-head h2 { font-size: 1.3rem; }
  .learning-section-description { flex-basis: 100%; padding: 0; font-size: 0.72rem; }
  .learning-library-grid { gap: 36px; }
  .learning-note-list { grid-template-columns: minmax(0, 1fr); gap: 20px; }
  .learning-note-copy { padding: 22px; }
  .learning-note-copy h3 { font-size: 1.08rem; }
  .learning-note-copy > p { font-size: 0.82rem; }
  .learning-note-resources a { min-height: 44px; font-size: 0.73rem; }
  .learning-materials { padding: 22px 18px; border-radius: var(--radius); }
  .learning-material-groups { grid-template-columns: minmax(0, 1fr); gap: 14px; }
  .learning-material-group { padding: 16px; }
  .learning-file-main { font-size: 0.8rem; }
  .learning-file-main small { font-size: 0.65rem; }
}
@media (prefers-reduced-motion: reduce) {
  .learning-note-card { transition: none; }
  .learning-note-card:hover { transform: none; }
}
</style>
