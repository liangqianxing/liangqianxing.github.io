<template>
  <div class="page-frame learning-notebook">
    <header class="learning-welcome">
      <div>
        <NuxtLink to="/blog" class="learning-back">← 回到博客</NuxtLink>
        <p class="learning-kicker">LEARNING NOTEBOOK / 学习专题</p>
        <h1>CMU 15-779 <span>学习专题</span></h1>
        <p class="learning-course-title" lang="en">{{ cmuLearning.courseTitle }}</p>
        <p class="learning-intro">{{ cmuLearning.description }}</p>
        <a :href="cmuLearning.courseUrl" class="learning-course-link" target="_blank" rel="noopener noreferrer">课程官网 <span aria-hidden="true">↗</span></a>
      </div>
      <div class="learning-cover" aria-hidden="true">
        <span>CARNEGIE MELLON</span>
        <strong>15<span>—</span>779</strong>
        <p>materials / learning notes</p>
      </div>
    </header>

    <nav class="learning-section-nav" aria-label="学习专题目录">
      <a href="#materials">学习资料 <span>{{ materialsError ? '—' : materials.length }}</span></a>
      <a href="#notes">学习记录 <span>{{ notesError ? '—' : notes.length }}</span></a>
    </nav>

    <div class="learning-library-grid">
      <section id="materials" class="learning-section learning-materials" aria-labelledby="materials-title">
        <header class="learning-section-head"><div><p class="learning-kicker">01 / MATERIALS</p><h2 id="materials-title">学习资料</h2></div><span>{{ materialsError ? '—' : materials.length }} 份</span></header>
        <div v-if="materialsError" class="learning-empty" role="alert"><p>资料暂时没能加载。</p><button type="button" @click="refreshMaterials()">重新加载</button></div>
        <ul v-else-if="materials.length" class="learning-file-list">
          <li v-for="file in materials" :key="file.path"><a :href="file.path" download><span>{{ file.name }}</span><small>{{ file.extension.toUpperCase() || 'FILE' }} · {{ formatBytes(file.bytes) }} ↓</small></a></li>
        </ul>
        <div v-else class="learning-empty"><span aria-hidden="true">＋</span><p>还没有学习资料</p><small>讲义、代码和实验附件，都可以收在这里。</small></div>
        <a :href="materialUploadUrl" class="learning-button" target="_blank" rel="noopener noreferrer">上传学习资料 <span>GitHub ↗</span></a>
      </section>

      <section id="notes" class="learning-section learning-notes" aria-labelledby="notes-title">
        <header class="learning-section-head"><div><p class="learning-kicker">02 / LEARNING LOG</p><h2 id="notes-title">学习记录</h2></div><span>{{ notesError ? '—' : notes.length }} 篇</span></header>
        <div v-if="notesError" class="learning-empty" role="alert"><p>学习记录暂时没能加载。</p><button type="button" @click="refreshNotes()">重新加载</button></div>
        <ol v-else-if="notes.length" class="learning-note-list">
          <li v-for="note in notes" :key="note.path"><NuxtLink :to="note.path"><span class="learning-note-number">{{ String(note.seriesOrder ?? 0).padStart(2, '0') }}</span><div><h3>{{ note.title }}</h3><p>{{ note.description || note.excerpt }}</p><time :datetime="formatDate(note.date)">{{ formatDate(note.date) }}</time></div><span aria-hidden="true">↗</span></NuxtLink></li>
        </ol>
        <div v-else class="learning-empty"><span aria-hidden="true">＋</span><p>还没有学习记录</p><small>从第一篇笔记开始，按学习顺序慢慢积累。</small></div>
        <a :href="noteUploadUrl" class="learning-button" target="_blank" rel="noopener noreferrer">上传学习记录 <span>GitHub ↗</span></a>
      </section>
    </div>

    <footer class="learning-update">
      <div class="learning-update-actions">
        <a href="/materials/cmu-15-779/templates/learning-note.md" download="cmu-15-779-learning-note.md">下载记录模板 ↓</a>
        <a :href="maintenanceUrl" target="_blank" rel="noopener noreferrer">更新方法 ↗</a>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { cmuLearning } from '~/data/learning/cmu-15-779'
import { formatDate } from '~/utils/blog'
import type { LearningMaterial } from '~/server/api/learning/cmu-15-779/materials.get'

const appConfig = useAppConfig()
const [{ data: allPosts, error: notesError, refresh: refreshNotes }, { data: uploadedFiles, error: materialsError, refresh: refreshMaterials }] = await Promise.all([
  usePublicArchive(),
  useAsyncData<LearningMaterial[]>('cmu-15-779-materials', () => $fetch('/api/learning/cmu-15-779/materials', { responseType: 'json' })),
])
const notes = computed(() => (allPosts.value ?? []).filter(post => post.series === cmuLearning.series).sort((a, b) => (a.seriesOrder ?? 999) - (b.seriesOrder ?? 999)))
const materials = computed(() => uploadedFiles.value ?? [])
const repoUrl = 'https://github.com/liangqianxing/liangqianxing.github.io'
const materialUploadUrl = `${repoUrl}/upload/main/public/materials/cmu-15-779`
const noteUploadUrl = `${repoUrl}/upload/main/content/posts`
const maintenanceUrl = `${repoUrl}/blob/main/docs/cmu-15-779-maintenance.md`
function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}
useHead({
  title: cmuLearning.title,
  meta: [{ name: 'description', content: cmuLearning.description }, { property: 'og:title', content: cmuLearning.title }, { property: 'og:description', content: cmuLearning.description }],
  link: [{ rel: 'canonical', href: `${appConfig.url}/learning/cmu-15-779` }],
})
</script>

<style scoped>
.learning-notebook :where(h1, h2, h3, p) { margin: 0; }
.learning-notebook a { text-decoration: none; }
.learning-welcome { display: grid; grid-template-columns: minmax(0, 1fr) 210px; gap: 46px; align-items: center; padding: 6px 4px 30px; }
.learning-back { display: inline-block; color: var(--muted); font-size: 0.74rem; padding-block: 8px; margin-bottom: 18px; }
.learning-kicker { color: var(--muted); font: 0.65rem/1.6 var(--font-mono); letter-spacing: 0; }
.learning-welcome h1 { margin-top: 12px; color: var(--ink); font-size: 2.65rem; font-weight: 500; letter-spacing: 0; line-height: 1.4; }
.learning-welcome h1 span { color: var(--accent); }
.learning-course-title { margin-top: 8px !important; color: var(--muted); font-size: 0.74rem; line-height: 1.6; }
.learning-intro { margin-top: 16px !important; color: var(--text); font-size: 0.88rem; line-height: 1.85; }
.learning-course-link { display: inline-flex; align-items: center; gap: 12px; min-height: 40px; margin-top: 12px; color: var(--accent); font-size: 0.8rem; }
.learning-cover { display: flex; flex-direction: column; gap: 16px; align-items: start; padding: 24px; border: 1px solid var(--line); border-radius: 6px 18px 18px 6px; background: var(--topic-purple); transform: rotate(3deg); color: var(--ink); box-shadow: 4px 6px 0 var(--topic-orange); }
.learning-cover > span { font: 0.57rem/1.6 var(--font-mono); letter-spacing: 0; color: var(--muted); }
.learning-cover strong { font: 500 2.8rem/1.05 var(--font-mono); letter-spacing: 0; }
.learning-cover strong span { display: block; font-size: 1.6rem; color: var(--accent); }
.learning-cover p { font: 0.65rem/1.7 var(--font-mono); }
.learning-section-nav { display: flex; flex-wrap: wrap; gap: 10px 26px; padding: 14px 4px; margin-bottom: 26px; border-block: 1px solid var(--line); }
.learning-section-nav a { display: inline-flex; align-items: center; gap: 8px; color: var(--text); font-size: 0.8rem; min-height: 28px; }
.learning-section-nav span { color: var(--muted); font: 0.65rem var(--font-mono); }
.learning-library-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; align-items: start; }
.learning-section { padding: 24px; border-radius: 16px; min-width: 0; scroll-margin-top: calc(var(--nav-h) + 20px); }
.learning-materials { background: var(--topic-orange); }
.learning-notes { background: var(--topic-blue); }
.learning-section-head { display: flex; align-items: end; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; }
.learning-section-head h2 { margin-top: 5px; font-size: 1.25rem; font-weight: 500; color: var(--ink); }
.learning-section-head > span { color: var(--muted); font-size: 0.75rem; }
.learning-empty { padding: 22px 0; color: var(--text); font-size: 0.85rem; line-height: 1.8; }
.learning-empty > span { display: block; color: var(--accent); font-size: 1.5rem; margin-bottom: 8px; }
.learning-empty small { display: block; color: var(--muted); font-size: 0.75rem; margin-top: 5px; }
.learning-empty button { margin-top: 10px; border: 0; background: none; color: var(--accent); cursor: pointer; }
.learning-button { display: flex; justify-content: space-between; gap: 12px; padding: 11px 14px; margin-top: 12px; color: var(--ink); border: 1px solid var(--line-strong); border-radius: 9px; font-size: 0.77rem; }
.learning-button span { color: var(--muted); font-size: 0.65rem; }
.learning-file-list, .learning-note-list { list-style: none; padding: 0; margin: 12px 0; }
.learning-file-list li + li, .learning-note-list li + li { border-top: 1px dashed var(--line-strong); }
.learning-file-list a { display: flex; flex-direction: column; gap: 4px; color: var(--ink); padding-block: 12px; font-size: 0.82rem; overflow-wrap: anywhere; }
.learning-file-list small { color: var(--muted); font-size: 0.65rem; }
.learning-note-list a { display: grid; grid-template-columns: 24px minmax(0, 1fr) 16px; gap: 12px; align-items: baseline; padding-block: 14px; color: var(--ink); }
.learning-note-number, .learning-note-list time { font: 0.67rem var(--font-mono); color: var(--muted); }
.learning-note-list h3 { font-size: 0.9rem; font-weight: 500; line-height: 1.7; overflow-wrap: anywhere; }
.learning-note-list p { font-size: 0.75rem; color: var(--muted); margin-top: 4px !important; line-height: 1.7; }
.learning-note-list time { display: block; margin-top: 8px; }
.learning-note-list a > span:last-child { color: var(--accent); }
.learning-update { margin-top: 24px; padding: 14px 4px; }
.learning-update-actions { display: flex; flex-wrap: wrap; gap: 12px 24px; margin-top: 6px; }
.learning-update-actions a { color: var(--accent); font-size: 0.75rem; padding-block: 7px; }
.learning-notebook a:hover { color: var(--accent); }
@media (max-width: 820px) { .learning-welcome { grid-template-columns: minmax(0, 1fr) 170px; gap: 26px; } .learning-cover { padding: 20px; } .learning-cover strong { font-size: 2.3rem; } }
@media (max-width: 640px) { .learning-welcome { grid-template-columns: minmax(0, 1fr); padding: 0 0 24px; } .learning-cover { display: none; } .learning-welcome h1 { font-size: 2rem; } .learning-library-grid { grid-template-columns: minmax(0, 1fr); gap: 16px; } .learning-section { padding: 22px; } }
</style>
