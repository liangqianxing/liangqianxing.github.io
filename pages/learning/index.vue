<template>
  <div class="page-frame learning-directory">
    <header class="learning-directory-head">
      <div>
        <p class="learning-directory-kicker">COURSES / 课程学习</p>
        <h1>课程学习</h1>
        <p class="learning-directory-intro">按课程整理资料、实验与学习记录。</p>
      </div>
      <span class="learning-directory-count">{{ learningCourses.length }} 门课程</span>
    </header>

    <ul class="learning-course-list" aria-label="课程列表">
      <li v-for="course in learningCourses" :key="course.slug">
        <NuxtLink :to="`/learning/${encodeURIComponent(course.slug)}`" class="learning-course-row">
          <div class="learning-course-visual">
            <img v-if="course.cover" :src="course.cover.src" :alt="course.cover.alt" width="640" height="400" decoding="async" class="learning-course-image">
            <span v-if="course.cover?.caption" class="learning-course-caption">{{ course.cover.caption }}</span>
            <div v-else class="learning-course-code" aria-hidden="true"><span>{{ course.institution }}</span><strong>{{ course.code }}</strong></div>
          </div>
          <div class="learning-course-copy">
            <h2>{{ course.title }}</h2>
            <p lang="en">{{ course.courseTitle }}</p>
            <div class="learning-course-topics"><span v-for="topic in course.topics" :key="topic">{{ topic }}</span></div>
          </div>
          <div class="learning-course-meta"><span>{{ notesError ? '—' : noteCount(course.series) }} 篇记录</span><span>查看专题 <span aria-hidden="true">→</span></span></div>
        </NuxtLink>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { learningCourses } from '~/data/learning/courses'

const appConfig = useAppConfig()
const { data: posts, error: notesError } = await usePublicArchive()
function noteCount(series: string) {
  return (posts.value ?? []).filter(post => post.series === series).length
}
useHead({
  title: '课程学习',
  meta: [{ name: 'description', content: '按课程整理学习资料、实验与学习记录。' }],
  link: [{ rel: 'canonical', href: `${appConfig.url}/learning` }],
})
</script>

<style scoped>
.learning-directory :where(h1, h2, p) { margin: 0; }
.learning-directory-head { display: flex; align-items: end; justify-content: space-between; gap: 24px; padding: 12px 4px 30px; border-bottom: 1px solid var(--line); }
.learning-directory-kicker { color: var(--muted); font: 0.68rem/1.6 var(--font-mono); letter-spacing: 0; }
.learning-directory h1 { margin-top: 10px; color: var(--ink); font-size: 2.3rem; font-weight: 500; line-height: 1.4; letter-spacing: 0; }
.learning-directory-intro { margin-top: 12px !important; color: var(--text); font-size: 0.86rem; line-height: 1.8; }
.learning-directory-count { color: var(--muted); font-size: 0.76rem; white-space: nowrap; padding-bottom: 4px; }
.learning-course-list { list-style: none; padding: 0; margin: 0; }
.learning-course-list li { border-bottom: 1px solid var(--line); }
.learning-course-row { display: grid; grid-template-columns: 224px minmax(0, 1fr) auto; align-items: center; gap: 26px; padding: 28px 4px; color: var(--ink); text-decoration: none; }
.learning-course-visual { min-width: 0; overflow: hidden; border-radius: 12px; background: var(--topic-purple); }
.learning-course-image { display: block; width: 100%; height: auto; aspect-ratio: 8 / 5; object-fit: contain; background: #fff; }
.learning-course-caption { display: block; padding: 7px 10px; color: var(--muted); font-size: 0.6rem; line-height: 1.6; }
.learning-course-code { display: flex; flex-direction: column; justify-content: center; gap: 8px; min-height: 104px; padding: 16px; border-radius: 12px; background: var(--topic-purple); }
.learning-course-code span { color: var(--muted); font: 0.67rem var(--font-mono); }
.learning-course-code strong { font: 500 1.3rem var(--font-mono); color: var(--accent); overflow-wrap: anywhere; }
.learning-course-copy { min-width: 0; }
.learning-course-copy h2 { color: var(--ink); font-size: 1.2rem; font-weight: 500; line-height: 1.6; }
.learning-course-copy p { margin-top: 5px !important; color: var(--muted); font-size: 0.8rem; line-height: 1.8; overflow-wrap: anywhere; }
.learning-course-topics { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 12px; color: var(--accent); font-size: 0.69rem; }
.learning-course-meta { display: flex; flex-direction: column; align-items: end; gap: 18px; font-size: 0.76rem; white-space: nowrap; }
.learning-course-meta > span:first-child { color: var(--muted); font-size: 0.7rem; }
.learning-course-meta > span:last-child { color: var(--accent); }
.learning-course-row:hover .learning-course-copy h2 { color: var(--accent); }
@media (max-width: 760px) { .learning-course-row { grid-template-columns: 180px minmax(0, 1fr); gap: 14px 20px; } .learning-course-meta { grid-column: 2; flex-direction: row; justify-content: space-between; align-items: center; gap: 12px; } .learning-course-visual { grid-row: 1 / 3; } }
@media (max-width: 560px) { .learning-directory-head { align-items: start; gap: 12px; } .learning-directory h1 { font-size: 1.9rem; } .learning-directory-count { padding-top: 5px; } .learning-course-row { grid-template-columns: minmax(0, 1fr); gap: 16px; padding-block: 24px; } .learning-course-visual { grid-row: auto; } .learning-course-image { aspect-ratio: 16 / 9; } .learning-course-code { flex-direction: row; justify-content: start; align-items: baseline; min-height: 0; gap: 12px; padding: 12px 14px; } .learning-course-meta { grid-column: 1; } }
</style>
