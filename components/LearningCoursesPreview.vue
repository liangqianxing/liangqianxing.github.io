<template>
  <section class="learning-preview" aria-labelledby="learning-preview-title">
    <div>
      <p class="learning-preview-kicker">LEARNING / 课程学习</p>
      <h2 id="learning-preview-title">课程学习</h2>
      <p class="learning-preview-description">{{ learningCourses.length }} 门课程 · 课程资料、实验与学习记录</p>
    </div>
    <NuxtLink to="/learning">查看课程 <span aria-hidden="true">→</span></NuxtLink>
  </section>
</template>

<script setup lang="ts">
import { learningCourses } from '~/data/learning/courses'
</script>

<style scoped>
.learning-preview { display: flex; align-items: center; justify-content: space-between; gap: 28px; padding: 22px 28px; margin-bottom: 38px; border: 1px solid var(--line); border-radius: 16px; background: var(--topic-green); }
.learning-preview :where(h2, p) { margin: 0; }
.learning-preview-kicker { color: var(--muted); font: 0.66rem var(--font-mono); letter-spacing: 0; }
.learning-preview h2 { margin-top: 7px; font-size: 1.2rem; font-weight: 500; color: var(--ink); }
.learning-preview-description { margin-top: 8px; color: var(--text); font-size: 0.82rem; line-height: 1.8; }
.learning-preview a { flex-shrink: 0; display: inline-flex; align-items: center; gap: 12px; min-height: 40px; color: var(--accent); font-size: 0.82rem; }
@media (max-width: 640px) { .learning-preview { align-items: start; flex-direction: column; gap: 12px; padding: 20px; margin-bottom: 30px; } }
</style>
