<template>
  <section class="learning-preview" aria-labelledby="learning-preview-title">
    <div class="learning-preview-head">
      <div>
        <p class="learning-preview-kicker">LEARNING / 课程学习与系统实践</p>
        <h2 id="learning-preview-title">把经典课程一行行写成代码</h2>
        <p class="learning-preview-description">{{ learningCourses.length }} 门硬核公开课 · 包含双语精读译稿、从零实现代码与 GPU 实验记录</p>
      </div>
      <NuxtLink to="/learning" class="learning-preview-all">查看全部课程 <span aria-hidden="true">→</span></NuxtLink>
    </div>
    <div class="learning-preview-grid">
      <NuxtLink
        v-for="course in learningCourses"
        :key="course.slug"
        :to="`/learning/${encodeURIComponent(course.slug)}`"
        class="learning-preview-card"
      >
        <div class="learning-preview-card-top">
          <span class="learning-preview-badge">{{ course.institution }} · {{ course.code }}</span>
          <span class="learning-preview-arrow" aria-hidden="true">↗</span>
        </div>
        <h3>{{ course.title }}</h3>
        <p class="learning-preview-sub" lang="en">{{ course.courseTitle }}</p>
        <div class="learning-preview-tags">
          <span v-for="topic in course.topics.slice(0, 3)" :key="topic">{{ topic }}</span>
        </div>
      </NuxtLink>
    </div>
  </section>
</template>

<script setup lang="ts">
import { learningCourses } from '~/data/learning/courses'
</script>

<style scoped>
.learning-preview {
  position: relative;
  overflow: hidden;
  padding: 24px 26px;
  margin-bottom: 48px;
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  background: var(--surface);
  box-shadow: var(--shadow-soft);
}
.learning-preview :where(h2, h3, p) { margin: 0; }
.learning-preview-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 22px;
}
.learning-preview-kicker {
  color: var(--accent);
  font: 600 0.68rem/1.5 var(--font-mono);
  letter-spacing: 0.06em;
}
.learning-preview h2 {
  margin-top: 6px;
  font-size: 1.28rem;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.02em;
}
.learning-preview-description {
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.75;
}
.learning-preview-all {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 36px;
  padding: 0 14px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 12%, var(--surface-2));
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 500;
  transition: transform 180ms ease, background-color 180ms ease;
}
.learning-preview-all:hover {
  transform: translateY(-1px);
  background: color-mix(in srgb, var(--accent) 18%, var(--surface-2));
}
.learning-preview-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.learning-preview-card {
  position: relative;
  display: flex;
  flex-direction: column;
  padding: 18px 20px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--bg-2);
  transition:
    transform 200ms cubic-bezier(0.22, 1, 0.36, 1),
    border-color 180ms ease,
    box-shadow 200ms ease;
}
.learning-preview-card:hover {
  transform: translateY(-2px);
  border-color: color-mix(in srgb, var(--accent) 35%, var(--line));
  box-shadow: 0 14px 30px -22px rgba(15, 23, 42, 0.35);
}
.learning-preview-card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
.learning-preview-badge {
  padding: 3px 9px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface);
  color: var(--muted);
  font: 500 0.66rem/1.4 var(--font-mono);
}
.learning-preview-arrow {
  color: var(--muted);
  font-size: 0.95rem;
  transition: transform 200ms cubic-bezier(0.22, 1, 0.36, 1), color 200ms ease;
}
.learning-preview-card:hover .learning-preview-arrow {
  color: var(--accent);
  transform: translate(3px, -3px);
}
.learning-preview-card h3 {
  color: var(--ink);
  font-size: 1.02rem;
  font-weight: 600;
  line-height: 1.4;
  letter-spacing: -0.015em;
  transition: color 180ms ease;
}
.learning-preview-card:hover h3 {
  color: var(--accent);
}
.learning-preview-sub {
  margin-top: 4px;
  color: var(--muted);
  font-size: 0.75rem;
  line-height: 1.5;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.learning-preview-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 14px;
}
.learning-preview-tags span {
  padding: 2px 8px;
  border-radius: 6px;
  background: var(--bg-2);
  color: var(--muted);
  font-size: 0.68rem;
}
@media (max-width: 720px) {
  .learning-preview { padding: 22px 20px; margin-bottom: 36px; }
  .learning-preview-head { flex-direction: column; align-items: flex-start; gap: 14px; }
  .learning-preview-grid { grid-template-columns: 1fr; }
}
</style>
