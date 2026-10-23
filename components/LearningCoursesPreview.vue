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
  padding: 28px 30px;
  margin-bottom: 48px;
  border: 1px solid color-mix(in srgb, var(--line-strong) 65%, transparent);
  border-radius: 26px;
  background:
    radial-gradient(ellipse at 95% 0%, color-mix(in srgb, var(--topic-purple) 70%, transparent), transparent 60%),
    radial-gradient(ellipse at 5% 100%, color-mix(in srgb, var(--topic-green) 85%, transparent), transparent 65%),
    var(--surface);
  box-shadow: 0 12px 34px rgba(32, 24, 44, 0.04);
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
  font-size: 1.32rem;
  font-weight: 500;
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
  padding: 20px 22px;
  border: 1px solid color-mix(in srgb, var(--line) 85%, transparent);
  border-radius: 18px;
  background: color-mix(in srgb, var(--surface) 88%, transparent);
  backdrop-filter: blur(8px);
  transition:
    transform 240ms cubic-bezier(0.22, 1, 0.36, 1),
    border-color 220ms ease,
    box-shadow 240ms cubic-bezier(0.22, 1, 0.36, 1);
}
.learning-preview-card:hover {
  transform: translateY(-3px);
  border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
  box-shadow: 0 14px 32px rgba(26, 20, 38, 0.08);
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
  border-radius: 999px;
  background: var(--topic-purple);
  color: var(--accent);
  font: 500 0.66rem/1.4 var(--font-mono);
}
.learning-preview-card:nth-child(2) .learning-preview-badge {
  background: var(--topic-blue);
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
  font-size: 1.04rem;
  font-weight: 500;
  line-height: 1.45;
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
