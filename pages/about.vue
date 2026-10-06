<template>
  <div class="page-frame about-journal">
    <section class="about-welcome" aria-labelledby="about-title">
      <div class="about-welcome-copy">
        <p class="about-kicker">
          <span aria-hidden="true">✳</span>
          ABOUT ME
          <span class="about-kicker-line" aria-hidden="true"></span>
          很高兴在这里遇见你
        </p>
        <h1 id="about-title">
          你好，我是
          <br />
          <span>
            {{ appConfig.authorCN }}
            <svg viewBox="0 0 220 16" fill="none" aria-hidden="true">
              <path d="M4 10Q110 1 216 9M30 14Q116 7 192 12" />
            </svg>
          </span>
          <i aria-hidden="true">.</i>
        </h1>
        <p class="about-introduction">{{ introduction }}</p>
        <div class="about-personal-tags" aria-label="兴趣">
          <span>AI Infra</span>
          <span>Backend</span>
          <span>XCPC</span>
          <span>旅行</span>
        </div>
        <div class="about-contact">
          <a :href="appConfig.github" target="_blank" rel="noopener noreferrer">
            GitHub
            <span aria-hidden="true">↗</span>
          </a>
          <NuxtLink to="/academic">
            学术主页
            <span aria-hidden="true">↗</span>
          </NuxtLink>
          <NuxtLink to="/posts">
            读我的笔记
            <span aria-hidden="true">→</span>
          </NuxtLink>
        </div>
      </div>
      <div class="about-keepsake">
        <svg
          class="about-flower"
          viewBox="0 0 64 64"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="M32 30C10 34 12 10 24 14c8 2 8 10 8 16Zm2 2C30 10 54 12 50 24c-2 8-10 8-16 8Zm-2 2c22-4 20 20 8 16-8-2-8-10-8-16Zm-2-2c4 22-20 20-16 8 2-8 10-8 16-8Z"
          />
          <circle cx="32" cy="32" r="4" />
        </svg>
        <figure class="about-photo">
          <span class="about-photo-tape" aria-hidden="true"></span>
          <img
            src="/avatar.jpg"
            :alt="`${appConfig.authorCN}的头像`"
            width="1080"
            height="1080"
          />
          <figcaption>
            {{ appConfig.authorEN }}
            <span aria-hidden="true">↗</span>
          </figcaption>
        </figure>
        <p class="about-photo-caption">在代码里探索，也在路上。</p>
        <svg
          class="about-spark"
          viewBox="0 0 40 40"
          fill="none"
          aria-hidden="true"
        >
          <path d="M20 2Q22 18 38 20Q22 22 20 38Q18 22 2 20Q18 18 20 2Z" />
        </svg>
      </div>
    </section>

    <div v-if="nextEducation" class="about-next-stop">
      <span class="about-stop-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none">
          <path d="m3 9 9-5 9 5-9 5-9-5Zm4 3v5q5 4 10 0v-5m4-3v8" />
        </svg>
      </span>
      <div>
        <span class="about-stop-label">
          下一站 · {{ nextEducation.school }}
        </span>
        <p>{{ nextEducation.major }} · {{ nextEducation.status }}</p>
      </div>
      <NuxtLink to="/academic">
        学术档案
        <span aria-hidden="true">↗</span>
      </NuxtLink>
    </div>

    <section
      id="about-interests"
      class="about-section"
      aria-labelledby="interests-title"
    >
      <header class="about-section-head">
        <div>
          <p class="about-section-kicker">CURIOSITY</p>
          <h2 id="interests-title">好奇心落在哪里</h2>
        </div>
        <p>从一个问题开始，慢慢弄明白。</p>
      </header>
      <div class="about-interest-grid">
        <article
          v-for="(item, index) in focusAreas"
          :key="item.key"
          class="about-interest-note"
          :data-tone="item.tone"
        >
          <div class="about-note-top">
            <span>0{{ index + 1 }} / {{ item.label }}</span>
            <svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
              <template v-if="item.key === 'AI'">
                <rect x="8" y="8" width="16" height="16" rx="5" />
                <path
                  d="M12 4v4m8-4v4M12 24v4m8-4v4M4 12h4m-4 8h4m16-8h4m-4 8h4M12 14h8m-8 4h5"
                />
              </template>
              <template v-else-if="item.key === 'SYS'">
                <rect x="5" y="5" width="22" height="9" rx="3" />
                <rect x="5" y="18" width="22" height="9" rx="3" />
                <path d="M10 9.5h.01M10 22.5h.01M16 14v4m3-8.5h4m-4 13h4" />
              </template>
              <path
                v-else-if="item.key === 'SRC'"
                d="m10 9-7 7 7 7m12-14 7 7-7 7M19 5l-6 22"
              />
              <path
                v-else
                d="M16 8c-4-3-8-3-12-2v20c4-1 8-1 12 2m0-20c4-3 8-3 12-2v20c-4-1-8-1-12 2V8ZM8 11l4 1m-4 5 4 1m8-6 4-1m-4 7 4-1"
              />
            </svg>
          </div>
          <h3>{{ item.title }}</h3>
          <p>{{ item.desc }}</p>
        </article>
      </div>
    </section>

    <div class="about-detail-grid">
      <section
        class="about-section about-journey"
        aria-labelledby="journey-title"
      >
        <header class="about-section-head">
          <div>
            <p class="about-section-kicker">ALONG THE WAY</p>
            <h2 id="journey-title">走过的路，和下一站</h2>
          </div>
        </header>
        <div class="about-journey-paper">
          <article
            v-for="entry in journeyEntries"
            :key="`${entry.kind}-${entry.name}`"
            class="about-journey-entry"
          >
            <div class="about-entry-logo">
              <img
                v-if="entry.logo"
                :src="entry.logo"
                alt=""
                width="48"
                height="48"
                loading="lazy"
              />
              <span v-else aria-hidden="true">{{ entry.name[0] }}</span>
            </div>
            <div class="about-entry-copy">
              <div class="about-entry-top">
                <h3>{{ entry.name }}</h3>
                <span v-if="entry.upcoming" class="about-upcoming">
                  即将开启
                </span>
              </div>
              <p v-if="entry.nameEN" class="about-entry-en">
                {{ entry.nameEN }}
              </p>
              <p class="about-entry-role">{{ entry.role }}</p>
              <p v-if="entry.desc" class="about-entry-desc">{{ entry.desc }}</p>
              <p class="about-entry-period">
                <span class="about-entry-time">{{ entry.period }}</span>
                <span>{{ entry.kind }}</span>
              </p>
            </div>
          </article>
        </div>
      </section>
      <aside class="about-tools" aria-labelledby="tools-title">
        <p class="about-section-kicker">IN MY TOOLBOX</p>
        <h2 id="tools-title">手边的工具</h2>
        <p class="about-tools-intro">把想法做出来时，常用这些。</p>
        <div class="about-tool-list">
          <span v-for="tech in appConfig.techStack" :key="tech.name">
            {{ tech.name }}
          </span>
        </div>
        <svg
          class="about-tool-doodle"
          viewBox="0 0 180 85"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="m25 8 100 7 6 45-99-5-7-47Zm7 47-19 12 127 10-9-17m-18-18-8 6-2-7m-11-14-7 5 6 6m21-10 7 7-7 5M39 63l67 5"
          />
          <path d="m151 8 2 8 8 2-8 2-2 8-2-8-8-2 8-2 2-8Z" />
        </svg>
        <p class="about-tools-foot">边做，边学，边记录。</p>
        <NuxtLink to="/tags">
          翻翻主题索引
          <span aria-hidden="true">→</span>
        </NuxtLink>
      </aside>
    </div>

    <section
      v-if="recentPosts.length"
      class="about-section about-writing"
      aria-labelledby="writing-title"
    >
      <header class="about-section-head">
        <div>
          <p class="about-section-kicker">LATEST NOTES</p>
          <h2 id="writing-title">最近写下的</h2>
        </div>
        <NuxtLink to="/posts">
          全部文章
          <span aria-hidden="true">↗</span>
        </NuxtLink>
      </header>
      <div class="about-writing-list">
        <NuxtLink
          v-for="post in recentPosts"
          :key="post.path"
          class="about-writing-link"
          :to="post.path"
        >
          <time class="about-writing-date" :datetime="formatDate(post.date)">
            {{ formatDate(post.date) }}
          </time>
          <h3>{{ post.title }}</h3>
          <span aria-hidden="true">↗</span>
        </NuxtLink>
      </div>
      <p class="about-writing-count">
        这里存着 {{ posts.length }} 篇公开笔记
        <span v-if="topicCount">，围绕 {{ topicCount }} 个标签慢慢生长</span>
        。
      </p>
    </section>
    <div class="about-signoff">
      <p>
        随手记录，
        <span>慢慢探索。</span>
      </p>
      <NuxtLink to="/blog">
        回到博客
        <span aria-hidden="true">→</span>
      </NuxtLink>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { PostMeta } from '~/server/api/posts.get'
import { formatDate } from '~/utils/blog'

const appConfig = useAppConfig()
const { data } = await useAsyncData<PostMeta[]>('about-posts', () =>
  $fetch('/api/posts'),
)
const posts = computed(() => data.value ?? [])
const recentPosts = computed(() => posts.value.slice(0, 3))
const topicCount = computed(
  () => new Set(posts.value.flatMap((post) => post.tags ?? [])).size,
)

// The admission status has its own section; retain the personal parts of the bio.
const introduction = computed(() =>
  appConfig.bio.replace(/[^。]*已保研录取[^。]*。?/g, '').trim(),
)
const isUpcoming = (period: string) =>
  Number.parseInt(period, 10) > new Date().getFullYear()
const nextEducation = computed(() =>
  appConfig.education.find((edu) => isUpcoming(edu.period)),
)
const journeyEntries = computed(() => [
  ...appConfig.education.map((edu) => ({
    name: edu.school,
    nameEN: edu.schoolEN,
    logo: edu.logo,
    role: edu.major,
    desc: edu.status,
    period: edu.period,
    kind: '教育',
    upcoming: isUpcoming(edu.period),
  })),
  ...appConfig.experience.map((exp) => ({
    name: exp.company,
    nameEN: exp.companyEN,
    logo: exp.logo,
    role: exp.role,
    desc: exp.desc,
    period: exp.period,
    kind: '经历',
    upcoming: false,
  })),
])

const focusAreas = [
  {
    key: 'AI',
    label: 'AI',
    tone: 'purple',
    title: 'AI Infra / Agent',
    desc: 'RAG、上下文工程、记忆系统、推理优化和 Agent 产品化。',
  },
  {
    key: 'SYS',
    label: 'SYSTEMS',
    tone: 'green',
    title: 'Backend Systems',
    desc: '高并发、缓存、数据库、消息队列和分布式系统设计。',
  },
  {
    key: 'SRC',
    label: 'SOURCE',
    tone: 'blue',
    title: 'Source Reading',
    desc: '从源码和项目结构里提炼可迁移的工程经验。',
  },
  {
    key: 'DOC',
    label: 'NOTES',
    tone: 'orange',
    title: 'Knowledge Base',
    desc: '把零散学习转成可搜索、可复盘、可长期维护的笔记库。',
  },
]

useHead({
  title: '关于',
  meta: [
    {
      name: 'description',
      content: `关于 ${appConfig.authorCN} — ${appConfig.role}`,
    },
  ],
})
</script>

<style scoped>
.about-journal {
  --about-shadow: 0 10px 32px #33264007;
}
.about-journal :where(h1, h2, h3, p, figure) {
  margin: 0;
}
.about-journal a {
  text-decoration: none;
}
.about-welcome {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 64px;
  align-items: center;
  padding: 30px 32px 42px;
}
.about-welcome-copy {
  min-width: 0;
}
.about-kicker {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  font-size: 0.7rem;
  letter-spacing: 0.04em;
  flex-wrap: wrap;
}
.about-kicker > span:first-child {
  color: var(--accent);
  font-size: 1.3rem;
  line-height: 1;
}
.about-kicker-line {
  width: 22px;
  height: 1px;
  background: var(--line-strong);
}
.about-journal h1 {
  margin-block: 20px 24px;
  color: var(--ink);
  font-size: clamp(2.6rem, 4.5vw, 3.3rem);
  font-weight: 500;
  line-height: 1.45;
  letter-spacing: -0.055em;
}
.about-journal h1 > span {
  position: relative;
  display: inline-block;
  color: var(--accent);
}
.about-journal h1 svg {
  position: absolute;
  bottom: -1px;
  left: -3%;
  width: 105%;
  height: 12px;
  stroke: var(--accent-2);
  stroke-width: 2;
  stroke-linecap: round;
  opacity: 0.55;
  pointer-events: none;
}
.about-journal h1 i {
  color: var(--accent-2);
  font-family: Georgia, serif;
  font-style: normal;
  margin-left: 4px;
}
.about-introduction {
  max-width: 450px;
  font-size: 0.94rem;
  line-height: 1.95;
  color: var(--text);
  text-wrap: pretty;
}
.about-personal-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
  font-size: 0.72rem;
  color: var(--text);
}
.about-personal-tags span {
  padding: 3px 10px;
  border-radius: 6px;
  background: var(--topic-purple);
  transform: rotate(-2deg);
}
.about-personal-tags span:nth-child(2) {
  background: var(--topic-green);
  transform: rotate(2deg);
}
.about-personal-tags span:nth-child(3) {
  background: var(--topic-orange);
  transform: rotate(-1deg);
}
.about-personal-tags span:nth-child(4) {
  background: var(--topic-blue);
  transform: rotate(2deg);
}
.about-contact {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 24px;
  margin-top: 30px;
}
.about-contact a {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 40px;
  color: var(--ink);
  font-size: 0.8rem;
  border-bottom: 1px solid var(--line-strong);
}
.about-contact a span,
.about-signoff a span,
.about-writing-link > span:last-child {
  transition: transform 180ms ease;
}
.about-contact a:hover {
  color: var(--accent);
  border-color: var(--accent);
}
.about-contact a:hover span,
.about-signoff a:hover span {
  transform: translate(2px, -2px);
}
.about-keepsake {
  position: relative;
  padding: 24px 14px;
}
.about-photo {
  position: relative;
  padding: 12px 12px 16px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 5px;
  box-shadow: 0 12px 32px #33264010;
  transform: rotate(5deg);
  transition: transform 280ms ease;
}
.about-photo img {
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: 2px;
}
.about-photo figcaption {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 8px 0;
  color: var(--ink);
  font:
    italic 1.3rem/1.4 Georgia,
    serif;
}
.about-photo figcaption span {
  color: var(--accent-2);
  font-size: 1rem;
}
.about-photo-tape {
  position: absolute;
  width: 88px;
  height: 27px;
  top: -13px;
  left: calc(50% - 44px);
  z-index: 1;
  background: color-mix(in srgb, var(--topic-orange) 80%, transparent);
  border-inline: 1px dashed var(--line-strong);
  transform: rotate(-9deg);
}
.about-journal .about-photo-caption {
  margin-top: 28px;
  text-align: center;
  color: var(--muted);
  font-size: 0.75rem;
  transform: rotate(-3deg);
}
.about-flower {
  position: absolute;
  width: 66px;
  height: 66px;
  z-index: 1;
  top: -4px;
  right: -8px;
  stroke: var(--accent-2);
  stroke-width: 1.4;
  fill: var(--topic-purple);
  transform: rotate(15deg);
  pointer-events: none;
  transition: transform 350ms ease;
}
.about-spark {
  position: absolute;
  left: -28px;
  bottom: 58px;
  width: 32px;
  height: 32px;
  fill: var(--topic-orange);
  stroke: var(--accent-2);
  stroke-width: 1;
  pointer-events: none;
}
.about-next-stop {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px 26px;
  border-radius: 18px;
  background: var(--topic-green);
  color: var(--ink);
  margin-block: 12px 64px;
}
.about-stop-icon {
  display: grid;
  place-items: center;
  flex-shrink: 0;
  width: 44px;
  height: 44px;
  background: color-mix(in srgb, var(--surface) 60%, transparent);
  border-radius: 50%;
}
.about-stop-icon svg {
  width: 25px;
  height: 25px;
  stroke: var(--text);
  stroke-width: 1.25;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.about-stop-label {
  font-size: 0.86rem;
  font-weight: 500;
}
.about-next-stop p {
  margin-top: 2px;
  font-size: 0.76rem;
  color: var(--text);
}
.about-next-stop a {
  margin-left: auto;
  flex-shrink: 0;
  padding-block: 8px;
  color: var(--ink);
  font-size: 0.76rem;
}
.about-section {
  margin-bottom: 60px;
  scroll-margin-top: calc(var(--nav-h) + 24px);
}
.about-section-head {
  display: flex;
  justify-content: space-between;
  align-items: end;
  gap: 20px;
  margin-bottom: 22px;
}
.about-section-kicker {
  color: var(--muted);
  font-size: 0.66rem;
  letter-spacing: 0.13em;
  font-weight: 400;
}
.about-journal h2 {
  margin-top: 6px;
  font-size: 1.35rem;
  color: var(--ink);
  font-weight: 500;
  line-height: 1.6;
  letter-spacing: -0.03em;
}
.about-section-head > p {
  font-size: 0.78rem;
  color: var(--muted);
}
.about-section-head > a {
  flex-shrink: 0;
  font-size: 0.8rem;
  color: var(--accent);
  padding-block: 6px;
}
.about-interest-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}
.about-interest-note {
  padding: 22px 26px 24px;
  border-radius: 18px 24px 20px 24px;
  background: var(--topic-purple);
  min-width: 0;
}
.about-interest-note[data-tone='green'] {
  background: var(--topic-green);
}
.about-interest-note[data-tone='blue'] {
  background: var(--topic-blue);
}
.about-interest-note[data-tone='orange'] {
  background: var(--topic-orange);
}
.about-note-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}
.about-note-top > span {
  color: var(--text);
  font: 0.67rem var(--font-mono);
  letter-spacing: 0.04em;
}
.about-note-top svg {
  width: 30px;
  height: 30px;
  color: var(--text);
  stroke: currentColor;
  stroke-width: 1.2;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.about-interest-note h3 {
  color: var(--ink);
  font-size: 1rem;
  font-weight: 500;
  line-height: 1.5;
}
.about-interest-note p {
  max-width: 340px;
  margin-top: 10px;
  color: var(--text);
  font-size: 0.82rem;
  line-height: 1.85;
}
.about-detail-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 32px;
  align-items: start;
  margin-bottom: 60px;
}
.about-journey {
  margin-bottom: 0;
  min-width: 0;
}
.about-journey-paper {
  padding: 4px 28px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: var(--about-shadow);
}
.about-journey-entry {
  display: grid;
  grid-template-columns: 46px minmax(0, 1fr);
  gap: 18px;
  padding-block: 26px;
}
.about-journey-entry + .about-journey-entry {
  border-top: 1px dashed var(--line);
}
.about-entry-logo {
  display: grid;
  place-items: center;
  align-self: start;
  width: 46px;
  height: 46px;
  padding: 5px;
  background: #fff;
  border-radius: 12px;
  border: 1px solid #eee;
}
.about-entry-logo img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
.about-entry-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
}
.about-entry-copy {
  min-width: 0;
}
.about-entry-copy h3 {
  color: var(--ink);
  font-size: 1rem;
  font-weight: 500;
  line-height: 1.6;
}
.about-upcoming {
  border-radius: 5px;
  padding: 2px 7px;
  color: var(--text);
  background: var(--topic-green);
  font-size: 0.64rem;
  white-space: nowrap;
}
.about-entry-en {
  color: var(--muted);
  font-size: 0.72rem;
  overflow-wrap: anywhere;
}
.about-journal .about-entry-role {
  color: var(--ink);
  font-size: 0.83rem;
  margin-top: 12px;
}
.about-journal .about-entry-desc {
  color: var(--text);
  font-size: 0.79rem;
  line-height: 1.85;
  margin-top: 4px;
}
.about-journal .about-entry-period {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--muted);
  font-size: 0.72rem;
  margin-top: 12px;
  flex-wrap: wrap;
}
.about-entry-period > span:last-child {
  padding-left: 12px;
  border-left: 1px solid var(--line-strong);
  font-size: 0.68rem;
}
.about-tools {
  position: relative;
  padding: 30px 26px;
  margin-top: 65px;
  background: var(--topic-orange);
  border-radius: 5px 20px 20px 20px;
  min-width: 0;
}
.about-tools::before {
  content: '';
  position: absolute;
  top: -10px;
  left: 26px;
  width: 70px;
  height: 22px;
  background: color-mix(in srgb, var(--topic-purple) 80%, transparent);
  transform: rotate(-5deg);
  pointer-events: none;
}
.about-tools-intro {
  margin-top: 12px;
  color: var(--text);
  font-size: 0.8rem;
}
.about-tool-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 24px;
}
.about-tool-list span {
  border: 1px solid color-mix(in srgb, var(--text) 12%, transparent);
  padding: 5px 10px;
  background: color-mix(in srgb, var(--surface) 55%, transparent);
  border-radius: 8px;
  font-size: 0.73rem;
  color: var(--ink);
  overflow-wrap: anywhere;
  max-width: 100%;
}
.about-tool-doodle {
  display: block;
  width: min(180px, 100%);
  height: auto;
  margin: 32px auto 20px;
  color: var(--accent-2);
  stroke: currentColor;
  stroke-width: 1.2;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.about-tools-foot {
  color: var(--text);
  font-size: 0.8rem;
}
.about-tools a {
  display: inline-flex;
  gap: 8px;
  margin-top: 18px;
  color: var(--ink);
  font-size: 0.78rem;
  min-height: 32px;
  align-items: center;
}
.about-writing {
  margin-bottom: 44px;
}
.about-writing-list {
  border-top: 1px solid var(--line);
}
.about-writing-link {
  display: grid;
  grid-template-columns: 88px minmax(0, 1fr) 20px;
  align-items: center;
  gap: 20px;
  padding: 22px 4px;
  border-bottom: 1px solid var(--line);
}
.about-writing-date {
  color: var(--muted);
  font: 0.69rem var(--font-mono);
}
.about-writing-link h3 {
  color: var(--ink);
  font-weight: 400;
  font-size: 0.92rem;
  line-height: 1.75;
  transition: color 180ms ease;
  overflow-wrap: anywhere;
}
.about-writing-link > span:last-child {
  color: var(--accent);
  text-align: right;
}
.about-writing-link:hover h3 {
  color: var(--accent);
}
.about-writing-link:hover > span:last-child {
  transform: translate(2px, -2px);
}
.about-journal .about-writing-count {
  margin-top: 18px;
  font-size: 0.75rem;
  color: var(--muted);
}
.about-signoff {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 28px 32px;
  background: var(--notebook-hero-bg);
  border-radius: 20px;
}
.about-signoff p {
  font-size: 1.05rem;
  color: var(--ink);
}
.about-signoff p span {
  color: var(--accent);
}
.about-signoff a {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding-block: 6px;
  font-size: 0.78rem;
  color: var(--ink);
  flex-shrink: 0;
}
@media (hover: hover) and (pointer: fine) {
  .about-keepsake:hover .about-photo {
    transform: rotate(2deg) translateY(-3px);
  }
  .about-keepsake:hover .about-flower {
    transform: rotate(35deg);
  }
}
@media (min-width: 821px) and (prefers-reduced-motion: no-preference) {
  .about-welcome-copy {
    animation: about-arrive 500ms ease both;
  }
  .about-keepsake {
    animation: about-arrive 650ms 90ms ease both;
  }
}
@keyframes about-arrive {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
@media (max-width: 820px) {
  .about-welcome {
    grid-template-columns: minmax(0, 1fr) 230px;
    gap: 32px;
    padding: 20px 12px 32px;
  }
  .about-detail-grid {
    grid-template-columns: minmax(0, 1fr) 250px;
    gap: 24px;
  }
  .about-journey-paper {
    padding-inline: 20px;
  }
  .about-tools {
    padding-inline: 20px;
  }
}
@media (max-width: 640px) {
  .about-welcome {
    grid-template-columns: minmax(0, 1fr);
    gap: 30px;
    padding: 4px 4px 30px;
  }
  .about-journal h1 {
    font-size: 2.7rem;
  }
  .about-keepsake {
    width: 230px;
    justify-self: center;
    padding: 12px;
    margin-bottom: 6px;
  }
  .about-photo-caption {
    font-size: 0.73rem;
  }
  .about-next-stop {
    margin-bottom: 44px;
    padding: 18px;
    gap: 12px;
    flex-wrap: wrap;
  }
  .about-next-stop > div {
    flex: 1;
    min-width: 0;
  }
  .about-next-stop a {
    margin-left: 56px;
  }
  .about-next-stop p {
    line-height: 1.8;
  }
  .about-section {
    margin-bottom: 44px;
  }
  .about-section-head {
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }
  .about-section-head > p {
    width: 100%;
  }
  .about-interest-grid {
    gap: 12px;
  }
  .about-interest-note {
    padding: 18px;
  }
  .about-interest-note h3 {
    font-size: 0.9rem;
    overflow-wrap: anywhere;
  }
  .about-note-top {
    gap: 8px;
  }
  .about-note-top span {
    font-size: 0.58rem;
  }
  .about-note-top svg {
    width: 25px;
    height: 25px;
    flex-shrink: 0;
  }
  .about-detail-grid {
    grid-template-columns: minmax(0, 1fr);
    gap: 24px;
    margin-bottom: 44px;
  }
  .about-journey {
    margin-bottom: 0;
  }
  .about-tools {
    margin-top: 8px;
    padding: 28px;
  }
  .about-tool-doodle {
    width: 135px;
    margin-block: 24px 16px;
  }
  .about-writing-link {
    grid-template-columns: minmax(0, 1fr) 18px;
    gap: 6px 12px;
    padding-block: 18px;
  }
  .about-writing-date {
    grid-column: 1;
  }
  .about-writing-link h3 {
    grid-row: 2;
  }
  .about-writing-link > span:last-child {
    grid-row: 2;
    grid-column: 2;
  }
  .about-signoff {
    padding: 24px;
    flex-wrap: wrap;
    gap: 10px;
  }
}
@media (max-width: 480px) {
  .about-interest-grid {
    grid-template-columns: minmax(0, 1fr);
  }
  .about-interest-note {
    padding: 20px 24px;
  }
  .about-interest-note h3 {
    font-size: 1rem;
  }
  .about-note-top span {
    font-size: 0.65rem;
  }
  .about-kicker {
    gap: 8px;
    font-size: 0.64rem;
  }
  .about-journal h1 {
    font-size: 2.45rem;
  }
  .about-journal h2 {
    font-size: 1.25rem;
  }
  .about-journey-entry {
    grid-template-columns: 36px minmax(0, 1fr);
    gap: 14px;
  }
  .about-entry-logo {
    width: 36px;
    height: 36px;
    padding: 4px;
    border-radius: 9px;
  }
  .about-journey-paper {
    padding-inline: 18px;
  }
  .about-contact {
    gap: 12px 20px;
  }
}
@media (prefers-reduced-motion: reduce) {
  .about-journal *,
  .about-journal *::before {
    animation: none !important;
    transition: none !important;
  }
  .about-keepsake:hover .about-photo {
    transform: rotate(5deg);
  }
  .about-keepsake:hover .about-flower {
    transform: rotate(15deg);
  }
}

.about-entry-time {
  font-variant-numeric: tabular-nums;
}
</style>
