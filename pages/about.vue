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
          <span>{{ appConfig.authorCN }}</span>
          <i aria-hidden="true">.</i>
        </h1>
        <p class="about-introduction">{{ introduction }}</p>
        <div class="about-personal-tags" aria-label="兴趣">
          <span>LLM Agent</span>
          <span>Post-Training</span>
          <span>AI Infra</span>
          <span>XCPC</span>
        </div>
        <div class="about-contact">
          <a :href="appConfig.github" target="_blank" rel="noopener noreferrer">
            GitHub
            <span aria-hidden="true">↗</span>
          </a>
          <NuxtLink to="/academic">
            学术与论文主页
            <span aria-hidden="true">↗</span>
          </NuxtLink>
          <NuxtLink to="/posts">
            浏览技术长文
            <span aria-hidden="true">→</span>
          </NuxtLink>
        </div>
      </div>
      <div class="about-keepsake">
        <figure class="about-photo">
          <SiteImage
            src="/avatar.jpg"
            :alt="`${appConfig.authorCN}的头像`"
            width="1080"
            height="1080"
            sizes="(max-width: 700px) 245px, 230px"
            priority
          />
          <figcaption>
            {{ appConfig.authorEN }}
            <span aria-hidden="true">↗</span>
          </figcaption>
        </figure>
        <p class="about-photo-caption">从公式推导到可复现的系统工程。</p>
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
          <p class="about-section-kicker">RESEARCH &amp; ENGINEERING FOCUS</p>
          <h2 id="interests-title">研究与系统工程方向</h2>
        </div>
        <p>连接算法建模、后训练对齐与底层推理系统。</p>
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
              <SiteImage
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
                <div class="about-entry-title">
                  <h3>{{ entry.name }}</h3>
                  <span v-if="entry.upcoming" class="about-upcoming">
                    即将开启
                  </span>
                </div>
                <p class="about-entry-period">
                  <span class="about-entry-time">{{ entry.period }}</span>
                  <span>{{ entry.kind }}</span>
                </p>
              </div>
              <div class="about-entry-details">
                <p class="about-entry-role">{{ entry.role }}</p>
                <span v-if="entry.nameEN" class="about-entry-en" lang="en">
                  {{ entry.nameEN }}
                </span>
              </div>
              <p v-if="entry.desc" class="about-entry-desc">
                <SiteImage
                  v-if="entry.jointLogo"
                  :src="entry.jointLogo"
                  :alt="`${entry.jointSchool}校徽`"
                  width="26"
                  height="26"
                  class="about-entry-joint-logo"
                  loading="lazy"
                />
                <span>{{ entry.desc }}</span>
              </p>
            </div>
          </article>
        </div>
      </section>
      <aside class="about-tools" aria-labelledby="tools-title">
        <p class="about-section-kicker">TECH STACK &amp; SYSTEMS</p>
        <h2 id="tools-title">常用技术栈</h2>
        <p class="about-tools-intro">在算法实验、推理加速与系统开发中最常打交道的工具。</p>
        <div class="about-tool-list">
          <span v-for="tech in appConfig.techStack" :key="tech.name">
            {{ tech.name }}
          </span>
        </div>
        <p class="about-tools-foot">在实验里验证假设，在代码里沉淀结论。</p>
        <NuxtLink to="/tags">
          浏览专题索引
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
          <h2 id="writing-title">近期深度长文</h2>
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
        目前公开收录 {{ posts.length }} 篇研究与工程长文
        <span v-if="topicCount">，覆盖 {{ topicCount }} 个技术标签</span>
        。
      </p>
    </section>
    <div class="about-signoff">
      <p>
        严谨建模，
        <span>扎实落地。</span>
      </p>
      <NuxtLink to="/blog">
        回到博客
        <span aria-hidden="true">→</span>
      </NuxtLink>
    </div>
  </div>
</template>

<script setup lang="ts">
import { formatDate } from '~/utils/blog'

const appConfig = useAppConfig()
const { data } = await usePublicArchive()
const posts = computed(() => data.value ?? [])
const recentPosts = computed(() => posts.value.slice(0, 3))
const topicCount = computed(
  () => new Set(posts.value.flatMap((post) => post.tags ?? [])).size,
)

// The admission status has its own section; retain the personal parts of the bio.
const introduction = computed(() =>
  appConfig.bio.replace(/[^。，]*已(?:保研|推免)录取[^。]*。?/g, '').trim(),
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
    jointSchool: edu.jointSchool ?? '',
    jointLogo: edu.jointLogo ?? '',
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
    jointSchool: '',
    jointLogo: '',
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
    label: 'AGENT & RL',
    tone: 'purple',
    title: 'LLM Agent 与后训练对齐',
    desc: '多约束 DAG 任务规划、两阶段工具路由、100K+ 长上下文与 MCP 沙箱；Verifiable GRPO、SFT → DPO 偏好对齐与多视角推理蒸馏。',
  },
  {
    key: 'SYS',
    label: 'AI INFRA',
    tone: 'green',
    title: 'AI 推理系统与算子优化',
    desc: 'SGLang / vLLM 高并发服务调优（Prefix Caching / Chunked Prefill）、单卡 FSDP2 训练栈、CUDA 融合算子与视频 DiT 时空缓存加速。',
  },
  {
    key: 'SRC',
    label: 'OPEN SOURCE',
    tone: 'blue',
    title: '开源构建与底层源码剖析',
    desc: 'DeepScientist（7,000+ 用户）与 MiniCode（1k+ Stars）核心研发，向 mlc-ai/xgrammar（PR #1011）等开源基础设施贡献底层并发修复。',
  },
  {
    key: 'DOC',
    label: 'ALGORITHMS',
    tone: 'orange',
    title: '算法竞赛与分布式后端',
    desc: 'ACM-ICPC / CCPC 区域赛与全国邀请赛奖牌、RAICOM 国一；扎实的数据结构、C++ / Go 高并发系统与分布式存储工程实践。',
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
  font-size: clamp(2.5rem, 4.4vw, 3.2rem);
  font-weight: 600;
  line-height: 1.24;
  letter-spacing: -0.042em;
}
.about-journal h1 > span {
  position: relative;
  display: inline-block;
  color: var(--accent);
}
.about-journal h1 i {
  color: var(--accent-2);
  font-style: normal;
  margin-left: 2px;
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
  padding: 4px 11px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--surface);
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.68rem;
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
  padding: 10px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
  transition: border-color 220ms ease, box-shadow 220ms ease;
}
.about-photo img {
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: 10px;
}
.about-photo figcaption {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 6px 2px;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.02em;
}
.about-photo figcaption span {
  color: var(--accent);
  font-size: 0.9rem;
}
.about-journal .about-photo-caption {
  margin-top: 20px;
  text-align: center;
  color: var(--muted);
  font-size: 0.75rem;
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
  padding: 22px 24px 24px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  min-width: 0;
  transition: border-color 200ms ease, box-shadow 200ms ease;
}
.about-interest-note[data-tone='purple'] { background: color-mix(in srgb, var(--topic-purple) 55%, var(--surface)); }
.about-interest-note[data-tone='green'] {
  background: color-mix(in srgb, var(--topic-green) 55%, var(--surface));
}
.about-interest-note[data-tone='blue'] {
  background: color-mix(in srgb, var(--topic-blue) 55%, var(--surface));
}
.about-interest-note[data-tone='orange'] {
  background: color-mix(in srgb, var(--topic-orange) 55%, var(--surface));
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
  padding: 2px 22px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: var(--about-shadow);
}
.about-journey-entry {
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr);
  gap: 14px;
  padding-block: 16px;
}
.about-journey-entry + .about-journey-entry {
  border-top: 1px dashed var(--line);
}
.about-entry-logo {
  display: grid;
  place-items: center;
  align-self: start;
  width: 38px;
  height: 38px;
  padding: 4px;
  background: #fff;
  border-radius: 10px;
  border: 1px solid #eee;
}
.about-entry-logo img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
.about-entry-top {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 4px 12px;
  flex-wrap: wrap;
}
.about-entry-title {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  min-width: 0;
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
  font-size: 0.7rem;
  overflow-wrap: anywhere;
}
.about-entry-details {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 2px 10px;
  margin-top: 4px;
  line-height: 1.6;
}
.about-journal .about-entry-role {
  color: var(--ink);
  font-size: 0.83rem;
}
.about-journal .about-entry-desc {
  display: flex;
  align-items: center;
  gap: 7px;
  color: var(--text);
  font-size: 0.79rem;
  line-height: 1.7;
  margin-top: 4px;
}
.about-entry-desc > span {
  min-width: 0;
  overflow-wrap: anywhere;
}
.about-entry-joint-logo {
  flex-shrink: 0;
  width: 26px;
  height: 26px;
  padding: 2px;
  border: 1px solid #eee;
  border-radius: 6px;
  background: #fff;
  object-fit: contain;
}
.about-journal .about-entry-period {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 0.72rem;
  line-height: 1.6;
}
.about-entry-period > span:last-child {
  padding-left: 8px;
  border-left: 1px solid var(--line-strong);
  font-size: 0.68rem;
}
.about-tools {
  position: relative;
  padding: 26px;
  margin-top: 65px;
  border: 1px solid var(--line);
  background: var(--surface);
  border-radius: var(--radius);
  min-width: 0;
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
  padding: 26px 28px;
  border: 1px solid var(--line);
  background: var(--surface);
  border-radius: var(--radius);
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
    border-color: color-mix(in srgb, var(--accent) 35%, var(--line));
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
    grid-template-columns: minmax(0, 1fr);
    gap: 24px;
  }
  .about-journey-paper {
    padding-inline: 20px;
  }
  .about-tools {
    padding-inline: 20px;
    margin-top: 8px;
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
    padding: 24px;
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
    grid-template-columns: 32px minmax(0, 1fr);
    gap: 12px;
    padding-block: 14px;
  }
  .about-entry-logo {
    width: 32px;
    height: 32px;
    padding: 3px;
    border-radius: 8px;
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
}

.about-entry-time {
  font-variant-numeric: tabular-nums;
}
</style>
