<script setup lang="ts">
import { academicProfile as profile } from '~/data/academic'
import { imageWithBase } from '~/utils/image-assets'

definePageMeta({ layout: 'profile' })

const appConfig = useAppConfig()
const imageBaseURL = useRuntimeConfig().app.baseURL
const publications = profile.publications.map(paper => ({
  ...paper,
  paperUrl: paper.links.find(link => link.label === 'Paper')?.url,
}))
const quickLinks = computed(() => [
  ...(profile.email ? [{ label: 'Email', url: `mailto:${profile.email}` }] : []),
  { label: 'GitHub', url: profile.github },
  ...(profile.scholar ? [{ label: 'Google Scholar', url: profile.scholar }] : []),
  ...(profile.cv ? [{ label: 'CV / 简历', url: profile.cv }] : []),
  { label: 'Blog', url: '/blog' },
])
const featuredProjects = profile.projects.slice(0, 2)
const heroStats = computed(() => [
  { value: String(profile.publications.length).padStart(2, '0'), label: 'Publications' },
  { value: String(featuredProjects.length).padStart(2, '0'), label: 'Open Systems' },
  { value: String(profile.interests.length).padStart(2, '0'), label: 'Research Tracks' },
])
const sections = computed(() => [
  { id: 'about', label: 'About' },
  { id: 'research', label: 'Research' },
  ...(profile.news.length ? [{ id: 'news', label: 'News' }] : []),
  { id: 'publications', label: 'Publications' },
  ...(featuredProjects.length ? [{ id: 'projects', label: 'Projects' }] : []),
  { id: 'experience', label: 'Experience' },
  { id: 'education', label: 'Education' },
  ...(profile.awards?.length ? [{ id: 'awards', label: 'Honors' }] : []),
])

function isExternal(url: string) {
  return /^https?:\/\//.test(url)
}

useHead({
  title: `${profile.name} | Academic Homepage`,
  titleTemplate: null,
  meta: [
    { name: 'description', content: `${profile.nameCN}的学术主页：研究兴趣、论文、教育背景与经历。` },
    { property: 'og:title', content: `${profile.name} · Academic Homepage` },
    { property: 'og:description', content: profile.bio[0] },
    { property: 'og:url', content: `${appConfig.url}/academic` },
    { property: 'og:image', content: `${appConfig.url}${profile.avatar}` },
  ],
  link: [{ rel: 'canonical', href: `${appConfig.url}/academic` }],
})
</script>

<template>
  <div class="academic-page">
    <div class="academic-glow" aria-hidden="true" />
    <header class="academic-identity">
      <div class="academic-heading">
        <p class="academic-kicker"><span class="academic-kicker-dot" aria-hidden="true" /> ACADEMIC HOMEPAGE</p>
        <h1>{{ profile.name }} <span lang="zh-CN">{{ profile.nameCN }}</span></h1>
        <p class="academic-tagline">{{ profile.tagline }}</p>
        <div class="academic-affiliation-row">
          <p class="academic-affiliation">{{ profile.affiliation }}</p>
          <span class="academic-affiliation-pill">{{ profile.affiliationNote }}</span>
        </div>
        <p v-if="profile.location" class="academic-location">{{ profile.location }}</p>
        <div class="academic-links" aria-label="个人链接">
          <a
            v-for="link in quickLinks"
            :key="link.label"
            :href="link.url"
            class="academic-link-pill"
            :target="isExternal(link.url) ? '_blank' : undefined"
            :rel="isExternal(link.url) ? 'noopener noreferrer' : undefined"
          >{{ link.label }}<span v-if="isExternal(link.url)" aria-hidden="true"> ↗</span><span v-else aria-hidden="true"> →</span></a>
        </div>
        <dl class="academic-hero-stats" aria-label="学术与开源工作概览">
          <div v-for="stat in heroStats" :key="stat.label">
            <dt>{{ stat.label }}</dt>
            <dd>{{ stat.value }}</dd>
          </div>
        </dl>
      </div>
      <div class="academic-portrait-wrap">
        <SiteImage class="academic-portrait" :src="profile.avatar" :alt="`${profile.nameCN}的头像`" width="208" height="208" priority />
        <span class="academic-portrait-caption">RESEARCH × ENGINEERING</span>
      </div>
    </header>

    <nav class="academic-sections" aria-label="学术主页栏目">
      <a v-for="section in sections" :key="section.id" :href="`#${section.id}`">{{ section.label }}</a>
    </nav>

    <section id="about" class="academic-section" aria-labelledby="about-heading">
      <h2 id="about-heading">About Me <span>个人简介</span></h2>
      <p v-for="paragraph in profile.bio" :key="paragraph">{{ paragraph }}</p>
    </section>

    <section id="research" class="academic-section" aria-labelledby="research-heading">
      <h2 id="research-heading">Research Interests <span>研究兴趣</span></h2>
      <div class="academic-interests">
        <article v-for="interest in profile.interests" :key="interest.title">
          <h3>{{ interest.english }}</h3>
          <span>{{ interest.title }}</span>
          <p>{{ interest.description }}</p>
        </article>
      </div>
    </section>

    <section v-if="profile.news.length" id="news" class="academic-section" aria-labelledby="news-heading">
      <h2 id="news-heading">News <span>近期动态</span></h2>
      <ul class="academic-news">
        <li v-for="item in profile.news" :key="`${item.date}-${item.text}`">
          <span class="academic-date">{{ item.date }}</span>
          <a v-if="item.url" :href="item.url" :target="isExternal(item.url) ? '_blank' : undefined" :rel="isExternal(item.url) ? 'noopener noreferrer' : undefined">{{ item.text }}</a>
          <p v-else>{{ item.text }}</p>
        </li>
      </ul>
    </section>

    <section id="publications" class="academic-section" aria-labelledby="publications-heading">
      <h2 id="publications-heading">Publications <span>论文</span></h2>
      <div v-if="profile.publications.length" class="academic-publications">
        <article v-for="paper in publications" :key="paper.title" class="academic-paper" :class="{ 'with-image': paper.image }">
          <figure v-if="paper.image" class="academic-paper-figure">
            <span class="academic-paper-badge">{{ paper.badge || paper.year }}</span>
            <a class="academic-paper-figure-link" :href="imageWithBase(paper.image, imageBaseURL)" target="_blank" rel="noopener noreferrer" :aria-label="`查看 ${paper.title} 的框架原图`">
              <SiteImage :src="paper.image" :alt="paper.imageAlt || `${paper.title} 框架概览`" width="334" sizes="(max-width: 468px) calc(100vw - 48px), (max-width: 760px) 420px, (max-width: 1008px) calc(36vw - 28.8px), 334px" loading="lazy" />
            </a>
            <figcaption v-if="paper.imageSource" class="academic-paper-caption">
              <a :href="paper.imageSource.url" :target="isExternal(paper.imageSource.url) ? '_blank' : undefined" :rel="isExternal(paper.imageSource.url) ? 'noopener noreferrer' : undefined" :aria-label="`${paper.title} 的图源：${paper.imageSource.label}`">{{ paper.imageSource.label }} <span aria-hidden="true">↗</span></a>
            </figcaption>
          </figure>
          <div class="academic-paper-copy">
            <span v-if="!paper.image" class="academic-paper-badge academic-paper-badge-inline">{{ paper.badge || paper.year }}</span>
            <h3>
              <a v-if="paper.paperUrl" :href="paper.paperUrl" :target="isExternal(paper.paperUrl) ? '_blank' : undefined" :rel="isExternal(paper.paperUrl) ? 'noopener noreferrer' : undefined">{{ paper.title }}</a>
              <template v-else>{{ paper.title }}</template>
            </h3>
            <p class="academic-authors">
              <template v-for="(author, index) in paper.authors" :key="author.name"><span v-if="index">, </span><strong v-if="author.self">{{ author.name }}</strong><span v-else>{{ author.name }}</span></template>
            </p>
            <p class="academic-venue">{{ paper.venue }} <span aria-hidden="true">·</span> {{ paper.year }}</p>
            <p v-if="paper.summary" class="academic-summary">{{ paper.summary }}</p>
            <div v-if="paper.links.length" class="academic-resource-links">
              <a v-for="link in paper.links" :key="link.label" :href="link.url" :target="isExternal(link.url) ? '_blank' : undefined" :rel="isExternal(link.url) ? 'noopener noreferrer' : undefined" :aria-label="`${paper.title} — ${link.label}`">{{ link.label }} <span aria-hidden="true">↗</span></a>
            </div>
          </div>
        </article>
      </div>
      <p v-else class="academic-empty">论文列表正在整理中，将在这里更新论文、预印本与相关资料。</p>
    </section>

    <section v-if="featuredProjects.length" id="projects" class="academic-section" aria-labelledby="projects-heading">
      <h2 id="projects-heading">Selected Systems &amp; Open Source <span>代表系统与开源工作</span></h2>
      <article v-for="project in featuredProjects" :key="project.name" class="academic-project">
        <h3>{{ project.name }}</h3>
        <p>{{ project.description }}</p>
        <div class="academic-project-meta">
          <span v-for="tag in project.tags" :key="tag" class="academic-project-tag">{{ tag }}</span>
          <a v-for="link in project.links" :key="link.label" :href="link.url" :target="isExternal(link.url) ? '_blank' : undefined" :rel="isExternal(link.url) ? 'noopener noreferrer' : undefined">{{ link.label }} <span aria-hidden="true">↗</span></a>
        </div>
      </article>
    </section>

    <section id="experience" class="academic-section" aria-labelledby="experience-heading">
      <h2 id="experience-heading">Research &amp; Industry Experience <span>科研与实习经历</span></h2>
      <article v-for="item in profile.experience" :key="item.institution" class="academic-entry" :class="{ 'with-logo': item.logo }">
        <SiteImage v-if="item.logo" :src="item.logo" :alt="item.institution" width="44" height="44" loading="lazy" />
        <div class="academic-entry-copy">
          <h3>{{ item.institution }} <span>{{ item.english }}</span></h3>
          <p>{{ item.role }}</p>
          <p v-if="item.description" class="academic-entry-description">{{ item.description }}</p>
        </div>
        <span class="academic-date">{{ item.period }}</span>
      </article>
    </section>

    <section id="education" class="academic-section" aria-labelledby="education-heading">
      <h2 id="education-heading">Education <span>教育背景</span></h2>
      <article v-for="item in profile.education" :key="item.institution" class="academic-entry" :class="{ 'with-logo': item.logo }">
        <SiteImage v-if="item.logo" :src="item.logo" :alt="item.institution" width="44" height="44" loading="lazy" />
        <div class="academic-entry-copy">
          <h3>{{ item.institution }} <span>{{ item.english }}</span></h3>
          <p>{{ item.role }}</p>
          <p v-if="item.description" class="academic-entry-description">{{ item.description }}</p>
        </div>
        <span class="academic-date">{{ item.period }}</span>
      </article>
    </section>

    <section v-if="profile.awards?.length" id="awards" class="academic-section" aria-labelledby="awards-heading">
      <h2 id="awards-heading">Honors &amp; Competitive Programming <span>算法竞赛与荣誉</span></h2>
      <article v-for="award in profile.awards" :key="award.title" class="academic-entry">
        <div class="academic-entry-copy">
          <h3>{{ award.title }}</h3>
          <p class="academic-entry-description">{{ award.detail }}</p>
        </div>
        <span v-if="award.year" class="academic-date">{{ award.year }}</span>
      </article>
    </section>

    <p class="academic-updated">Last updated · {{ profile.lastUpdated }}</p>
  </div>
</template>

<style scoped>
/* The decorative glow bleeds past the column; clip it so narrow screens never scroll sideways. */
:global(.profile-site-academic) { overflow-x: clip; }
.academic-page { --academic-muted-strong: #5f6d64; position: relative; width: min(960px, calc(100% - 48px)); margin: 0 auto; padding: 52px 0 44px; color: #1e293b; font-size: 15px; line-height: 1.9; }
.academic-glow { position: absolute; top: -20px; left: -10%; right: -10%; height: 340px; z-index: -1; pointer-events: none; background: radial-gradient(ellipse at 25% 20%, rgba(40, 91, 69, .08), transparent 62%), radial-gradient(ellipse at 82% 35%, rgba(114, 158, 132, .11), transparent 60%); filter: blur(18px); }
.academic-identity { display: flex; justify-content: space-between; align-items: flex-start; gap: 48px; padding-bottom: 38px; animation: academic-fade-up 520ms cubic-bezier(.16, 1, .3, 1) both; }
.academic-heading { min-width: 0; }
.academic-kicker { display: inline-flex; align-items: center; gap: 8px; margin: 0 0 12px; font-size: 10px; font-weight: 650; letter-spacing: .17em; color: #54665c; }
.academic-kicker-dot { width: 6px; height: 6px; border-radius: 50%; background: #166550; box-shadow: 0 0 0 4px rgba(40, 91, 69, .14); }
.academic-heading h1 { margin: 0; color: #0f172a; font-size: 38px; font-weight: 650; letter-spacing: -.045em; line-height: 1.3; }
.academic-heading h1 span { margin-left: 12px; font-size: 22px; font-weight: 450; letter-spacing: .025em; color: #3c4d44; }
.academic-tagline { margin: 12px 0 18px; color: #52606d; font-size: 15px; line-height: 1.75; }
.academic-affiliation-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; }
.academic-affiliation { margin: 0; font-weight: 600; color: #0f172a; }
.academic-affiliation-pill { display: inline-flex; align-items: center; padding: 2px 10px; border-radius: 999px; background: #e6efe9; border: 1px solid #cce0d2; color: #24533e; font-size: 12px; font-weight: 550; }
.academic-location { margin: 4px 0 0; font-size: 13px; color: #69756e; }
.academic-links { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
.academic-page a { color: #166550; text-decoration: none; text-underline-offset: 4px; }
.academic-page a:hover { text-decoration: underline; }
.academic-page a:focus-visible { outline: 2px solid #166550; outline-offset: 4px; border-radius: 3px; }
.academic-link-pill { display: inline-flex; align-items: center; gap: 5px; padding: 6px 14px; border: 1px solid #d6e0d8; border-radius: 999px; background: #ffffff; color: #24533e; font-size: 12.5px; font-weight: 550; line-height: 1.4; box-shadow: 0 2px 8px rgba(32, 40, 36, .04); transition: transform 200ms cubic-bezier(.22, 1, .36, 1), border-color 200ms ease, background-color 200ms ease, box-shadow 200ms ease; }
.academic-link-pill:hover { text-decoration: none !important; transform: translateY(-2px); border-color: #8eb09b; background: #f4f8f5; box-shadow: 0 8px 18px rgba(40, 91, 69, .1); }
.academic-portrait-wrap { position: relative; flex-shrink: 0; margin-top: 16px; padding: 5px; border-radius: 50%; background: linear-gradient(135deg, #d7e5dc, #ffffff); box-shadow: 0 14px 36px rgba(32, 45, 38, .1), 0 0 0 1px #dce5de; transition: transform 300ms cubic-bezier(.22, 1, .36, 1); }
.academic-portrait-wrap:hover { transform: translateY(-2px); }
.academic-portrait { display: block; width: 152px; height: 152px; border-radius: 50%; object-fit: cover; background: #e8ede6; }
.academic-sections { position: sticky; top: 0; z-index: 40; display: flex; flex-wrap: wrap; gap: 10px 24px; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; padding: 13px 0; background: rgb(248 250 249 / 92%); box-shadow: 0 4px 20px rgb(32 40 36 / 4%); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }
.academic-sections a { position: relative; padding: 3px 0; color: var(--academic-muted-strong); font-size: 12.5px; font-weight: 550; transition: color 180ms ease; }
.academic-sections a:hover { color: #166550; text-decoration: none; }
.academic-section { margin-top: 48px; scroll-margin-top: 88px; }
.academic-section h2 { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px; margin: 0 0 22px; color: #0f172a; font-size: 21px; font-weight: 600; letter-spacing: -.025em; line-height: 1.4; }
.academic-section h2 > span { color: var(--academic-muted-strong); font-size: 12px; font-weight: 400; letter-spacing: .025em; }
.academic-section > p { max-width: 72ch; margin: 0 0 14px; }
.academic-section h3 { margin: 0; font-size: 15px; font-weight: 600; line-height: 1.6; color: #27362e; }
.academic-interests { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; }
.academic-interests article { padding: 20px 22px; border: 1px solid #e1e8e2; border-left: 3px solid #166550; border-radius: 14px; background: #ffffff; box-shadow: 0 6px 20px rgba(32, 40, 36, .035); transition: transform 240ms cubic-bezier(.22, 1, .36, 1), box-shadow 240ms ease, border-color 240ms ease; }
.academic-interests article:hover { transform: translateY(-3px); border-color: #b7cebf; border-left-color: #166550; box-shadow: 0 14px 30px rgba(40, 91, 69, .08); }
.academic-interests h3 { font-size: 14.5px; }
.academic-interests article > span { display: inline-block; margin-top: 2px; color: #166550; font-size: 12px; font-weight: 500; }
.academic-interests p { margin: 8px 0 0; color: #52606d; font-size: 13px; line-height: 1.8; }
.academic-empty { color: #69756e; font-size: 14px; padding: 16px 0; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; }
.academic-news { list-style: none; margin: 0; padding: 0; }
.academic-news li { display: grid; grid-template-columns: 100px minmax(0, 1fr); gap: 16px; padding: 6px 0; }
.academic-news p { margin: 0; }
.academic-date { color: var(--academic-muted-strong); font-size: 12px; white-space: nowrap; font-variant-numeric: tabular-nums; }
.academic-publications { display: grid; gap: 22px; }
.academic-paper { padding: 28px; border: 1px solid #e1e8e2; border-radius: 18px; background: #ffffff; box-shadow: 0 8px 26px rgba(32, 40, 36, .04); transition: transform 260ms cubic-bezier(.22, 1, .36, 1), box-shadow 260ms ease, border-color 260ms ease; }
.academic-paper:hover { transform: translateY(-3px); border-color: #b7cebf; box-shadow: 0 16px 38px rgba(40, 91, 69, .09); }
.academic-paper.with-image { display: grid; grid-template-columns: minmax(0, .38fr) minmax(0, .62fr); align-items: start; gap: 30px; }
.academic-paper-figure { position: relative; min-width: 0; margin: 0; }
.academic-paper-badge { position: absolute; top: -12px; left: 10px; z-index: 1; padding: 4px 11px; border-radius: 999px; background: #166550; color: #fff; font-size: 11px; font-weight: 650; letter-spacing: .04em; line-height: 1.4; box-shadow: 0 4px 12px rgba(40, 91, 69, .26); }
.academic-paper-badge-inline { position: static; display: inline-block; margin-bottom: 10px; }
.academic-paper-figure-link { display: block; overflow: hidden; border: 1px solid #e2e8f0; border-radius: 10px; background: #fff; box-shadow: 0 4px 14px rgb(32 40 36 / 6%); cursor: zoom-in; transition: border-color 220ms ease, box-shadow 220ms ease; }
.academic-paper-figure-link > img { display: block; width: 100%; height: auto; object-fit: contain; border-radius: 9px; transition: transform 360ms cubic-bezier(.22, 1, .36, 1); }
.academic-paper-figure-link:hover { border-color: #8eb09b; box-shadow: 0 8px 22px rgba(40, 91, 69, .12); }
.academic-paper-figure-link:hover > img { transform: scale(1.025); }
.academic-paper-caption { margin-top: 8px; font-size: 11px; line-height: 1.6; text-align: center; }
.academic-paper-caption a { color: var(--academic-muted-strong); }
.academic-paper-copy { min-width: 0; }
.academic-paper h3 { color: #0f172a; font-size: 17.5px; line-height: 1.5; overflow-wrap: anywhere; }
.academic-paper p { margin: 8px 0; font-size: 13px; }
.academic-authors { color: #52606d; line-height: 1.8; }
.academic-authors strong { color: #1f4d39; font-weight: 650; background: #ebf3ee; padding: 1px 6px; border-radius: 4px; }
.academic-venue { color: var(--academic-muted-strong); font-style: italic; }
.academic-summary { max-width: 72ch; color: #58655e; line-height: 1.85; }
.academic-resource-links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; font-size: 12px; font-weight: 550; }
.academic-resource-links a { display: inline-flex; align-items: center; gap: 4px; padding: 4px 11px; border: 1px solid #d8e2da; border-radius: 999px; background: #f8faf8; color: #166550; transition: background-color 180ms ease, border-color 180ms ease, transform 180ms ease; }
.academic-resource-links a:hover { text-decoration: none; background: #e8f1eb; border-color: #9bb8a6; transform: translateY(-1px); }
.academic-entry { display: flex; align-items: flex-start; gap: 18px; padding: 18px 20px; margin-bottom: 12px; border: 1px solid #e3eae4; border-radius: 14px; background: #ffffff; box-shadow: 0 4px 16px rgba(32, 40, 36, .03); transition: transform 220ms ease, border-color 220ms ease; }
.academic-entry:hover { transform: translateY(-2px); border-color: #bfd2c5; }
.academic-entry > img { width: 44px; height: 44px; padding: 3px; object-fit: contain; background: white; border: 1px solid #e8ece6; border-radius: 9px; flex-shrink: 0; }
.academic-entry-copy { flex: 1; min-width: 0; }
.academic-entry h3 { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0 10px; }
.academic-entry h3 span { color: var(--academic-muted-strong); font-size: 12px; font-weight: 400; }
.academic-entry p { margin: 3px 0 0; font-size: 13px; }
.academic-entry .academic-entry-description { margin-top: 6px; font-size: 12.5px; color: #5f6c65; line-height: 1.75; }
.academic-entry > .academic-date { padding: 3px 9px; border-radius: 999px; background: #f2f5f2; font-size: 11.5px; }
.academic-project { padding: 16px 0; border-bottom: 1px solid #e2e8f0; }
.academic-project:first-of-type { padding-top: 0; }
.academic-project p { margin: 5px 0; font-size: 13px; color: #52606d; }
.academic-project-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 10px; font-size: 12px; }
.academic-project-tag { padding: 1px 7px; border: 1px solid #e2e8f0; border-radius: 4px; font-size: 11px; color: #52606d; }
.academic-updated { margin: 36px 0 0; color: var(--academic-muted-strong); font-size: 11px; }
@keyframes academic-fade-up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@media (max-width: 760px) {
  .academic-paper.with-image { grid-template-columns: minmax(0, 1fr); gap: 20px; }
  .academic-paper-figure { width: min(100%, 420px); }
  .academic-paper h3 { font-size: 17px; }
}
@media (max-width: 640px) {
  .academic-page { padding-top: 32px; font-size: 14px; }
  .academic-identity { align-items: flex-start; gap: 18px; padding-bottom: 28px; }
  .academic-heading h1 { font-size: 29px; }
  .academic-heading h1 span { display: block; margin: 5px 0 0; font-size: 18px; }
  .academic-kicker { font-size: 8px; letter-spacing: .13em; }
  .academic-portrait-wrap { margin-top: 22px; }
  .academic-portrait { width: 88px; height: 88px; }
  .academic-tagline { font-size: 12px; margin-top: 10px; }
  .academic-affiliation { font-size: 13px; }
  .academic-affiliation-pill { font-size: 11px; }
  .academic-links { gap: 8px; }
  .academic-link-pill { padding: 5px 11px; font-size: 12px; }
  .academic-sections { gap: 10px 20px; padding: 13px 0; }
  .academic-section { margin-top: 34px; scroll-margin-top: 104px; }
  .academic-section h2 { font-size: 20px; }
  .academic-interests { grid-template-columns: 1fr; gap: 14px; }
  .academic-entry { position: relative; gap: 13px; padding: 16px; flex-wrap: wrap; }
  .academic-entry > .academic-date { width: fit-content; margin-top: 2px; }
  .academic-entry-copy { flex-basis: 100%; }
  .academic-entry.with-logo .academic-entry-copy { flex-basis: calc(100% - 57px); }
  .academic-paper { padding: 22px 18px; }
  .academic-news li { grid-template-columns: 70px minmax(0, 1fr); gap: 12px; }
}
@media (pointer: coarse) {
  .academic-resource-links a { min-height: 40px; padding-inline: 14px; }
  .academic-paper-caption a { display: inline-flex; align-items: center; min-height: 36px; }
  .academic-project-meta { gap: 4px 12px; }
  .academic-project-meta a { display: inline-flex; align-items: center; min-height: 40px; }
}
/* 2026 visual refresh — a quieter editorial canvas with one strong identity panel. */
.academic-page {
  --academic-muted-strong: #647168;
  width: min(1080px, calc(100% - 48px));
  padding: 36px 0 72px;
  color: #2c3932;
  font-size: 16px;
  counter-reset: academic-section;
}
.academic-glow { top: -80px; height: 520px; background: radial-gradient(ellipse at 16% 30%, rgb(73 120 91 / 10%), transparent 58%), radial-gradient(ellipse at 84% 15%, rgb(193 157 91 / 9%), transparent 52%); }
.academic-identity {
  position: relative;
  isolation: isolate;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 220px;
  align-items: center;
  gap: 56px;
  overflow: hidden;
  padding: 52px 56px;
  border: 1px solid rgb(255 255 255 / 8%);
  border-radius: 28px;
  background: radial-gradient(circle at 84% 18%, rgb(108 157 123 / 22%), transparent 35%), linear-gradient(135deg, #142b23, #0d1e19 72%);
  box-shadow: 0 28px 70px -42px rgb(20 43 35 / 65%);
}
.academic-identity::before {
  position: absolute;
  inset: 0;
  z-index: -1;
  background-image: linear-gradient(rgb(255 255 255 / 3%) 1px, transparent 1px), linear-gradient(90deg, rgb(255 255 255 / 3%) 1px, transparent 1px);
  background-size: 44px 44px;
  mask-image: linear-gradient(90deg, transparent, #000);
  content: '';
}
.academic-kicker { margin-bottom: 16px; color: #a9c5b3; font: 600 10px/1.5 var(--font-mono); letter-spacing: .2em; }
.academic-kicker-dot { background: #9ed5af; box-shadow: 0 0 0 4px rgb(158 213 175 / 12%); }
.academic-heading h1 { color: #f7faf7; font-size: clamp(40px, 4.4vw, 54px); font-weight: 620; letter-spacing: -.05em; }
.academic-heading h1 span { color: #c2d1c7; font-size: 23px; font-weight: 400; }
.academic-tagline { max-width: 620px; margin: 16px 0 20px; color: #b8c9bf; font-size: 16px; line-height: 1.8; }
.academic-affiliation { color: #edf4ef; font-size: 14px; }
.academic-affiliation-pill { border-color: rgb(190 220 198 / 18%); background: rgb(190 220 198 / 9%); color: #c8dfcf; font-size: 11px; }
.academic-location { color: #9fb5a7; }
.academic-links { gap: 9px; margin-top: 24px; }
.academic-page .academic-link-pill { padding: 7px 14px; border-color: rgb(230 241 233 / 18%); border-radius: 9px; background: rgb(255 255 255 / 6%); color: #eff6f0; box-shadow: none; font-size: 12px; }
.academic-page .academic-link-pill:hover { border-color: rgb(230 241 233 / 40%); background: #f2f7f3; color: #173e2d; box-shadow: none; }
.academic-hero-stats { display: flex; gap: 36px; margin: 30px 0 0; padding: 22px 0 0; border-top: 1px solid rgb(255 255 255 / 12%); }
.academic-hero-stats div { display: flex; flex-direction: column; }
.academic-hero-stats dt { order: 2; margin-top: 2px; color: #8fa99a; font: 500 9px/1.5 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }
.academic-hero-stats dd { order: 1; margin: 0; color: #e6d3a7; font: 500 23px/1.3 var(--font-mono); letter-spacing: -.04em; }
.academic-portrait-wrap { width: 220px; margin: 0; padding: 10px 10px 15px; border: 1px solid rgb(255 255 255 / 16%); border-radius: 28px; background: rgb(255 255 255 / 7%); box-shadow: 0 22px 44px -24px rgb(0 0 0 / 60%); }
.academic-portrait-wrap:hover { transform: translateY(-4px) rotate(1deg); }
.academic-portrait { width: 198px; height: 198px; border-radius: 19px; background: #244438; }
.academic-portrait-caption { display: block; margin-top: 12px; color: #a9c1b2; font: 500 8px/1.4 var(--font-mono); letter-spacing: .13em; text-align: center; }
.academic-sections { top: 12px; flex-wrap: nowrap; gap: 2px; overflow-x: auto; margin-top: 18px; padding: 7px; border: 1px solid #dfe5dd; border-radius: 14px; background: rgb(250 251 248 / 90%); box-shadow: 0 10px 30px -22px rgb(31 53 40 / 35%); scrollbar-width: none; }
.academic-sections::-webkit-scrollbar { display: none; }
.academic-sections a { flex-shrink: 0; padding: 7px 13px; border-radius: 8px; color: #5b6960; font-size: 12px; }
.academic-sections a:hover { background: #e9efe8; color: #1d5d43; }
.academic-section { margin-top: 68px; scroll-margin-top: 96px; counter-increment: academic-section; }
.academic-section h2 { gap: 12px; margin-bottom: 26px; color: #17271f; font-size: 25px; font-weight: 620; letter-spacing: -.035em; }
.academic-section h2::before { color: #8a978e; font: 500 10px/1.5 var(--font-mono); letter-spacing: .06em; content: counter(academic-section, decimal-leading-zero); }
.academic-section h2::after { flex: 1; min-width: 40px; height: 1px; margin-left: 8px; background: #dfe5dd; content: ''; }
.academic-section h2 > span { color: #7b877f; font-size: 12px; }
.academic-section h3 { color: #1c3025; font-size: 16px; }
.academic-section > p { max-width: 84ch; color: #435149; }
.academic-interests { gap: 0; overflow: hidden; border: 1px solid #dfe5dd; border-radius: 18px; background: #fff; }
.academic-interests article { padding: 28px; border: 0; border-radius: 0; box-shadow: none; }
.academic-interests article + article { border-left: 1px solid #e4e9e2; }
.academic-interests article:hover { transform: none; border-color: #e4e9e2; background: #f7f9f5; box-shadow: none; }
.academic-interests article + article:hover { border-left-color: #e4e9e2; }
.academic-interests h3 { font-size: 15px; line-height: 1.55; }
.academic-interests article > span { margin-top: 8px; color: #2a6b4e; }
.academic-interests p { margin-top: 12px; color: #5d6b62; font-size: 13px; }
.academic-news { border-top: 1px solid #dfe5dd; }
.academic-news li { grid-template-columns: 90px minmax(0, 1fr); gap: 22px; padding: 15px 4px; border-bottom: 1px solid #e4e9e2; }
.academic-news .academic-date { padding-top: 3px; color: #2a6b4e; font-family: var(--font-mono); }
.academic-news a { color: #34443a !important; }
.academic-news a:hover { color: #1f6b4f !important; }
.academic-publications { gap: 0; border-top: 1px solid #dfe5dd; }
.academic-paper { padding: 36px 0; border: 0; border-bottom: 1px solid #dfe5dd; border-radius: 0; background: transparent; box-shadow: none; }
.academic-paper:hover { transform: none; border-color: #dfe5dd; box-shadow: none; }
.academic-paper.with-image { grid-template-columns: minmax(0, .36fr) minmax(0, .64fr); gap: 36px; }
.academic-paper-figure { padding-top: 5px; }
.academic-paper-badge { top: 16px; left: 12px; padding: 4px 9px; border-radius: 6px; background: #1d5d43; box-shadow: none; font-family: var(--font-mono); font-size: 9px; }
.academic-paper-figure-link { border-color: #dce3db; border-radius: 12px; box-shadow: 0 12px 30px -24px rgb(31 53 40 / 55%); }
.academic-paper h3 { color: #17271f; font-size: 20px; line-height: 1.5; letter-spacing: -.02em; }
.academic-paper h3 a { color: inherit; }
.academic-paper p { font-size: 14px; }
.academic-authors { color: #5e6b63; }
.academic-authors strong { background: #e3ede5; color: #1d5d43; }
.academic-summary { color: #55645b; line-height: 1.9; }
.academic-resource-links { gap: 7px; margin-top: 18px; }
.academic-resource-links a { padding: 5px 11px; border-color: #d7e0d7; border-radius: 7px; background: transparent; font-size: 11px; }
.academic-resource-links a:first-child { border-color: #1d5d43; background: #1d5d43; color: #fff; }
.academic-resource-links a:hover { border-color: #8dac98; background: #e8efe8; color: #1d5d43; transform: none; }
.academic-project { padding: 26px 0; }
.academic-project:first-of-type { padding-top: 26px; border-top: 1px solid #dfe5dd; }
.academic-project h3 { font-size: 17px; letter-spacing: -.015em; }
.academic-project p { max-width: 94ch; margin-top: 8px; color: #536158; font-size: 14px; line-height: 1.9; }
.academic-project-meta { gap: 8px 14px; margin-top: 14px; }
.academic-project-tag { border-color: #dfe5dd; background: #edf1eb; color: #5c6960; }
.academic-entry { gap: 20px; margin: 0; padding: 24px 0; border: 0; border-bottom: 1px solid #dfe5dd; border-radius: 0; background: transparent; box-shadow: none; }
.academic-entry:first-of-type { border-top: 1px solid #dfe5dd; }
.academic-entry:hover { transform: none; border-color: #cbd6cc; }
.academic-entry > img { width: 48px; height: 48px; border-color: #dfe5dd; border-radius: 12px; }
.academic-entry h3 { font-size: 16px; }
.academic-entry h3 span { color: #7b877f; }
.academic-entry p { color: #3f5045; font-size: 14px; }
.academic-entry .academic-entry-description { color: #5b6960; font-size: 13.5px; line-height: 1.9; }
.academic-entry > .academic-date { padding: 4px 0; background: transparent; color: #66756b; font-family: var(--font-mono); font-size: 11px; }
.academic-updated { margin-top: 48px; font-family: var(--font-mono); }
@media (max-width: 800px) {
  .academic-identity { grid-template-columns: minmax(0, 1fr) 176px; gap: 30px; padding: 42px 38px; }
  .academic-portrait-wrap { width: 176px; border-radius: 23px; }
  .academic-portrait { width: 154px; height: 154px; border-radius: 15px; }
  .academic-portrait-caption { font-size: 6.5px; }
  .academic-hero-stats { gap: 24px; }
}
@media (max-width: 640px) {
  .academic-page { width: calc(100% - 32px); padding-top: 20px; font-size: 14.5px; }
  .academic-identity { display: grid; grid-template-columns: minmax(0, 1fr); gap: 0; padding: 28px 24px 30px; border-radius: 22px; }
  .academic-heading { display: contents; }
  .academic-kicker { max-width: calc(100% - 84px); font-size: 8px; }
  .academic-heading h1 { max-width: calc(100% - 80px); font-size: 32px; }
  .academic-heading h1 span { font-size: 17px; }
  .academic-portrait-wrap { position: absolute; top: 28px; right: 22px; width: 72px; margin: 0; padding: 5px; border-radius: 15px; }
  .academic-portrait { width: 60px; height: 60px; border-radius: 10px; }
  .academic-portrait-caption { display: none; }
  .academic-tagline { margin: 18px 0; font-size: 13.5px; }
  .academic-affiliation { font-size: 13px; }
  .academic-links { margin-top: 22px; }
  .academic-hero-stats { justify-content: space-between; gap: 12px; margin-top: 26px; padding-top: 18px; }
  .academic-hero-stats dd { font-size: 20px; }
  .academic-hero-stats dt { font-size: 7.5px; }
  .academic-sections { top: 8px; margin-top: 12px; padding: 5px; border-radius: 12px; }
  .academic-sections a { padding: 7px 11px; font-size: 11px; }
  .academic-section { margin-top: 48px; scroll-margin-top: 88px; }
  .academic-section h2 { gap: 8px; margin-bottom: 22px; font-size: 21px; }
  .academic-section h2::after { display: none; }
  .academic-section h2 > span { font-size: 11px; }
  .academic-interests { grid-template-columns: 1fr; }
  .academic-interests article { padding: 24px; }
  .academic-interests article + article { border-top: 1px solid #e4e9e2; border-left: 0; }
  .academic-news li { grid-template-columns: 66px minmax(0, 1fr); gap: 12px; padding: 14px 0; }
  .academic-paper { padding: 28px 0; }
  .academic-paper.with-image { grid-template-columns: minmax(0, 1fr); gap: 24px; }
  .academic-paper-figure { width: 100%; max-width: 480px; }
  .academic-paper h3 { font-size: 18px; }
  .academic-paper p { font-size: 13px; }
  .academic-project p { font-size: 13px; }
  .academic-entry { gap: 13px; padding: 22px 0; }
  .academic-entry > img { width: 44px; height: 44px; }
  .academic-entry.with-logo .academic-entry-copy { flex-basis: calc(100% - 57px); }
  .academic-entry > .academic-date { margin-left: 57px; }
  .academic-entry:not(.with-logo) > .academic-date { margin-left: 0; }
}
@media print {
  :global(body.profile-body) { background: #fff; }
  :global(.profile-header), :global(.profile-footer), .academic-sections, .academic-glow { display: none; }
  .academic-page { width: auto; padding: 0; color: #000; }
  .academic-identity { display: block; padding: 0 0 20px; border: 0; border-bottom: 2px solid #222; border-radius: 0; background: #fff; box-shadow: none; }
  .academic-identity::before, .academic-portrait-wrap { display: none; }
  .academic-heading h1, .academic-heading h1 span, .academic-tagline, .academic-affiliation, .academic-location,
  .academic-section h2, .academic-section h3, .academic-paper h3 { color: #000; }
  .academic-page .academic-link-pill, .academic-affiliation-pill { border-color: #aaa; background: transparent; color: #000; box-shadow: none; }
  .academic-hero-stats { padding-top: 12px; border-top-color: #ccc; }
  .academic-hero-stats dt, .academic-hero-stats dd { color: #000; }
  .academic-section { margin-top: 28px; break-inside: auto; }
  .academic-paper, .academic-entry { break-inside: avoid; box-shadow: none; }
  .academic-page a { color: inherit; text-decoration: underline; }
  @page { margin: 16mm; }
}
</style>
