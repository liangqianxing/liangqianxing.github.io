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
const sections = computed(() => [
  { id: 'about', label: 'About' },
  { id: 'research', label: 'Research' },
  ...(profile.news.length ? [{ id: 'news', label: 'News' }] : []),
  { id: 'publications', label: 'Publications' },
  ...(profile.projects.length ? [{ id: 'projects', label: 'Projects' }] : []),
  { id: 'education', label: 'Education' },
  { id: 'experience', label: 'Experience' },
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
      </div>
      <div class="academic-portrait-wrap">
        <SiteImage class="academic-portrait" :src="profile.avatar" :alt="`${profile.nameCN}的头像`" width="156" height="156" priority />
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

    <section v-if="profile.projects.length" id="projects" class="academic-section" aria-labelledby="projects-heading">
      <h2 id="projects-heading">Selected Projects <span>项目</span></h2>
      <article v-for="project in profile.projects" :key="project.name" class="academic-project">
        <h3>{{ project.name }}</h3>
        <p>{{ project.description }}</p>
        <div class="academic-project-meta">
          <span v-for="tag in project.tags" :key="tag" class="academic-project-tag">{{ tag }}</span>
          <a v-for="link in project.links" :key="link.label" :href="link.url" :target="isExternal(link.url) ? '_blank' : undefined" :rel="isExternal(link.url) ? 'noopener noreferrer' : undefined">{{ link.label }} <span aria-hidden="true">↗</span></a>
        </div>
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

    <section id="experience" class="academic-section" aria-labelledby="experience-heading">
      <h2 id="experience-heading">Experience <span>经历</span></h2>
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

    <p class="academic-updated">Last updated · {{ profile.lastUpdated }}</p>
  </div>
</template>

<style scoped>
.academic-page { --academic-muted-strong: #5f6d64; position: relative; width: min(960px, calc(100% - 48px)); margin: 0 auto; padding: 52px 0 44px; color: #303b35; font-size: 15px; line-height: 1.9; }
.academic-glow { position: absolute; top: -20px; left: -10%; right: -10%; height: 340px; z-index: -1; pointer-events: none; background: radial-gradient(ellipse at 25% 20%, rgba(40, 91, 69, .08), transparent 62%), radial-gradient(ellipse at 82% 35%, rgba(114, 158, 132, .11), transparent 60%); filter: blur(18px); }
.academic-identity { display: flex; justify-content: space-between; align-items: flex-start; gap: 48px; padding-bottom: 38px; animation: academic-fade-up 520ms cubic-bezier(.16, 1, .3, 1) both; }
.academic-heading { min-width: 0; }
.academic-kicker { display: inline-flex; align-items: center; gap: 8px; margin: 0 0 12px; font-size: 10px; font-weight: 650; letter-spacing: .17em; color: #54665c; }
.academic-kicker-dot { width: 6px; height: 6px; border-radius: 50%; background: #285b45; box-shadow: 0 0 0 4px rgba(40, 91, 69, .14); }
.academic-heading h1 { margin: 0; color: #1c2621; font-size: 38px; font-weight: 650; letter-spacing: -.045em; line-height: 1.3; }
.academic-heading h1 span { margin-left: 12px; font-size: 22px; font-weight: 450; letter-spacing: .025em; color: #3c4d44; }
.academic-tagline { margin: 12px 0 18px; color: #5d6a63; font-size: 15px; font-family: Georgia, serif; font-style: italic; }
.academic-affiliation-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; }
.academic-affiliation { margin: 0; font-weight: 600; color: #202824; }
.academic-affiliation-pill { display: inline-flex; align-items: center; padding: 2px 10px; border-radius: 999px; background: #e6efe9; border: 1px solid #cce0d2; color: #24533e; font-size: 12px; font-weight: 550; }
.academic-location { margin: 4px 0 0; font-size: 13px; color: #69756e; }
.academic-links { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
.academic-page a { color: #285b45; text-decoration: none; text-underline-offset: 4px; }
.academic-page a:hover { text-decoration: underline; }
.academic-page a:focus-visible { outline: 2px solid #285b45; outline-offset: 4px; border-radius: 3px; }
.academic-link-pill { display: inline-flex; align-items: center; gap: 5px; padding: 6px 14px; border: 1px solid #d6e0d8; border-radius: 999px; background: #ffffff; color: #24533e; font-size: 12.5px; font-weight: 550; line-height: 1.4; box-shadow: 0 2px 8px rgba(32, 40, 36, .04); transition: transform 200ms cubic-bezier(.22, 1, .36, 1), border-color 200ms ease, background-color 200ms ease, box-shadow 200ms ease; }
.academic-link-pill:hover { text-decoration: none !important; transform: translateY(-2px); border-color: #8eb09b; background: #f4f8f5; box-shadow: 0 8px 18px rgba(40, 91, 69, .1); }
.academic-portrait-wrap { position: relative; flex-shrink: 0; margin-top: 16px; padding: 5px; border-radius: 50%; background: linear-gradient(135deg, #d7e5dc, #ffffff); box-shadow: 0 14px 36px rgba(32, 45, 38, .1), 0 0 0 1px #dce5de; transition: transform 300ms cubic-bezier(.22, 1, .36, 1); }
.academic-portrait-wrap:hover { transform: translateY(-3px) rotate(2deg); }
.academic-portrait { display: block; width: 152px; height: 152px; border-radius: 50%; object-fit: cover; background: #e8ede6; }
.academic-sections { position: sticky; top: 0; z-index: 40; display: flex; flex-wrap: wrap; gap: 10px 24px; border-top: 1px solid #dfe5de; border-bottom: 1px solid #dfe5de; padding: 13px 0; background: rgb(250 250 248 / 92%); box-shadow: 0 4px 20px rgb(32 40 36 / 4%); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }
.academic-sections a { position: relative; padding: 3px 0; color: var(--academic-muted-strong); font-size: 12.5px; font-weight: 550; transition: color 180ms ease; }
.academic-sections a:hover { color: #285b45; text-decoration: none; }
.academic-section { margin-top: 48px; scroll-margin-top: 88px; }
.academic-section h2 { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px; margin: 0 0 22px; color: #202824; font-size: 21px; font-weight: 600; letter-spacing: -.025em; line-height: 1.4; }
.academic-section h2 > span { color: var(--academic-muted-strong); font-size: 12px; font-weight: 400; letter-spacing: .025em; }
.academic-section > p { max-width: 72ch; margin: 0 0 14px; }
.academic-section h3 { margin: 0; font-size: 15px; font-weight: 600; line-height: 1.6; color: #27362e; }
.academic-interests { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; }
.academic-interests article { padding: 20px 22px; border: 1px solid #e1e8e2; border-left: 3px solid #285b45; border-radius: 14px; background: #ffffff; box-shadow: 0 6px 20px rgba(32, 40, 36, .035); transition: transform 240ms cubic-bezier(.22, 1, .36, 1), box-shadow 240ms ease, border-color 240ms ease; }
.academic-interests article:hover { transform: translateY(-3px); border-color: #b7cebf; border-left-color: #285b45; box-shadow: 0 14px 30px rgba(40, 91, 69, .08); }
.academic-interests h3 { font-size: 14.5px; }
.academic-interests article > span { display: inline-block; margin-top: 2px; color: #285b45; font-size: 12px; font-weight: 500; }
.academic-interests p { margin: 8px 0 0; color: #65716b; font-size: 13px; line-height: 1.8; }
.academic-empty { color: #69756e; font-size: 14px; padding: 16px 0; border-top: 1px solid #e0e5df; border-bottom: 1px solid #e0e5df; }
.academic-news { list-style: none; margin: 0; padding: 0; }
.academic-news li { display: grid; grid-template-columns: 100px minmax(0, 1fr); gap: 16px; padding: 6px 0; }
.academic-news p { margin: 0; }
.academic-date { color: var(--academic-muted-strong); font-size: 12px; white-space: nowrap; font-variant-numeric: tabular-nums; }
.academic-publications { display: grid; gap: 22px; }
.academic-paper { padding: 28px; border: 1px solid #e1e8e2; border-radius: 18px; background: #ffffff; box-shadow: 0 8px 26px rgba(32, 40, 36, .04); transition: transform 260ms cubic-bezier(.22, 1, .36, 1), box-shadow 260ms ease, border-color 260ms ease; }
.academic-paper:hover { transform: translateY(-3px); border-color: #b7cebf; box-shadow: 0 16px 38px rgba(40, 91, 69, .09); }
.academic-paper.with-image { display: grid; grid-template-columns: minmax(0, .38fr) minmax(0, .62fr); align-items: start; gap: 30px; }
.academic-paper-figure { position: relative; min-width: 0; margin: 0; }
.academic-paper-badge { position: absolute; top: -12px; left: 10px; z-index: 1; padding: 4px 11px; border-radius: 999px; background: #285b45; color: #fff; font-size: 11px; font-weight: 650; letter-spacing: .04em; line-height: 1.4; box-shadow: 0 4px 12px rgba(40, 91, 69, .26); }
.academic-paper-badge-inline { position: static; display: inline-block; margin-bottom: 10px; }
.academic-paper-figure-link { display: block; overflow: hidden; border: 1px solid #e0e5df; border-radius: 10px; background: #fff; box-shadow: 0 4px 14px rgb(32 40 36 / 6%); cursor: zoom-in; transition: border-color 220ms ease, box-shadow 220ms ease; }
.academic-paper-figure-link > img { display: block; width: 100%; height: auto; object-fit: contain; border-radius: 9px; transition: transform 360ms cubic-bezier(.22, 1, .36, 1); }
.academic-paper-figure-link:hover { border-color: #8eb09b; box-shadow: 0 8px 22px rgba(40, 91, 69, .12); }
.academic-paper-figure-link:hover > img { transform: scale(1.025); }
.academic-paper-caption { margin-top: 8px; font-size: 11px; line-height: 1.6; text-align: center; }
.academic-paper-caption a { color: var(--academic-muted-strong); }
.academic-paper-copy { min-width: 0; }
.academic-paper h3 { color: #202824; font-size: 17.5px; line-height: 1.5; overflow-wrap: anywhere; }
.academic-paper p { margin: 8px 0; font-size: 13px; }
.academic-authors { color: #65716b; line-height: 1.8; }
.academic-authors strong { color: #1f4d39; font-weight: 650; background: #ebf3ee; padding: 1px 6px; border-radius: 4px; }
.academic-venue { color: var(--academic-muted-strong); font-style: italic; }
.academic-summary { max-width: 72ch; color: #58655e; line-height: 1.85; }
.academic-resource-links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; font-size: 12px; font-weight: 550; }
.academic-resource-links a { display: inline-flex; align-items: center; gap: 4px; padding: 4px 11px; border: 1px solid #d8e2da; border-radius: 999px; background: #f8faf8; color: #285b45; transition: background-color 180ms ease, border-color 180ms ease, transform 180ms ease; }
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
.academic-project { padding: 16px 0; border-bottom: 1px solid #e0e5df; }
.academic-project:first-of-type { padding-top: 0; }
.academic-project p { margin: 5px 0; font-size: 13px; color: #65716b; }
.academic-project-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 10px; font-size: 12px; }
.academic-project-tag { padding: 1px 7px; border: 1px solid #e0e5df; border-radius: 4px; font-size: 11px; color: #65716b; }
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
@media print {
  :global(body.profile-body) { background: #fff; }
  :global(.profile-header), :global(.profile-footer), .academic-sections, .academic-glow { display: none; }
  .academic-page { width: auto; padding: 0; color: #000; }
  @page { margin: 16mm; }
  .academic-section, .academic-paper, .academic-entry { break-inside: avoid; box-shadow: none; }
  .academic-page a { color: inherit; text-decoration: underline; }
}
</style>
