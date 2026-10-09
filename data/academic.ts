export interface AcademicLink {
  label: string
  url: string
}

export interface Publication {
  title: string
  authors: { name: string; self?: boolean }[]
  venue: string
  year: string
  badge?: string
  image?: string
  imageAlt?: string
  imageSource?: AcademicLink
  summary?: string
  links: AcademicLink[]
}

export interface AcademicProject {
  name: string
  description: string
  tags: string[]
  links: AcademicLink[]
}

// 学术主页的内容集中在这里修改；只填写已经确认、适合公开的信息。
// 空的邮箱、Scholar、简历链接不会显示，避免出现无效按钮。
export const academicProfile = {
  name: 'Enhao Gu',
  nameCN: '古恩豪',
  avatar: '/avatar.jpg',
  tagline: 'Language, intelligence & systems.',
  affiliation: '华东师范大学 · 软件工程',
  affiliationNote: '已保研录取，2027 年入学',
  location: '',
  email: '',
  scholar: 'https://scholar.google.com/citations?user=M6ht2eQAAAAJ&hl=zh-CN',
  cv: '',
  github: 'https://github.com/LiangQianXing',
  lastUpdated: '2026 年 10 月',
  bio: [
    '你好，我是古恩豪。我关注大语言模型、智能体与 AI 系统，也喜欢从工程实践出发，理解模型与系统如何协同工作。',
    '我已保研录取华东师范大学软件工程，将于 2027 年入学。此前曾在西湖大学自然语言处理实验室担任访问学生，目前在美团参与开发实习。',
    '研究之外，我会参加算法竞赛，也喜欢旅行。在博客中，我记录论文阅读、源码分析和工程实践中的思考。',
  ],
  interests: [
    { title: '大语言模型', english: 'Large Language Models', description: '模型原理、学习与推理，以及语言模型的实际应用。' },
    { title: '智能体', english: 'LLM Agents', description: '工具使用、记忆与上下文工程，以及智能体系统的构建。' },
    { title: 'AI 系统', english: 'AI Infrastructure', description: '推理服务、系统优化，以及模型与后端基础设施的结合。' },
  ],
  // 添加真实动态后，页面会自动显示 News 栏目。
  news: [] as { date: string; text: string; url?: string }[],
  publications: [
    {
      title: 'AutoFigure-Edit: Generating Editable Scientific Illustrations via Reference-Guided Styling',
      authors: [
        { name: 'Zhen Lin' },
        { name: 'Qiujie Xie' },
        { name: 'Minjun Zhu' },
        { name: 'Shichen Li' },
        { name: 'QiYao Sun' },
        { name: 'Enhao Gu', self: true },
        { name: 'Yiran Ding' },
        { name: 'Ke Sun' },
        { name: 'Fang Guo' },
        { name: 'Panzhong Lu' },
        { name: 'Zhiyuan Ning' },
        { name: 'Yixuan Weng' },
        { name: 'Yue Zhang' },
      ],
      venue: 'ACL (System Demonstrations)',
      year: '2026',
      badge: 'ACL 2026',
      image: '/media/publications/autofigure-edit-overview.png',
      imageAlt: 'AutoFigure-Edit 的五阶段框架：风格生成、结构索引、素材提取、SVG 模板细化与素材注入。',
      imageSource: { label: 'Figure 1 · arXiv', url: 'https://arxiv.org/html/2603.06674v1/method_v1.png' },
      summary: '结合长文本理解、参考图风格控制与原生 SVG 编辑，生成可编辑的科研插图。',
      links: [
        { label: 'Paper', url: 'https://aclanthology.org/2026.acl-demo.6/' },
        { label: 'Code', url: 'https://github.com/ResearAI/AutoFigure-Edit' },
        { label: 'Demo', url: 'https://autofigure.cc/' },
        { label: 'Scholar', url: 'https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=M6ht2eQAAAAJ&citation_for_view=M6ht2eQAAAAJ:d1gkVwhDpl0C' },
      ],
    },
    {
      title: 'DeepReviewer 2.0: A Traceable Agentic System for Auditable Scientific Peer Review',
      authors: [
        { name: 'Yixuan Weng' },
        { name: 'Minjun Zhu' },
        { name: 'Qiujie Xie' },
        { name: 'Zhiyuan Ning' },
        { name: 'Shichen Li' },
        { name: 'Panzhong Lu' },
        { name: 'Zhen Lin' },
        { name: 'Enhao Gu', self: true },
        { name: 'Qiyao Sun' },
        { name: 'Yue Zhang' },
      ],
      venue: 'arXiv preprint · arXiv:2604.09590',
      year: '2026',
      badge: 'arXiv 2026',
      image: '/media/publications/deepreviewer-v2-overview.png',
      imageAlt: 'DeepReviewer 2.0 框架：论文解析与证据锚定、两阶段认知评审链、可追溯评审报告与批注。',
      imageSource: { label: 'Figure 2 · arXiv', url: 'https://arxiv.org/html/2604.09590v1/final.png' },
      summary: '面向可审计的科学同行评审，生成包含锚定批注、局部证据与后续验证动作的可追溯评审结果。',
      links: [
        { label: 'Paper', url: 'https://arxiv.org/abs/2604.09590' },
        { label: 'Code', url: 'https://github.com/ResearAI/DeepReviewer-v2' },
        { label: 'Scholar', url: 'https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=M6ht2eQAAAAJ&citation_for_view=M6ht2eQAAAAJ:qjMakFHDy7sC' },
      ],
    },
  ] as Publication[],
  projects: [] as AcademicProject[],
  education: [
    {
      institution: '华东师范大学',
      english: 'East China Normal University',
      role: '软件工程 · 已保研录取',
      period: '2027 入学',
      description: '即将入学。',
      logo: '/logos/ecnu.png',
    },
  ],
  experience: [
    {
      institution: '美团',
      english: 'Meituan',
      role: '暑期开发实习生',
      period: '2026.06 — 至今',
      description: '参与内部平台开发，涉及 LLM 相关工程与后端服务。',
      logo: '/logos/meituan.svg',
    },
    {
      institution: '西湖大学',
      english: 'Westlake University',
      role: '访问学生 · 自然语言处理实验室',
      period: '2025.12 — 2026.03',
      description: '',
      logo: '/logos/westlake.png',
    },
  ],
}
