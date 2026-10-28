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

export interface AcademicAward {
  title: string
  detail: string
  year?: string
}

// 学术主页的内容集中在这里修改；只填写已经确认、适合公开的信息。
// 空的邮箱、Scholar、简历链接不会显示，避免出现无效按钮。
export const academicProfile = {
  name: 'Enhao Gu',
  nameCN: '古恩豪',
  avatar: '/avatar.jpg',
  tagline: 'LLM Agents, Post-Training Alignment, Multimodal Generation & AI Systems.',
  affiliation: '华东师范大学 · 软件工程（硕士推免录取）',
  affiliationNote: '已保研录取，2027 年入学',
  location: '',
  email: '',
  scholar: 'https://scholar.google.com/citations?user=M6ht2eQAAAAJ&hl=zh-CN',
  cv: '',
  github: 'https://github.com/LiangQianXing',
  lastUpdated: '2026 年 10 月',
  bio: [
    '我是古恩豪（Enhao Gu），现就读于新疆大学软件工程专业（成绩前 10%，曾赴中南大学计算机学院交换学习），已获推免录取至华东师范大学软件工程攻读硕士学位（2027 年入学）。我的研究与工程工作聚焦于智能体系统（LLM Agents）、大模型后训练与偏好对齐（Post-Training & RLHF/GRPO/DPO）、多模态推理蒸馏与扩散生成加速，以及底层 AI 推理与训练系统（AI Infrastructure）。',
    '在学术研究方面，我曾在西湖大学自然语言处理实验室（张岳教授课题组）担任访问学生，负责科研自主智能体平台 DeepScientist（服务 7,000+ 用户）的核心研发，并合作完成可审计同行评审系统 DeepReviewer 2.0（ICLR 盲评基准 Elo 2057，对人类评审委员会胜率 71.6%）与可编辑科研插图系统 AutoFigure-Edit（ACL 2026 System Demonstrations，FigureBench 胜率 83.0%）。此前我在北京大学 IFlab 开展视频扩散模型（DiT）推理加速研究（EditCache），通过可重入 Latent 与 Block Hidden-State 跨步缓存复用及局部区域重算，在单卡 H100 上实现 2.46× 模型计算加速与 1.49× 严格同步端到端加速；并在多视角推理蒸馏（LLM-MRD，DASFAA 2026 Oral）与扩散模型自校正引导（In-situ Autoguidance，ICML 2025 Workshop 第一作者）上发表研究成果。',
    '在工业界与开源系统实践方面，我目前在美团海外酒旅业务研发平台担任 Agent 算法工程实习生，负责多约束 DAG 任务规划与局部重规划、两阶段工具路由（接口调用合法率 96%+）、上下文状态剪枝（Token 开销降低约 40%），以及基于持久化沙箱轨迹的 Rubrics 评测与 SFT/DPO 偏好对齐训练。同时，我是开源终端编程智能体 MiniCode（1k+ GitHub Stars）的核心开发者，并向结构化解码引擎 mlc-ai/xgrammar（PR #1011）等开源基础设施贡献底层并发修复。',
  ],
  interests: [
    {
      title: '智能体系统与长程规划',
      english: 'Agentic Systems & Planning',
      description: '多约束 DAG 任务分解与局部子图重规划、两阶段工具路由与结构化约束解码、超 100K 长上下文记忆压缩，以及可追溯、可审计的多智能体协同与沙箱闭环。',
    },
    {
      title: '大模型后训练、对齐与蒸馏',
      english: 'Post-Training, Alignment & Distillation',
      description: '可验证规则奖励强化学习（GRPO / RLVR）、生成式奖励模型（GRM / CycleReviewer）、轨迹级 SFT → DPO 偏好对齐与对齐税评测，以及跨模态多视角推理蒸馏（Reasoning Distillation）。',
    },
    {
      title: 'AI 推理系统与扩散加速',
      english: 'AI Infrastructure & Diffusion Acceleration',
      description: 'SGLang / vLLM 高并发推理调优（Prefix Caching / PagedAttention / Chunked Prefill）、FSDP2 分布式训练、CUDA / Triton 融合算子，以及视频 Diffusion Transformer（DiT）时空缓存复用。',
    },
  ],
  news: [
    {
      date: '2026.10',
      text: '开源贡献：定位并修复 mlc-ai/xgrammar 无限容量语法编译缓存并发 Clear 竞态缺陷（PR #1011），引入代际计数器与读写锁同步并通过全部 87 项 C++ GoogleTest。',
      url: 'https://github.com/mlc-ai/xgrammar/pull/1011',
    },
    {
      date: '2026.09',
      text: '预印本发布：可追溯、可审计的科学同行评审智能体系统 DeepReviewer 2.0 公开（arXiv:2604.09590），在 134 篇 ICLR 盲评基准上取得 Elo 2057（胜率 71.6%）。',
      url: 'https://arxiv.org/abs/2604.09590',
    },
    {
      date: '2026.06',
      text: '加入美团（Meituan）海外酒旅业务研发平台担任 Agent 算法工程实习生，负责多约束 DAG 规划、两阶段工具路由与轨迹级 SFT / DPO 偏好对齐。',
    },
    {
      date: '2026.05',
      text: '论文录用：AutoFigure-Edit 被 ACL 2026 System Demonstrations 录用，支持参考风格引导的原生 SVG 可编辑科研插图生成（FigureBench 胜率 83.0%）。',
      url: 'https://aclanthology.org/2026.acl-demo.6/',
    },
    {
      date: '2026.03',
      text: '论文录用：多模态多视角推理蒸馏框架 LLM-MRD 被 DASFAA 2026 录用为 Oral 报告（arXiv:2603.19293），将 7B VLM 推理能力压缩至 358M 模型（单样本延迟 17.2 ms）。',
      url: 'https://arxiv.org/abs/2603.19293',
    },
    {
      date: '2025.07',
      text: '论文录用：第一作者工作 In-situ Autoguidance 被 ICML 2025 Workshop 录用（arXiv:2510.17136），以单模型双前向构造扩散自引导，免除额外引导网络开销。',
      url: 'https://arxiv.org/abs/2510.17136',
    },
  ] as { date: string; text: string; url?: string }[],
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
      badge: 'ACL 2026 Demo',
      image: '/media/publications/autofigure-edit-overview.png',
      imageAlt: 'AutoFigure-Edit 的五阶段框架：风格生成、结构索引、素材提取、SVG 模板细化与素材注入。',
      imageSource: { label: 'Figure 1 · arXiv', url: 'https://arxiv.org/html/2603.06674v1/method_v1.png' },
      summary: '面向科研论文可编辑插图生成，设计“风格生成—结构索引—素材提取—SVG 模板精修—矢量注入”五阶段流水线，结合 MinerU v4 版面锚点解析与多模态参考风格控制生成原生可编辑 SVG，在 FigureBench 200 题人工盲评基准上取得 83.0% 胜率。',
      links: [
        { label: 'Paper', url: 'https://aclanthology.org/2026.acl-demo.6/' },
        { label: 'Code', url: 'https://github.com/ResearAI/AutoFigure-Edit' },
        { label: 'Demo', url: 'https://autofigure.cc/' },
        { label: 'Scholar', url: 'https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=M6ht2eQAAAAJ&citation_for_view=M6ht2eQAAAAJ:d1gkVwhDpl0C' },
      ],
    },
    {
      title: 'LLM-MRD: LLM-Guided Multi-View Reasoning Distillation for Fake News Detection',
      authors: [
        { name: 'Weilin Zhou' },
        { name: 'Shanwen Tan' },
        { name: 'Enhao Gu', self: true },
        { name: 'Yurong Qian' },
      ],
      venue: 'DASFAA (Oral Presentation) · arXiv:2603.19293',
      year: '2026',
      badge: 'DASFAA 2026 Oral',
      image: '/images/posts/llm-mrd-distillation-auditable-review-agents/llm-mrd-calibration-distillation-20261016.webp',
      imageAlt: 'LLM-MRD 多视角推理校准蒸馏与跨注意力融合架构图。',
      imageSource: { label: 'Architecture · Blog Deep Dive', url: '/posts/llm-mrd-distillation-auditable-review-agents' },
      summary: '利用 Qwen2.5-VL 离线生成文本操纵、图像伪影与图文一致性三视角推理链并编码为稠密语义监督；结合全局上下文 MLP 残差校准、温度缩放 KL 蒸馏与分支深监督，将 7B VLM 推理能力压缩至 358M 三路编码学生网络（BERT + MAE + CLIP），脱离教师独立推理延迟仅 17.2 ms，在 Weibo / Weibo-21 / GossipCop 取得 95.3% / 95.9% / 90.2% 准确率。',
      links: [
        { label: 'Paper', url: 'https://arxiv.org/abs/2603.19293' },
        { label: 'Code', url: 'https://github.com/Nasuro55/LLM-MRD' },
        { label: 'Blog', url: '/posts/llm-mrd-distillation-auditable-review-agents' },
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
      venue: 'arXiv preprint · arXiv:2604.09590 (Under Review)',
      year: '2026',
      badge: 'arXiv 2026',
      image: '/media/publications/deepreviewer-v2-overview.png',
      imageAlt: 'DeepReviewer 2.0 框架：论文解析与证据锚定、两阶段认知评审链、可追溯评审报告与批注。',
      imageSource: { label: 'Figure 2 · arXiv', url: 'https://arxiv.org/html/2604.09590v1/final.png' },
      summary: '面向可审计的科学同行评审，构建“主张—证据—风险”结构化记忆台账、PASA 文献检索可比性门控与导出预算校验，输出带 PDF 坐标锚定批注、局部证据与后续验证动作的可追溯评审报告；在 134 篇 ICLR 盲评基准上取得 Elo 2057（对人类评审委员会胜率 71.6%）。',
      links: [
        { label: 'Paper', url: 'https://arxiv.org/abs/2604.09590' },
        { label: 'Code', url: 'https://github.com/ResearAI/DeepReviewer-v2' },
        { label: 'Blog', url: '/posts/llm-mrd-distillation-auditable-review-agents' },
        { label: 'Scholar', url: 'https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=M6ht2eQAAAAJ&citation_for_view=M6ht2eQAAAAJ:qjMakFHDy7sC' },
      ],
    },
    {
      title: 'In-situ Autoguidance: Eliciting Self-Correction in Diffusion Models',
      authors: [
        { name: 'Enhao Gu', self: true },
        { name: 'Haolin Hou' },
      ],
      venue: 'ICML Workshop · arXiv:2510.17136',
      year: '2025',
      badge: 'ICML 2025 W',
      summary: '针对扩散模型 Autoguidance 需额外部署退化引导网络带来的双倍参数与显存开销，提出基于同一扩散模型确定性前向与推理期局部随机微扰（Stochastic Dropout Perturbation）的原位自引导机制，无需任何额外训练或辅助权重即可构造指向高密度数据流形的自校正引导向量，在 EDM2-S / ImageNet-512（50K 样本）上验证有效。',
      links: [
        { label: 'Paper', url: 'https://arxiv.org/abs/2510.17136' },
      ],
    },
  ] as Publication[],
  projects: [
    {
      name: 'DeepScientist · 科研自主智能体与容器沙箱平台（7,000+ 用户）',
      description: '面向全流程自主科学发现的 Agent 平台。集成 Python daemon、模块化 Skill 库与 MCP 协议栈（memory / artifact / bash_exec），支持超 100K Token 长上下文多轮规划、Copilot / Autonomous 双模续接与 JSONL 事件回放；基于 Docker SDK 动态编排隔离沙箱容器，结合 Redis、SSE / WebSocket 双流通信与 Prometheus 构建高并发运行时。',
      tags: ['LLM Agent', 'MCP Sandbox', 'Docker SDK', '100K+ Context', 'SSE / WebSocket'],
      links: [
        { label: 'GitHub', url: 'https://github.com/ResearAI/DeepScientist' },
        { label: 'Website', url: 'https://deepscientist.cc' },
        { label: '架构解析', url: '/posts/deepscientist-tech-stack' },
      ],
    },
    {
      name: 'MiniCode · 开源终端 Agentic Coding 引擎与评测基准（1k+ Stars）',
      description: '轻量级终端 AI 编程智能体核心开发。实现 ReAct 多步推理与工具编排循环（文件 Diff 编辑、Shell 执行、网络检索）、声明式工具注册表与统一权限拦截层、分层记忆压缩与 MCP 客户端自动协商；构建 30 项可执行 Tool-Use 基准并引入结构化错误回传，任务完成率达 86.7%、错误恢复率 100%（10/10）。',
      tags: ['TypeScript', 'Agentic Coding', 'ReAct', 'MCP', 'Tool-Use Benchmark'],
      links: [
        { label: 'GitHub', url: 'https://github.com/LiuMengxuan04/MiniCode' },
        { label: '架构与评测解析', url: '/posts/minicode-source-analysis' },
      ],
    },
    {
      name: 'mlc-ai/xgrammar · 结构化解码语法缓存并发竞态修复（PR #1011）',
      description: '定位并修复 XGrammar 无限容量语法编译缓存（Unlimited-Capacity Grammar Cache）在多线程并发执行 Get 与 Clear 时的字节计数（Byte Accounting）竞态覆盖缺陷。引入代际计数器（Generation Counter）与读写锁同步原语，编写确定性多线程并发 GoogleTest 并通过全部 87 项原生 C++ 测试。',
      tags: ['C++', 'Structured Decoding', 'Concurrency', 'XGrammar', 'Open Source'],
      links: [
        { label: 'PR #1011', url: 'https://github.com/mlc-ai/xgrammar/pull/1011' },
        { label: '技术长文', url: '/posts/xgrammar-concurrent-cache-generation-fix' },
      ],
    },
    {
      name: 'Qwen2.5-VL · SGLang 推理加速与单卡 FSDP2 可验证奖励 GRPO 训练栈',
      description: '在单卡 32 GB 显存下打通 Qwen2.5-VL-3B 多模态后训练与服务调优全链路：基于 verl 设计 JSON 格式门控 + 空间计数可验证规则奖励（Outcome Reward 0/0.5/1），冻结视觉塔以 FSDP2 + CPU Offload 完成 GRPO 更新与 704 次请求 Paired Bootstrap 评测；调优 SGLang RadixAttention 前缀缓存（TTFT p50 194→75 ms）、并发吞吐（3.47×）与 Chunked Prefill 显存池（峰值显存降低 20.3%）。',
      tags: ['Qwen2.5-VL', 'GRPO / verl', 'SGLang', 'FSDP2', 'Prefix Caching'],
      links: [
        { label: '实验报告', url: '/posts/qwen25-vl-sglang-verl-single-gpu-lab' },
      ],
    },
    {
      name: 'Fast-LLM-Kernels & LLM-Gateway · CUDA 融合算子与动态批处理推理网关',
      description: '底层算子与服务网关工程：基于 CUDA C++ 实现向量化 `float4` 访存 + Warp Shuffle + Shared Memory 归约的 Fused Residual-Add + RMSNorm 算子（RTX 4090D 热缓存 4.27× 加速，PyTorch C++ 扩展零拷贝绑定）；构建 FastAPI + Redis 语义缓存与 50 ms 动态批处理网关（精确/语义命中 P99 延迟 5307→17 ms / 21963→37 ms）。',
      tags: ['CUDA C++', 'Triton', 'Kernel Fusion', 'Dynamic Batching', 'Semantic Cache'],
      links: [
        { label: 'CUDA Kernels', url: 'https://github.com/LiangQianXing/fast-llm-kernels' },
        { label: 'LLM Gateway', url: 'https://github.com/LiangQianXing/llm-gateway' },
        { label: '算子解析', url: '/posts/cuda-fused-rmsnorm-kernel' },
      ],
    },
  ] as AcademicProject[],
  education: [
    {
      institution: '华东师范大学',
      english: 'East China Normal University (985)',
      role: '软件工程 · 硕士研究生（推免录取）',
      period: '2027.09 — 2030.06',
      description: '已通过全国优秀大学生夏令营 / 推免选拔录取，将于 2027 年秋季入学。',
      logo: '/logos/ecnu.png',
    },
    {
      institution: '新疆大学',
      english: 'Xinjiang University (211) · 中南大学（985）联合培养交换',
      role: '软件工程 · 工学学士在读（专业前 10%）',
      period: '2022.09 — 2027.06',
      description: '主修数据结构与算法、操作系统、计算机网络、编译原理与软件工程；曾赴中南大学计算机学院交流学习（获中南大学机器人校赛二等奖）。',
      logo: '/logos/xju.jpg',
    },
  ],
  experience: [
    {
      institution: '美团',
      english: 'Meituan',
      role: 'Agent 算法工程实习生 · 海外酒旅业务研发平台',
      period: '2026.06 — 至今',
      description: '面向跨境酒旅复杂多约束场景设计分层任务分解、DAG 拓扑调度与局部子图重规划（Local Replanning）；构建向量 Top-K 召回 + LLM 精排两阶段工具路由与 JSON Schema 强校验（线上接口调用合法率 96%+）；设计 Observation 状态剪枝策略降低约 40% 上下文 Token 开销；打通内部 AI Coding 持久化 Sandbox 与七阶段状态机，基于多步交互轨迹构建过程检查点与结果验收 Rubrics，开展 Pairwise 偏好数据构造与 SFT / DPO 对齐训练。',
      logo: '/logos/meituan.svg',
    },
    {
      institution: '西湖大学',
      english: 'Westlake University · NLP Lab (Prof. Yue Zhang)',
      role: '访问学生 · 自然语言处理实验室（张岳教授课题组）',
      period: '2025.12 — 2026.03',
      description: '负责科研自主智能体平台 DeepScientist（7,000+ 用户）核心研发，搭建超 100K Token 长上下文多轮规划、并发 Tool-use、Docker SDK 容器沙箱与 MCP 工具链（memory / artifact / bash_exec）；参与可审计同行评审系统 DeepReviewer 2.0（arXiv:2604.09590，ICLR 盲评基准 Elo 2057 / 胜率 71.6%）与可编辑科研插图系统 AutoFigure-Edit（ACL 2026 Demo，FigureBench 胜率 83.0%）研究。',
      logo: '/logos/westlake.png',
    },
    {
      institution: '北京大学',
      english: 'Peking University · IFlab',
      role: '科研实习生 · 视频扩散模型（DiT）推理加速',
      period: '2025.09 — 2025.12',
      description: '参与视频扩散模型局部编辑加速项目 EditCache（共同第一作者，面向 Wan2.1-VACE 多模态 Diffusion Transformer），设计可重入 Latent 与 Block Hidden-State 跨步缓存复用、局部区域重算与自适应推理起点（Adaptive Late Start）；在单卡 H100（33 帧 480×480）上实现模型计算延迟 978.23 → 398.07 ms（2.46× 加速）、严格同步端到端延迟 979.0 → 658.5 ms（1.49× 加速），并完成缓存传输开销与组件消融实验。',
      logo: '/logos/pku.png',
    },
  ],
  awards: [
    {
      title: 'ACM-ICPC 国际大学生程序设计竞赛',
      detail: '亚洲区域赛（成都站）铜牌、全国邀请赛（武汉站）银牌、新疆维吾尔自治区赛金牌（亚军）',
      year: '2024 – 2025',
    },
    {
      title: 'CCPC 中国大学生程序设计竞赛 & 百度之星程序设计大赛',
      detail: 'CCPC 全国邀请赛银牌、百度之星程序设计大赛省赛金牌（冠军）',
      year: '2024 – 2025',
    },
    {
      title: '机器人与高校计算机综合赛事',
      detail: '睿抗机器人开发者大赛（RAICOM）全国总决赛编程赛道国家一等奖、团体程序设计天梯赛全国一等奖、中南大学机器人校赛二等奖',
      year: '2024 – 2025',
    },
  ] as AcademicAward[],
}
