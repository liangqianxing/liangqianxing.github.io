---
title: DeepScientist 技术栈全解析：一个 AI 科研平台的架构设计
date: 2026-04-06
categories: 技术
cover: /images/posts/deepscientist.jpg
tags:
  - 全栈
  - FastAPI
  - Next.js
  - PostgreSQL
  - 面试
---

DeepScientist 是一个 AI 驱动的科研管理平台，用户 5000+。这篇文章梳理整个项目的技术选型和架构设计，作为面试准备的参考。

<!-- more -->

## 整体架构

```
┌─────────────────────────────────────────┐
│           Next.js 前端 (port 1288)       │
│   React 18 + TypeScript + Tailwind CSS  │
└──────────────────┬──────────────────────┘
                   │ HTTP / WebSocket
┌──────────────────▼──────────────────────┐
│          FastAPI 后端 (port 18080)       │
│        Python 3.11 + Uvicorn/Gunicorn   │
└──────┬───────────┬────────────┬─────────┘
       │           │            │
┌──────▼──┐  ┌─────▼────┐  ┌───▼──────────┐
│PostgreSQL│  │  MinIO   │  │  Sandbox     │
│  (数据库) │  │ (对象存储) │  │ (AI 沙箱容器) │
└─────────┘  └──────────┘  └─────────────┘
```

Monorepo 结构，前后端分离，全部通过 Docker Compose 编排。

## 前端技术栈

### 核心框架

| 技术 | 版本 | 用途 |
|---|---|---|
| Next.js | 15.0 | SSR/SSG 框架，App Router |
| React | 18.3 | UI 框架 |
| TypeScript | 5.3 | 类型安全 |

Next.js 15 使用 App Router，支持 Server Components，减少客户端 JS 体积。

### UI 与样式

- **Tailwind CSS** — 原子化 CSS，快速构建响应式布局
- **Radix UI** — 无样式的无障碍组件库（Dialog、Dropdown、Tooltip 等）
- **shadcn/ui** — 基于 Radix UI 的组件集合，可直接复制到项目里修改
- **Framer Motion** — 动画库，处理页面过渡和交互动效
- **GSAP** — 复杂时间轴动画

### 编辑器

这是项目的核心功能之一，用了两套编辑器：

**Tiptap 2** — 富文本编辑器
- 基于 ProseMirror，支持协作编辑
- 扩展：表格、代码块、数学公式（KaTeX）、图片上传
- 配合 Yjs 实现多人实时协同

**Monaco Editor** — 代码编辑器
- VS Code 同款编辑器内核
- 支持语法高亮、代码补全、多语言

### 实时协作

```
用户 A ──┐
         ├── Yjs CRDT ── Socket.IO ── 后端广播 ── 用户 B
用户 C ──┘
```

- **Yjs** — CRDT（无冲突复制数据类型）算法，多人同时编辑不冲突
- **Socket.IO Client** — WebSocket 封装，处理断线重连

### 数据可视化

- **Recharts** — 基于 D3 的 React 图表库（折线图、柱状图、饼图）
- **React Flow** — 节点图/流程图，用于知识图谱展示
- **Dagre** — 有向图自动布局算法

### 终端与远程桌面

- **XTerm.js** — 浏览器内终端模拟器，支持 WebGL 渲染加速
- **noVNC** — 纯 JS 的 VNC 客户端，可在浏览器里操作远程桌面

这两个组件支撑了 AI 沙箱的交互界面——用户可以直接在网页里操作 AI 运行的容器环境。

### 状态管理

| 库 | 用途 |
|---|---|
| Zustand | 全局客户端状态（用户信息、UI 状态） |
| Jotai | 原子化状态，细粒度订阅 |
| TanStack Query | 服务端状态，缓存 + 自动重新请求 |

三者分工明确：Zustand 管全局，Jotai 管局部，TanStack Query 管异步数据。

### PDF 处理

- **PDF.js** — Mozilla 出品，浏览器内渲染 PDF
- **UnPDF** — PDF 文本提取，供 AI 分析

## 后端技术栈

### 核心框架

**FastAPI** — Python 异步 Web 框架
- 基于 Pydantic 的自动参数校验
- 自动生成 OpenAPI 文档（/docs）
- 原生支持 async/await

**Uvicorn + Gunicorn** — 生产部署组合
- Uvicorn：ASGI 服务器，处理异步请求
- Gunicorn：进程管理，多 worker 提升并发

### 数据库层

```python
# SQLAlchemy 2.0 async 写法示例
async with AsyncSession(engine) as session:
    result = await session.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
```

| 技术 | 版本 | 用途 |
|---|---|---|
| PostgreSQL | 16 | 主数据库 |
| SQLAlchemy | 2.0 | ORM，async 模式 |
| AsyncPG | 0.29 | 异步 PostgreSQL 驱动 |
| Alembic | 1.13 | 数据库迁移 |

**为什么用 async？**
FastAPI 是 ASGI 框架，同步数据库驱动会阻塞事件循环。AsyncPG 是纯异步驱动，数据库 I/O 不阻塞，高并发下吞吐量显著提升。

**Alembic 迁移流程：**
```bash
# 修改 SQLAlchemy Model 后
alembic revision --autogenerate -m "add paper table"
alembic upgrade head
```

### 文件存储

**MinIO** — S3 兼容的自托管对象存储

架构原则：数据库只存元数据（文件名、路径、大小），二进制文件（PDF、图片、PPT）全部存 MinIO。

```
上传文件 → FastAPI → MinIO（存文件）
                  → PostgreSQL（存路径）

下载文件 → FastAPI → PostgreSQL（查路径）
                  → MinIO（取文件）→ 返回客户端
```

同时接入了 Boto3（AWS S3 SDK），可以无缝切换到 AWS S3。

### 认证与安全

- **JWT（python-jose）** — 无状态 Token，适合前后端分离
- **BCrypt** — 密码哈希，不可逆
- **Google OAuth 2.0** — 第三方登录
- **Pydantic** — 所有入参自动校验，防止非法数据进入业务层

### AI 集成

- **Google Gemini API** — 文献分析、内容生成
- **TikToken** — Token 计数，控制 LLM 请求成本
- **Docker SDK** — 动态创建/销毁 AI 沙箱容器

### 文档处理

| 库 | 用途 |
|---|---|
| PyMuPDF | PDF 解析、文本提取、页面渲染 |
| python-pptx | 生成 PowerPoint 导出 |
| CairoSVG | SVG 转 PNG/PDF |
| markdown-it-py | Markdown 渲染 |

### 监控

**Prometheus Client** — 暴露 `/metrics` 端点，记录请求数、延迟、错误率等指标，配合 Grafana 可视化。

## 基础设施

### Docker Compose 服务编排

```yaml
services:
  frontend:   # Next.js，port 1288
  backend:    # FastAPI，port 18080
  postgres:   # PostgreSQL 16，port 5432
  minio:      # 对象存储，port 9000/9001
  sandbox:    # AI 沙箱，port 15900-15901（VNC）
```

所有服务在同一个 `deepscientist-network` bridge 网络内，服务间通过容器名互相访问。

### 测试

- **Jest + Testing Library** — 前端单元测试 / 组件测试
- **Playwright** — 端到端测试，模拟真实用户操作
- **Pytest** — 后端单元测试 / 集成测试

## 面试常见问题

**Q：为什么选 FastAPI 而不是 Django/Flask？**

FastAPI 原生 async，性能接近 Node.js。自动生成 OpenAPI 文档省去大量手写工作。Pydantic 的类型校验比 Django REST Framework 的 Serializer 更简洁。适合 I/O 密集型的 API 服务。

**Q：Yjs 的 CRDT 是什么原理？**

CRDT（Conflict-free Replicated Data Type）是一种数据结构，多个节点可以独立修改，合并时保证最终一致性，不需要锁或中心协调。Yjs 用 CRDT 实现文档协同，即使网络断开离线编辑，重连后也能自动合并，不会丢失任何人的修改。

**Q：MinIO 和直接用数据库存文件有什么区别？**

数据库存二进制文件会导致：表体积膨胀、备份困难、无法利用 CDN 加速。对象存储专门为大文件设计，支持分片上传、断点续传、直接生成预签名 URL 让客户端直传，绕过后端减少带宽压力。

**Q：JWT 和 Session 的区别？**

Session 把状态存服务端（内存或 Redis），JWT 把状态编码在 Token 里由客户端持有。JWT 无状态，适合分布式部署，不需要共享 Session 存储。缺点是 Token 签发后无法主动失效，需要配合短过期时间 + Refresh Token 机制。

**Q：SQLAlchemy async 和同步有什么区别？**

同步模式下每次数据库查询都会阻塞当前线程，FastAPI 的事件循环被占用，无法处理其他请求。async 模式下查询变成协程，等待数据库响应时事件循环可以去处理其他请求，相同硬件下并发能力大幅提升。
