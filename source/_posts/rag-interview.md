---
title: RAG 面经：从原理到工程落地的高频问题整理
date: 2026-04-13
categories: 技术
tags:
  - 面试
  - RAG
  - LLM
  - 向量数据库
---

RAG（Retrieval-Augmented Generation，检索增强生成）是目前 AI 应用面试中出现频率最高的技术点之一。这篇文章基于 DeepScientist 项目的实际落地经验，整理面试中关于 RAG 的高频问题和回答思路。

<!-- more -->

## RAG 是什么，为什么需要它

大模型有两个核心局限：**知识截止日期**和**上下文窗口限制**。

直接把整个知识库塞进 prompt 不现实——成本高、速度慢、超出窗口。RAG 的思路是：**先检索相关内容，再把检索结果交给模型生成答案**。

```
用户提问
   ↓
向量检索（从知识库找最相关的 N 个片段）
   ↓
构建 prompt（问题 + 检索到的片段）
   ↓
大模型生成答案
```

这样模型只需要处理少量高度相关的内容，既省 token，又能基于真实文档回答，减少幻觉。

---

## RAG 的完整流程

分两个阶段：**离线索引**和**在线检索**。

### 离线阶段（数据准备）

```
原始文档（PDF/Word/Markdown）
   ↓ 1. 文档加载
解析为纯文本
   ↓ 2. 文本分块（Chunking）
切成若干文本片段
   ↓ 3. Embedding（向量化）
每个片段 → 高维向量
   ↓ 4. 存储
向量 + 原文 → 向量数据库
```

### 在线阶段（回答生成）

```
用户提问
   ↓ 1. 问题向量化
问题 → 向量
   ↓ 2. 相似度检索
在向量数据库中找 Top-K 最相似片段
   ↓ 3. 重排（可选）
用 Reranker 对召回结果精排
   ↓ 4. 构建 prompt
系统提示 + 参考文档 + 用户问题
   ↓ 5. 大模型生成
输出最终答案
```

---

## 高频面试题

### 文档分块策略怎么选，chunk size 怎么定

分块是 RAG 效果的基础，策略选错后面怎么调都没用。

**常见策略：**

| 策略 | 适用场景 | 优缺点 |
|------|---------|--------|
| 固定长度（512/1024 token） | 通用场景 | 简单，但可能切断语义 |
| 按段落/标题切分 | 结构化文档（技术文档、手册） | 语义完整，推荐 |
| 滑动窗口（有重叠） | 内容连续性强的文档 | 避免边界信息丢失，但冗余 |
| 递归字符切分 | 混合格式文档 | 灵活，LangChain 默认方案 |

**chunk size 怎么定：**

没有万能答案，需要实验。我的做法是：

1. 从历史问答中整理 50-100 个测试问题，标注每个问题对应的标准文档片段
2. 用不同 chunk size（256/512/1024）分别建索引，跑测试集
3. 看两个指标：**召回率**（标准片段是否在 Top-K 里）和**精确率**（Top-1 是否最相关）
4. 选召回率和精确率最优的组合

在 DeepScientist 里，按一级标题切分 + Top-3 召回效果最好。原因是科研文档结构清晰，按标题切分语义完整，不会把一个方法的描述切成两半。

---

### Embedding 模型怎么选

Embedding 模型决定了向量空间的质量，直接影响检索效果。

**评估标准：** 看 [MTEB 榜单](https://huggingface.co/spaces/mteb/leaderboard)，这是业界公认的 embedding 评测基准，覆盖检索、分类、聚类等多个任务。

**中文场景推荐：**

- `text-embedding-v4`（阿里）：MTEB 中文榜前列，国内部署无网络问题
- `bge-m3`（智源）：开源，支持多语言，可本地部署
- `text-embedding-3-large`（OpenAI）：英文效果好，中文一般

**选型原则：**

- 国内生产环境：优先阿里/智源，避免网络依赖
- 需要本地部署：bge 系列
- 英文为主：OpenAI 或 Cohere

---

### 向量数据库怎么选

| 数据库 | 特点 | 适用场景 |
|--------|------|---------|
| Milvus | 专用向量库，高性能，42k star | 生产环境，大规模数据 |
| pgvector | PostgreSQL 插件，SQL 兼容 | 已有 PG 基础设施，数据量中等 |
| Chroma | 轻量，纯 Python | 原型开发，本地测试 |
| Weaviate | 内置 BM25+向量混合检索 | 需要混合检索的场景 |
| Qdrant | Rust 实现，低内存占用 | 资源受限环境 |

**实际选型考虑：**

- 数据量 < 100 万：pgvector 够用，省去维护独立向量库的成本
- 数据量 > 100 万或高并发：Milvus/Qdrant
- 快速验证：Chroma，5 行代码跑起来

---

### 召回效果不好怎么优化

召回效果差通常有几个原因，逐一排查：

**1. 文档质量问题（最常见）**

文档写得乱、语义分散，是召回差的最大原因。解决方案：清洗文档，规范写法，同一个问题的描述集中在一个段落。

**2. 分块策略不合适**

chunk 太小：语义不完整，模型拿到片段无法回答。
chunk 太大：相似度被稀释，不相关内容混入。

调整分块策略，重新建索引测试。

**3. Embedding 模型不匹配**

专业领域（医疗、法律、代码）用通用 embedding 效果差。考虑用领域数据 fine-tune embedding 模型，或换专门的领域模型。

**4. 只用向量检索，缺少关键词匹配**

向量检索擅长语义相似，但对精确关键词（错误码、API 名称、专有名词）效果差。

解决方案：**混合检索（Hybrid Search）**，向量检索 + BM25 关键词检索，用 RRF（Reciprocal Rank Fusion）融合两路结果：

```python
def hybrid_search(query: str, top_k: int = 10):
    # 向量检索
    vector_results = vector_db.search(embed(query), top_k=top_k)
    
    # BM25 关键词检索
    bm25_results = bm25_index.search(query, top_k=top_k)
    
    # RRF 融合
    scores = {}
    for rank, doc in enumerate(vector_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (60 + rank)
    for rank, doc in enumerate(bm25_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (60 + rank)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

**5. 缺少 Reranker**

向量检索的 Top-K 结果顺序不一定准确。加一个 Reranker（交叉编码器）对召回结果精排，能显著提升 Top-1 准确率：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query: str, candidates: list[str]) -> list[str]:
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)
    return [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]
```

---

### 相似度阈值怎么设

阈值太低：召回不相关内容，模型被干扰，容易幻觉。
阈值太高：召回为空，模型无法回答。

**我的做法：**

用测试集跑不同阈值（0.5/0.6/0.7/0.75/0.8），画出召回率-精确率曲线，选拐点。

在 DeepScientist 里设的是 0.75。低于这个值直接返回"知识库中暂无相关信息"，不让模型凭空猜测。

```python
results = vector_db.search(query_vector, top_k=3)
filtered = [r for r in results if r.score >= 0.75]

if not filtered:
    return "抱歉，知识库中暂无相关信息，建议直接查阅官方文档。"
```

---

### 文档更新了怎么处理，增量索引怎么做

**方案：维护文档元数据表**

```sql
CREATE TABLE doc_index (
    doc_id      VARCHAR PRIMARY KEY,
    file_path   VARCHAR,
    file_hash   VARCHAR,   -- 文件内容 hash，用于判断是否变更
    indexed_at  TIMESTAMP,
    status      VARCHAR    -- pending / indexed / failed
);
```

**增量更新流程：**

```python
def sync_documents(doc_dir: str):
    for file in list_files(doc_dir):
        current_hash = md5(file)
        record = db.get(file.path)
        
        if record and record.file_hash == current_hash:
            continue  # 未变更，跳过
        
        # 删除旧向量
        if record:
            vector_db.delete(filter=f"doc_id == '{record.doc_id}'")
        
        # 重新索引
        chunks = split(load(file))
        vectors = embed(chunks)
        vector_db.insert(vectors, metadata={"doc_id": file.id})
        
        # 更新元数据
        db.upsert(doc_id=file.id, file_hash=current_hash, status="indexed")
```

整个过程异步执行，不影响在线查询。

---

### RAG 和 Fine-tuning 怎么选

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| 知识更新 | 实时，改文档即生效 | 需要重新训练 |
| 成本 | 低（只需向量化） | 高（GPU 训练） |
| 可解释性 | 高（能追溯来源） | 低（黑盒） |
| 适合场景 | 知识频繁更新、需要溯源 | 固定领域、风格/格式要求 |
| 幻觉风险 | 低（基于检索内容） | 高（依赖训练数据） |

**结论：** 大多数企业知识库场景用 RAG，知识更新频繁且需要溯源。Fine-tuning 适合固定领域的风格迁移（比如让模型学会特定的回复格式）。两者也可以结合：先 fine-tune 让模型理解领域语言，再用 RAG 提供实时知识。

---

### Agentic RAG 和普通 RAG 的区别

普通 RAG：**固定流程**，每次提问都先检索再生成，检索只发生一次。

Agentic RAG：**把检索作为工具**，交给 Agent 自主决定什么时候检索、检索什么。

```
普通 RAG：
  用户问 → 检索 → 生成答案

Agentic RAG（ReAct 模式）：
  用户问
    → Agent 思考：需要什么信息？
    → 调用检索工具（可能多次，每次 query 不同）
    → 观察结果，判断是否足够
    → 生成答案
```

**Agentic RAG 的优势：**

- 复杂问题可以多轮检索，每轮 query 根据上一轮结果动态调整
- Agent 可以决定"这个问题不需要检索，直接回答"，避免无效检索
- 可以混合使用多个知识源（内部文档 + 外部搜索）

**混合方案（我在 DeepScientist 里用的）：**

- 对话开始时做一次普通 RAG，召回最相关的背景文档
- ReAct 推理过程中，Agent 可以按需再次调用检索工具

这样结合了普通 RAG 的质量控制（避免每轮都检索的成本）和 Agentic RAG 的灵活性。

---

### 如何评估 RAG 系统的效果

**离线评估（有标注数据）：**

```python
# 核心指标
# 1. 召回率：标准答案片段是否在 Top-K 里
recall = len(relevant_in_topk) / len(all_relevant)

# 2. MRR（Mean Reciprocal Rank）：标准片段排在第几位
mrr = sum(1 / rank for rank in first_relevant_ranks) / len(queries)

# 3. 答案准确率：最终生成的答案是否正确（需要人工标注或 LLM 评判）
```

**在线评估（生产环境）：**

- 用户反馈（👎 按钮）：点击率反映答案质量
- 追问率：用户追问"你说的不对"说明答案有问题
- 会话完成率：用户是否在一次会话内解决了问题

**用 LLM 自动评估（RAGAS 框架）：**

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
# faithfulness：答案是否忠实于检索内容（防幻觉）
# answer_relevancy：答案是否回答了问题
# context_recall：检索内容是否覆盖了标准答案
```

---

## 一句话总结各个优化点

| 问题 | 解决方案 |
|------|---------|
| 召回不到相关内容 | 检查分块策略、换更好的 embedding 模型 |
| 召回到了但排序靠后 | 加 Reranker |
| 关键词/专有名词召回差 | 混合检索（向量 + BM25） |
| 模型答案和文档不符 | 检查 prompt 约束，加相似度阈值过滤 |
| 文档更新不及时 | 增量索引 + 文件 hash 变更检测 |
| 复杂问题一次检索不够 | 升级为 Agentic RAG，多轮检索 |
