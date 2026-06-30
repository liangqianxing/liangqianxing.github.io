---
title: 从零实现多模态 RAG：BM25、Dense 检索、RRF 融合、MMR 重排全部手写
date: 2026-06-30 18:00:00
tags:
  - RAG
  - 多模态
  - AI Infra
  - BM25
  - 向量检索
  - 混合检索
  - 实习求职
categories:
  - AI
---

> 项目地址：[github.com/liangqianxing/multimodal-rag](https://github.com/liangqianxing/multimodal-rag)  
> 91 个单元测试，全部通过 ✅ · 纯 numpy + PIL，无需 GPU

上一个项目写了 LLM 推理引擎，这次做一个互补的方向：**多模态 RAG**。

两者的关系是：推理引擎负责"怎么跑模型"，RAG 负责"给模型喂什么信息"。面试时两个合在一起讲，能覆盖整个 LLM 应用栈。

---

## 一、RAG 解决什么问题？

LLM 有两个硬伤：
1. **知识截止**：训练数据有时效性，不知道最新信息
2. **幻觉**：对不确定的内容会瞎编

RAG 的解法：查询时先检索相关文档，把文档内容拼进 prompt，让模型"有据可查"而不是凭空生成。

```
传统 LLM：
  用户提问 → LLM 直接回答（靠训练时的记忆，可能幻觉）

RAG 流程：
  用户提问 → 检索相关文档 → 文档 + 提问拼接成 prompt → LLM 回答
```

**多模态 RAG** 进一步支持图像检索：用文字查图，用图查文字，或者图文混合检索。

---

## 二、系统架构

```
Documents (文本 + 图像)
   │
   ├── 文本 chunks → TF-IDF embedding
   ├── 图像 chunks → 颜色直方图特征
   │                    ↓
   │              随机投影对齐 → 共享 embedding 空间
   │
   ▼
┌─────────────────────────────────┐
│  向量库 (cosine)  │  BM25 索引  │  ← 双路索引
└─────────────────────────────────┘
   │
   ▼ 查询时
Dense 检索 + BM25 检索
       ↓
   RRF 融合
       ↓
   MMR 重排
       ↓
  Generator → 答案
```

---

## 三、四个核心算法，逐一拆解

### 3.1 BM25 —— 从零实现的关键词检索

BM25 Okapi 是搜索引擎领域最经典的排名函数，比简单 TF-IDF 更好的地方在于对长文档做了惩罚：

```
BM25(q, d) = Σ IDF(tᵢ) · f(tᵢ,d)·(k₁+1) / [f(tᵢ,d) + k₁·(1-b+b·|d|/avgdl)]

参数：k₁=1.5（词频饱和），b=0.75（文档长度惩罚）
```

直觉：一篇 10000 词的文章里出现 5 次"机器学习"，和一篇 100 词的文章里出现 5 次，相关性应该不同。BM25 用 `|d|/avgdl` 做了归一化。

```python
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        ...
    
    def fit(self, corpus: List[str]) -> None:
        # 建立倒排索引：词 → 包含它的文档数（df）
        # 计算每个词的 IDF
        ...
    
    def get_top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        # 分词 → 对每个词查 IDF → 计算 BM25 得分 → 排序
        ...
```

纯 Python，零依赖，支持增量添加文档。

---

### 3.2 Dense Retrieval —— 嵌入空间的语义搜索

关键词检索的缺点：必须用相同的词才能匹配。"汽车"查不到"automobile"，"机器学习"查不到"ML"。

Dense Retrieval 把查询和文档都映射到同一个向量空间，用余弦相似度：

```
similarity(q, d) = (q · d) / (‖q‖ · ‖d‖)
```

在嵌入空间里，语义相近的词/句子距离也近，即使用词不同。

**本项目的 TF-IDF 嵌入器**（纯 numpy，无需 sentence-transformers）：

```python
class TFIDFEmbedder:
    def fit(self, corpus: List[str]) -> None:
        # 建立词表，计算 IDF 权重
        ...
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # 对每个文本：TF × IDF，L2 归一化
        # 返回 shape (N, vocab_size) 的矩阵
        ...
```

---

### 3.3 Hybrid Retrieval —— RRF 融合两路检索

BM25 和 Dense 的优劣是互补的：
- BM25：精确关键词匹配，速度快，不依赖嵌入质量
- Dense：语义理解，能处理同义词，但词表外词效果差

**Reciprocal Rank Fusion（RRF）** 是最简单有效的融合方法，不需要分数归一化：

```
RRF(d) = Σᵣ  1 / (k + rankᵣ(d))    k = 60
```

直觉：不关心绝对得分，只看在每个检索器的排名。第 1 名得 1/(60+1)≈0.016，第 10 名得 1/(60+10)≈0.014。把两路的分数加起来重新排序。

**实测结果**：

| 策略 | Recall@5 | MRR@5 |
|------|---------|-------|
| BM25 Only | 0.42 | 0.35 |
| Dense Only | 0.61 | 0.54 |
| **Hybrid (RRF)** | **0.71** | **0.63** |

混合检索比单一策略分别高出 +10pp 和 +17pp。

```python
class HybridRetriever:
    def retrieve(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        dense_results = self.dense.retrieve(query, k * 2)  # 候选池放大
        bm25_results  = self.sparse.retrieve(query, k * 2)
        
        # RRF 融合
        scores = {}
        for rank, (chunk, _) in enumerate(dense_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1/(self.rrf_k + rank + 1)
        for rank, (chunk, _) in enumerate(bm25_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1/(self.rrf_k + rank + 1)
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]
```

---

### 3.4 MMR —— 最大边际相关性，解决结果同质化

Top-5 结果全是说"机器学习是数据驱动方法"的不同说法，对 LLM 没有帮助。

**Maximal Marginal Relevance（MMR）** 在选每个结果时，不仅考虑与 query 的相关度，还惩罚与已选结果的相似度：

```
MMR(dᵢ) = argmax [ λ·sim(dᵢ, q) - (1-λ)·max_{dⱼ∈S} sim(dᵢ, dⱼ) ]

λ=0.5：相关性和多样性各占一半
S：已选结果集合
```

实现：贪心迭代，每次选 MMR 分最高的文档加入结果集：

```python
class MMRRetriever:
    def retrieve(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        # 1. 先用 base_retriever 取候选池（k * 3 个）
        candidates = self.base_retriever.retrieve(query, k * 3)
        
        # 2. 贪心 MMR 选取 k 个
        selected = []
        q_emb = self.embedder.embed_query(query)
        
        while len(selected) < k and candidates:
            best_chunk, best_score = None, -float("inf")
            for chunk, rel_score in candidates:
                div_penalty = max(
                    self._cosine(chunk.embedding, s.embedding)
                    for s, _ in selected
                ) if selected else 0.0
                mmr_score = self.lambda_param * rel_score - (1 - self.lambda_param) * div_penalty
                if mmr_score > best_score:
                    best_chunk, best_score = chunk, mmr_score
            selected.append((best_chunk, best_score))
            candidates.remove((best_chunk, _))
        
        return selected
```

---

## 四、多模态嵌入：不用 GPU 的跨模态对齐

CLIP 等模型能把文本和图像映射到同一空间，但需要 GPU 和大模型。

本项目用**随机投影对齐**：

1. **文本**：TF-IDF 向量（vocab_size 维）
2. **图像**：颜色直方图（R/G/B 各 64 bins + 灰度 32 bins = 224 维）
3. **对齐**：固定随机矩阵 W（seed=42，可复现）将两种特征投影到同一 64 维空间

```python
class MultimodalEmbedder:
    def fit(self, text_corpus, image_samples):
        # 拟合 TF-IDF 词表
        self.text_emb.fit(text_corpus)
        
        # 随机投影矩阵（固定种子，确保复现）
        rng = np.random.RandomState(42)
        self.text_proj  = rng.randn(self.text_emb.dim, self.target_dim) / np.sqrt(self.target_dim)
        self.image_proj = rng.randn(self.image_emb.dim, self.target_dim) / np.sqrt(self.target_dim)
    
    def embed_texts(self, texts):
        raw = self.text_emb.embed_texts(texts)   # (N, vocab_size)
        return self._l2_norm(raw @ self.text_proj)  # (N, target_dim)
    
    def embed_images(self, images):
        raw = self.image_emb.embed_images(images) # (N, 224)
        return self._l2_norm(raw @ self.image_proj) # (N, target_dim)
```

效果比真实 CLIP 差，但完整展示了跨模态 RAG 的机制，且换成真实 encoder 只需替换 `embed_texts` 和 `embed_images` 两个方法。

---

## 五、评估指标

### Recall@K

在前 K 个检索结果中，有多少比例的相关文档被找到：

```
Recall@K = |retrieved[:K] ∩ relevant| / |relevant|
```

### MRR@K（Mean Reciprocal Rank）

第一个相关文档排在第几：

```
MRR = 1 / rank_of_first_relevant_doc
```

第 1 名：MRR=1.0；第 2 名：MRR=0.5；第 5 名：MRR=0.2。

### NDCG@K

考虑相关文档的排序位置，排名越靠前得分越高。

---

## 六、面试高频考点

**Q：RAG 和 Fine-tuning 的区别？什么时候用哪个？**  
A：Fine-tuning 把知识"烧进"模型权重，适合固定领域、大量训练数据。RAG 在推理时动态检索，适合频繁更新的知识库、需要引用来源、或者不想训练成本的场景。实际上两者不互斥，可以先 RAG 再 fine-tune。

**Q：为什么 Hybrid 比单一检索好？**  
A：BM25 和 Dense 的错误不相关。BM25 在精确匹配、专有名词上强，Dense 在语义理解、同义词上强。RRF 融合时，两路都认为相关的文档得分最高，是一种"投票"机制，比任何单一方法更鲁棒。

**Q：MMR 什么时候比 top-K 好？**  
A：当 corpus 里有大量相似文档时，top-K 会全返回近似重复的内容。MMR 强制多样性，让 generator 拿到不同角度的信息。但当 corpus 天然多样时，MMR 反而可能降低 Recall（因为为了多样性牺牲了最相关的结果），所以 λ 是一个需要根据数据调的超参数。

**Q：如何扩展到亿级文档？**  
A：本项目用 numpy 暴力计算，适合 ~50K chunks。生产中换成 FAISS（Facebook 的向量库，支持 IVF 分区 + PQ 压缩）或 Milvus/Weaviate 等向量数据库。BM25 层可以换成 Elasticsearch。整体 pipeline 不变，只替换底层实现。

---

## 七、与 LangChain / LlamaIndex 的对比

| 维度 | 本项目 | LangChain / LlamaIndex |
|------|--------|----------------------|
| BM25 实现 | 从零，每行可读 | 封装的 rank-bm25 库 |
| RRF 融合 | 公式直接对应代码 | 可配置的抽象层 |
| MMR 重排 | 手写，λ 可调 | 内置 helper，内部隐藏 |
| 多模态对齐 | 显式随机投影 | 依赖外部 encoder |
| 评估指标 | 内置 Recall/MRR/NDCG | 需要外部评估框架 |
| 学习价值 | 高，公式 = 代码 | 低，抽象遮蔽细节 |

---

代码在 GitHub：[liangqianxing/multimodal-rag](https://github.com/liangqianxing/multimodal-rag)

```bash
git clone https://github.com/liangqianxing/multimodal-rag
cd multimodal-rag
pip install pytest numpy Pillow
pytest tests/ -v                    # 91 tests, all pass
python examples/basic_usage.py      # 端到端 demo
python run_all_benchmarks.py --fast # 检索质量 + 延迟 benchmark
```

---

两个项目合起来，基本覆盖了 AI Infra 面试的核心考察点：

| 项目 | 覆盖的面试考点 |
|------|-------------|
| [mini-llm-engine](https://github.com/liangqianxing/mini-llm-engine) | KV Cache、Continuous Batching、GPU 显存管理、投机解码 |
| [multimodal-rag](https://github.com/liangqianxing/multimodal-rag) | 向量检索、BM25、混合检索、多模态对齐、评估指标 |
