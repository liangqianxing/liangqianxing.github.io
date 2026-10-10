---
title: 从零实现 BPE Tokenizer：现代大语言模型分词器的核心原理
date: 2026-10-07
description: 深入理解并手写实现 GPT、LLaMA 等大语言模型使用的 Byte Pair Encoding (BPE) 分词算法，包含完整代码和可视化讲解
cover: /images/posts/bpe-tokenizer-from-scratch/bpe-cover.webp
coverAlt: 字节积木逐步组合成更长的子词 Token 的示意插画
series: LLM 从零实现
seriesOrder: 1
categories:
  - 技术
tags:
  - NLP
  - LLM
  - Tokenizer
  - Python
  - 深度学习
---

在学习斯坦福 CS336 课程时，我从零实现了一个字节级 BPE（Byte Pair Encoding）分词器。最容易写错的地方不是频率统计，而是训练与编码使用不同的选择规则，以及中文、空格和特殊 token 的边界处理。

这篇文章从 UTF-8 字节开始，依次实现 BPE 训练、按 rank 编码、预分词和 JSON 持久化。读者需要熟悉 Python 的列表、字典和基本类定义，不需要先了解 Transformer。正文保留核心代码，完整项目中的边界处理和测试可以在 [lm-lab 仓库](https://github.com/liangqianxing/lm-lab)中查看。

这是「LLM 从零实现」系列的第一篇。分词器把文本转换为 token ID；下一篇 [从零实现 Transformer](/posts/transformer-from-scratch)会把这些 ID 送入语言模型，[LM Lab 项目总结](/posts/lm-lab-project-complete)则串起分词、模型与训练流程。

## 什么是 BPE？

BPE 是一种子词分词算法，通过反复合并高频的相邻单元来构建词表。本文实现的是字节级 BPE：初始单元是 UTF-8 字节，合并后一个 token 可以包含多个字节。不同模型的分词器还会有不同的预分词、特殊 token 和词表配置，本文代码用于理解算法，并不复刻某个模型的分词结果。

### 为什么需要 BPE？

分词粒度会影响词表大小和输入序列长度：

| 方法 | 问题 |
|------|------|
| 按词分词 | 需要维护较大的词表，未收录的词需要额外处理；按空格切分也不适用于所有语言 |
| 字符级分词 | 词表较小，但同一段文本通常需要更多 token |
| BPE 分词 | 用子词表示文本，在词表大小与序列长度之间做取舍 |

### BPE 的核心思想

训练过程如下：
```
初始：256 个可能的字节值 (0-255)
循环：
  1. 统计所有相邻对的频率
  2. 选择最频繁的对（本文在平局时选 token ID 对的字典序最小者）
  3. 合并为新 token，分配新 ID (256, 257, ...)
  4. 更新所有序列
直到：达到目标词表大小，或没有相邻对可合并
```

### 为什么是字节级而不是字符级？

以 256 个字节为基础词表，合法的 UTF-8 文本都可以表示，不需要为未见过的字符引入 UNK。中文、emoji 和其他符号也能使用同一套编码流程。这个性质来自基础字节词表，并不代表每种语言的 token 数量都同样少。

```python
# UTF-8 编码示例
list("A".encode("utf-8"))   # [65]                1 字节
list("中".encode("utf-8"))  # [228, 184, 173]     3 字节
list("🙂".encode("utf-8")) # [240, 159, 153, 130] 4 字节
```

字符与字节不是一一对应的；单个 token 的字节也未必构成完整字符。解码时要先拼接全部 token 的字节，再做 UTF-8 解码。

---

## 实现路线图

我按照 T1 到 T5 五个阶段逐步实现。每一阶段只增加一类行为，先用小样本检查，再接入后续功能：

```
T1: 字节分词器 ─→ 字符、字节与无损往返
T2: BPE 训练   ─→ 频率统计和非重叠合并
T3: 编码解码   ─→ rank 优先级
T4: 预分词     ─→ 片段边界和特殊 token
T5: 持久化     ─→ 保存模型设置并验证一致性
```

---

## T1: 字节分词器

### 目标

理解字符、UTF-8 字节和 token 之间的关系。

### 实现

```python
from typing import Iterable, Literal, Sequence

class ByteTokenizer:
    """最简单的分词器：直接使用 UTF-8 字节"""
    
    def encode(self, text: str) -> list[int]:
        """字符 → 字节 ID"""
        return list(text.encode("utf-8"))
    
    def decode(
        self, 
        token_ids: Sequence[int],
        *, 
        errors: Literal["strict", "replace"] = "strict"
    ) -> str:
        """字节 ID → 字符"""
        if errors not in ("strict", "replace"):
            raise ValueError(f"Unsupported error mode: {errors}")

        # 关键：先拼接所有字节，再统一解码
        for token_id in token_ids:
            if not 0 <= token_id <= 255:
                raise ValueError(f"Invalid token ID: {token_id}")
        
        byte_sequence = bytes(token_ids)
        return byte_sequence.decode("utf-8", errors=errors)
```

### 关键点

逐字节解码会在多字节字符处失败：
```python
# 逐字节解码 → 会报错！
result = ""
for byte_id in [228, 184, 173]:  # "中"
    result += bytes([byte_id]).decode("utf-8")  # UnicodeDecodeError
```

先拼接字节再统一解码：
```python
# 先拼接，再统一解码
bytes([228, 184, 173]).decode("utf-8")  # "中"
```

---

## T2: BPE 训练算法

### 目标

实现频率统计和迭代合并。

### 数据结构

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Merge:
    pair: tuple[int, int]  # 要合并的 token 对
    new_id: int            # 合并后的新 ID

@dataclass
class BPEModel:
    vocab: dict[int, bytes]              # ID → 原始字节
    merges: tuple[Merge, ...] = ()       # 合并规则（有序）
    special_tokens: dict[str, int] = field(default_factory=dict)
    pretokenization: Literal["none", "simple"] = "none"
```

### 核心函数 1：统计相邻对

```python
def count_pairs(token_ids: Sequence[int]) -> dict[tuple[int, int], int]:
    """统计所有相邻 token 对的出现次数
    
    例：[97, 98, 97, 98] → {(97,98): 2, (98,97): 1}
    """
    if len(token_ids) < 2:
        return {}
    
    pair_counts = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    return pair_counts
```

**可视化**：
```
序列: [1, 2, 1, 2]
      ↓  ↓  ↓  ↓
对:  (1,2) (2,1) (1,2)
计数: {(1,2): 2, (2,1): 1}
```

### 核心函数 2：合并 token 对

```python
def merge_pair(
    token_ids: Sequence[int], 
    pair: tuple[int, int], 
    new_id: int
) -> list[int]:
    """从左到右非重叠合并
    
    例：[97,97,97,97], (97,97)→256 得 [256,256]
    """
    result = []
    i = 0
    
    while i < len(token_ids):
        if (i < len(token_ids) - 1 and 
            (token_ids[i], token_ids[i + 1]) == pair):
            result.append(new_id)
            i += 2  # 关键：跳过两个位置
        else:
            result.append(token_ids[i])
            i += 1
    
    return result
```

**非重叠合并可视化**：

`[a, a, a, a]` 中有三个相邻的 `(a, a)` 窗口，但合并时只能消费两个不重叠的匹配，结果为 `[x, x]`。统计时窗口可以重叠，合并时每个位置只能使用一次。

### 核心函数 3：BPE 训练

下面的基础版本分别处理每条文本，不跨文本合并。T4 会解释预分词和特殊 token；它们需要在训练和编码两端一致地接入，不能只加在编码端。

```python
def train_bpe(
    texts: Iterable[str],
    vocab_size: int
) -> BPEModel:
    """训练 BPE 模型"""
    if vocab_size < 256:
        raise ValueError("vocab_size must include all 256 base bytes")

    # 1. 初始化：0-255 所有字节
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    
    # 2. 文本 → 字节序列
    token_sequences = [
        list(text.encode("utf-8")) 
        for text in texts
    ]
    
    next_id = 256
    
    # 3. 训练循环
    while next_id < vocab_size:
        # 统计所有相邻对
        all_pairs = {}
        for seq in token_sequences:
            for pair, count in count_pairs(seq).items():
                all_pairs[pair] = all_pairs.get(pair, 0) + count
        
        if not all_pairs:
            break
        
        # 选择最频繁的对（平局选字典序最小）
        best_pair = min(
            all_pairs.items(),
            key=lambda x: (-x[1], x[0])
        )[0]
        
        # 记录合并规则
        merges.append(Merge(best_pair, next_id))
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        # 应用合并
        token_sequences = [
            merge_pair(seq, best_pair, next_id) 
            for seq in token_sequences
        ]
        
        next_id += 1
    
    return BPEModel(vocab=vocab, merges=tuple(merges))
```

### 训练过程可视化

![字节序列 97、98、97、98 先合并成两个 256，再合并成 257，对应 a b a b 到 ab ab 再到 abab](/images/posts/bpe-tokenizer-from-scratch/bpe-merge-process.webp)

*图 1 · `abab` 的两轮合并。第一轮得到两个 `ab`，第二轮得到 `abab`；编码时仍需按训练得到的规则顺序执行。原创示意图，AI 辅助绘制。[查看大图](/images/posts/bpe-tokenizer-from-scratch/bpe-merge-process.webp)。*

以 `"abab"` 为例：

```text
初始： [97, 98, 97, 98]
第 1 轮：(97, 98) 出现 2 次，合并为 256，序列变为 [256, 256]
第 2 轮：(256, 256) 出现 1 次，合并为 257，序列变为 [257]
```

仅用 `"abab"` 作为语料、目标词表大小设为 258 时，两轮训练分别加入 `256: b'ab'` 和 `257: b'abab'`。基础的 256 个字节仍保留在词表中。目标词表大小是上限：语料中没有可合并的相邻对时，训练会提前结束。

---

## T3: 编码和解码

### 核心：Rank 优先级

训练时，频率决定下一条合并规则；编码时，模型已经固定，要使用规则的 rank，也就是训练时记录的先后顺序。输入文本中的局部频率不能改变这个顺序，词表中的最长匹配也不是这里使用的算法。

对前面的训练算法产生的规则，下面按顺序扫描并应用 `merges` 的写法便于理解。它会多次遍历序列，实际使用时还可以优化查找和合并过程。

### 实现

```python
class BPETokenizer:
    def __init__(self, model: BPEModel):
        self.model = model
    
    def encode(self, text: str) -> list[int]:
        """按 rank 顺序应用规则"""
        token_ids = list(text.encode("utf-8"))
        
        # 关键：按 merges 的顺序（不是按频率！）
        for merge in self.model.merges:
            token_ids = merge_pair(
                token_ids, 
                merge.pair, 
                merge.new_id
            )
        
        return token_ids
    
    def decode(self, token_ids: Sequence[int]) -> str:
        """拼接字节并解码"""
        byte_sequence = b""
        for token_id in token_ids:
            if token_id not in self.model.vocab:
                raise ValueError(f"Unknown token ID: {token_id}")
            byte_sequence += self.model.vocab[token_id]
        
        return byte_sequence.decode("utf-8")
```

### Rank 优先级可视化

假设 merges 为：
```python
[
    Merge((98, 99), 256),   # Rank 0: b+c → bc
    Merge((97, 98), 257),   # Rank 1: a+b → ab  
    Merge((97, 256), 258),  # Rank 2: a+bc → abc
]
```

编码 `"abc"` 的过程：

```text
初始： [97, 98, 99]
rank 0：b + c → bc，得到 [97, 256]
rank 1：a + b 已经不匹配，序列保持 [97, 256]
rank 2：a + bc → abc，得到 [258]
```

如果忽略 rank，先合并 `a+b`，结果会停在 `[257,99]`，因为模型没有 `ab+c` 这条规则。这个对照说明合并顺序的重要性；这里的词表包含 `abc`，因此不能用这个例子证明最长匹配与 BPE 的结果不同。

---

## T4: 预分词和特殊 Token

### 预分词（Pretokenization）

预分词先把文本切成片段，每个片段内部独立应用 BPE。本文用 `alnum / space / other` 三类连续字符做教学示例，保留每个空格与换行。它按字符类别划分边界，不等同于语义分词，也不等同于 GPT-2 等分词器使用的完整正则规则。

```python
def pretokenize(text: str) -> list[str]:
    """分割为 alnum / space / other 片段"""
    if not text:
        return []
    
    def get_category(char: str) -> str:
        if char.isalnum():    return "alnum"
        elif char.isspace():  return "space"
        else:                 return "other"
    
    pieces = []
    current_piece = ""
    current_category = None
    
    for char in text:
        category = get_category(char)
        if category == current_category:
            current_piece += char
        else:
            if current_piece:
                pieces.append(current_piece)
            current_piece = char
            current_category = category
    
    if current_piece:
        pieces.append(current_piece)
    
    return pieces
```

**示例**：
```python
pretokenize("Hi,  世界!\n🙂")
# → ["Hi", ",", "  ", "世界", "!", "\n", "🙂"]

# 验证无损性
text = "Hi,  世界!\n🙂"
assert "".join(pretokenize(text)) == text
```

### 特殊 Token

特殊 token（如 `<eos>`, `<pad>`）的处理规则：

1. **默认不识别**：当作普通文本
2. **显式许可**：只识别 `allowed_special` 中列出的
3. **最长匹配**：同位置优先匹配更长的
4. **原子边界**：不参与 BPE 合并

下面展示项目的扩展 API，不能直接传给 T2、T3 的基础版本。正确的处理顺序是：训练时先隔离已配置的特殊 token，再对普通文本片段做预分词；编码时先识别显式允许的特殊 token，再处理普通文本。特殊 token 的 ID 在实际学到的 BPE ID 之后分配，不能假定总是 256；`vocab_size` 包含基础字节、BPE token 和特殊 token 的总数。

这条边界规则需要单独测试。仓库当前的 `simple` 训练分支先预分词，再比较片段是否等于特殊标记，可能把 `<eos>` 中的 `eos` 纳入普通训练语料。要满足上述规则，需要先识别完整特殊标记；文末列出本次核对的代码版本和复现边界。

```python
# 训练时隔离特殊 token
model = train_bpe(
    texts,
    vocab_size=300,
    special_tokens=("<eos>", "<pad>")
)

# 编码时显式允许
ids = tokenizer.encode(
    "Hello<eos>", 
    allowed_special={"<eos>"}
)
# 普通文本按模型的 BPE 规则编码，<eos> 作为单个专用 token
```

例如允许 `<eos>` 时，`"Hi,<eos> 世界!"` 先隔离出特殊 token，其余文本分成 `"Hi"`、`","`、`" "`、`"世界"`、`"!"`。这些片段各自执行 BPE，保留空格，也不跨过特殊 token 合并。

---

## T5: 模型持久化

### JSON 格式

下面只展示字段结构，省略了大部分基础字节条目；它不是可直接加载的完整模型。字节串使用十六进制保存，例如 `6161` 表示 `b"aa"`，`3c656f733e` 表示 `b"<eos>"`。`merges` 用数组保存，顺序就是 rank。

```json
{
  "format": "lm-lab-bpe",
  "version": 1,
  "pretokenization": "simple",
  "vocab": {
    "0": "00",
    "1": "01",
    "256": "6161",
    "257": "3c656f733e"
  },
  "merges": [
    {"pair": [97, 97], "new_id": 256}
  ],
  "special_tokens": {
    "<eos>": 257
  }
}
```

### 实现

以下方法放入 `BPETokenizer` 类中。保存时保留预分词设置和特殊 token；加载时除了检查基础字节，还要确保合并输入已在当前 rank 之前可用、新 ID 未被复用、字节内容一致。

```python
from pathlib import Path

def save(self, path: str | Path) -> None:
    """保存为 JSON"""
    import json
    path = Path(path)
    
    payload = {
        "format": "lm-lab-bpe",
        "version": 1,
        "pretokenization": self.model.pretokenization,
        "vocab": {
            str(tid): tbytes.hex()
            for tid, tbytes in self.model.vocab.items()
        },
        "merges": [
            {"pair": list(m.pair), "new_id": m.new_id}
            for m in self.model.merges
        ],
        "special_tokens": self.model.special_tokens,
    }
    
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8"
    )

@classmethod
def load(cls, path: str | Path) -> "BPETokenizer":
    """加载并验证"""
    import json
    path = Path(path)
    
    payload = json.loads(path.read_text(encoding="utf-8"))
    
    # 验证格式
    if payload.get("format") != "lm-lab-bpe":
        raise ValueError("Unsupported format")
    if payload.get("version") != 1:
        raise ValueError("Unsupported version")

    pretokenization = payload["pretokenization"]
    if pretokenization not in ("none", "simple"):
        raise ValueError("Unsupported pretokenization")
    
    # 加载 vocab
    vocab = {
        int(id_str): bytes.fromhex(hex_str)
        for id_str, hex_str in payload["vocab"].items()
    }
    
    # 验证基础字节 (0-255)
    for i in range(256):
        if i not in vocab or vocab[i] != bytes([i]):
            raise ValueError(f"Invalid base byte {i}")
    
    # 加载 merges 并验证
    merges = []
    available_ids = set(range(256))
    for m in payload["merges"]:
        pair = tuple(m["pair"])
        new_id = m["new_id"]

        if len(pair) != 2:
            raise ValueError("Invalid merge pair")
        
        # 输入必须在此条规则之前可用，不能引用未来规则
        if pair[0] not in available_ids or pair[1] not in available_ids:
            raise ValueError("Merge references unavailable ID")

        if new_id in available_ids or new_id not in vocab:
            raise ValueError("Invalid or reused merge ID")
        
        # 验证合并字节一致性
        expected = vocab[pair[0]] + vocab[pair[1]]
        if vocab[new_id] != expected:
            raise ValueError("Merge bytes mismatch")
        
        merges.append(Merge(pair, new_id))
        available_ids.add(new_id)

    special_tokens = payload.get("special_tokens", {})
    for marker, token_id in special_tokens.items():
        if not marker or token_id in available_ids or token_id not in vocab:
            raise ValueError("Invalid special token ID")
        if vocab[token_id] != marker.encode("utf-8"):
            raise ValueError("Special token bytes mismatch")
        available_ids.add(token_id)
    
    model = BPEModel(
        vocab=vocab,
        merges=tuple(merges),
        special_tokens=special_tokens,
        pretokenization=pretokenization
    )
    
    return cls(model)
```

---

## 完整使用示例

下面是项目 API 的组合方式，包含 T4 扩展。运行前需要检查包导出与特殊 token 的训练边界；它与前文的基础教学版不是同一套完整实现。

```python
from lm_lab.tokenization import train_bpe, BPETokenizer

# 1. 训练
corpus = [
    "The quick brown fox",
    "这是中文句子",
    "🙂 Emoji!"
]

model = train_bpe(
    corpus,
    vocab_size=400,
    pretokenization="simple",
    special_tokens=("<s>", "</s>", "<pad>")
)

# 2. 使用
tokenizer = BPETokenizer(model)

text = "<s>Hello world!</s>"
tokens = tokenizer.encode(text, allowed_special={"<s>", "</s>"})
print(f"Tokens: {tokens}")
print(f"Vocab size: {len(model.vocab)}")

decoded = tokenizer.decode(tokens)
assert decoded == text

# 3. 持久化
tokenizer.save("tokenizer.json")
loaded = BPETokenizer.load("tokenizer.json")
assert loaded.model == model
```

---

## 测试与复现边界

71 项分词器测试是原稿写作时的阶段性记录，不代表当前仓库的运行结果。复现时应以对应版本实际运行的测试为准。

2026 年 10 月 10 日核对的 [代码版本 `bf854b1`](https://github.com/liangqianxing/lm-lab/tree/bf854b11027bb430843c0d5b72e0a94cf95232dd)仍有两个需要处理的边界：包入口的导出与 `bpe.py` 定义不一致，会阻止普通导入；`simple` 训练分支需要先识别完整特殊标记，再做预分词。前面的使用示例展示 API 组合，运行前应检查这两点。

核心测试应覆盖这些行为：

| 测试类型 | 典型用例 |
|---------|---------|
| UTF-8 正确性 | 中文、emoji、组合字符 |
| 非重叠合并 | `[a,a,a,a]` → `[x,x]` |
| Rank 优先级 | 复杂合并顺序 |
| 预分词无损性 | `"".join(pieces) == text` |
| 特殊 token 隔离 | 特殊标记不进入普通对统计，训练与编码遵守各自的许可规则 |
| 持久化一致性 | 保存前后的词表、rank、预分词和特殊 token 配置一致 |

本文的基础代码适合手算和调试。它逐轮重新统计训练语料，编码时逐条扫描规则，解码时反复拼接字节；处理较大语料前还需要优化数据结构与内存使用。测试也应验证错误输入，不能只检查一次往返成功。

## 如何比较分词效果

token 数取决于实际训练语料、词表和预分词规则，仅给出 `vocab_size` 无法推导一句话会得到多少 token。字节级表示能覆盖多语言字符，也不代表它在中文或 emoji 上一定比其他分词方案更高效。

比较方案时，应固定训练与评估语料，分别记录实际词表大小、评估文本的 UTF-8 字节数和 token 数，并检查解码能否还原原文。速度比较还需要注明硬件、输入规模和计时方式。本文的小例子用于检验规则，不作为分词压缩率或吞吐量的实测结果。

## 容易写错的地方

这次实现让我把三个概念分清了：字符不是字节，训练频率不是编码 rank，预分词边界也不是语义边界。回到代码里，最值得反复检查的是下面这些位置：

| 容易错的点 | 正确做法 |
|-----------|---------|
| 逐字节解码中文 | 先拼接再统一解码 |
| 将统计次数当成替换次数 | 统计可重叠，实际合并从左到右且不重叠 |
| 按频率编码 | 按 rank 顺序编码 |
| 预分词丢失空格 | 保持无损性 |
| 特殊 token 默认识别 | 编码时只识别显式允许的标记 |
| 只在编码时加入预分词 | 训练与编码沿用同一普通文本分段规则 |
| 加载时只检查词表中存在 ID | 按 rank 检查输入已可用、新 ID 未复用及字节一致性 |

---

## 下一步学习

分词器输出的是整数 ID，模型还需要把它们转换成向量。接下来可以阅读 [从零实现 Transformer](/posts/transformer-from-scratch)，理解 embedding、注意力与前馈网络如何处理这些输入，再到 [LM Lab 项目总结](/posts/lm-lab-project-complete)查看优化器和训练流程。

---

## 参考资料

- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [GPT-2 Tokenizer 实现](https://github.com/openai/gpt-2)
- [HuggingFace Tokenizers 文档](https://huggingface.co/docs/tokenizers/)

完整代码与教学规范：[GitHub - lm-lab](https://github.com/liangqianxing/lm-lab)。
