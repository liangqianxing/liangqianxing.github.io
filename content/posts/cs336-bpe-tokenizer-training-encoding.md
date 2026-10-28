---
title: 'Stanford CS336 学习记录 07：Byte-Level BPE 分词器训练、倒排索引加速与流式编码'
date: '2026-10-25'
description: '实现 GPT-2 正则预分词、特殊 token 硬边界切分、基于 pair-to-word 倒排索引的增量 BPE 合并算法（< 0.25s 通过速度基准），以及与 tiktoken 完全对齐的 Tokenizer。'
series: Stanford CS336
seriesOrder: 7
cover: /images/posts/cs336-bpe-tokenizer-training-encoding/cs336-bpe-tokenizer-training-encoding-cover.webp
coverAlt: UTF-8 字节块经过正则切分闸口后沿合并树聚合成子词单元的 AI 辅助概念插画
codeUrl: https://github.com/liangqianxing/cs336-labs/tree/b5c93ad0cc73178b2b2a62e518a0900da47a0fb7
attachmentUrl: /materials/cs336/experiments/cs336-07-bpe-tokenizer-20261025.zip
categories:
  - 学习笔记
tags:
  - CS336
  - Tokenizer
  - BPE
  - Tiktoken
  - 算法优化
draft: false
hidden: false
haloSync: false
haloPublished: false
---

前面六篇完成了 Transformer 语言模型本身的全部神经网络组件。要让模型真正处理自然语言文本，还需要打通入口处的 **Byte-Level BPE（Byte-Pair Encoding）分词器**。这一篇记录 `cs336_basics/bpe.py`（BPE 训练）与 `cs336_basics/tokenizer.py`（编码、解码与流式迭代）的完整实现与性能优化。

[查看代码仓库](https://github.com/liangqianxing/cs336-labs/tree/b5c93ad0cc73178b2b2a62e518a0900da47a0fb7) · [下载本节代码与测试 ZIP](/materials/cs336/experiments/cs336-07-bpe-tokenizer-20261025.zip)

## 为什么选择 UTF-8 Byte-Level BPE（Problem `unicode1` & `unicode2`）

直接按 Unicode 码点建词表会遇到两个致命问题：Unicode 17.0 定义了近 16 万个字符，初始词表过大且长尾字符极度稀疏；一旦遇到训练未见的字符就会退化为 `<unk>`（Out-of-Vocabulary）。

而将文本编码为 **UTF-8 字节序列**后，任何的字符串都可以表示为 $0 \sim 255$ 之间的整数序列，基础词表大小固定为 256，天然实现 **零 OOV**。相比 UTF-16 或 UTF-32，UTF-8 对英文和常见 ASCII 符号只需 1 个字节，不会引入大量无意义的零字节（`\x00`），字节流更加紧凑。

需要特别注意解码时的多字节边界问题：在 UTF-8 中，一个汉字或 Emoji 往往由 2～4 个字节组成（例如 `🙃` 对应 4 个字节 `\xf0\x9f\x99\x83`）。如果像讲义反例那样逐字节调用 `bytes([b]).decode("utf-8")`，任何非 ASCII 多字节字符都会立即抛出 `UnicodeDecodeError`。正确做法是先把所有 token 对应的 `bytes` 拼接成完整字节串 `b"".join(...)`，最后一次性调用 `.decode("utf-8", errors="replace")`。

## 预分词（Pre-tokenization）与特殊 Token 硬边界隔离

如果在整篇语料上直接统计相邻字节对，BPE 会把跨词边界的标点与单词合并（例如把 `dog.` 和 `dog!` 合并成完全不同的 token ID），而且每次合并都要扫描整篇文档。

因此训练和编码的第一步都是先做两级切分：

### 1. 先按 Special Tokens 切分，防止跨文档合并

像 `<|endoftext|>` 这样的特殊 token 用来标记文档边界。训练时，它们既不能被拆成普通字节参与频次统计，也不能让前一篇文档的末尾字节与后一篇文档的开头字节跨边界合并。

而在分词编码时，如果用户注册了互相包含的重叠特殊 token（例如同时注册 `<|endoftext|>` 和 `<|endoftext|><|endoftext|>`，对应官方测试 `test_overlapping_special_tokens`），正则切分必须遵循**最长匹配优先**：

```python
sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
pattern_str = "(" + "|".join(re.escape(tok) for tok in sorted_specials) + ")"
```

先按长度降序排列并用 `re.escape` 转义 `|` 等元字符，才能保证连续的两个 `<|endoftext|><|endoftext|>` 被优先识别为单个长特殊 token，而不会被短 token 提前截断。

### 2. 再用 GPT-2 正则 `re.finditer` 提取预分词频次表

在每个普通文本片段内部，使用 `regex` 库的 GPT-2 预分词模式：

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

这个正则把英文缩写后缀（`'s`、`'ll`、`'ve`）、带前导空格的字母串、数字串、标点串以及连续空白切分成独立的 pre-token。注意末尾的 `\s+(?!\S)|\s+`：当遇到多行换行或连续空格后跟非空白字符时，它会把最后一个前导空格留给后面的单词（如 `" how"`），其余前导空白单独成块，这正是与 `tiktoken` 在双换行与尾随换行测试（`test_encode_special_token_trailing_newlines`）上保持 100% 一致的关键。

## 训练加速：从 3 秒到 0.19 秒的倒排索引增量更新

官方 `test_train_bpe_speed` 要求在 `corpus.en` 上完成 500 词表训练的耗时必须小于 **1.5 秒**。如果每次合并都遍历所有唯一单词重新统计字节对频次，耗时通常在 3 秒左右。

观察发现：**一次合并 `(p0, p1) -> new_token`，只会改变那些原本就包含 `(p0, p1)` 的单词的内部字节对计数**，其余 99% 的单词根本不受影响。因此我们在 `cs336_basics/bpe.py` 中维护两张全局表：
1. `pair_counts: dict[tuple[bytes, bytes], int]`：每个相邻字节对的当前总频次；
2. `pair_to_words: dict[tuple[bytes, bytes], set[int]]`：包含该字节对的唯一单词下标集合（倒排索引）。

每轮迭代时：
1. 按频次最高、平局时字典序更大（`(count, pair)` 最大）选出 `best_pair = (p0, p1)`；
2. 仅从 `pair_to_words.pop(best_pair)` 取出受影响的单词下标 `w_idx`；
3. 对每个受影响单词，先将其旧相邻对的计数减去 `freq`，再执行 `(p0, p1) -> new_token` 合并，最后将新单词的相邻对计数加回 `pair_counts` 并更新 `pair_to_words`。

```python
for w_idx in affected_word_indices:
    word = words[w_idx]
    freq = freqs[w_idx]
    # 1) 减去该词原有的非 best_pair 相邻对频次
    for i in range(len(word) - 1):
        old_pair = (word[i], word[i + 1])
        if old_pair != best_pair:
            ...
    # 2) 在该词内部合并 (p0, p1) -> new_token
    ...
    # 3) 将合并后新词的相邻对频次加回全局表
    for i in range(len(new_word) - 1):
        new_pair = (new_word[i], new_word[i + 1])
        pair_counts[new_pair] += freq
        pair_to_words[new_pair].add(w_idx)
```

需要特别注意平局决胜规则（Tie-breaking）：当多个字节对频次并列第一时，讲义要求选择**字典序更大**的字节对 `max(pairs)`。加上倒排索引后，`test_train_bpe_speed` 实测仅需约 **0.18s**（远低于 1.5s 上限），生成的词表和 merges 列表与官方 `train-bpe-reference-merges.txt` 逐项完全一致。

## 分词器编码：按 Merge Rank 顺序合并与流式 `encode_iterable`

在 `Tokenizer.encode` 中，拿到一个 pre-token 的初始单字节序列 `[b_0, b_1, ...]` 后，不能遍历全部 5 万条 merges 规则逐条检查（那会是 $O(|M| \cdot L)$），而应当预先构建哈希表 `merge_ranks = {pair: rank}`。

每次在当前字节序列的相邻对中，找出在 `merge_ranks` 里 **rank 最小（即训练时最早生成）** 的那一对比邻块进行合并，循环直到没有任何相邻对出现在 `merge_ranks` 中，再配上高频词结果缓存 `_word_cache`，就能在瞬间完成整篇文档的编码。

针对超大文件無法一次性塞入内存的场景（对应 `encode_iterable` 与 1MB 内存上限测试），`Tokenizer.encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]` 使用生成器按行/按块惰性产出 token ID：

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    for chunk in iterable:
        if not chunk:
            continue
        yield from self.encode(chunk)
```

对于离线语料预处理，为什么讲义推荐将 token ID 序列化为 `np.uint16`？因为 16 位无符号整数的表示范围是 $0 \sim 65{,}535$，刚好能无损容纳 TinyStories（10K 词表）和 OpenWebText（32K 词表）的所有 token ID，同时相比 Python `int` 或 `int64` 节省 75% 的磁盘与内存占用。

## 官方测试验证结果

运行 `test_train_bpe.py` 与 `test_tokenizer.py` 全套测试：

```text
tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
tests/test_tokenizer.py (23 passed, 2 skipped on macOS)
```

所有 ASCII、多语言 Unicode（德语、Emoji）、特殊 token 边界以及与 OpenAI `tiktoken` GPT-2 编码器的逐 token 对照测试全部通过。最后一篇将继续完成 [Cross-Entropy、AdamW 优化器、余弦退火调度与完整训练闭环](/posts/cs336-cross-entropy-adamw-training-loop/)。

*封面使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 辅助生成，属于概念插画；准确的算法细节与代码实现以正文为准。*

## 参考资料

- [Sennrich et al., 2016: Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [OpenAI tiktoken GPT-2 regex pattern](https://github.com/openai/tiktoken)
- [Stanford CS336 Assignment 1 官方仓库](https://github.com/stanford-cs336/assignment1-basics)
