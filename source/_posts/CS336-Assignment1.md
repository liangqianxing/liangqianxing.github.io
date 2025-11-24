---
title: CS336 Assignment1
date: 2025-11-24 09:41:11
tags:
  - NLP
  - Tokenizer
  - BPE
  - 大语言模型
---

# CS336 Assignment1
## BPE (Byte Pair Encoding) 原理与从零实现
BPE（Byte Pair Encoding，字节对编码）是一种非常流行的子词（subword）分词算法，最初用于数据压缩，后来被广泛应用于自然语言处理领域，尤其是在 GPT 系列、LLaMA、RoBERTa 等大语言模型的分词器中。

## BPE 的核心思想

从语料中最频繁出现的相邻符号对（最初是单个字符或字节）开始，逐步合并它们，形成更大的子词单元，直到达到指定的词汇表大小为止。

## BPE 训练过程（经典示例）

假设我们有如下小型语料库（每个词后加 `</w>` 表示词尾）：

```text
low</w>:    5
lower</w>:  2
newest</w>: 6
widest</w>: 3
```

字符级拆分后：
```text
l o w </w> ×5
l o w e r </w> ×2
n e w e s t </w> ×6
w i d e s t </w> ×3
```
统计所有相邻 pair 频率 → 发现 (e, s) 和 (s, t) 都是 9 次 → 任选其一合并（如 es）→ 继续迭代 → 最终得到 est、low、lowest</w> 等高频子词


BPE 在实际模型中的两个重要变体
原始 BPE（OpenAI GPT-2 用）

操作在字符级别（UTF-8 bytes）
基础词汇表是 256 个 byte + 合并规则
优点：能处理任何 Unicode 字符，永不出现 OOV（未知词）
SentencePiece BPE（Google、LLaMA、T5 等用）

直接在原始文本（不分词）上训练
支持 unigram 模式（BPE 的变种）
可以加入特殊控制符号（如 ▁ 表示空格）
BPE 分词时的贪心规则
应用所有合并规则时，按合并顺序从先到后贪心应用（即先训练时学的合并优先级更高）。

例如：

如果先学会了 “un” → “un”
再学会了 “un” + “##able” → “unable”
看到 “unable” 时就会先合并成 “un” + “##able” → “unable”，而不会拆成别的
BPE 的优缺点
优点：

有效解决 OOV 问题（尤其对稀有词、拼写错误、新词）
能把常见词保持完整（high frequency → 合并早 → 成为单个 token）
稀有词被拆成子词，仍有意义
词汇表大小可控
缺点：

分词不一定符合语义或形态学（纯统计）
对低资源语言可能产生很碎的分词
“token 效率”不如 WordPiece 或 Unigram LM 在某些语言上高
总结一句话
BPE 是通过不断合并语料中最常见的相邻符号对，来构建一个大小适中、覆盖广泛的子词词汇表的无监督分词算法，是目前主流大模型分词器的基石之一。

```python
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the input_path,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 1. 参数校验与初始化
    pat_str = kwargs.get("pat_str", GPT2_PRETOKENIZER_PATTERN)
    special_tokens = special_tokens or []
    unique_special_tokens: list[str] = []
    seen_specials: set[str] = set()

    # 这里的逻辑是去重并保持顺序
    for token in special_tokens:
        if not isinstance(token, str):
            msg = f"Expected special tokens to be strings, got {type(token)!r}"
            raise TypeError(msg)
        if token not in seen_specials:
            seen_specials.add(token)
            unique_special_tokens.append(token)
    
    special_tokens_bytes = [token.encode("utf-8") for token in unique_special_tokens]
    num_special_tokens = len(special_tokens_bytes)

    # 基础词表大小为 256 (字节范围)
    if vocab_size < 2**8 + num_special_tokens:
        msg = "vocab_size must be at least 256 + number of special tokens"
        raise ValueError(msg)

    merges_target = vocab_size - num_special_tokens - 2**8
    pretokenizer = regex.compile(pat_str)

    # 2. 读取文件
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    words: list[list[int]] = []
    word_frequencies: list[int] = []
    word_lookup: dict[str, int] = {}

    # 3. 预分词 (Pre-tokenization)
    # 首先按特殊 token 切分，防止特殊 token 被正则拆散
    removable_specials = [token for token in unique_special_tokens if token]
    segments = [text]
    if removable_specials:
        escaped = [regex.escape(token) for token in removable_specials]
        split_pattern = regex.compile("|".join(escaped))
        segments = [segment for segment in split_pattern.split(text) if segment]

    for segment in segments:
        for match in pretokenizer.finditer(segment):
            token = match.group(0)
            if not token:
                continue
            
            idx = word_lookup.get(token)
            if idx is None:
                token_bytes = token.encode("utf-8")
                if not token_bytes:
                    continue
                idx = len(words)
                word_lookup[token] = idx
                words.append(list(token_bytes))
                word_frequencies.append(0)
            
            word_frequencies[idx] += 1

    # 4. 初始化 BPE 统计结构
    token_id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    next_token_id = 256

    pair_stats: Counter[tuple[int, int]] = Counter()
    pair_indices: dict[tuple[int, int], set[int]] = {}
    word_pair_counters: list[Counter[tuple[int, int]]] = []

    # 初次统计所有单词中的 pair
    for idx, token_ids in enumerate(words):
        freq = word_frequencies[idx]
        if freq == 0 or len(token_ids) < 2:
            word_pair_counters.append(Counter())
            continue
        
        pair_counter = Counter(zip(token_ids[:-1], token_ids[1:]))
        word_pair_counters.append(pair_counter)
        
        for pair, count in pair_counter.items():
            pair_stats[pair] += count * freq
            pair_indices.setdefault(pair, set()).add(idx)

    # --- 内部辅助函数 (闭包) ---
    def remove_word_from_stats(word_idx: int) -> None:
        counter = word_pair_counters[word_idx]
        if not counter:
            return
        freq = word_frequencies[word_idx]
        for pair, count in counter.items():
            pair_stats[pair] -= count * freq
            if pair_stats[pair] <= 0:
                pair_stats.pop(pair, None)
            
            indices = pair_indices.get(pair)
            if indices is not None:
                indices.discard(word_idx)
                if not indices:
                    pair_indices.pop(pair, None)

    def add_word_to_stats(word_idx: int) -> None:
        tokens = words[word_idx]
        if len(tokens) < 2:
            word_pair_counters[word_idx] = Counter()
            return
        
        counter = Counter(zip(tokens[:-1], tokens[1:]))
        word_pair_counters[word_idx] = counter
        freq = word_frequencies[word_idx]
        for pair, count in counter.items():
            pair_stats[pair] += count * freq
            pair_indices.setdefault(pair, set()).add(word_idx)

    def merge_word(word_idx: int, pair: tuple[int, int], new_token_id: int) -> None:
        tokens = words[word_idx]
        if len(tokens) < 2:
            return
        
        merged: list[int] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        words[word_idx] = merged

    # 5. BPE 训练主循环
    for _ in range(max(0, merges_target)):
        if not pair_stats:
            break

        # 定义优先级：优先频次高，频次相同比较字节内容（为了确定性）
        def pair_priority(item: tuple[tuple[int, int], int]) -> tuple[int, bytes, bytes]:
            (left_id, right_id), count = item
            return count, token_id_to_bytes[left_id], token_id_to_bytes[right_id]

        best_pair, _ = max(pair_stats.items(), key=pair_priority)
        
        left_bytes = token_id_to_bytes[best_pair[0]]
        right_bytes = token_id_to_bytes[best_pair[1]]
        
        merges.append((left_bytes, right_bytes))
        
        new_token_id = next_token_id
        token_id_to_bytes[new_token_id] = left_bytes + right_bytes

        affected_words = pair_indices.pop(best_pair, set())
        
        # 如果没有单词受到影响（理论上不应发生，因为 stats 里有），直接跳过
        if not affected_words:
            next_token_id += 1
            pair_stats.pop(best_pair, None)
            continue

        # 更新受影响单词的统计信息
        for word_idx in sorted(affected_words):
            remove_word_from_stats(word_idx)
            merge_word(word_idx, best_pair, new_token_id)
            add_word_to_stats(word_idx)
        
        pair_stats.pop(best_pair, None)
        next_token_id += 1

    # 6. 构建最终词表
    vocab: dict[int, bytes] = {
        idx: token for idx, token in token_id_to_bytes.items() if idx < next_token_id
    }

    # 添加特殊 Token
    for token_bytes in special_tokens_bytes:
        if len(vocab) >= vocab_size:
            break
        vocab[next_token_id] = token_bytes
        next_token_id += 1

    return vocab, merges

```