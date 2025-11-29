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


## BPE 在实际模型中的两个重要变体
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

## 代码实现
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

## ALL code
```python
from __future__ import annotations

import builtins
import locale
import math
import os
from collections import Counter
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import regex
import tiktoken
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn.utils import clip_grad_norm_


def _ensure_utf8_locale() -> None:
    try:
        preferred = locale.getpreferredencoding(False)
    except Exception:
        preferred = "utf-8"
    if preferred.lower() != "utf-8":
        locale.getpreferredencoding = lambda *_args, **_kwargs: "utf-8"  # type: ignore[assignment]


_ensure_utf8_locale()

_ORIGINAL_OPEN = builtins.open


def _utf8_default_open(
    file,
    mode="r",
    buffering=-1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    closefd: bool = True,
    opener=None,
):
    if "b" not in mode and encoding is None:
        encoding = "utf-8"
    return _ORIGINAL_OPEN(file, mode, buffering, encoding, errors, newline, closefd, opener)


builtins.open = _utf8_default_open  # type: ignore[assignment]


GPT2_PRETOKENIZER_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    if tuple(weights.shape) != (d_out, d_in):
        msg = f"weights shape {tuple(weights.shape)} does not match ({d_out}, {d_in})"
        raise ValueError(msg)

    return F.linear(in_features, weights, bias=None)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    if tuple(weights.shape) != (vocab_size, d_model):
        msg = f"weights shape {tuple(weights.shape)} does not match ({vocab_size}, {d_model})"
        raise ValueError(msg)

    token_ids = token_ids.to(torch.long)
    return F.embedding(token_ids, weights)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    if d_model <= 0 or d_ff <= 0:
        raise ValueError("d_model and d_ff must be positive")

    gate = F.linear(in_features, w1_weight, bias=None)
    up = F.linear(in_features, w3_weight, bias=None)
    activated = F.silu(gate) * up
    return F.linear(activated, w2_weight, bias=None)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    if d_k == 0:
        raise ValueError("d_k must be positive")

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        fill = torch.finfo(scores.dtype).min
        mask = mask.to(dtype=torch.bool, device=scores.device)
        if mask.shape != scores.shape:
            mask = mask.expand(scores.shape)
        scores = scores.masked_fill(~mask, fill)

    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, V)


def _build_causal_mask(
    batch_dims: tuple[int, ...], num_heads: int, seq_len: int, device: torch.device
) -> Bool[Tensor, " ..."]:
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    view_shape = (1,) * len(batch_dims) + (1, seq_len, seq_len)
    return mask.view(view_shape).expand(*batch_dims, num_heads, seq_len, seq_len)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    head_dim = d_model // num_heads
    batch_dims = tuple(in_features.shape[:-2])
    seq_len = in_features.shape[-2]

    def _project(weight: Tensor) -> Tensor:
        proj = F.linear(in_features, weight, bias=None)
        new_shape = (*batch_dims, seq_len, num_heads, head_dim)
        proj = proj.reshape(new_shape)
        permute_order = list(range(len(batch_dims))) + [len(batch_dims) + 1, len(batch_dims), len(batch_dims) + 2]
        return proj.permute(permute_order)

    q = _project(q_proj_weight)
    k = _project(k_proj_weight)
    v = _project(v_proj_weight)

    mask = _build_causal_mask(batch_dims, num_heads, seq_len, in_features.device)
    attn_output = run_scaled_dot_product_attention(q, k, v, mask=mask)
    permute_order = list(range(len(batch_dims))) + [len(batch_dims) + 1, len(batch_dims), len(batch_dims) + 2]
    attn_output = attn_output.permute(permute_order)
    merged = attn_output.reshape(*batch_dims, seq_len, d_model)
    return F.linear(merged, o_proj_weight, bias=None)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    head_dim = d_model // num_heads
    batch_dims = tuple(in_features.shape[:-2])
    seq_len = in_features.shape[-2]
    device = in_features.device

    def _project(weight: Tensor) -> Tensor:
        proj = F.linear(in_features, weight, bias=None)
        new_shape = (*batch_dims, seq_len, num_heads, head_dim)
        proj = proj.reshape(new_shape)
        permute_order = list(range(len(batch_dims))) + [len(batch_dims) + 1, len(batch_dims), len(batch_dims) + 2]
        return proj.permute(permute_order)

    q = _project(q_proj_weight)
    k = _project(k_proj_weight)
    v = _project(v_proj_weight)

    if token_positions is None:
        base = torch.arange(seq_len, device=device, dtype=torch.long)
        view_shape = (1,) * len(batch_dims) + (seq_len,)
        token_positions = base.view(view_shape)
    else:
        token_positions = torch.as_tensor(token_positions, dtype=torch.long, device=device)
    target_shape = batch_dims + (seq_len,)
    if token_positions.shape != target_shape:
        missing = len(target_shape) - token_positions.ndim
        if missing < 0:
            raise ValueError("token_positions has too many dimensions for the provided input")
        shape = (1,) * missing + tuple(token_positions.shape)
        token_positions = token_positions.reshape(shape)
        token_positions = token_positions.expand(target_shape)

    rope_positions = token_positions.unsqueeze(-2).expand(*batch_dims, num_heads, seq_len)
    q = run_rope(head_dim, theta, max_seq_len, q, rope_positions)
    k = run_rope(head_dim, theta, max_seq_len, k, rope_positions)

    mask = _build_causal_mask(batch_dims, num_heads, seq_len, device)
    attn_output = run_scaled_dot_product_attention(q, k, v, mask=mask)
    permute_order = list(range(len(batch_dims))) + [len(batch_dims) + 1, len(batch_dims), len(batch_dims) + 2]
    attn_output = attn_output.permute(permute_order)
    merged = attn_output.reshape(*batch_dims, seq_len, d_model)
    return F.linear(merged, o_proj_weight, bias=None)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    if d_k % 2 != 0:
        raise ValueError("d_k must be even for RoPE")
    if theta <= 0:
        raise ValueError("theta must be positive")

    x = in_query_or_key
    device = x.device
    dtype = x.dtype
    seq_len = x.shape[-2]

    if token_positions is None:
        base = torch.arange(seq_len, device=device, dtype=torch.long)
        view_shape = (1,) * (x.ndim - 2) + (seq_len,)
        token_positions = base.view(view_shape)
    else:
        token_positions = torch.as_tensor(token_positions, dtype=torch.long, device=device)
        expected_prefix = x.shape[:-1]
        if token_positions.shape != expected_prefix:
            missing = len(expected_prefix) - token_positions.ndim
            if missing < 0:
                raise ValueError("token_positions incompatible with input shape")
            shape = (1,) * missing + tuple(token_positions.shape)
            token_positions = token_positions.reshape(shape)
            token_positions = token_positions.expand(expected_prefix)

    half_dim = d_k // 2
    freq_exponents = torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim
    inv_freq = torch.exp(-math.log(theta) * freq_exponents).to(dtype)
    angles = token_positions.to(dtype).unsqueeze(-1) * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    reshaped = x.reshape(*x.shape[:-1], half_dim, 2)
    x_even = reshaped[..., 0]
    x_odd = reshaped[..., 1]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    prefix_shape = in_query_or_key.shape[:-1]
    return torch.stack((rotated_even, rotated_odd), dim=-1).reshape(*prefix_shape, d_k)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    eps = 1e-5
    batch_dims = tuple(in_features.shape[:-2])
    seq_len = in_features.shape[-2]
    device = in_features.device

    base_positions = torch.arange(seq_len, device=device, dtype=torch.long)
    view_shape = (1,) * len(batch_dims) + (seq_len,)
    token_positions = base_positions.view(view_shape).expand(*batch_dims, seq_len)

    attn_input = run_rmsnorm(d_model=d_model, eps=eps, weights=weights["ln1.weight"], in_features=in_features)
    attn_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=attn_input,
        token_positions=token_positions,
    )
    residual = in_features + attn_output

    ffn_input = run_rmsnorm(d_model=d_model, eps=eps, weights=weights["ln2.weight"], in_features=residual)
    ffn_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=ffn_input,
    )
    return residual + ffn_output


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    if in_indices.shape[-1] > context_length:
        raise ValueError("sequence length exceeds context length")

    x = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices,
    )

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        layer_weights = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
        )

    x = run_rmsnorm(d_model=d_model, eps=1e-5, weights=weights["ln_final.weight"], in_features=x)
    logits = run_linear(
        d_in=d_model,
        d_out=vocab_size,
        weights=weights["lm_head.weight"],
        in_features=x,
    )
    return logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    if weights.shape != (d_model,):
        msg = f"weights shape {tuple(weights.shape)} does not match ({d_model},)"
        raise ValueError(msg)
    if in_features.shape[-1] != d_model:
        msg = f"Input features last dimension {in_features.shape[-1]} does not equal d_model {d_model}"
        raise ValueError(msg)

    variance = in_features.pow(2).mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(variance + eps)
    return in_features * scale * weights


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return F.silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    data = torch.as_tensor(dataset, dtype=torch.long)
    if data.ndim != 1:
        raise ValueError("dataset must be 1D")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if context_length >= data.shape[0]:
        raise ValueError("context_length must be smaller than dataset length")

    max_start = data.shape[0] - context_length
    starts = torch.randint(0, max_start, (batch_size,))
    offsets = torch.arange(context_length)
    x = data[starts.unsqueeze(1) + offsets]
    y = data[starts.unsqueeze(1) + offsets + 1]

    target_device = torch.device(device)
    return x.to(target_device), y.to(target_device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    shifted = in_features - in_features.max(dim=dim, keepdim=True).values
    exps = shifted.exp()
    return exps / exps.sum(dim=dim, keepdim=True)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    logits = inputs.to(torch.float32)
    targets = targets.to(torch.long)
    log_probs = logits.log_softmax(dim=-1)
    return F.nll_loss(log_probs, targets, reduction="mean")


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    clip_grad_norm_(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return torch.optim.AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if warmup_iters < 0 or cosine_cycle_iters < 0:
        raise ValueError("warmup_iters and cosine_cycle_iters must be non-negative")

    if warmup_iters > 0 and it <= warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    if cosine_cycle_iters <= 0:
        return min_learning_rate

    if it >= cosine_cycle_iters:
        return min_learning_rate

    cosine_span = max(cosine_cycle_iters - warmup_iters, 1)
    progress = (it - warmup_iters) / cosine_span
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(state, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


class _BPETokenizer:
    """Simple GPT-2 style BPE tokenizer supporting streaming inputs."""

    _STREAM_CHUNK_SIZE = 8192

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None,
    ) -> None:
        self._pretokenizer = regex.compile(GPT2_PRETOKENIZER_PATTERN)

        self._id_to_token_bytes: dict[int, bytes] = {}
        self._token_bytes_to_id: dict[bytes, int] = {}
        for token_id, token_bytes in vocab.items():
            idx = int(token_id)
            if not isinstance(token_bytes, (bytes, bytearray)):
                token_bytes = bytes(token_bytes)
            else:
                token_bytes = bytes(token_bytes)
            self._id_to_token_bytes[idx] = token_bytes
            self._token_bytes_to_id[token_bytes] = idx

        self._pair_ranks: dict[tuple[bytes, bytes], int] = {}
        for rank, pair in enumerate(merges):
            if len(pair) != 2:
                continue
            left, right = pair
            if not isinstance(left, (bytes, bytearray)):
                left = bytes(left)
            else:
                left = bytes(left)
            if not isinstance(right, (bytes, bytearray)):
                right = bytes(right)
            else:
                right = bytes(right)
            self._pair_ranks[(left, right)] = rank

        self._bpe_cache: dict[bytes, tuple[int, ...]] = {}

        deduped_specials: list[str] = []
        seen_specials: set[str] = set()
        if special_tokens:
            for token in special_tokens:
                if not isinstance(token, str):
                    msg = f"Expected special tokens to be strings, got {type(token)!r}"
                    raise TypeError(msg)
                if not token:
                    raise ValueError("Special tokens must be non-empty strings.")
                if token in seen_specials:
                    continue
                seen_specials.add(token)
                deduped_specials.append(token)

        self._special_tokens = deduped_specials
        self._special_token_to_id: dict[str, int] = {}
        self._special_regex: regex.Pattern[str] | None = None
        self._special_prefixes: dict[int, set[str]] = {}
        self._max_special_prefix_len = 0

        if self._special_tokens:
            regex_tokens = sorted(self._special_tokens, key=len, reverse=True)
            pattern = "|".join(regex.escape(token) for token in regex_tokens)
            self._special_regex = regex.compile(pattern)
            for token in self._special_tokens:
                token_bytes = token.encode("utf-8")
                token_id = self._token_bytes_to_id.get(token_bytes)
                if token_id is None:
                    msg = f"Special token {token!r} does not exist in the vocabulary."
                    raise ValueError(msg)
                self._special_token_to_id[token] = token_id
                for prefix_len in range(1, len(token)):
                    self._special_prefixes.setdefault(prefix_len, set()).add(token[:prefix_len])
                if len(token) > 1:
                    self._max_special_prefix_len = max(self._max_special_prefix_len, len(token) - 1)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            msg = f"Tokenizer.encode expects a string, got {type(text)!r}"
            raise TypeError(msg)
        return list(self._encode_from_chunks([text]))

    def encode_iterable(self, iterable: Iterable[str] | IO[str]) -> Iterable[int]:
        chunks = self._chunk_source(iterable)

        def generator() -> Iterable[int]:
            yield from self._encode_from_chunks(chunks)

        return generator()

    def decode(self, token_ids: Iterable[int]) -> str:
        byte_segments: list[bytes] = []
        for token_id in token_ids:
            idx = int(token_id)
            try:
                token_bytes = self._id_to_token_bytes[idx]
            except KeyError as exc:
                raise KeyError(f"Unknown token id {idx}") from exc
            byte_segments.append(token_bytes)
        data = b"".join(byte_segments)
        if not data:
            return ""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            # Decoding individual tokens may produce incomplete multi-byte sequences.
            # Fall back to a byte-preserving decode so callers can still inspect tokens.
            return data.decode("latin-1")

    def _chunk_source(self, source: Iterable[str] | IO[str]) -> Iterable[str]:
        read_method = getattr(source, "read", None)
        if callable(read_method):
            while True:
                chunk = read_method(self._STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                if not isinstance(chunk, str):
                    chunk = chunk.decode("utf-8")
                if chunk:
                    yield chunk
            return
        for chunk in source:
            if not isinstance(chunk, str):
                msg = f"encode_iterable expects strings, got {type(chunk)!r}"
                raise TypeError(msg)
            if chunk:
                yield chunk

    def _encode_from_chunks(self, chunks: Iterable[str]) -> Iterable[int]:
        for segment, is_special in self._split_on_special(chunks):
            if not segment:
                continue
            if is_special:
                yield self._special_token_to_id[segment]
                continue
            for match in self._pretokenizer.finditer(segment):
                piece = match.group(0)
                if not piece:
                    continue
                token_bytes = piece.encode("utf-8")
                if not token_bytes:
                    continue
                yield from self._bpe(token_bytes)

    def _split_on_special(self, chunks: Iterable[str]) -> Iterable[tuple[str, bool]]:
        if not self._special_regex:
            for chunk in chunks:
                if chunk:
                    yield chunk, False
            return

        buffer = ""
        for chunk in chunks:
            if not chunk:
                continue
            buffer += chunk
            while True:
                match = self._special_regex.search(buffer)
                if not match:
                    break
                start, end = match.span()
                if start:
                    yield buffer[:start], False
                yield match.group(0), True
                buffer = buffer[end:]
            keep = self._pending_special_prefix_length(buffer)
            if keep == 0:
                if buffer:
                    yield buffer, False
                    buffer = ""
            else:
                safe_len = len(buffer) - keep
                if safe_len > 0:
                    yield buffer[:safe_len], False
                    buffer = buffer[safe_len:]
        if buffer:
            yield buffer, False

    def _pending_special_prefix_length(self, text: str) -> int:
        if self._max_special_prefix_len == 0 or not text:
            return 0
        upto = min(len(text), self._max_special_prefix_len)
        for length in range(upto, 0, -1):
            suffix = text[-length:]
            prefixes = self._special_prefixes.get(length)
            if prefixes and suffix in prefixes:
                return length
        return 0

    def _bpe(self, token_bytes: bytes) -> tuple[int, ...]:
        cached = self._bpe_cache.get(token_bytes)
        if cached is not None:
            return cached

        if token_bytes in self._token_bytes_to_id:
            result = (self._token_bytes_to_id[token_bytes],)
            self._bpe_cache[token_bytes] = result
            return result

        word = tuple(token_bytes[i : i + 1] for i in range(len(token_bytes)))
        pairs = self._get_pairs(word)

        while pairs:
            best_pair = min(
                pairs,
                key=lambda pair: self._pair_ranks.get(pair, float("inf")),
            )
            if best_pair not in self._pair_ranks:
                break
            first, second = best_pair
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == first
                    and word[i + 1] == second
                ):
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        result = tuple(self._token_bytes_to_id[symbol] for symbol in word)
        self._bpe_cache[token_bytes] = result
        return result

    @staticmethod
    def _get_pairs(word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
        pairs: set[tuple[bytes, bytes]] = set()
        if len(word) < 2:
            return pairs
        prev = word[0]
        for symbol in word[1:]:
            pairs.add((prev, symbol))
            prev = symbol
        return pairs


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    if vocab is None:
        raise ValueError("vocab must be provided.")
    if merges is None:
        raise ValueError("merges must be provided.")
    return _BPETokenizer(vocab, merges, special_tokens or [])



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
    # 修正：范围应该是 256 (0-255)，原文的 28 可能是笔误
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