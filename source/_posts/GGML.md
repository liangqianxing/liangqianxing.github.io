---
title: GGML
date: 2025-11-28 10:18:58
categories:
  - ???
tags:
  - GGML
  - LLM
  - GPT-2
mathjax: true
toc:
  enable: true
  number: false
---

# GGML ??????

?????????????????? -> ???? -> ??? -> ?????????? GPT-2 ?????????????????

## ????

- ?? `vocab` ? token ???????????? token id
- ???? `gpt2_eval` ???/????? logits ???
- `main-ctx.cpp` ????????????? -> ?? -> ???
- `ggml_tensor` ??????????
- ?????`mul_mat` / `norm` / `softmax` / `rope`????????

## ????

| ?? | ?? |
| --- | --- |
| vocab | token ? id ????????? id -> token ?? |
| tokenization | ???????? token??/??/??? |
| lookup | ? token ????? id |
| logits | ??????????????? `n_vocab` |
| KV cache | ???? token ? K/V????????? |
| `n_past` | KV cache ???? token ? |
| `n_batch` | ??????? token ??? |

## ?? Prompt?gpt_random_prompt?

???? prompt ?????????????????????

```c++
std::string gpt_random_prompt(std::mt19937 & rng) {
    const int r = rng() % 10;
    switch (r) {
        case 0: return "So";
        case 1: return "Once upon a time";
        case 2: return "When";
        case 3: return "The";
        case 4: return "After";
        case 5: return "If";
        case 6: return "import";
        case 7: return "He";
        case 8: return "She";
        case 9: return "They";
    }

    return "The";
}
```

## gpt2_eval?????

?????

```c++
bool gpt2_eval(
        const gpt2_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token);
```

?????

| ?? | ?? |
| --- | --- |
| `model` | GPT-2 ??????? embedding??????KV ???? |
| `n_threads` | ggml ???????`-t/--threads`? |
| `n_past` | ??? token ???? KV cache ??? causal mask |
| `embd_inp` | ????? token id ?? |
| `embd_w` | ?? logits????? token ??? `n_vocab` ??? |
| `mem_per_token` | ?? token ???????????????? |

?????

- ????????????? `mem_per_token`????????
- ????????? `gpt2_eval`??? logits ???? token?

## ??????gpt2-ctx-main?

??????

1. ??????? seed????????
2. Tokenize prompt??? `n_predict` ????????
3. ???? `gpt2_eval` ??? `mem_per_token`
4. ?????`gpt2_eval` -> ?? -> ?? context -> ?? token
5. ?? `50256`??? token???????????

```c++
int main(int argc, char ** argv) {
    ggml_time_init();

    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "models/gpt-2-117M/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    gpt2_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!gpt2_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;

        test_gpt_tokenizer(vocab, params.token_test);
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, embd_inp.size());
    for (int i = 0; i < std::min(8, (int) embd_inp.size()); i++) {
        printf("%d ", embd_inp[i]);
    }
    printf("\n\n");

    // submit the input prompt token-by-token
    // this reduces the memory usage during inference, at the cost of a bit of speed at the beginning
    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    gpt2_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (size_t i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!gpt2_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (size_t k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (int32_t(embd.size()) >= params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 50256) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx_w);

    return 0;
}
```

## ggml_tensor???????

```c++
struct ggml_tensor {
    enum ggml_type type;
    struct ggml_backend_buffer * buffer;
    int64_t ne[GGML_MAX_DIMS]; // ????
    size_t  nb[GGML_MAX_DIMS]; // ?? stride
    enum ggml_op op;           // ????????
    int32_t op_params[...];
    int32_t flags;
    struct ggml_tensor * src[GGML_MAX_SRC]; // ?????
    struct ggml_tensor * view_src;
    size_t view_offs;
    void * data;               // ??????
    char name[GGML_MAX_NAME];
    void * extra;
    char padding[8];
};
```

?????

- `ne` / `nb`?????? stride??????????
- `op` / `src`????????????????
- `view_src` / `view_offs`????????????????
- `data`???????????????????

## ggml_op?????

`ggml_op` ????????????????? op ????????

| op | ?? |
| --- | --- |
| `GGML_OP_MUL_MAT` | ???? |
| `GGML_OP_NORM` | ????LayerNorm/RMSNorm ????? |
| `GGML_OP_SOFT_MAX` | Softmax |
| `GGML_OP_ROPE` | Rotary Positional Embedding |

## ???????

**??**?? C++ ??? ggml API ??? + ?????????????????? `ggml_mul_mat`?`ggml_norm`?`ggml_soft_max_inplace` ??????? `ggml_build_forward_expand` ? `ggml_graph_compute_with_ctx` ???

**??**?`ggml_compute_backward()` ?? `ggml_op` ???????????????? `ggml/src/ggml.c`??? `GGML_OP_MUL_MAT` ?????????????

## ??????

### ggml_mul_mat

```c++
// ggml_mul_mat
static inline bool ggml_can_mul_mat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
    return (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));
    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}
void ggml_mul_mat_set_prec(
        struct ggml_tensor * a,
        enum ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);
    const int32_t prec_i32 = (int32_t) prec;
    ggml_set_op_params_i32(a, 0, prec_i32);
}
```

???`ggml_can_mul_mat` ???????????`ne` ????????????? `F32`?

### ggml_norm

```c++
// ggml_norm

static struct ggml_tensor * ggml_norm_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 eps,
        bool                  inplace) {
    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, sizeof(eps));

    result->op     = GGML_OP_NORM;
    result->src[0] = a;

    return result;
}

struct ggml_tensor * ggml_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 eps) {
    return ggml_norm_impl(ctx, a, eps, false);
}

struct ggml_tensor * ggml_norm_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 eps) {
    return ggml_norm_impl(ctx, a, eps, true);
}
```

???`inplace` ??? view?????????`eps` ?? `op_params` ???

### ggml_softmax

```c++
// ggml_soft_max

static struct ggml_tensor * ggml_soft_max_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * mask,
        float                 scale,
        float                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    if (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    if (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    float params[] = { scale, max_bias };
    ggml_set_op_params(result, params, sizeof(params));

    result->op     = GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    return result;
}

struct ggml_tensor * ggml_soft_max(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, false);
}

struct ggml_tensor * ggml_soft_max_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, true);
}

struct ggml_tensor * ggml_soft_max_ext(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * mask,
        float                 scale,
        float                 max_bias) {
    return ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

struct ggml_tensor * ggml_soft_max_ext_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * mask,
        float                 scale,
        float                 max_bias) {
    return ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

void ggml_soft_max_add_sinks(
        struct ggml_tensor * a,
        struct ggml_tensor * sinks) {
    if (!sinks) {
        a->src[2] = NULL;
        return;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == NULL);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}
```

C ??????`struct ggml_tensor *` ???? C ???????????????????ggml ???? `typedef` ? `struct ggml_tensor` ??????????????? `struct ggml_tensor *`?

### ggml_rope

```c++
// ggml_rope

static struct ggml_tensor * ggml_rope_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * c,
        int                   n_dims,
        int                   sections[GGML_MROPE_SECTIONS],
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    if (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } else {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    if (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[15] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    sizeof(float));
    memcpy(params +  6, &freq_scale,   sizeof(float));
    memcpy(params +  7, &ext_factor,   sizeof(float));
    memcpy(params +  8, &attn_factor,  sizeof(float));
    memcpy(params +  9, &beta_fast,    sizeof(float));
    memcpy(params + 10, &beta_slow,    sizeof(float));
    if (mrope_used && sections) {
        memcpy(params + 11, sections,  sizeof(int32_t) * GGML_MROPE_SECTIONS);
    } else {
        memset(params + 11, 0,         sizeof(int32_t) * GGML_MROPE_SECTIONS);
    }
    ggml_set_op_params(result, params, sizeof(params));

    result->op     = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

struct ggml_tensor * ggml_rope(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int                   n_dims,
        int                   mode) {
    return ggml_rope_impl(
        ctx, a, b, NULL, n_dims, NULL, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

struct ggml_tensor * ggml_rope_multi(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * c,
        int                   n_dims,
        int                   sections[GGML_MROPE_SECTIONS],
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

struct ggml_tensor * ggml_rope_multi_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * c,
        int                   n_dims,
        int                   sections[GGML_MROPE_SECTIONS],
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

struct ggml_tensor * ggml_rope_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int                   n_dims,
        int                   mode) {
    return ggml_rope_impl(
        ctx, a, b, NULL, n_dims, NULL, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

struct ggml_tensor * ggml_rope_ext(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ggml_rope_impl(
        ctx, a, b, c, n_dims, NULL, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

struct ggml_tensor * ggml_rope_ext_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ggml_rope_impl(
        ctx, a, b, c, n_dims, NULL, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

struct ggml_tensor * ggml_rope_custom(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ggml_rope_impl(
        ctx, a, b, NULL, n_dims, NULL, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

struct ggml_tensor * ggml_rope_custom_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ggml_rope_impl(
        ctx, a, b, NULL, n_dims, NULL, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

void ggml_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

struct ggml_tensor * ggml_rope_ext_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    struct ggml_tensor * result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    return result;
}

struct ggml_tensor * ggml_rope_multi_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * c,
        int                   n_dims,
        int                   sections[4],
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    struct ggml_tensor * result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    return result;
}
```

???

- `mode` ?????? MROPE?`sections` ?? MROPE ???
- `op_params` ?? `n_dims`?`n_ctx_orig` ? `freq_*` ????????
