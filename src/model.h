#pragma once
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// ===========================================================================
// KV 缓存：生成时缓存每层注意力算出的 K / V，避免重复计算历史 token
// ===========================================================================

struct KVCache {
    torch::Tensor k, v;   // [B, num_heads, T, d_k]

    bool empty() const { return !k.defined(); }
    int64_t len() const { return k.defined() ? k.size(2) : 0; }
};

// ===========================================================================
// GPT-2 模型结构：一个最小可用的 GPT-2（124M 配置）
// ===========================================================================

struct MultiHeadAttention : torch::nn::Module {
    int64_t d_model, num_heads, d_k;
    torch::nn::Linear W_q{nullptr}, W_k{nullptr}, W_v{nullptr}, W_o{nullptr};

    MultiHeadAttention(int64_t d_model, int64_t num_heads)
        : d_model(d_model), num_heads(num_heads) {
        TORCH_CHECK(d_model % num_heads == 0, "d_model 必须能被 num_heads 整除");
        d_k = d_model / num_heads;
        W_q = register_module("W_q", torch::nn::Linear(d_model, d_model));
        W_k = register_module("W_k", torch::nn::Linear(d_model, d_model));
        W_v = register_module("W_v", torch::nn::Linear(d_model, d_model));
        W_o = register_module("W_o", torch::nn::Linear(d_model, d_model));
    }

    // 完整前向：一次性算整段序列（训练 / 预填充的对照实现）
    torch::Tensor forward(torch::Tensor x) {
        int64_t B = x.size(0), T = x.size(1);

        // [B, T, d_model] -> [B, num_heads, T, d_k]
        auto Q = W_q->forward(x).view({B, T, num_heads, d_k}).transpose(1, 2);
        auto K = W_k->forward(x).view({B, T, num_heads, d_k}).transpose(1, 2);
        auto V = W_v->forward(x).view({B, T, num_heads, d_k}).transpose(1, 2);

        // 缩放点积注意力 + 因果掩码（只能看到当前及之前的 token）
        auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt(double(d_k));
        auto mask = torch::triu(torch::ones({T, T}, x.options().dtype(torch::kBool)), 1);
        scores = scores.masked_fill(mask, -std::numeric_limits<float>::infinity());

        auto attn = torch::matmul(torch::softmax(scores, -1), V);

        // 拼回头，再过输出投影
        attn = attn.transpose(1, 2).contiguous().view({B, T, d_model});
        return W_o->forward(attn);
    }

    // 增量前向：只算 x 里的新 token（[B, T]），历史 K/V 直接用缓存，避免重复计算
    torch::Tensor forward(torch::Tensor x, KVCache& cache) {
        int64_t B = x.size(0), T = x.size(1);
        int64_t hist = cache.len();   // 本次新增之前的缓存长度

        auto Q = W_q->forward(x).view({B, T, num_heads, d_k}).transpose(1, 2);
        auto K = W_k->forward(x).view({B, T, num_heads, d_k}).transpose(1, 2);
        auto V = W_v->forward(x).view({B, T, num_heads, d_k}).transpose(1, 2);

        // 把新 token 的 K/V 追加进缓存
        if (cache.empty()) {
            cache.k = K;
            cache.v = V;
        } else {
            cache.k = torch::cat({cache.k, K}, 2);
            cache.v = torch::cat({cache.v, V}, 2);
        }

        // 因果掩码：新 token 只能看到自己及之前的位置（缓存里全是 ≤ 当前的位置）
        int64_t total = cache.k.size(2);
        auto mask = torch::triu(torch::ones({T, total}, x.options().dtype(torch::kBool)), hist + 1);
        auto scores = torch::matmul(Q, cache.k.transpose(-2, -1)) / std::sqrt(double(d_k));
        scores = scores.masked_fill(mask, -std::numeric_limits<float>::infinity());

        auto attn = torch::matmul(torch::softmax(scores, -1), cache.v);
        attn = attn.transpose(1, 2).contiguous().view({B, T, d_model});
        return W_o->forward(attn);
    }
};

struct FeedForward : torch::nn::Module {
    torch::nn::Linear W1{nullptr}, W2{nullptr};

    FeedForward(int64_t d_model, int64_t d_ff) {
        W1 = register_module("W1", torch::nn::Linear(d_model, d_ff));
        W2 = register_module("W2", torch::nn::Linear(d_ff, d_model));
    }

    torch::Tensor forward(torch::Tensor x) {
        return W2->forward(torch::gelu(W1->forward(x)));
    }
};

struct TransformerBlock : torch::nn::Module {
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    std::shared_ptr<MultiHeadAttention> attn;
    std::shared_ptr<FeedForward> ffn;

    TransformerBlock(int64_t d_model, int64_t n_heads, int64_t d_ff) {
        ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        attn = register_module("attn", std::make_shared<MultiHeadAttention>(d_model, n_heads));
        ffn = register_module("ffn", std::make_shared<FeedForward>(d_model, d_ff));
    }

    // Pre-LayerNorm + 残差连接
    torch::Tensor forward(torch::Tensor x) {
        x = x + attn->forward(ln1->forward(x));
        x = x + ffn->forward(ln2->forward(x));
        return x;
    }

    torch::Tensor forward(torch::Tensor x, KVCache& cache) {
        x = x + attn->forward(ln1->forward(x), cache);
        x = x + ffn->forward(ln2->forward(x));
        return x;
    }
};

struct GPT2 : torch::nn::Module {
    torch::nn::Embedding wte{nullptr}, wpe{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::LayerNorm final_ln{nullptr};
    torch::nn::Linear lm_head{nullptr};
    int64_t n_layers, max_len;

    GPT2(int64_t vocab_size, int64_t d_model, int64_t n_heads,
         int64_t d_ff, int64_t n_layers, int64_t max_len = 1024) {
        this->n_layers = n_layers;
        this->max_len = max_len;

        wte = register_module("wte", torch::nn::Embedding(vocab_size, d_model));
        wpe = register_module("wpe", torch::nn::Embedding(max_len, d_model));

        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int64_t i = 0; i < n_layers; ++i)
            blocks->push_back(register_module("block_" + std::to_string(i),
                std::make_shared<TransformerBlock>(d_model, n_heads, d_ff)));

        final_ln = register_module("final_ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        lm_head = register_module("lm_head", torch::nn::Linear(d_model, vocab_size));

        // 权重共享：输出头和词嵌入共用同一张权重
        lm_head->weight = wte->weight;
    }

    torch::Tensor forward(torch::Tensor tokens) {
        int64_t T = tokens.size(1);
        auto pos = torch::arange(T, torch::kLong).to(tokens.device());
        auto x = wte->forward(tokens) + wpe->forward(pos).unsqueeze(0);

        for (const auto& block : *blocks)
            x = block->as<TransformerBlock>()->forward(x);

        return lm_head->forward(final_ln->forward(x));
    }

    // 增量前向：tokens 是新 token（[B, T]），caches 每层一个并就地更新。
    // 位置由缓存长度决定，因此只对不超过 max_len 的序列有效。
    torch::Tensor forward(torch::Tensor tokens, std::vector<KVCache>& caches) {
        int64_t T = tokens.size(1);
        int64_t pos = caches.empty() ? 0 : caches[0].len();
        auto pos_ids = torch::arange(pos, pos + T, torch::kLong).to(tokens.device());
        auto x = wte->forward(tokens) + wpe->forward(pos_ids).unsqueeze(0);

        for (int64_t i = 0; i < n_layers; ++i)
            x = (*blocks)[i]->as<TransformerBlock>()->forward(x, caches[i]);

        return lm_head->forward(final_ln->forward(x));
    }
};
