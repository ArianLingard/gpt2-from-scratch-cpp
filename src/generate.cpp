#include "generate.h"

#include <algorithm>
#include <limits>
#include <tuple>

std::vector<int64_t> generate(
    std::shared_ptr<GPT2> model,
    torch::Tensor input_tokens,
    int max_new_tokens,
    float temperature,
    int top_k,
    int64_t eos_id,
    const std::vector<int64_t>& banned)
{
    torch::Device device = input_tokens.device();
    std::vector<int64_t> out;

    torch::NoGradGuard no_grad;
    float temp = std::max(temperature, 1e-5f);
    int64_t max_len = model->max_len;

    // 每层一个 KV 缓存，生成过程中复用历史 K/V，只算新 token
    std::vector<KVCache> caches(model->n_layers);

    // 预填充：先把整段 prompt 送进去（超过 max_len 只保留最后 max_len 个）
    int64_t seq = input_tokens.size(1);
    torch::Tensor next = seq > max_len ? input_tokens.slice(1, seq - max_len, seq) : input_tokens;

    for (int step = 0; step < max_new_tokens; ++step) {
        if (caches[0].len() >= max_len) break;   // 达到 GPT-2 位置编码上限

        // 增量前向：第一轮算整段 prompt，之后每轮只算 1 个新 token
        auto logits = model->forward(next, caches)[0][-1] / temp;

        // Top-K 过滤：只保留概率最高的 K 个候选
        if (top_k > 0) {
            float th = std::get<0>(torch::topk(logits, top_k))[-1].item<float>();
            logits = torch::where(logits < th,
                                  torch::full_like(logits, -std::numeric_limits<float>::infinity()),
                                  logits);
        }

        // 屏蔽指定 token（换行符 198/628），让输出保持连续
        if (!banned.empty())
            logits = logits.index_fill(0, torch::tensor(banned, torch::kLong).to(device),
                                       -std::numeric_limits<float>::infinity());

        int64_t tok;
        if (temperature <= 1e-5)
            tok = logits.argmax().item<int64_t>();                          // 贪心
        else
            tok = torch::multinomial(torch::softmax(logits, -1), 1).item<int64_t>();

        if (tok == eos_id) break;
        out.push_back(tok);
        next = torch::tensor({{tok}}, torch::kLong).to(device);   // 下一轮只算这 1 个 token
    }
    return out;
}
