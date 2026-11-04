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
    torch::Tensor tokens = input_tokens.clone();
    std::vector<int64_t> out;

    torch::NoGradGuard no_grad;
    float temp = std::max(temperature, 1e-5f);

    for (int step = 0; step < max_new_tokens; ++step) {
        // GPT-2 最多 1024 个位置，超长只保留最后 1024 个
        int64_t seq = tokens.size(1);
        auto input = seq > 1024 ? tokens.slice(1, seq - 1024, seq) : tokens;

        // 取最后一个 token 的 logits，并做温度缩放
        auto logits = model->forward(input)[0][-1] / temp;

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

        int64_t next;
        if (temperature <= 1e-5)
            next = logits.argmax().item<int64_t>();                          // 贪心
        else
            next = torch::multinomial(torch::softmax(logits, -1), 1).item<int64_t>();

        if (next == eos_id) break;
        out.push_back(next);
        tokens = torch::cat({tokens, torch::tensor({{next}}, torch::kLong).to(device)}, 1);
    }
    return out;
}
