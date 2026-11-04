#include "model.h"
#include "weights.h"
#include "generate.h"
#include "bpe.h"

#include <torch/torch.h>
#include "json.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char* argv[]) {
    try {
        // 用户输入提示词：优先取命令行参数，否则从标准输入读一行
        std::string prompt_text;
        if (argc > 1) {
            for (int a = 1; a < argc; ++a) {
                if (a > 1) prompt_text += ' ';
                prompt_text += argv[a];
            }
        } else {
            std::cout << "请输入提示词（回车确认）: " << std::flush;
            std::getline(std::cin, prompt_text);
        }
        if (prompt_text.empty()) {
            std::cout << "提示词为空，使用默认值 \"Hello, I am\"\n";
            prompt_text = "Hello, I am";
        }

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

        auto model = std::make_shared<GPT2>(50257, 768, 12, 3072, 12);
        model->to(device);
        load_weights(model, "checkpoint/model.safetensors", device);

        std::ifstream f("checkpoint/vocab.json");
        if (!f) throw std::runtime_error("找不到 checkpoint/vocab.json");
        nlohmann::json vocab;
        f >> vocab;
        std::unordered_map<int64_t, std::string> id2token;
        std::unordered_map<std::string, int64_t> token2id;
        for (auto& [tok, id] : vocab.items()) {
            int64_t vid = id.get<int64_t>();
            id2token[vid] = tok;
            token2id[tok] = vid;
        }

        // 读 BPE 合并表，用于把文本编码成 token id
        auto ranks = load_merges("checkpoint/merges.txt");

        // 把自然语言提示词编码成 token id（BPE）
        auto prompt_ids = encode(prompt_text, token2id, ranks);
        auto prompt = torch::tensor(prompt_ids, torch::TensorOptions().dtype(torch::kLong))
                          .unsqueeze(0).to(device);
        std::cout << "编码 \"" << prompt_text << "\" ->";
        for (auto id : prompt_ids) std::cout << " " << id;
        std::cout << "\n\n";

        struct Strategy { std::string name; float temp; int top_k; };
        std::vector<Strategy> strategies = {
            {"贪心 (T->0)", 0.01f, 0},
            {"采样 (T=0.7, Top-K=50)", 0.7f, 50},
            {"发散 (T=1.2, Top-K=100)", 1.2f, 100},
        };

        for (const auto& s : strategies) {
            auto ids = generate(model, prompt, 100, s.temp, s.top_k, 50256, {198, 628});
            std::cout << "[" << s.name << "]\n"
                      << prompt_text << decode(ids, id2token) << "\n\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
