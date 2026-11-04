#pragma once
#include "model.h"

#include <cstdint>
#include <vector>

// 自回归生成：给定输入 token，逐 token 采样出后续文本
std::vector<int64_t> generate(
    std::shared_ptr<GPT2> model,
    torch::Tensor input_tokens,
    int max_new_tokens,
    float temperature = 1.0f,
    int top_k = 0,
    int64_t eos_id = 50256,                        // GPT-2 的结束符 <|endoftext|>
    const std::vector<int64_t>& banned = {});      // 需要屏蔽的 token（如换行符）
