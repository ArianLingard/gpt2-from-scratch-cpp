#pragma once
#include "model.h"

#include <string>

// 把 HuggingFace 的 GPT-2 权重（safetensors）加载到模型中
void load_weights(std::shared_ptr<GPT2> model, const std::string& path, torch::Device device);
