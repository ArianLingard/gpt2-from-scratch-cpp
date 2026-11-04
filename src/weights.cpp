#include "weights.h"

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

#include <stdexcept>
#include <string>
#include <vector>

// 把 HuggingFace 的 GPT-2 命名映射到上面的模型
void load_weights(std::shared_ptr<GPT2> model, const std::string& path, torch::Device device) {
    torch::NoGradGuard no_grad; // copy_ 是 in-place 操作，避免 autograd 报错

    safetensors::safetensors_t st;
    std::string err;
    if (!safetensors::load_from_file(path, &st, nullptr, &err))
        throw std::runtime_error("加载 safetensors 失败: " + err);

    // 按名字取张量，并搬到目标设备
    auto get = [&](const std::string& name) {
        safetensors::tensor_t t;
        if (!st.tensors.at(name, &t))
            throw std::runtime_error("缺少权重: " + name);

        std::vector<int64_t> shape(t.shape.begin(), t.shape.end());
        return torch::from_blob(
                   st.storage.data() + t.data_offsets[0],      // 数据缓冲区的字节偏移
                   shape,
                   torch::TensorOptions().dtype(torch::kFloat32)) // GPT-2 权重均为 F32
            .clone()
            .to(device);
    };

    model->wte->weight.copy_(get("wte.weight"));
    model->wpe->weight.copy_(get("wpe.weight"));

    for (int64_t i = 0; i < model->n_layers; ++i) {
        std::string p = "h." + std::to_string(i) + ".";
        auto b = (*model->blocks)[i]->as<TransformerBlock>();

        b->ln1->weight.copy_(get(p + "ln_1.weight"));
        b->ln1->bias.copy_(get(p + "ln_1.bias"));
        b->ln2->weight.copy_(get(p + "ln_2.weight"));
        b->ln2->bias.copy_(get(p + "ln_2.bias"));

        // HF 把 Q/K/V 拼成一个矩阵 c_attn，这里按 1/3 切回三份
        auto qkv_w = get(p + "attn.c_attn.weight");
        auto qkv_b = get(p + "attn.c_attn.bias");
        b->attn->W_q->weight.copy_(qkv_w.chunk(3, 1)[0].transpose(0, 1));
        b->attn->W_k->weight.copy_(qkv_w.chunk(3, 1)[1].transpose(0, 1));
        b->attn->W_v->weight.copy_(qkv_w.chunk(3, 1)[2].transpose(0, 1));
        b->attn->W_q->bias.copy_(qkv_b.chunk(3, 0)[0]);
        b->attn->W_k->bias.copy_(qkv_b.chunk(3, 0)[1]);
        b->attn->W_v->bias.copy_(qkv_b.chunk(3, 0)[2]);

        b->attn->W_o->weight.copy_(get(p + "attn.c_proj.weight").transpose(0, 1));
        b->attn->W_o->bias.copy_(get(p + "attn.c_proj.bias"));

        b->ffn->W1->weight.copy_(get(p + "mlp.c_fc.weight").transpose(0, 1));
        b->ffn->W1->bias.copy_(get(p + "mlp.c_fc.bias"));
        b->ffn->W2->weight.copy_(get(p + "mlp.c_proj.weight").transpose(0, 1));
        b->ffn->W2->bias.copy_(get(p + "mlp.c_proj.bias"));
    }

    model->final_ln->weight.copy_(get("ln_f.weight"));
    model->final_ln->bias.copy_(get("ln_f.bias"));
}
