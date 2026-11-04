# GPT-2 从零实现 Demo（C++ / LibTorch）

一个用 **C++ 17 + LibTorch** 从零手写的 GPT-2（124M 配置）推理示例。不依赖 HuggingFace 的 `transformers`，只用 LibTorch 的底层算子实现了 Transformer 结构、safetensors 权重加载、字节级 BPE 分词，以及三种采样策略的自回归文本生成。

## 功能

- **模型结构**：`MultiHeadAttention`（缩放点积 + 因果掩码）、`FeedForward`、`TransformerBlock`（Pre-LayerNorm + 残差）、`GPT2`
- **权重加载**：从 HuggingFace 的 `model.safetensors` 读取权重，并把 `c_attn`（Q/K/V 拼合矩阵）按 1/3 拆回三份
- **BPE 编码/解码**：完整实现 GPT-2 的字节级 BPE —— 预分词（复刻 GPT-2 正则）、字节↔Unicode 映射、`merges.txt` 合并、`vocab.json` 查表
- **文本生成**：基于 KV Cache 的自回归生成，支持贪心、温度采样 + Top-K 三种策略，可屏蔽指定 token（如换行符）
- **用户输入提示词**：支持命令行参数或标准输入

## 目录结构

```
GPT-2_demo/
├── CMakeLists.txt          # 构建脚本
├── README.md
├── include/                # 第三方头文件（已随项目提供）
│   ├── json.hpp            # nlohmann/json（解析 vocab.json）
│   └── safetensors.hh      # safetensors 读取器
├── src/
│   ├── model.h             # GPT-2 模型结构（header-only）
│   ├── weights.h/.cpp      # safetensors 权重加载
│   ├── generate.h/.cpp     # 自回归文本生成
│   ├── bpe.h/.cpp          # 字节级 BPE 编码/解码
│   └── main.cpp            # 程序入口
└── checkpoint/             # 模型与词表（需自行准备，见下文）
    ├── model.safetensors   # GPT-2 124M 权重
    ├── vocab.json          # token -> id 映射
    └── merges.txt          # BPE 合并表
```

## 依赖

- **LibTorch**（C++ 发行版），本项目的 `CMakeLists.txt` 中写死了 `D:/LibTorch`，请按需修改
- **CMake** ≥ 3.18
- **编译器**：Windows 上使用 MSVC（Visual Studio 2022 / 2026）；CUDA 可选，未安装时会自动退回到 CPU
- `json.hpp` 与 `safetensors.hh` 已放在 `include/`，无需额外下载

### 准备 checkpoint 文件

在 `checkpoint/` 下放置三个文件（GPT-2 124M 的 HuggingFace 权重）：

- `model.safetensors`
- `vocab.json`
- `merges.txt`

## 构建

在项目根目录执行：

```bash
# Visual Studio 生成器（会生成到 build/Release/）
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release

# 或使用 Ninja（生成到 build/ 根目录）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

> 若你的 Visual Studio 是 2022 版，把生成器改为 `"Visual Studio 17 2022"`；若 LibTorch 不在 `D:/LibTorch`，改 `CMakeLists.txt` 里的 `CMAKE_PREFIX_PATH`。

## 运行

程序从**当前工作目录**读取 `checkpoint/`，请确保运行目录下能找到这三个文件（以及 LibTorch 的 DLL）。

```bash
# 方式一：命令行参数传入提示词（多个参数会自动用空格拼接）
.\build\Release\libtorch_demo.exe "The capital of France is"
.\build\Release\libtorch_demo.exe Hello, I am

# 方式二：不带参数，从标准输入读取一行
.\build\Release\libtorch_demo.exe
```

示例输出：

```
编码 "Hello, I am" -> 15496 11 314 716

[贪心 (T->0)]
Hello, I am a little bit of a fan of the original series. ...

[采样 (T=0.7, Top-K=50)]
Hello, I am not a human. I am a zombie, ...

[发散 (T=1.2, Top-K=100)]
Hello, I am so glad that I found you. ...
```

### Windows 小贴士

- 若终端中文乱码，运行前先执行 `chcp 65001` 切换到 UTF-8 代码页
- LibTorch 的 DLL（`torch.dll`、`torch_cpu.dll`、`c10.dll` 等）需放在 exe 同目录、运行目录或 `PATH` 中

## 采样策略说明

| 策略 | 温度 T | Top-K | 说明 |
|------|--------|-------|------|
| 贪心 | ~0    | 0     | 每步取概率最大的 token，输出确定但易重复 |
| 采样 | 0.7   | 50    | 温度缩放 + 从 Top-50 里随机采样，较平衡 |
| 发散 | 1.2   | 100   | 高温 + 更大的候选集，输出更具随机性 |

> 生成时通过 `banned` 屏蔽了换行符 token（198/628），让输出保持连续段落。

## 实现要点

- **KV Cache**：生成时缓存每层注意力算出的 K/V，每个新 token 只做一次增量前向（`forward(x, caches)`），避免每步重复计算整段历史
- **权重共享**：`lm_head->weight` 与 `wte->weight` 指向同一张权重
- **c_attn 拆分**：HF 把 Q/K/V 拼成一个 `[3*d, d]` 矩阵，加载时按列切成三份并转置
- **字节级 BPE**：每个 UTF-8 字节先映射成一个 Unicode 字符（不可打印字节映射到 `256+` 的码点，如空格 `Ġ`），再做 BPE 合并；编码与解码共用同一套 `unicode_to_byte` 映射
- **位置编码**：最多 1024 个位置，超长输入只保留最后 1024 个 token
