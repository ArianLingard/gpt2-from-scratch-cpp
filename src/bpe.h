#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// 读 merges.txt，返回 "合并后的 token" -> 优先级（rank 越小越先合并）
std::unordered_map<std::string, int> load_merges(const std::string& path);

// 字节级 BPE 解码：token id -> 文本
std::string decode(const std::vector<int64_t>& ids,
                   const std::unordered_map<int64_t, std::string>& id2token);

// 字节级 BPE 编码：文本 -> token id
std::vector<int64_t> encode(const std::string& text,
                            const std::unordered_map<std::string, int64_t>& token2id,
                            const std::unordered_map<std::string, int>& ranks);
