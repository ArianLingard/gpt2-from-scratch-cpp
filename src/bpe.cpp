#include "bpe.h"

#include <fstream>
#include <limits>
#include <stdexcept>

// ===========================================================================
// 内部工具函数（均为文件级 static）
// ===========================================================================

// Unicode 码点 -> 原始字节：GPT-2 的可打印字节直接映射到自身，
// 不可打印字节映射到 256 以上的码点（与 HF 的 bytes_to_unicode 一致）
static std::unordered_map<int, uint8_t> unicode_to_byte() {
    bool printable[256] = {};
    for (int b = 33; b <= 126; ++b) printable[b] = true;   // ! ~
    for (int b = 161; b <= 172; ++b) printable[b] = true;  // ¡ ¬
    for (int b = 174; b <= 255; ++b) printable[b] = true;  // ® ÿ

    std::unordered_map<int, uint8_t> m;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (printable[b]) m[b] = uint8_t(b);
        else              m[256 + n++] = uint8_t(b);
    }
    return m;
}

// UTF-8 字符串 -> 码点序列
static std::vector<int> utf8_to_codepoints(const std::string& s) {
    std::vector<int> cps;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = unsigned char(s[i]);
        int cp, len;
        if (c < 0x80)                { cp = c;        len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else                         { cp = c;        len = 1; }
        for (int k = 1; k < len && i + k < s.size(); ++k)
            cp = (cp << 6) | (unsigned char(s[i + k]) & 0x3F);
        cps.push_back(cp);
        i += len;
    }
    return cps;
}

std::string decode(const std::vector<int64_t>& ids,
                   const std::unordered_map<int64_t, std::string>& id2token) {
    static const auto u2b = unicode_to_byte();

    std::string text;
    for (int64_t id : ids) {
        auto it = id2token.find(id);
        if (it == id2token.end()) continue;

        for (int cp : utf8_to_codepoints(it->second)) {
            auto b = u2b.find(cp);
            text.push_back(b != u2b.end() ? char(b->second) : '?');
        }
    }
    return text;
}

// 把 Unicode 码点编码成 UTF-8 字符串（与 unicode_to_byte 互逆）
static std::string cp_to_utf8(int cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back(char(cp));
    } else if (cp < 0x800) {
        s.push_back(char(0xC0 | (cp >> 6)));
        s.push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(char(0xE0 | (cp >> 12)));
        s.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(char(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(char(0xF0 | (cp >> 18)));
        s.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(char(0x80 | (cp & 0x3F)));
    }
    return s;
}

// byte -> Unicode 字符（字节级 BPE 里每个字节都对应一个"字符"）
static std::unordered_map<uint8_t, std::string> byte_to_unicode() {
    static const auto u2b = unicode_to_byte();
    std::unordered_map<uint8_t, std::string> m;
    for (const auto& [cp, byte] : u2b)
        m[byte] = cp_to_utf8(cp);
    return m;
}

std::unordered_map<std::string, int> load_merges(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("找不到 " + path);
    std::unordered_map<std::string, int> ranks;
    std::string line;
    std::getline(f, line); // 跳过 "#version: 0.2"
    int rank = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        ranks[line.substr(0, sp) + line.substr(sp + 1)] = rank++;
    }
    return ranks;
}

// 把字符串拆成码点，并记录每个码点的字节区间（按码点分类、按字节截取）
struct Codepoint { int cp; size_t start, end; };

static std::vector<Codepoint> utf8_split(const std::string& s) {
    std::vector<Codepoint> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = unsigned char(s[i]);
        int cp, len;
        if (c < 0x80)                { cp = c;        len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else                         { cp = c;        len = 1; }
        for (int k = 1; k < len && i + k < s.size(); ++k)
            cp = (cp << 6) | (unsigned char(s[i + k]) & 0x3F);
        out.push_back({cp, i, i + size_t(len)});
        i += len;
    }
    return out;
}

// 模仿 GPT-2 预分词正则里的 \s、\p{L}、\p{N} 分类
static bool is_space_cp(int cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}
static bool is_letter_cp(int cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    // 常见拉丁扩展字母（à/é 等），让非纯英文场景也能大致正确
    return cp >= 0xC0 && cp <= 0xFF && cp != 0xD7 && cp != 0xF7;
}
static bool is_number_cp(int cp) { return cp >= '0' && cp <= '9'; }
static bool is_symbol_cp(int cp) {
    return !is_space_cp(cp) && !is_letter_cp(cp) && !is_number_cp(cp);
}

// 缩写形式 's 't 're 've 'm 'll 'd 单独成词（正则的第一条分支）
static bool try_contraction(const std::vector<Codepoint>& cps, size_t i, size_t n, size_t& len) {
    if (cps[i].cp != '\'') return false;
    static const std::string sufs[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const auto& s : sufs) {
        size_t L = s.size();
        if (i + L >= n) continue;
        bool ok = true;
        for (size_t k = 0; k < L; ++k)
            if (cps[i + 1 + k].cp != (unsigned char)s[k]) { ok = false; break; }
        if (ok) { len = L; return true; }
    }
    return false;
}

// 预分词：按 GPT-2 的正则把文本切成若干段（每段是原始 UTF-8 子串）
// 正则: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<std::string> pretokenize(const std::string& text) {
    auto cps = utf8_split(text);
    size_t n = cps.size(), i = 0;
    std::vector<std::string> tokens;

    auto take = [&](size_t a, size_t b) { // 码点区间 [a, b]，取对应字节
        return text.substr(cps[a].start, cps[b].end - cps[a].start);
    };

    while (i < n) {
        size_t clen;
        if (try_contraction(cps, i, n, clen)) {
            tokens.push_back(take(i, i + clen));
            i += 1 + clen;
            continue;
        }

        // ? 空格 + 字母 / 纯字母
        if (cps[i].cp == ' ' && i + 1 < n && is_letter_cp(cps[i + 1].cp)) {
            size_t j = i + 1;
            while (j < n && is_letter_cp(cps[j].cp)) ++j;
            tokens.push_back(take(i, j - 1));
            i = j;
            continue;
        }
        if (is_letter_cp(cps[i].cp)) {
            size_t j = i;
            while (j < n && is_letter_cp(cps[j].cp)) ++j;
            tokens.push_back(take(i, j - 1));
            i = j;
            continue;
        }

        // ? 空格 + 数字 / 纯数字
        if (cps[i].cp == ' ' && i + 1 < n && is_number_cp(cps[i + 1].cp)) {
            size_t j = i + 1;
            while (j < n && is_number_cp(cps[j].cp)) ++j;
            tokens.push_back(take(i, j - 1));
            i = j;
            continue;
        }
        if (is_number_cp(cps[i].cp)) {
            size_t j = i;
            while (j < n && is_number_cp(cps[j].cp)) ++j;
            tokens.push_back(take(i, j - 1));
            i = j;
            continue;
        }

        // ? 空格 + 其他符号 / 纯符号
        if (cps[i].cp == ' ' && i + 1 < n && is_symbol_cp(cps[i + 1].cp)) {
            size_t j = i + 1;
            while (j < n && is_symbol_cp(cps[j].cp)) ++j;
            tokens.push_back(take(i, j - 1));
            i = j;
            continue;
        }
        if (is_symbol_cp(cps[i].cp)) {
            size_t j = i;
            while (j < n && is_symbol_cp(cps[j].cp)) ++j;
            tokens.push_back(take(i, j - 1));
            i = j;
            continue;
        }

        // 纯空白 \s+(?!\S)|\s+：优先匹配"尾部空白"，再退化为整段空白
        size_t j = i;
        while (j < n && is_space_cp(cps[j].cp)) ++j;
        if (j == n) {                   // 空白一直到字符串结尾
            tokens.push_back(take(i, j - 1));
            i = j;
        } else if (j - 1 > i) {         // 多字符空白且后面还有内容：留最后一个空白
            tokens.push_back(text.substr(cps[i].start, cps[j - 1].start - cps[i].start));
            i = j - 1;
        } else {                        // 单个空白后面还有内容
            tokens.push_back(take(i, i));
            i = j;
        }
    }
    return tokens;
}

// 对单个预分词结果做 BPE 合并
static std::vector<std::string> bpe_merge(
    const std::vector<std::string>& token,
    const std::unordered_map<std::string, int>& ranks) {
    std::vector<std::string> word = token;
    while (word.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        std::string first, second;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = ranks.find(word[i] + word[i + 1]);
            int r = (it == ranks.end()) ? std::numeric_limits<int>::max() : it->second;
            if (r < best_rank) {
                best_rank = r;
                first = word[i];
                second = word[i + 1];
            }
        }
        if (best_rank == std::numeric_limits<int>::max()) break;

        // 一趟合并所有（不重叠的）该 pair，与 HF 实现一致
        std::vector<std::string> next;
        size_t i = 0;
        while (i < word.size()) {
            size_t j = i;
            while (j < word.size() && word[j] != first) ++j;
            if (j == word.size()) {
                for (size_t k = i; k < word.size(); ++k) next.push_back(word[k]);
                break;
            }
            for (size_t k = i; k < j; ++k) next.push_back(word[k]);
            if (j + 1 < word.size() && word[j + 1] == second) {
                next.push_back(first + second);
                i = j + 2;
            } else {
                next.push_back(word[j]);
                i = j + 1;
            }
        }
        word = std::move(next);
    }
    return word;
}

std::vector<int64_t> encode(
    const std::string& text,
    const std::unordered_map<std::string, int64_t>& token2id,
    const std::unordered_map<std::string, int>& ranks) {
    static const auto b2u = byte_to_unicode();

    std::vector<int64_t> ids;
    for (const auto& raw : pretokenize(text)) {
        // 每个字节映射成对应的"Unicode 字符"
        std::vector<std::string> word;
        for (unsigned char b : raw) {
            auto it = b2u.find(b);
            word.push_back(it != b2u.end() ? it->second : std::string(1, char(b)));
        }
        for (const auto& w : bpe_merge(word, ranks)) {
            auto it = token2id.find(w);
            if (it != token2id.end()) ids.push_back(it->second);
        }
    }
    return ids;
}
