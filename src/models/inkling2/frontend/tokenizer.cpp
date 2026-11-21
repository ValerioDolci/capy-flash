#include "models/inkling2/frontend/tokenizer.h"

#include "text/unicode.h"

#include <nlohmann/json.hpp>
#include <utf8proc/utf8proc.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ninfer::models::inkling2::frontend {
namespace {

using Json    = nlohmann::json;
namespace uni = ninfer::text::unicode_internal;

constexpr std::int64_t kMaxTokenId = 1'000'000;

// The o200k pre-tokenization of Inkling 2 (tokenizer.json pre_tokenizer.pretokenizers[0]).
// o200k_word_end below is this expression written out; the loader requires the exact text.
constexpr std::string_view kO200kSplitPattern =
    R"o200k([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)o200k";

void validate_pipeline(const Json& root, const Json& model) {
    const auto require = [](bool valid, const char* field) {
        if (!valid) {
            throw std::invalid_argument(std::string("tokenizer.json ") + field +
                                        " is not supported by the Inkling 2 tokenizer");
        }
    };
    const auto type = [](const Json& value, const char* name) {
        return value.is_object() && value.contains("type") && value["type"] == name;
    };
    require(!root.contains("normalizer") || root["normalizer"].is_null(), "normalizer");
    require(root.contains("pre_tokenizer"), "pre_tokenizer");
    const auto& pre = root["pre_tokenizer"];
    require(type(pre, "Sequence") && pre.contains("pretokenizers") &&
                pre["pretokenizers"].is_array() && pre["pretokenizers"].size() == 2,
            "pre_tokenizer");
    const auto& split = pre["pretokenizers"][0];
    const auto& bytes = pre["pretokenizers"][1];
    require(type(split, "Split") && split.contains("pattern") && split["pattern"].is_object() &&
                split["pattern"].value("Regex", std::string{}) == kO200kSplitPattern &&
                split.value("behavior", std::string{}) == "Isolated" &&
                split.value("invert", Json(false)) == false,
            "pre_tokenizer.Split");
    require(type(bytes, "ByteLevel") && bytes.value("add_prefix_space", Json(true)) == false &&
                bytes.value("use_regex", Json(true)) == false,
            "pre_tokenizer.ByteLevel");
    require(root.contains("decoder") && type(root["decoder"], "ByteLevel"), "decoder");
    if (root.contains("post_processor") && !root["post_processor"].is_null()) {
        require(type(root["post_processor"], "ByteLevel"), "post_processor");
    }
    if (model.contains("dropout") && !model["dropout"].is_null()) {
        require(model["dropout"].is_number() && model["dropout"] == 0, "model.dropout");
    }
    for (const char* field : {"continuing_subword_prefix", "end_of_word_suffix", "unk_token"}) {
        require(!model.contains(field) || model[field].is_null() || model[field] == "", field);
    }
    require(model.value("byte_fallback", Json(false)) == false, "model.byte_fallback");
    require(model.value("ignore_merges", Json(false)) == true, "model.ignore_merges");
}

struct VocabMetadata {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_set<int> occupied_ids;
};

Json read_json_asset(std::string_view contents, std::string_view label) {
    try {
        return Json::parse(contents);
    } catch (const nlohmann::json::exception& ex) {
        throw std::invalid_argument("malformed " + std::string(label) + ": " + ex.what());
    }
}

const Json& require_object_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_object()) {
        throw std::invalid_argument("field " + std::string(field) + " must be object in " +
                                    std::string(label));
    }
    return object.at(field);
}

const Json& require_array_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_array()) {
        throw std::invalid_argument("field " + std::string(field) + " must be array in " +
                                    std::string(label));
    }
    return object.at(field);
}

int parse_token_id(const Json& value, const char* field, std::string_view label) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument("field " + std::string(field) + " must be integer in " +
                                    std::string(label));
    }
    const std::int64_t id = value.is_number_unsigned()
                                ? static_cast<std::int64_t>(std::min<std::uint64_t>(
                                      value.get<std::uint64_t>(), kMaxTokenId + 1))
                                : value.get<std::int64_t>();
    if (id < 0 || id > kMaxTokenId) {
        throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                    std::string(label));
    }
    return static_cast<int>(id);
}

std::string require_string_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_string()) {
        throw std::invalid_argument("field " + std::string(field) + " must be string in " +
                                    std::string(label));
    }
    return object.at(field).get<std::string>();
}

bool require_bool_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_boolean()) {
        throw std::invalid_argument("field " + std::string(field) + " must be boolean in " +
                                    std::string(label));
    }
    return object.at(field).get<bool>();
}

VocabMetadata load_vocab(const Json& model, std::string_view label) {
    if (!model.contains("type") || model.at("type") != "BPE") {
        throw std::invalid_argument("field model.type must be BPE in " + std::string(label));
    }
    const Json& vocab = require_object_field(model, "vocab", label);
    if (vocab.empty()) {
        throw std::invalid_argument("field model.vocab must not be empty in " + std::string(label));
    }
    int max_id = -1;
    VocabMetadata metadata;
    metadata.token_to_id.reserve(vocab.size());
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        if (!metadata.occupied_ids.insert(id).second) {
            throw std::invalid_argument("field model.vocab has duplicate id in " +
                                        std::string(label));
        }
        max_id = std::max(max_id, id);
    }
    metadata.id_to_token.resize(static_cast<std::size_t>(max_id + 1));
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        metadata.id_to_token.at(static_cast<std::size_t>(id)) = item.key();
        metadata.token_to_id.emplace(item.key(), id);
    }
    return metadata;
}

AddedToken parse_added_token_fields(int id, const Json& item, std::string_view label) {
    if (!item.is_object()) {
        throw std::invalid_argument("added token item must be object in " + std::string(label));
    }
    AddedToken token;
    token.id          = id;
    token.content     = require_string_field(item, "content", label);
    token.single_word = require_bool_field(item, "single_word", label);
    token.lstrip      = require_bool_field(item, "lstrip", label);
    token.rstrip      = require_bool_field(item, "rstrip", label);
    token.normalized  = require_bool_field(item, "normalized", label);
    token.special     = require_bool_field(item, "special", label);
    if (token.content.empty()) {
        throw std::invalid_argument("added token content must not be empty in " +
                                    std::string(label));
    }
    // With no normalizer, normalized=true would still split identically; the flags that change
    // matching are rejected.
    if (token.single_word || token.lstrip || token.rstrip || token.normalized) {
        throw std::invalid_argument("Tokenizer only supports added tokens with single_word=false, "
                                    "lstrip=false, rstrip=false, and normalized=false in " +
                                    std::string(label));
    }
    return token;
}

bool same_added_token(const AddedToken& lhs, const AddedToken& rhs) noexcept {
    return lhs.id == rhs.id && lhs.content == rhs.content && lhs.single_word == rhs.single_word &&
           lhs.lstrip == rhs.lstrip && lhs.rstrip == rhs.rstrip &&
           lhs.normalized == rhs.normalized && lhs.special == rhs.special;
}

int parse_added_token_decoder_id(std::string_view key, std::string_view label) {
    std::int64_t parsed     = -1;
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), parsed);
    if (error != std::errc{} || end != key.data() + key.size() || parsed < 0 ||
        parsed > kMaxTokenId || std::to_string(parsed) != key) {
        throw std::invalid_argument("added_tokens_decoder key must be a nonnegative token id in " +
                                    std::string(label));
    }
    return static_cast<int>(parsed);
}

// added_tokens of tokenizer.json, then added_tokens_decoder of tokenizer_config.json, which must
// agree on shared ids; the result is sorted by id.
std::vector<AddedToken> load_added_tokens(const Json& root, const Json& config,
                                          std::vector<std::string>& id_to_token,
                                          const VocabMetadata& vocab) {
    constexpr std::string_view label        = "tokenizer.json";
    constexpr std::string_view config_label = "tokenizer_config.json";
    std::vector<AddedToken> tokens;
    std::unordered_map<int, std::size_t> by_id;
    std::unordered_set<std::string> contents;
    const auto add = [&](AddedToken token, std::string_view source) {
        if (const auto existing = by_id.find(token.id); existing != by_id.end()) {
            if (!same_added_token(tokens[existing->second], token)) {
                throw std::invalid_argument("conflicting added-token definition for id " +
                                            std::to_string(token.id) + " in " +
                                            std::string(source));
            }
            return;
        }
        if (vocab.occupied_ids.contains(token.id) || vocab.token_to_id.contains(token.content) ||
            !contents.insert(token.content).second) {
            throw std::invalid_argument("added token " + token.content +
                                        " overlaps the vocabulary or another added token in " +
                                        std::string(source));
        }
        const auto index = static_cast<std::size_t>(token.id);
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        id_to_token[index] = token.content;
        by_id.emplace(token.id, tokens.size());
        tokens.push_back(std::move(token));
    };
    for (const Json& item : require_array_field(root, "added_tokens", label)) {
        if (!item.is_object() || !item.contains("id")) {
            throw std::invalid_argument("missing field added_tokens.id in tokenizer.json");
        }
        add(parse_added_token_fields(parse_token_id(item.at("id"), "added_tokens.id", label), item,
                                     label),
            label);
    }
    for (const auto& item :
         require_object_field(config, "added_tokens_decoder", config_label).items()) {
        add(parse_added_token_fields(parse_added_token_decoder_id(item.key(), config_label),
                                     item.value(), config_label),
            config_label);
    }
    std::sort(tokens.begin(), tokens.end(),
              [](const AddedToken& lhs, const AddedToken& rhs) { return lhs.id < rhs.id; });
    return tokens;
}

std::vector<int> load_default_stop_token_ids(std::string_view contents) {
    constexpr std::string_view label = "generation_config.json";
    const Json root                  = read_json_asset(contents, label);
    if (!root.is_object() || !root.contains("eos_token_id")) {
        throw std::invalid_argument("missing field eos_token_id in generation_config.json");
    }
    const Json& eos = root.at("eos_token_id");
    if (eos.is_number_integer()) { return {parse_token_id(eos, "eos_token_id", label)}; }
    if (eos.is_array() && !eos.empty()) {
        std::vector<int> ids;
        for (const Json& item : eos) { ids.push_back(parse_token_id(item, "eos_token_id", label)); }
        return ids;
    }
    throw std::invalid_argument(
        "field eos_token_id must be integer or non-empty array in generation_config.json");
}

std::uint64_t merge_pair_key(int left, int right) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32U) |
           static_cast<std::uint32_t>(right);
}

BpeMergeTable load_bpe_merge_rules(const Json& model,
                                   const std::unordered_map<std::string, int>& token_to_id) {
    const Json& merges = require_array_field(model, "merges", "tokenizer.json");
    BpeMergeTable rules;
    rules.reset(merges.size());
    // Rank order: linear probing lets the first claimant keep its home slot, and low ranks are
    // the frequent merges.
    int rank = 0;
    for (const Json& item : merges) {
        std::string left;
        std::string right;
        if (item.is_array() && item.size() == 2 && item[0].is_string() && item[1].is_string()) {
            left  = item[0].get<std::string>();
            right = item[1].get<std::string>();
        } else if (item.is_string()) {
            const std::string pair  = item.get<std::string>();
            const std::size_t space = pair.find(' ');
            if (space == std::string::npos || space == 0 || space + 1 >= pair.size() ||
                pair.find(' ', space + 1) != std::string::npos) {
                throw std::invalid_argument("malformed model.merges entry in tokenizer.json");
            }
            left  = pair.substr(0, space);
            right = pair.substr(space + 1);
        } else {
            throw std::invalid_argument(
                "field model.merges must contain symbol pairs in tokenizer.json");
        }
        const auto left_id   = token_to_id.find(left);
        const auto right_id  = token_to_id.find(right);
        const auto result_id = token_to_id.find(left + right);
        if (left_id == token_to_id.end() || right_id == token_to_id.end() ||
            result_id == token_to_id.end()) {
            throw std::invalid_argument(
                "model.merges references a symbol outside model.vocab in tokenizer.json");
        }
        if (!rules.insert(merge_pair_key(left_id->second, right_id->second),
                          BpeMergeRule{.rank = rank++, .result = result_id->second})) {
            throw std::invalid_argument("duplicate merge pair in model.merges");
        }
    }
    return rules;
}

// GPT-2 byte-level alphabet: printable Latin-1 bytes map to themselves, the others to U+0100+.
std::array<std::uint32_t, 256> byte_level_codepoints() {
    std::array<std::uint32_t, 256> codepoints{};
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        codepoints[static_cast<std::size_t>(byte)] =
            visible ? static_cast<std::uint32_t>(byte) : next++;
    }
    return codepoints;
}

std::string decode_byte_level_token(std::string_view token, int id) {
    static const std::unordered_map<std::uint32_t, char> byte_decoder = [] {
        std::unordered_map<std::uint32_t, char> decoder;
        const auto codepoints = byte_level_codepoints();
        for (std::size_t byte = 0; byte < codepoints.size(); ++byte) {
            decoder.emplace(codepoints[byte], static_cast<char>(static_cast<unsigned char>(byte)));
        }
        return decoder;
    }();
    std::string bytes;
    for (const uni::CodepointSpan& codepoint :
         uni::utf8_codepoints(token, "Tokenizer token id " + std::to_string(id))) {
        const auto byte = byte_decoder.find(static_cast<std::uint32_t>(codepoint.value));
        if (byte == byte_decoder.end()) {
            throw std::invalid_argument("Tokenizer token id " + std::to_string(id) +
                                        " contains a character outside the byte-level alphabet");
        }
        bytes.push_back(byte->second);
    }
    return bytes;
}

std::array<int, 256> load_byte_token_ids(const std::unordered_map<std::string, int>& token_to_id) {
    const auto codepoints = byte_level_codepoints();
    std::array<int, 256> ids;
    for (std::size_t byte = 0; byte < ids.size(); ++byte) {
        const auto token =
            token_to_id.find(uni::codepoint_to_utf8(static_cast<std::int32_t>(codepoints[byte])));
        if (token == token_to_id.end()) {
            throw std::invalid_argument("tokenizer.json vocabulary lacks a byte-level symbol");
        }
        ids[byte] = token->second;
    }
    return ids;
}

// Character classes of the o200k expression, one bit each.
enum CharClass : std::uint8_t {
    kLetter  = 1U << 0U, // \p{L}
    kNumber  = 1U << 1U, // \p{N}
    kUpper   = 1U << 2U, // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
    kLower   = 1U << 3U, // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
    kSpace   = 1U << 4U, // \s: the Unicode White_Space property
    kNewline = 1U << 5U, // [\r\n]
};

bool is_white_space(std::int32_t cp) noexcept {
    return (cp >= 0x09 && cp <= 0x0d) || cp == 0x20 || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f ||
           cp == 0x205f || cp == 0x3000;
}

std::uint8_t char_class(std::int32_t cp) noexcept {
    std::uint8_t bits = 0;
    switch (utf8proc_category(cp)) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LT:
        bits = kLetter | kUpper;
        break;
    case UTF8PROC_CATEGORY_LL:
        bits = kLetter | kLower;
        break;
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
        bits = kLetter | kUpper | kLower;
        break;
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
    case UTF8PROC_CATEGORY_ME:
        bits = kUpper | kLower;
        break;
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO:
        bits = kNumber;
        break;
    default:
        break;
    }
    if (is_white_space(cp)) { bits |= kSpace; }
    if (cp == '\r' || cp == '\n') { bits |= kNewline; }
    return bits;
}

struct ClassifiedText {
    std::string_view text;
    std::vector<std::size_t> offsets; // byte offset of each codepoint, plus text.size()
    std::vector<std::uint8_t> classes;
    std::vector<std::int32_t> values;

    explicit ClassifiedText(std::string_view input) : text(input) {
        for (std::size_t offset = 0; offset < input.size();) {
            const uni::CodepointSpan cp =
                uni::utf8_codepoint_at(input, offset, "Tokenizer::encode input");
            offsets.push_back(offset);
            classes.push_back(char_class(cp.value));
            values.push_back(cp.value);
            offset += cp.length;
        }
        offsets.push_back(input.size());
    }

    [[nodiscard]] std::size_t size() const noexcept { return classes.size(); }

    [[nodiscard]] bool has(std::size_t i, std::uint8_t bits) const noexcept {
        return i < classes.size() && (classes[i] & bits) != 0;
    }

    [[nodiscard]] std::size_t run(std::size_t i, std::uint8_t bits) const noexcept {
        while (has(i, bits)) { ++i; }
        return i;
    }
};

// [^\r\n\p{L}\p{N}]
bool contraction_prefix_class(const ClassifiedText& t, std::size_t i) noexcept {
    return i < t.size() && !t.has(i, kNewline | kLetter | kNumber);
}

// [^\s\p{L}\p{N}]
bool punctuation_class(const ClassifiedText& t, std::size_t i) noexcept {
    return i < t.size() && !t.has(i, kSpace | kLetter | kNumber);
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d)? after a letter run ending at i.
std::size_t contraction_end(const ClassifiedText& t, std::size_t i) noexcept {
    if (i >= t.size() || t.values[i] != '\'') { return i; }
    const auto lower = [&](std::size_t k) -> std::int32_t {
        if (k >= t.size()) { return 0; }
        const std::int32_t cp = t.values[k];
        return cp >= 'A' && cp <= 'Z' ? cp - 'A' + 'a' : cp;
    };
    constexpr std::string_view suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (std::string_view suffix : suffixes) {
        bool match = true;
        for (std::size_t k = 0; k < suffix.size() && match; ++k) {
            match = lower(i + 1 + k) == suffix[k];
        }
        if (match) { return i + 1 + suffix.size(); }
    }
    return i;
}

// Codepoint index after the o200k match starting at codepoint `begin`. The alternatives are tried
// in order, each with the backtracking of the leftmost-first regex engine resolved in closed form.
std::size_t o200k_word_end(const ClassifiedText& t, std::size_t begin) {
    const std::size_t prefixed = begin + 1;
    // 1: P? [Lu Lt Lm Lo M]* [Ll Lm Lo M]+ contraction?
    for (const bool use_prefix : {true, false}) {
        if (use_prefix && !contraction_prefix_class(t, begin)) { continue; }
        const std::size_t start = use_prefix ? prefixed : begin;
        const std::size_t upper = t.run(start, kUpper);
        if (t.has(upper, kLower)) { return contraction_end(t, t.run(upper, kLower)); }
        for (std::size_t k = upper; k > start; --k) {
            if (t.has(k - 1, kLower)) { return contraction_end(t, k); }
        }
    }
    // 2: P? [Lu Lt Lm Lo M]+ [Ll Lm Lo M]* contraction?
    for (const bool use_prefix : {true, false}) {
        if (use_prefix && !contraction_prefix_class(t, begin)) { continue; }
        const std::size_t start = use_prefix ? prefixed : begin;
        const std::size_t upper = t.run(start, kUpper);
        if (upper > start) { return contraction_end(t, t.run(upper, kLower)); }
    }
    // 3: \p{N}{1,3}
    if (t.has(begin, kNumber)) {
        std::size_t end = begin;
        while (end < begin + 3 && t.has(end, kNumber)) { ++end; }
        return end;
    }
    // 4: ' '? [^\s\p{L}\p{N}]+ [\r\n/]*
    {
        std::size_t start = begin;
        if (t.values[begin] == ' ' && punctuation_class(t, prefixed)) { start = prefixed; }
        if (punctuation_class(t, start)) {
            std::size_t end = start;
            while (punctuation_class(t, end)) { ++end; }
            while (end < t.size() && (t.has(end, kNewline) || t.values[end] == '/')) { ++end; }
            return end;
        }
    }
    // 5: \s*[\r\n]+   6: \s+(?!\S)   7: \s+
    if (t.has(begin, kSpace)) {
        const std::size_t end = t.run(begin, kSpace);
        for (std::size_t k = end; k > begin; --k) {
            if (t.has(k - 1, kNewline)) { return k; }
        }
        if (end == t.size() || end - begin == 1) { return end; }
        return end - 1;
    }
    // Every codepoint matches one alternative; kept for a character outside all classes.
    return prefixed;
}

struct BpeNode {
    int symbol               = -1;
    int previous             = -1;
    int next                 = -1;
    std::uint32_t generation = 0;
    bool live                = true;
};

struct BpeCandidate {
    int rank                       = 0;
    int left                       = -1;
    int right                      = -1;
    int result                     = -1;
    std::uint32_t left_generation  = 0;
    std::uint32_t right_generation = 0;
};

struct LaterBpeCandidate {
    bool operator()(const BpeCandidate& lhs, const BpeCandidate& rhs) const noexcept {
        if (lhs.rank != rhs.rank) { return lhs.rank > rhs.rank; }
        return lhs.left > rhs.left;
    }
};

// Lowest-rank merge first, leftmost on equal rank.
void append_bpe_word(std::vector<int>& out, std::string_view word, const BpeMergeTable& merge_rules,
                     const std::array<int, 256>& byte_token_ids) {
    std::vector<BpeNode> nodes(word.size());
    for (std::size_t index = 0; index < word.size(); ++index) {
        nodes[index] = BpeNode{.symbol   = byte_token_ids[static_cast<unsigned char>(word[index])],
                               .previous = index == 0 ? -1 : static_cast<int>(index - 1),
                               .next = index + 1 == word.size() ? -1 : static_cast<int>(index + 1)};
    }
    std::priority_queue<BpeCandidate, std::vector<BpeCandidate>, LaterBpeCandidate> queue;
    const auto push_candidate = [&](int left) {
        if (left < 0 || !nodes[static_cast<std::size_t>(left)].live) { return; }
        const int right = nodes[static_cast<std::size_t>(left)].next;
        if (right < 0) { return; }
        const BpeMergeEntry* rule =
            merge_rules.find(merge_pair_key(nodes[static_cast<std::size_t>(left)].symbol,
                                            nodes[static_cast<std::size_t>(right)].symbol));
        if (rule == nullptr) { return; }
        queue.push(BpeCandidate{
            .rank             = rule->rank,
            .left             = left,
            .right            = right,
            .result           = rule->result,
            .left_generation  = nodes[static_cast<std::size_t>(left)].generation,
            .right_generation = nodes[static_cast<std::size_t>(right)].generation,
        });
    };
    for (std::size_t index = 0; index + 1 < nodes.size(); ++index) {
        push_candidate(static_cast<int>(index));
    }
    while (!queue.empty()) {
        const BpeCandidate candidate = queue.top();
        queue.pop();
        BpeNode& left  = nodes[static_cast<std::size_t>(candidate.left)];
        BpeNode& right = nodes[static_cast<std::size_t>(candidate.right)];
        if (!left.live || !right.live || left.next != candidate.right ||
            left.generation != candidate.left_generation ||
            right.generation != candidate.right_generation) {
            continue;
        }
        left.symbol = candidate.result;
        ++left.generation;
        left.next  = right.next;
        right.live = false;
        ++right.generation;
        if (left.next >= 0) {
            nodes[static_cast<std::size_t>(left.next)].previous = candidate.left;
        }
        push_candidate(left.previous);
        push_candidate(candidate.left);
    }
    for (int node = nodes.empty() ? -1 : 0; node >= 0;
         node     = nodes[static_cast<std::size_t>(node)].next) {
        out.push_back(nodes[static_cast<std::size_t>(node)].symbol);
    }
}

bool is_stop_token_id(std::span<const int> stop_token_ids, int id) {
    return std::find(stop_token_ids.begin(), stop_token_ids.end(), id) != stop_token_ids.end();
}

constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";

// Complete UTF-8 sequences of `pending`, which keeps an incomplete trailing sequence.
std::string consume_generated_utf8(std::string& pending) {
    std::string decoded;
    decoded.reserve(pending.size());
    std::size_t offset = 0;
    while (offset < pending.size()) {
        const auto lead    = static_cast<unsigned char>(pending[offset]);
        std::size_t length = 0;
        if (lead <= 0x7fU) {
            decoded.push_back(pending[offset]);
            ++offset;
            continue;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
        } else {
            decoded.append(kUtf8Replacement);
            ++offset;
            continue;
        }
        bool malformed = false;
        for (std::size_t index = 1; index < length; ++index) {
            if (offset + index >= pending.size()) {
                pending.erase(0, offset);
                return decoded;
            }
            const auto byte      = static_cast<unsigned char>(pending[offset + index]);
            unsigned int minimum = 0x80U;
            unsigned int maximum = 0xbfU;
            if (index == 1) {
                if (lead == 0xe0U) {
                    minimum = 0xa0U;
                } else if (lead == 0xedU) {
                    maximum = 0x9fU;
                } else if (lead == 0xf0U) {
                    minimum = 0x90U;
                } else if (lead == 0xf4U) {
                    maximum = 0x8fU;
                }
            }
            if (byte < minimum || byte > maximum) {
                // One replacement per maximal subpart; the offending byte starts the next one.
                decoded.append(kUtf8Replacement);
                offset += index;
                malformed = true;
                break;
            }
        }
        if (malformed) { continue; }
        decoded.append(pending, offset, length);
        offset += length;
    }
    pending.clear();
    return decoded;
}

} // namespace

Tokenizer::Tokenizer(TokenizerResources resources) {
    if (resources.tokenizer_json.empty() || resources.tokenizer_config_json.empty() ||
        resources.generation_config_json.empty()) {
        throw std::invalid_argument("tokenizer resources are empty");
    }
    const Json root   = read_json_asset(resources.tokenizer_json, "tokenizer.json");
    const Json config = read_json_asset(resources.tokenizer_config_json, "tokenizer_config.json");
    const Json& model = require_object_field(root, "model", "tokenizer.json");
    validate_pipeline(root, model);

    VocabMetadata vocab  = load_vocab(model, "tokenizer.json");
    decoded_token_bytes_ = std::move(vocab.id_to_token);
    added_tokens_        = load_added_tokens(root, config, decoded_token_bytes_, vocab);
    valid_token_ids_.assign(decoded_token_bytes_.size(), false);
    special_token_ids_.assign(decoded_token_bytes_.size(), false);
    for (const int id : vocab.occupied_ids) {
        auto& bytes = decoded_token_bytes_[static_cast<std::size_t>(id)];
        bytes       = decode_byte_level_token(bytes, id);
        valid_token_ids_[static_cast<std::size_t>(id)] = true;
    }
    bytes_to_id_.reserve(vocab.occupied_ids.size());
    for (const int id : vocab.occupied_ids) {
        if (!bytes_to_id_.emplace(decoded_token_bytes_[static_cast<std::size_t>(id)], id).second) {
            throw std::invalid_argument(
                "tokenizer.json vocabulary has two tokens with equal bytes");
        }
    }
    for (std::size_t index = 0; index < added_tokens_.size(); ++index) {
        const AddedToken& token                                = added_tokens_[index];
        valid_token_ids_[static_cast<std::size_t>(token.id)]   = true;
        special_token_ids_[static_cast<std::size_t>(token.id)] = token.special;
        added_token_candidates_[static_cast<unsigned char>(token.content.front())].push_back(index);
    }
    bpe_merge_rules_        = load_bpe_merge_rules(model, vocab.token_to_id);
    byte_token_ids_         = load_byte_token_ids(vocab.token_to_id);
    default_stop_token_ids_ = load_default_stop_token_ids(resources.generation_config_json);
}

bool Tokenizer::append_ordinary(std::vector<int>& ids, std::string_view text,
                                std::size_t max_tokens) const {
    if (text.empty()) { return true; }
    const ClassifiedText classified(text);
    std::vector<int> word_ids;
    for (std::size_t begin = 0; begin < classified.size();) {
        const std::size_t end       = o200k_word_end(classified, begin);
        const std::string_view word = text.substr(
            classified.offsets[begin], classified.offsets[end] - classified.offsets[begin]);
        word_ids.clear();
        if (const auto whole = bytes_to_id_.find(std::string(word)); whole != bytes_to_id_.end()) {
            word_ids.push_back(whole->second);
        } else {
            append_bpe_word(word_ids, word, bpe_merge_rules_, byte_token_ids_);
        }
        for (const int id : word_ids) {
            ids.push_back(id);
            if (ids.size() == max_tokens) { return false; }
        }
        begin = end;
    }
    return true;
}

std::vector<int> Tokenizer::encode(std::string_view text, EncodeOptions options,
                                   std::span<const text::ByteSpan> literal_spans) const {
    std::size_t previous_end = 0;
    for (const auto span : literal_spans) {
        if (span.begin < previous_end || span.begin >= span.end || span.end > text.size()) {
            throw std::invalid_argument(
                "Tokenizer literal spans must be ordered disjoint input ranges");
        }
        previous_end = span.end;
    }
    std::vector<int> ids;
    if (options.max_tokens == 0) { return ids; }
    if (!options.parse_added_tokens) {
        (void)append_ordinary(ids, text, options.max_tokens);
        return ids;
    }
    std::size_t ordinary_begin = 0;
    std::size_t pos            = 0;
    while (pos < text.size()) {
        const AddedToken* match = nullptr;
        for (const std::size_t index :
             added_token_candidates_[static_cast<unsigned char>(text[pos])]) {
            const AddedToken& token = added_tokens_[index];
            if ((match == nullptr || token.content.size() > match->content.size()) &&
                token.content.size() <= text.size() - pos &&
                text.compare(pos, token.content.size(), token.content) == 0 &&
                !text::overlaps(literal_spans, pos, pos + token.content.size())) {
                match = &token;
            }
        }
        if (match == nullptr) {
            ++pos;
            continue;
        }
        if (!append_ordinary(ids, text.substr(ordinary_begin, pos - ordinary_begin),
                             options.max_tokens)) {
            return ids;
        }
        ids.push_back(match->id);
        if (ids.size() == options.max_tokens) { return ids; }
        pos += match->content.size();
        ordinary_begin = pos;
    }
    (void)append_ordinary(ids, text.substr(ordinary_begin), options.max_tokens);
    return ids;
}

std::string Tokenizer::decode(std::span<const int> ids, DecodeOptions options) const {
    std::string text;
    const std::size_t terminal_stop_index =
        (!ids.empty() && is_stop_token_id(options.stop_token_ids, ids.back())) ? ids.size() - 1
                                                                               : ids.size();
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i == terminal_stop_index) { continue; }
        text += decode_token_bytes(ids[i], options.skip_special_tokens);
    }
    (void)uni::utf8_codepoints(text, "Tokenizer::decode reconstructed output");
    return text;
}

DecodedTokenView Tokenizer::decoded_token(int id) const {
    if (!is_valid_token(id)) {
        throw std::out_of_range("Tokenizer::decode token id " + std::to_string(id) +
                                " is outside loaded vocabulary");
    }
    const auto index = static_cast<std::size_t>(id);
    return DecodedTokenView{.bytes   = decoded_token_bytes_[index],
                            .special = special_token_ids_[index]};
}

std::string_view Tokenizer::decode_token_bytes(int id, bool skip_special_tokens) const {
    const DecodedTokenView token = decoded_token(id);
    return skip_special_tokens && token.special ? std::string_view{} : token.bytes;
}

std::optional<int> Tokenizer::added_token_id(std::string_view content) const {
    for (const AddedToken& token : added_tokens_) {
        if (token.content == content) { return token.id; }
    }
    return std::nullopt;
}

bool Tokenizer::is_special_token(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < special_token_ids_.size() &&
           special_token_ids_[static_cast<std::size_t>(id)];
}

bool Tokenizer::is_valid_token(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < valid_token_ids_.size() &&
           valid_token_ids_[static_cast<std::size_t>(id)];
}

bool Tokenizer::has_exact_token_domain(std::size_t size) const noexcept {
    return valid_token_ids_.size() == size &&
           std::find(valid_token_ids_.begin(), valid_token_ids_.end(), false) ==
               valid_token_ids_.end();
}

std::string TokenStreamDecoder::push(int id, bool skip_special_tokens) {
    return push_bytes(tokenizer_->decode_token_bytes(id, skip_special_tokens));
}

std::string TokenStreamDecoder::push_bytes(std::string_view bytes) {
    pending_.append(bytes);
    return consume_generated_utf8(pending_);
}

std::string TokenStreamDecoder::finish() {
    std::string text = consume_generated_utf8(pending_);
    if (!pending_.empty()) {
        pending_.clear();
        text.append(kUtf8Replacement);
    }
    return text;
}

} // namespace ninfer::models::inkling2::frontend
