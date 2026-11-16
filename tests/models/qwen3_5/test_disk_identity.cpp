// The disk tier's token identity (program/storage/disk_identity.h). A wrong match there resumes
// another conversation's KV and gives a wrong answer without any error, so: pictures of the same
// size must differ, a text-only prompt must keep its token stream (the blocks and files of every
// conversation saved before media support), and a frontier inside a Vision item must be refused.
#include "models/qwen3_5/program/storage/disk_identity.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

namespace q = ninfer::models::qwen3_5;
using q::detail::disk_identity;
using q::detail::splits_vision;
using ninfer::TokenId;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

constexpr TokenId kImagePad = 151655;
constexpr TokenId kVideoPad = 151656;

// A 2x4 merged grid: 8 placeholder tokens, the way the processor lays out one image.
q::VisionItem image(std::size_t begin, std::uint8_t content) {
    q::VisionItem item;
    item.modality    = q::PromptModality::Image;
    item.grid        = {.temporal = 1, .height = 4, .width = 8};
    item.patch_begin = 0;
    item.patch_count = 32;
    item.content_digest.fill(content);
    item.token_spans = {{.begin = begin, .count = 8}};
    return item;
}

struct Prompt {
    std::vector<TokenId> tokens;
    std::vector<std::uint8_t> types;
    std::vector<q::VisionItem> items;
};

void text(Prompt& prompt, std::size_t count) {
    for (std::size_t k = 0; k < count; ++k) {
        prompt.tokens.push_back(static_cast<TokenId>(1000 + prompt.tokens.size()));
        prompt.types.push_back(0);
    }
}

void picture(Prompt& prompt, std::uint8_t content) {
    q::VisionItem item = image(prompt.tokens.size(), content);
    for (std::size_t k = 0; k < item.token_spans.front().count; ++k) {
        prompt.tokens.push_back(kImagePad);
        prompt.types.push_back(static_cast<std::uint8_t>(q::PromptModality::Image));
    }
    prompt.items.push_back(std::move(item));
}

bool identity(const Prompt& prompt, std::vector<TokenId>& out, std::size_t tokens = SIZE_MAX) {
    const std::size_t count = tokens == SIZE_MAX ? prompt.tokens.size() : tokens;
    return disk_identity(std::span<const TokenId>(prompt.tokens.data(), count),
                         std::span<const std::uint8_t>(prompt.types.data(), count), prompt.items,
                         out);
}

// Text, one 8-token picture at [10, 18), text.
Prompt one_picture(std::uint8_t content) {
    Prompt prompt;
    text(prompt, 10);
    picture(prompt, content);
    text(prompt, 12);
    return prompt;
}

void test_text_only_is_the_token_stream() {
    Prompt prompt;
    text(prompt, 300);
    std::vector<TokenId> out{7, 7, 7};
    expect(identity(prompt, out), "text-only prompt refused");
    expect(out == prompt.tokens, "text-only identity differs from the token stream");

    std::vector<TokenId> empty_out{7};
    const Prompt empty;
    expect(identity(empty, empty_out), "empty prompt refused");
    expect(empty_out.empty(), "empty prompt has a non-empty identity");
}

void test_same_size_pictures_differ() {
    const Prompt a = one_picture(0x11);
    const Prompt b = one_picture(0x22);
    expect(a.tokens == b.tokens && a.types == b.types,
           "setup: the two prompts must have the same tokens and types");
    std::vector<TokenId> ia, ib, ia2;
    expect(identity(a, ia) && identity(b, ib) && identity(a, ia2), "picture prompt refused");
    expect(ia == ia2, "the same picture gives two identities");
    expect(ia != ib, "two pictures of the same size give the same identity");
    for (std::size_t k = 0; k < a.tokens.size(); ++k) {
        const bool vision = k >= 10 && k < 18;
        if (vision) {
            expect(ia[k] < 0 && ib[k] < 0, "a placeholder keeps a real (non-negative) token id");
            expect(ia[k] != ib[k], "a placeholder of two different pictures has the same id");
            for (std::size_t j = 10; j < k; ++j) {
                expect(ia[j] != ia[k], "two placeholders of one picture have the same id");
            }
        } else {
            expect(ia[k] == a.tokens[k] && ib[k] == b.tokens[k], "a text token was changed");
        }
    }
}

void test_item_fields_bind_the_identity() {
    const Prompt base = one_picture(0x11);
    std::vector<TokenId> reference;
    expect(identity(base, reference), "picture prompt refused");
    const auto differs = [&](const Prompt& changed, std::string_view what) {
        std::vector<TokenId> out;
        expect(identity(changed, out), "changed picture prompt refused");
        expect(out != reference, what);
    };
    Prompt grid        = base;
    grid.items[0].grid = {.temporal = 1, .height = 8, .width = 4};
    differs(grid, "a transposed grid (same token count) gives the same identity");
    Prompt patches               = base;
    patches.items[0].patch_begin = 32;
    differs(patches, "another patch range gives the same identity");
    Prompt video            = base;
    video.items[0].modality = q::PromptModality::Video;
    for (std::size_t k = 10; k < 18; ++k) {
        video.tokens[k] = kVideoPad;
        video.types[k]  = static_cast<std::uint8_t>(q::PromptModality::Video);
    }
    video.items[0].timestamps = {0.0, 0.5};
    std::vector<TokenId> video_out;
    expect(identity(video, video_out), "video prompt refused");
    Prompt later              = video;
    later.items[0].timestamps = {1.0, 1.5};
    std::vector<TokenId> later_out;
    expect(identity(later, later_out), "video prompt refused");
    expect(video_out != later_out, "other timestamps give the same identity");
}

void test_picture_order_matters() {
    Prompt ab, ba;
    text(ab, 4);
    picture(ab, 0x11);
    text(ab, 4);
    picture(ab, 0x22);
    text(ba, 4);
    picture(ba, 0x22);
    text(ba, 4);
    picture(ba, 0x11);
    std::vector<TokenId> iab, iba;
    expect(identity(ab, iab) && identity(ba, iba), "two-picture prompt refused");
    expect(iab != iba, "swapped pictures give the same identity");
}

void test_prefix_is_stable() {
    // The disk names a block by the identity of its prefix, so the identity of a prefix must be
    // the prefix of the identity: a longer conversation finds the blocks of a shorter one.
    const Prompt prompt = one_picture(0x11);
    std::vector<TokenId> full;
    expect(identity(prompt, full), "picture prompt refused");
    for (const std::size_t count :
         {std::size_t{0}, std::size_t{5}, std::size_t{10}, std::size_t{18}, std::size_t{25}}) {
        std::vector<TokenId> part;
        expect(identity(prompt, part, count), "a prefix that does not split the picture refused");
        expect(part.size() == count && std::equal(part.begin(), part.end(), full.begin()),
               "the identity of a prefix is not the prefix of the identity");
    }
}

void test_frontier_inside_a_picture_is_refused() {
    const Prompt prompt = one_picture(0x11);
    for (const std::size_t frontier : {std::size_t{11}, std::size_t{14}, std::size_t{17}}) {
        std::vector<TokenId> out;
        expect(!identity(prompt, out, frontier), "an identity cut inside a picture accepted");
        expect(splits_vision(prompt.items, frontier), "a frontier inside a picture not reported");
    }
    for (const std::size_t frontier :
         {std::size_t{0}, std::size_t{1}, std::size_t{10}, std::size_t{18}, std::size_t{30}}) {
        expect(!splits_vision(prompt.items, frontier),
               "a frontier at or outside a picture's edges reported as inside");
    }
    expect(!splits_vision({}, 5), "a text-only prompt reported as split");

    // A video item with two spans: a frontier between them is still inside the item.
    q::VisionItem video;
    video.modality    = q::PromptModality::Video;
    video.token_spans = {{.begin = 10, .count = 4}, {.begin = 20, .count = 4}};
    expect(splits_vision({video}, 17), "a frontier between the spans of one item not reported");
    expect(!splits_vision({video}, 24), "the end of a two-span item reported as inside");

    q::VisionItem spanless;
    expect(splits_vision({spanless}, 0), "an item without spans is not refused");
}

void test_types_and_items_must_agree() {
    const Prompt good = one_picture(0x11);
    std::vector<TokenId> out;

    Prompt no_item = good;
    no_item.items.clear();
    expect(!identity(no_item, out), "placeholder types without an item accepted");

    Prompt text_typed    = good;
    text_typed.types[12] = 0;
    expect(!identity(text_typed, out), "an item over a text-typed token accepted");

    Prompt stray   = good;
    stray.types[3] = static_cast<std::uint8_t>(q::PromptModality::Image);
    expect(!identity(stray, out), "a placeholder type outside every item accepted");

    Prompt overlap = good;
    overlap.items.push_back(image(14, 0x22));
    expect(!identity(overlap, out), "two items over the same tokens accepted");

    Prompt spanless = good;
    spanless.items[0].token_spans.clear();
    expect(!identity(spanless, out), "an item without spans accepted");

    Prompt past_end = good;
    past_end.items[0].token_spans.push_back({.begin = 25, .count = 100});
    expect(!identity(past_end, out), "an item crossing the end of the tokens accepted");
}

} // namespace

int main() {
    test_text_only_is_the_token_stream();
    test_same_size_pictures_differ();
    test_item_fields_bind_the_identity();
    test_picture_order_matters();
    test_prefix_is_stable();
    test_frontier_inside_a_picture_is_refused();
    test_types_and_items_must_agree();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "disk identity: all checks passed\n";
    return 0;
}
