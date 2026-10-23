// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
#include "../csrc/HostPacker.h"
#include "reference_packer.h"
#include <cassert>
#include <iostream>
#include <random>
#include <string>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(     std::string("line ") + std::to_string(__LINE__) + ": " + #condition); } while (0)

// Decode by inverting each small-tile index, independently of the encoder's
// nested traversal. Check both offset levels as well as all original BF16 bits.
static void check_decode(const std::vector<uint16_t>& words, int m, int k,
                         const zipserv::HostPacked& packed) {
    std::vector<uint16_t> decoded(size_t(m) * k);
    const int* global = packed.global.get();
    for (int tile = 0; tile < packed.tiles; ++tile) {
        int common = global[2 * tile], rare = global[2 * tile + 1];
        for (int s = 0; s < 64; ++s) {
            if (s % 16 == 0) {
                CHECK(packed.medium.get()[tile * 8 + (s / 16) * 2] == common - global[2 * tile]);
                CHECK(packed.medium.get()[tile * 8 + (s / 16) * 2 + 1] == rare - global[2 * tile + 1]);
            }
            const int row = (tile / (k / 64)) * 64 + (s / 16) * 16 + (s % 2) * 8;
            const int col = (tile % (k / 64)) * 64 + ((s / 4) % 4) * 16 + ((s / 2) % 2) * 8;
            const int small = tile * 64 + s;
            for (int bit = 0; bit < 64; ++bit) {
                unsigned code = ((packed.b1.get()[small] >> bit) & 1) |
                    (((packed.b2.get()[small] >> bit) & 1) << 1) |
                    (((packed.b3.get()[small] >> bit) & 1) << 2);
                uint16_t word;
                if (code) {
                    const uint8_t sm = packed.common.get()[common++];
                    word = uint16_t(((sm & 128) << 8) | ((packed.base + code - 1) << 7) | (sm & 127));
                } else word = packed.rare.get()[rare++];
                decoded[(row + bit / 8) * k + col + bit % 8] = word;
            }
        }
        CHECK((common + 15) / 16 * 16 == global[2 * (tile + 1)]);
        CHECK((rare + 7) / 8 * 8 == global[2 * (tile + 1) + 1]);
    }
    CHECK(decoded == words);
}

static int cases = 0;
static void compare(const std::vector<uint16_t>& words, int m, int k, int base,
                    unsigned workers, bool include_small = true) {
    uint64_t histogram[256] = {};
    for (uint16_t word : words) ++histogram[(word >> 7) & 255];
    const int expected_base = base < 0 ? zipserv::best_exponent_window(histogram) : base;
    zipserv_test::ReferencePacked legacy(words.data(), m, k, expected_base);
    auto packed = zipserv::pack_bf16(words.data(), m, k, base, include_small, workers);
    CHECK(packed.base == expected_base);
    CHECK(packed.tiles == legacy.tiles);
    CHECK(packed.max_common == legacy.max_common);
    CHECK(packed.max_rare == legacy.max_rare);
    CHECK(packed.global.get()[2 * packed.tiles] == legacy.global[2 * legacy.tiles]);
    CHECK(packed.global.get()[2 * packed.tiles + 1] == legacy.global[2 * legacy.tiles + 1]);
    const auto lengths = packed.lengths();
    const auto actual = packed.buffers(), expected = legacy.buffers();
    for (size_t i = 0; i < lengths.size(); ++i)
        if (lengths[i]) CHECK(std::memcmp(actual[i], expected[i], lengths[i]) == 0);
    CHECK(bool(packed.small) == include_small);
    check_decode(words, m, k, packed);
    ++cases;
}

int main() try {
    std::mt19937 rng(42);
    std::vector<uint16_t> all_words(65536);
    for (int i = 0; i < 65536; ++i) all_words[i] = uint16_t(i);
    for (int base : {-1, 0, 120, 249})
        for (unsigned workers : {1U, 2U, 4U}) compare(all_words, 256, 256, base, workers);
    // Every exponent window, with both payload streams and nontrivial padding.
    for (int base = 0; base <= 249; ++base) {
        std::vector<uint16_t> words(4096);
        for (auto& word : words) {
            uint32_t r = rng();
            const int e = r % 5 ? base + int(r % 7) : int(r % 256);
            word = uint16_t((r & 0x807f) | (e << 7));
        }
        compare(words, 64, 64, base, 1);
    }
    // Empty streams, endpoint exponents, infinities/NaNs, signed zero, and
    // alignment boundaries on both sides of the common/exception threshold.
    for (uint16_t word : {uint16_t(0), uint16_t(0x8000), uint16_t(0x7f80), uint16_t(0xffff)}) {
        std::vector<uint16_t> words(16384, word);
        for (int base : {-1, 0, 120, 249}) compare(words, 128, 128, base, 4);
    }
    for (int count : {0, 1, 7, 8, 15, 16, 17, 4080, 4081, 4088, 4095, 4096}) {
        std::vector<uint16_t> words(4096, uint16_t(30 << 7));
        std::fill_n(words.begin(), count, uint16_t(120 << 7));
        compare(words, 64, 64, 120, 1);
    }
    for (const auto& shape : {std::pair<int,int>{64,192}, {128,320}, {320,128}, {1024,1024}}) {
        std::vector<uint16_t> words(size_t(shape.first) * shape.second);
        for (auto& word : words) word = uint16_t(rng());
        for (unsigned workers : {0U, 1U, 4U})
            for (bool small : {false, true}) compare(words, shape.first, shape.second, -1, workers, small);
    }
    // Adjacent tiles may have empty common/rare streams and different padding.
    std::vector<uint16_t> mixed_tiles(256 * 256);
    for (int row = 0; row < 256; ++row)
        for (int col = 0; col < 256; ++col) {
            const int tile = (row / 64) * 4 + col / 64;
            mixed_tiles[row * 256 + col] = tile % 3 == 0 ? uint16_t(120 << 7) :
                tile % 3 == 1 ? uint16_t(40 << 7) : uint16_t(rng());
        }
    for (unsigned workers : {1U, 4U})
        for (bool small : {false, true}) compare(mixed_tiles,256,256,120,workers,small);
    // Concurrent independent calls must not share offsets or histogram state.
    std::array<std::thread, 4> calls;
    std::array<bool, 4> matched{};
    for (unsigned i = 0; i < calls.size(); ++i) calls[i] = std::thread([&, i] {
        auto packed = zipserv::pack_bf16(all_words.data(), 256, 256, 249, false, 2);
        check_decode(all_words, 256, 256, packed);
        matched[i] = true;
    });
    for (auto& call : calls) call.join();
    for (bool value : matched) CHECK(value);
    for (const auto& shape : {std::pair<int,int>{0,64}, {63,64}, {64,65}, {INT_MAX,64}}) {
        bool rejected = false;
        try { (void)zipserv::pack_bf16(all_words.data(), shape.first, shape.second); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
    }
    for (int base : {-2,250}) {
        bool rejected = false;
        try { (void)zipserv::pack_bf16(all_words.data(), 256,256,base); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
    }
    bool rejected = false;
    try { (void)zipserv::pack_bf16(nullptr, 64,64); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { (void)zipserv::pack_bf16(all_words.data(),64,64,0,false,65); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    std::cout << "PASS: " << cases << " byte-identical legacy comparisons, independent CPU decode, all BF16 words/windows, concurrent packing and invalid inputs\n";
} catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
