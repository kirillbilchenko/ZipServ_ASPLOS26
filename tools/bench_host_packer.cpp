// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
// Each invocation measures one implementation in a fresh process so peak RSS
// does not carry over from the reference encoder. Synthetic data, no model I/O.
#include "../csrc/HostPacker.h"
#include "../tests/reference_packer.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <sys/resource.h>

int main(int argc, char** argv) try {
    if (argc < 2 || argc > 7) throw std::invalid_argument(
        "usage: bench_host_packer verify|legacy|serial|parallel [rows=4096] [cols=4096] [iterations=3] [rare_percent=5] [workers=0]");
    const std::string mode = argv[1];
    const int m = argc > 2 ? std::stoi(argv[2]) : 4096;
    const int k = argc > 3 ? std::stoi(argv[3]) : 4096;
    const int iterations = argc > 4 ? std::stoi(argv[4]) : 3;
    const int rare_percent = argc > 5 ? std::stoi(argv[5]) : 5;
    const int workers = argc > 6 ? std::stoi(argv[6]) : 0;
    if ((mode != "verify" && mode != "legacy" && mode != "serial" && mode != "parallel") ||
        !zipserv::valid_matrix(m,k) || iterations < 1 || rare_percent < 0 || rare_percent > 100 ||
        workers < 0 || workers > 64) throw std::invalid_argument("invalid benchmark arguments");
    std::vector<uint16_t> words(size_t(m) * k);
    std::mt19937 rng(42);
    for (auto& word : words) {
        uint32_t r = rng();
        const unsigned exponent = r % 100 < unsigned(rare_percent) ? 40 + (r >> 8) % 7 : 120 + (r >> 8) % 7;
        word = uint16_t((r & 0x807f) | (exponent << 7));
    }
    if (mode == "verify") {
        uint64_t counts[256] = {};
        for (uint16_t word : words) ++counts[(word >> 7) & 255];
        zipserv_test::ReferencePacked reference(words.data(),m,k,zipserv::best_exponent_window(counts));
        auto packed = zipserv::pack_bf16(words.data(),m,k,-1,false,unsigned(workers));
        if (packed.tiles != reference.tiles || packed.max_common != reference.max_common ||
            packed.max_rare != reference.max_rare ||
            packed.global.get()[2*packed.tiles] != reference.global[2*reference.tiles] ||
            packed.global.get()[2*packed.tiles+1] != reference.global[2*reference.tiles+1])
            throw std::runtime_error("packed metadata differs from reference");
        const auto lengths = packed.lengths();
        const auto actual = packed.buffers(), expected = reference.buffers();
        for (size_t i = 0; i < lengths.size(); ++i)
            if (lengths[i] && std::memcmp(actual[i],expected[i],lengths[i]))
                throw std::runtime_error("packed bytes differ from reference");
        std::cout << "{\"mode\":\"verify\",\"matches\":true,\"packed_bytes\":"
                  << std::accumulate(lengths.begin(),lengths.end(),size_t(0)) << "}\n";
        return 0;
    }
    // Include global exponent selection in all timings, as real bridge calls do.
    std::vector<double> seconds;
    size_t packed_bytes = 0;
    uint64_t checksum = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        if (mode == "legacy") {
            uint64_t counts[256] = {};
            for (uint16_t word : words) ++counts[(word >> 7) & 255];
            const int base = zipserv::best_exponent_window(counts);
            zipserv_test::ReferencePacked packed(words.data(),m,k,base);
            seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
            packed_bytes = size_t(packed.global[2*packed.tiles]) + size_t(packed.global[2*packed.tiles+1])*2 +
                size_t(packed.tiles)*1568 + size_t(packed.tiles+1)*8;
            checksum += packed.b1[0] + packed.b2[packed.tiles*64-1] + packed.max_common;
        } else {
            auto packed = zipserv::pack_bf16(words.data(),m,k,-1,false,mode == "serial" ? 1U : unsigned(workers));
            seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
            const auto lengths = packed.lengths();
            packed_bytes = std::accumulate(lengths.begin(),lengths.end(),size_t(0));
            checksum += packed.b1.get()[0] + packed.b2.get()[packed.tiles*64-1] + packed.max_common;
        }
    }
    std::sort(seconds.begin(),seconds.end());
    const double elapsed = seconds[seconds.size()/2];
    rusage usage{};
    if (getrusage(RUSAGE_SELF,&usage)) throw std::runtime_error("getrusage failed");
#ifdef __APPLE__
    const double peak_mib = double(usage.ru_maxrss)/(1024*1024);
#else
    const double peak_mib = double(usage.ru_maxrss)/1024;
#endif
    std::cout << std::fixed << std::setprecision(3)
        << "{\"mode\":\"" << mode << "\",\"m\":" << m << ",\"k\":" << k
        << ",\"iterations\":" << iterations << ",\"rare_percent\":" << rare_percent
        << ",\"workers\":" << (mode == "legacy" || mode == "serial" ? 1U : zipserv::host_detail::worker_count((m/64)*(k/64),unsigned(workers)))
        << ",\"median_ms\":" << elapsed*1000 << ",\"input_gb_s\":" << double(words.size()*2)/elapsed/1e9
        << ",\"peak_rss_mib\":" << peak_mib << ",\"packed_bytes\":" << packed_bytes
        << ",\"checksum\":" << checksum << "}\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
