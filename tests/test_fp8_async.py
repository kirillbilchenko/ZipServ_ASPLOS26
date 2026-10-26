"""Exact async-tile format, native fragment bounds and Marlin tile mapping."""
from __future__ import annotations

import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fp8_async

HARNESS = r'''
#include "fp8_async_decode.h"
#include <fstream>
#include <iterator>
#include <vector>
int main(int argc, char** argv) {
  for (unsigned common = 0; common < 2; common++)
    for (unsigned flips = 0; flips < 16; flips++)
      for (unsigned position = 0; position < 4; position++)
        for (unsigned value = 0; value < 128; value++) {
          uint32_t codes = 0, reference = 0;
          for (unsigned j = 0; j < 4; j++) {
            const unsigned v = j == position ? value : ((value * 31 + j * 7) & 127);
            codes |= v << (j * 7);
            reference |= ((v & 63) | ((v & 64) << 1) |
                ((common ^ ((flips >> j) & 1)) << 6)) << (j * 8);
          }
          if (zipserv_fp8::finish7<true>(codes, common * 0x40404040u, flips) != reference) return 2;
          if (!flips && zipserv_fp8::finish7<false>(codes, common * 0x40404040u, 0) != reference) return 3;
          const auto bf = zipserv_fp8::direct7(codes, flips, common);
          const uint32_t odd = (reference & 0x80008000u) | ((reference & 0x7f007f00u) >> 4);
          const uint32_t shifted = reference << 8;
          const uint32_t even = (shifted & 0x80008000u) | ((shifted & 0x7f007f00u) >> 4);
          if (bf.even != even || bf.odd != odd) return 6;
        }
  if (argc != 3) return 4;
  std::ifstream file(argv[1], std::ios::binary);
  std::vector<uint8_t> blob((std::istreambuf_iterator<char>(file)), {});
  const auto size = zipserv_fp8::load64(blob.data() + 8);
  std::ofstream out(argv[2], std::ios::binary);
  for (uint64_t tile = 0; tile < size / 1024; tile++) {
    const uint64_t entry = zipserv_fp8::load64(blob.data() + 64 + tile * 8);
    const size_t payload_size = 1024 - ((entry >> 1) & 7) * 16;
    auto* payload = new uint8_t[payload_size];
    std::memcpy(payload, blob.data() + (entry & ~uint64_t(15)), payload_size);
    for (unsigned pair = 0; pair < 32; pair++) {
      const auto result = zipserv_fp8::async_fragment(payload, pair, entry & 1, blob[4]);
      const auto uniform = zipserv_fp8::async_fragment<true>(payload, pair, entry & 1, blob[4]);
      if (std::memcmp(&result, &uniform, sizeof(result))) return 6;
      const auto indexed = zipserv_fp8::async_fragment<true, false, true>(payload, pair, entry & 1, blob[4]);
      if (std::memcmp(&result, &indexed, sizeof(result))) return 7;
      out.write(reinterpret_cast<const char*>(&result), 32);
    }
    delete[] payload;  // Isolated allocation exposes any reads into padding/next tile.
  }
  return out.good() ? 0 : 5;
}
'''


class AsyncFP8Tests(unittest.TestCase):
    codec = fp8_async
    harness = HARNESS
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("Require a C++ compiler for native fragment checks")
        cls.scratch = tempfile.TemporaryDirectory(prefix="zipserv-async-cpu-")
        cls.root = Path(cls.scratch.name)
        source = cls.root / "check.cpp"
        source.write_text(cls.harness)
        cls.binary = cls.root / "check"
        subprocess.run([compiler, "-std=c++17", "-O2", "-g", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                        "-I", str(ROOT / "tools"), str(source), "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def native(self, raw, force_raw=False, alignment=16):
        blob, accounting = self.codec.encode(raw, force_raw=force_raw, alignment=alignment)
        self.assertEqual(self.codec.decode(blob), raw)
        self.assertEqual(sum(accounting[k] for k in ("header_bytes", "index_bytes", "payload_bytes")), len(blob))
        source, output = self.root / "input.zfa", self.root / "output.bin"
        source.write_bytes(blob)
        subprocess.run([str(self.binary), str(source), str(output)], check=True,
                       env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0", "UBSAN_OPTIONS": "halt_on_error=1"})
        self.assertEqual(output.read_bytes(), raw)
        return blob, accounting

    def test_patterns_sparse_exceptions_and_raw_fallback(self):
        rng = random.Random(183)
        mixed = bytearray(b"\x78\xf9\x62\xe3" * 2048)
        for group in (0, 1, 30, 31, 32, 33, 62, 63, 64, 127, 511):
            mixed[group * 16:group * 16 + 16] = bytes(range(0, 256, 16))
        for raw in (b"\0" * 8192, b"\xff" * 8192, bytes(range(256)) * 32,
                    rng.randbytes(8192), bytes(mixed), bytes(mixed[:4096]) + rng.randbytes(4096)):
            for force in (False, True):
                for alignment in (16, 32):
                    self.native(raw, force, alignment)

    def test_every_partial_copy_length(self):
        for exceptions in (0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 64):
            raw = bytearray(b"\x78" * 1024)
            for group in range(exceptions):
                raw[group * 16] ^= 64
            blob, account = self.native(bytes(raw))
            if exceptions < 49:
                self.assertEqual(account["raw_fallback_tiles"], 0)
                entry = struct.unpack_from("<Q", blob, 64)[0]
                self.assertEqual(len(blob) - (entry & ~15), 1024 - ((entry >> 1) & 7) * 16)
            else:
                self.assertEqual(account["raw_fallback_tiles"], 1)

    def test_malformed_inputs(self):
        for raw in (b"", b"x" * 1023):
            with self.assertRaises(ValueError): self.codec.encode(raw)
        raw = bytearray(b"\x78" * 4096); raw[0] = 0
        blob, _ = self.codec.encode(bytes(raw))
        payload = struct.unpack_from("<Q", blob, 64)[0] & ~15
        broken = [blob[:-1], blob + bytes(16), b"BAD!" + blob[4:]]
        for offset, value in ((4, 2), (5, 32), (24, 2), (25, 1), (32, 1), (64, 1),
                              (payload + 8, 1), (payload + 912, 0), (payload + 914, 1)):
            data = bytearray(blob); data[offset] = value; broken.append(bytes(data))
        for data in broken:
            with self.assertRaises(ValueError): self.codec.decode(data)

    def test_global_shared_tile_mapping(self):
        # Entire contiguous 1024-byte compression tiles must stay together across
        # Marlin's strided K/N loads, including N not divisible by 128.
        for n in (64, 128, 192, 256, 384, 12288):
            for tile_k, tile_n, threads in ((128, 128, 256), (64, 256, 256),
                                           (64, 128, 128), (128, 64, 128)):
                if n % tile_n: continue
                stride, stage, vecs = tile_n, tile_n * tile_k // 16, 2
                iterations = stage // (threads * vecs)
                for column in range(n // tile_n):
                    mapping = {}
                    for thread in range(threads):
                        first = thread if threads <= stride else n * (thread // stride) + thread % stride
                        first += stride * column
                        count = (stride + threads - 1) // threads
                        for i in range(iterations * vecs):
                            source = first + (i % count) * threads + n * (i // count) * ((threads + stride - 1) // stride)
                            destination = threads * i + thread
                            mapping[destination] = source
                    self.assertEqual(set(mapping), set(range(stage)))
                    # Producer ownership changes the issuing thread, not the
                    # native tile contents. Each warp must own both copies of
                    # every lane in its entire 1024-byte tile, even at K slices.
                    for slice_row in (0, 1, 7):
                        origin = stride * column + n * (tile_k // 16) * slice_row
                        producer, owners = {}, {}
                        for thread in range(threads):
                            for i in range(iterations):
                                local = 2 * threads * i + (thread // 32) * 64 + thread % 32
                                source = origin + n * (local // stride) + local % stride
                                for j in (0, 32):
                                    producer[local + j] = source + j
                                    owners.setdefault((local + j) // 64, set()).add(thread // 32)
                        offset = n * (tile_k // 16) * slice_row
                        self.assertEqual(producer, {j: v + offset for j, v in mapping.items()})
                        self.assertTrue(all(len(warps) == 1 for warps in owners.values()))
                        # Each individual copy instruction is contiguous across
                        # a warp, avoiding extra 32-byte sector transactions.
                        for warp in range(threads // 32):
                            for i in range(iterations):
                                for half in (0, 32):
                                    vectors = [2 * threads * i + warp * 64 + lane + half for lane in range(32)]
                                    self.assertEqual([producer[j] for j in vectors],
                                                     list(range(producer[vectors[0]], producer[vectors[0]] + 32)))
                    for start in range(0, stage, 64):
                        self.assertEqual([mapping[j] for j in range(start, start + 64)],
                                         list(range(mapping[start], mapping[start] + 64)))
                        self.assertEqual(mapping[start] % 64, 0)


if __name__ == "__main__":
    unittest.main()
