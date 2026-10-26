"""Native range decoder vs independent Python codec, with ASan/UBSan bounds checks."""
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
import fp8_range


HARNESS = r'''
#include "fp8_marlin_decode.h"
#include <fstream>
#include <iterator>
#include <vector>
int main(int argc, char** argv) {
  // Exhaustive exponent combinations at every legal base; varied residuals.
  for (unsigned base = 0; base <= 8; base++) {
    for (unsigned exp = 0; exp < 4096; exp++) {
      for (unsigned pattern = 0; pattern < 16; pattern++) {
        const unsigned low = (exp * 137 + pattern * 0x1111) & 0xffff;
        uint32_t reference = 0;
        for (unsigned i = 0; i < 4; i++) {
          const unsigned r = (low >> (4*i)) & 15;
          reference |= ((r & 7) | ((r & 8) << 4) |
                        ((((exp >> (3*i)) & 7) + base) << 3)) << (8*i);
        }
        if (zipserv_fp8::expand4(low, exp, base) != reference) return 2;
      }
    }
  }
  if (argc != 3) return 3;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<uint8_t> blob((std::istreambuf_iterator<char>(input)), {});
  if (blob.size() < 64) return 4;
  const auto bytes = zipserv_fp8::load64(blob.data() + 8);
  std::ofstream output(argv[2], std::ios::binary);
  for (uint64_t i = 0; i < bytes / 16; i++) {
    const auto address = zipserv_fp8::locate(blob.data(), i);
    const auto words = zipserv_fp8::decode(address, blob[4]);
    output.write(reinterpret_cast<const char*>(&words), 16);
    // Isolate each block: ASan catches reads into adjacent blocks or padding.
    const int size = address.raw ? 16 : 14;
    auto* isolated = new uint8_t[size];
    std::memcpy(isolated, address.data, size);
    const auto checked = zipserv_fp8::decode({isolated, address.raw, false}, blob[4]);
    if (std::memcmp(&words, &checked, 16)) return 5;
    delete[] isolated;
  }
  return output.good() ? 0 : 6;
}
'''


class MarlinDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("Native decoder checks require clang++ or g++")
        cls.scratch = tempfile.TemporaryDirectory(prefix="zipserv-marlin-cpu-")
        cls.root = Path(cls.scratch.name)
        source = cls.root / "check.cpp"
        source.write_text(HARNESS)
        cls.binary = cls.root / "check"
        subprocess.run([compiler, "-std=c++17", "-O2", "-g", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                        "-I", str(ROOT / "tools"), str(source), "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def check_native(self, raw):
        blob, accounting = fp8_range.encode(raw, 16, 2048)
        self.assertEqual(fp8_range.decode(blob), raw)
        source, output = self.root / "input.zfr", self.root / "output.bin"
        source.write_bytes(blob)
        subprocess.run([str(self.binary), str(source), str(output)], check=True,
                       env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0",
                            "UBSAN_OPTIONS": "halt_on_error=1"})
        self.assertEqual(output.read_bytes(), raw)
        return blob, accounting

    def test_all_fp8_patterns_and_raw_tiles(self):
        rng = random.Random(31)
        for raw in (bytes(range(256)) * 16, rng.randbytes(8192), b"\0" * 4096,
                    b"\x7f\xff" * 2048):
            with self.subTest(pattern=raw[:8]):
                self.check_native(raw)

    def test_both_mask_halves_and_two_byte_aligned_fallbacks(self):
        raw = bytearray(b"\x38\xb9\x42\xc3" * 2048)
        # Raw groups before/after mask boundary, final group and tile boundary.
        for block in (1, 3, 62, 63, 64, 65, 127, 128, 255, 511):
            raw[block * 16:block * 16 + 16] = bytes(range(0, 256, 16))
        blob, account = self.check_native(bytes(raw))
        self.assertGreater(account["compressed_blocks"], 0)
        offset = struct.unpack_from("<Q", blob, 64)[0]
        self.assertEqual(offset & 1, 0)
        # The first raw block follows one 14-byte block and is only 2-aligned.
        self.assertEqual((offset + 16 + 14) % 4, 2)

    def test_windows_and_alternating_raw_compressed_tiles(self):
        rng = random.Random(32)
        for base in range(9):
            values = bytes(((base + rng.randrange(8)) << 3) |
                           rng.randrange(8) | (rng.randrange(2) << 7) for _ in range(4096))
            raw = values[:2048] + rng.randbytes(2048) + values[2048:]
            with self.subTest(base=base):
                _, account = self.check_native(raw)
                self.assertEqual(account["base_exponent"], base)
                self.assertGreater(account["raw_fallback_tiles"], 0)


if __name__ == "__main__":
    unittest.main()
