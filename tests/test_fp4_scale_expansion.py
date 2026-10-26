"""Exhaustively compare the actual affine expansion against scalar palette lookup."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ExpansionTests(unittest.TestCase):
    def test_every_consecutive_palette_and_valid_half_record(self) -> None:
        """All 254 palettes and 3**8 selector halves, including byte carry boundaries."""
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "A C++17 compiler is required for the native helper check")
        header = Path(__file__).resolve().parents[1] / "tools" / "fp4_scale_expansion.cuh"
        code = r'''
#include <cstdint>
#include <cstdio>
#define __device__
#define __forceinline__ inline
struct int2 { int x, y; };
int2 make_int2(int x, int y) { return {x, y}; }
#include "fp4_scale_expansion.cuh"
int main() {
  constexpr int order[] = {0,1,4,5,2,3,6,7};
  for (unsigned base = 0; base <= 253; ++base) {
    const uint32_t palette = base | ((base+1)<<8) | ((base+2)<<16) | (1u<<24);
    for (unsigned pattern = 0; pattern < 6561; ++pattern) {
      unsigned cursor = pattern;
      uint32_t q = 0;
      for (unsigned j = 0; j < 8; ++j) { q |= (cursor % 3) << (2*j); cursor /= 3; }
      const int2 result = zipserv_fp4_affine_half(palette, q);
      const uint32_t words[] = {uint32_t(result.x), uint32_t(result.y)};
      for (unsigned j = 0; j < 8; ++j) {
        const unsigned actual = (words[j/4] >> (8*(j%4))) & 255;
        const unsigned expected = base + ((q >> (2*order[j])) & 3);
        if (actual != expected) {
          std::fprintf(stderr, "base=%u q=%u byte=%u actual=%u expected=%u\n",
                       base,q,j,actual,expected);
          return 1;
        }
      }
    }
  }
  std::puts("1666494 half records; 13331952 bytes exact");
}
'''
        with tempfile.TemporaryDirectory(prefix="zipserv-fp4-check.") as temporary:
            root = Path(temporary)
            (root / "check.cpp").write_text(code)
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(header.parent),
                            str(root / "check.cpp"), "-o", str(root / "check")], check=True)
            result = subprocess.run([str(root / "check")], check=True, text=True, capture_output=True)
            self.assertIn("13331952 bytes exact", result.stdout)


if __name__ == "__main__":
    unittest.main()
