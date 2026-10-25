"""Verify the bundled, model-independent MXFP4 scale representation on CPU."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
from zipserv_vllm.mxfp4_codec import pack_sparse, unpack_sparse


class CodecTests(unittest.TestCase):
    def test_consecutive_and_generic_palettes(self) -> None:
        """The data selects the palette; it is never tied to a model exponent range."""
        for palette in ((0, 1, 2), (119, 120, 121), (252, 253, 254), (3, 91, 254)):
            raw = np.resize(np.array(palette, dtype=np.uint8), 4096)
            packed = pack_sparse(raw)
            self.assertEqual(packed.palette, palette)
            self.assertEqual(bool(packed.descriptor & (1 << 24)), palette[1] == palette[0] + 1)
            self.assertEqual(packed.retained_bytes, 1024 + 24)
            np.testing.assert_array_equal(unpack_sparse(packed), raw)

    def test_all_bytes_and_exceptions(self) -> None:
        """Preserve all U8 bit patterns in the byte codec, including the reserved float code."""
        raw = np.full(8192, 120, dtype=np.uint8)
        raw[255:511] = np.arange(256, dtype=np.uint8)
        packed = pack_sparse(raw)
        self.assertGreater(packed.exceptions.size, 0)
        np.testing.assert_array_equal(unpack_sparse(packed), raw)

    def test_chunk_boundary_uses_global_exception_indices(self) -> None:
        raw = np.resize(np.array([119, 120, 121], dtype=np.uint8), 2**20 + 32)
        raw[2**20 - 16:2**20 + 16] = np.arange(32, dtype=np.uint8)
        packed = pack_sparse(raw)
        np.testing.assert_array_equal(unpack_sparse(packed), raw)
        escaped = packed.selectors[packed.selectors >> 30 == 3]
        np.testing.assert_array_equal(escaped & 0x3fffffff, np.arange(len(escaped)))

    def test_single_value_and_random_raw_fallback_cost(self) -> None:
        rng = np.random.default_rng(174)
        for raw in (np.full(4096, 127, dtype=np.uint8), rng.integers(0, 256, 4096, dtype=np.uint8)):
            packed = pack_sparse(raw)
            np.testing.assert_array_equal(unpack_sparse(packed), raw)
        self.assertGreater(packed.retained_bytes, raw.size)

    def test_invalid_source_layout(self) -> None:
        for raw in (np.zeros(0, np.uint8), np.zeros(17, np.uint8), np.zeros(16, np.int32),
                    np.zeros((4, 4), np.uint8), np.zeros(32, np.uint8)[::2]):
            with self.subTest(shape=raw.shape), self.assertRaises(ValueError):
                pack_sparse(raw)

    def test_noncanonical_and_invalid_exceptions(self) -> None:
        raw = np.full(4096, 120, dtype=np.uint8)
        raw[:16] = np.arange(16, dtype=np.uint8)
        packed = pack_sparse(raw)
        corrupt = packed.selectors.copy()
        corrupt[1] = 3  # Selector 3 requires a complete vector escape.
        with self.assertRaises(ValueError):
            unpack_sparse(replace(packed, selectors=corrupt))
        corrupt = packed.selectors.copy()
        corrupt[0] = 0xc0000001  # Skipped or out-of-range exception index.
        with self.assertRaises(ValueError):
            unpack_sparse(replace(packed, selectors=corrupt))
        with self.assertRaises(ValueError):
            unpack_sparse(replace(packed, exceptions=np.concatenate([packed.exceptions, packed.exceptions])))
        with self.assertRaises(ValueError):
            unpack_sparse(replace(packed, palette=(120, 120, 120)))


if __name__ == "__main__":
    unittest.main()
