"""ZFB1 byte-exact Python/native reconstruction, format isolation and bounds."""
from __future__ import annotations

import random
import unittest
import numpy as np
import test_fp8_async as async_tests
import fp8_async
import fp8_byteplane


class ByteplaneFP8Tests(async_tests.AsyncFP8Tests):
    codec = fp8_byteplane
    harness = async_tests.HARNESS.replace(
        'async_fragment(payload', 'async_fragment<false, true>(payload').replace(
        'async_fragment<true>(', 'async_fragment<true, true>(').replace(
        'async_fragment<true, false, true>(', 'async_fragment<true, true, true>(')

    def test_every_byte_value_at_every_position_in_compressed_tiles(self):
        for common in (0, 1):
            tiles = np.full((32, 256, 1024), common << 6, dtype=np.uint8)
            for position in range(32):
                tiles[position, :, position] = np.arange(256, dtype=np.uint8)
                # Exercise the last group and both exception-mask halves too.
                tiles[position, :, 512 + position] = np.arange(256, dtype=np.uint8)
                tiles[position, :, 992 + position] = np.arange(256, dtype=np.uint8)
            _, account = self.native(tiles.tobytes())
            self.assertEqual(account['raw_fallback_tiles'], 0)
            self.assertEqual(account['common_bit6'], common)

    def test_same_storage_cost_and_distinct_format(self):
        rng = random.Random(818)
        raw = bytearray(b'\x78\xf9\x62\xe3' * 2048)
        raw[::127] = rng.randbytes(len(raw[::127]))
        raw.extend(rng.randbytes(4096))
        for alignment in (16, 32):
            old, old_account = fp8_async.encode(bytes(raw), alignment=alignment)
            new, new_account = self.codec.encode(bytes(raw), alignment=alignment)
            self.assertNotEqual(old[:4], new[:4])
            for key in ('container_bytes', 'common_bit6', 'tiles', 'raw_fallback_tiles',
                        'exception_groups', 'payload_bytes', 'source_sha256'):
                self.assertEqual(old_account[key], new_account[key])
            with self.assertRaises(ValueError): fp8_async.decode(new)
            with self.assertRaises(ValueError): self.codec.decode(old)
            self.assertEqual(self.codec.decode(new), fp8_async.decode(old))


if __name__ == '__main__':
    unittest.main()
