"""Preservation of every FP8 pattern and strict GPU input validation."""
import random
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fp8_palette as codec
import fp8_range as range_codec


class FP8PaletteTests(unittest.TestCase):
    def test_range_block_formats(self):
        rng = random.Random(177)
        repeated = bytearray(b"\x78\xf8\x69\xe9" * 1024)
        for i in range(256):
            repeated[(i * 17) % len(repeated)] = i
        for group in (16, 32):
            for tile in (1024, 2048):
                for raw in (bytes(range(256)) * 16, bytes(repeated), rng.randbytes(4096), b"\0" * 4096):
                    blob, sizes = range_codec.encode(raw, group, tile)
                    self.assertEqual(range_codec.decode(blob), raw)
                    self.assertEqual(sum(sizes[k] for k in ("header_bytes", "index_bytes", "payload_bytes")), len(blob))
                blob, _ = range_codec.encode(bytes(repeated), group, tile)
                for bad in (blob[:-1], blob + bytes(16), b"BAD!" + blob[4:], blob[:5] + b"\1" + blob[6:]):
                    with self.assertRaises(ValueError):
                        range_codec.decode(bad)

    def test_all_patterns_exceptions_fallback_and_accounting(self):
        rng = random.Random(174)
        patterns = [bytes(range(256)) * 16, b"\xb8" * 4096, rng.randbytes(4096)]
        mixed = bytearray(b"\xb8" * 4096)
        for i, value in enumerate(range(256)):
            mixed[(i * 17) % len(mixed)] = value
        patterns.append(bytes(mixed))
        for bits in (2, 3):
            for tile in (1024, 2048):
                for raw in patterns:
                    with self.subTest(bits=bits, tile=tile, pattern=raw[:8]):
                        encoded, sizes = codec.encode(raw, bits, tile)
                        self.assertEqual(codec.decode(encoded), raw)
                        self.assertEqual(sum(sizes[k] for k in (
                            "header_bytes", "index_bytes", "payload_bytes")), len(encoded))
                self.assertEqual(codec.encode(patterns[0], bits, tile)[1]["raw_fallback_tiles"],
                                 len(patterns[0]) // tile)
                self.assertGreater(codec.encode(patterns[3], bits, tile)[1]["exception_symbols"], 0)

    def test_bad_inputs_and_corruption(self):
        for raw, bits, tile in ((b"", 3, 1024), (b"x" * 1023, 3, 1024),
                                (b"x" * 1024, 4, 1024), (b"x" * 1024, 3, 512)):
            with self.assertRaises(ValueError):
                codec.encode(raw, bits, tile)
        encoded, _ = codec.encode(b"\x38" * 4096)
        for broken in (encoded[:20], encoded[:-1], encoded + b"\0" * 16,
                       b"BAD!" + encoded[4:]):
            with self.assertRaises(ValueError):
                codec.decode(broken)
        for offset, value in ((5, 1), (24, 255), (31, 1), (32, 1), (64, 1)):
            broken = bytearray(encoded)
            broken[offset] = value
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                codec.decode(bytes(broken))
        payload = struct.unpack_from("<Q", encoded, 64)[0] & ~1
        broken = bytearray(encoded)
        broken[payload + 2] = 1
        with self.assertRaises(ValueError):
            codec.decode(bytes(broken))


if __name__ == "__main__":
    unittest.main()
