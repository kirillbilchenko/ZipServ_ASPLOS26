"""Test experimental opt-in boundaries and byte-format compatibility without CUDA."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "integrations/vllm"), str(ROOT / "tools")]
from zipserv_vllm.mixed_plugin import settings
from zipserv_vllm import fp8_codec
import fp8_byteplane


class MixedTests(unittest.TestCase):
    def test_inert_default_and_disabled_controls(self):
        self.assertIsNone(settings({}))
        parsed = settings({"ZIPSERV_MIXED_FP8": "off"})
        self.assertEqual((parsed.bf16, parsed.fp8), ("off", "off"))

    def test_native_path_and_reviewed_targets_required(self):
        for kind, library in (("BF16", "ZIPSERV_LIBRARY"), ("FP8", "ZIPSERV_FP8_LIBRARY")):
            with tempfile.NamedTemporaryFile() as f:
                env = {"ZIPSERV_MIXED_" + kind: "on", library: f.name}
                with self.assertRaises(ValueError):
                    settings(env)
                env["ZIPSERV_" + kind + "_TARGETS"] = '["layers.*.out_proj"]'
                self.assertEqual(getattr(settings(env), kind.lower()), "on")

    def test_invalid_savings_modes_and_targets(self):
        for value in ("nan", "inf", "0", "100", "-1"):
            with self.assertRaises(ValueError):
                settings({"ZIPSERV_MIXED_FP8": "off", "ZIPSERV_MIXED_MIN_SAVING_PERCENT": value})
        with self.assertRaises(ValueError):
            settings({"ZIPSERV_MIXED_FP8": "auto"})
        for value in ('"layer"', '{"layer":1}', '[1]', '[""]'):
            with self.assertRaises(ValueError):
                settings({"ZIPSERV_MIXED_BF16": "off", "ZIPSERV_BF16_TARGETS": value})

    def test_packaged_codec_consumes_existing_experiment_format(self):
        rng = np.random.default_rng(42)
        fixtures = [bytes(range(256)) * 16, rng.bytes(8192), b'\0' * 8192,
                    b'\x78\xf9\x62\xe3' * 2048, b'\x38' * 4096 + rng.bytes(4096)]
        for raw in fixtures:
            for alignment in (16, 32):
                original, _ = fp8_byteplane.encode(raw, alignment=alignment)
                packed, _ = fp8_codec.encode(raw, alignment=alignment)
                self.assertEqual(packed, original)
                self.assertEqual(fp8_codec.decode(original), raw)
                self.assertEqual(fp8_byteplane.decode(packed), raw)
                with self.assertRaises(ValueError):
                    fp8_codec.decode(packed[:-1])


if __name__ == "__main__":
    unittest.main()
