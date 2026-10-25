"""Bit-exact reference codecs, honest accounting, bounded readers and selection gates."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compression_policy as policy
import low_precision_codec as codec
import screen_low_precision as screen


class CodecTests(unittest.TestCase):
    def test_all_float_patterns_preserved(self) -> None:
        raw = bytes(range(256)) * 4 + bytes(range(17))
        skewed = b"\x38\xb8\x39\xb9" * 256 + bytes(range(17))
        for kind in codec.KINDS:
            for name in codec.CODECS:
                for streams in ((1, 32) if name in ("rans", "huffman") else (1,)):
                    for data in (raw, skewed):
                        with self.subTest(kind=kind, codec=name, streams=streams, uniform=data is raw):
                            packed, accounting = codec.encode_container(data, kind, name, 256, streams)
                            self.assertEqual(codec.decode_container(packed), data)
                            self.assertEqual(sum(accounting[k] for k in
                                                 ("header_bytes", "codebook_bytes", "index_and_tag_bytes",
                                                  "aligned_payload_bytes")), len(packed))
                            self.assertEqual(accounting["container_sha256"], hashlib.sha256(packed).hexdigest())

    def test_fields_are_bitwise_inverses_including_single_byte(self) -> None:
        for kind in codec.KINDS:
            for byte in range(256):
                raw = bytes([byte])
                symbols, residual, _, _ = codec.split_fields(raw, kind)
                self.assertEqual(codec.join_fields(symbols, residual, kind), raw)
        for width in range(1, 9):
            values = bytes(range(1 << width)) + b"\0"
            self.assertEqual(codec.unpack_bits(codec.pack_bits(values, width), len(values), width), values)
        for width in (0, 9):
            with self.assertRaises(ValueError):
                codec.pack_bits(b"\0", width)

    def test_entropy_streams_and_rejected_truncations(self) -> None:
        rng = random.Random(57)
        for data in (b"\xff" * 4096, bytes(range(256)) * 8,
                     bytes(rng.choices(range(256), weights=[10000] + [1] * 255, k=4097))):
            counts = [data.count(i) for i in range(256)]
            freq = codec.frequencies(counts)
            payload = codec.rans_encode(data, freq)
            self.assertEqual(codec.rans_decode(payload, len(data), freq), data)
            lengths = codec.huffman_lengths(counts)
            huffman = codec.huffman_encode(data, lengths)
            self.assertEqual(codec.huffman_decode(huffman, len(data), lengths), data)
            for broken in (payload[:3], payload + b"\0", payload[:-1]):
                with self.assertRaises(ValueError):
                    codec.rans_decode(broken, len(data), freq)
            for broken in (huffman[:-1], huffman + b"\0"):
                with self.assertRaises(ValueError):
                    codec.huffman_decode(broken, len(data), lengths)

    def test_container_rejects_corrupt_boundaries(self) -> None:
        packed, _ = codec.encode_container(b"\x7f" * 4099, "scale-u8", "rans", 2048, 32)
        for payload in (b"", packed[:20], packed[:-1], packed + b"\0", b"BAD!" + packed[4:],
                        packed[:24] + b"\1" + packed[25:]):
            with self.subTest(length=len(payload)):
                with self.assertRaises(ValueError):
                    codec.decode_container(payload)

    def test_raw_fallback_and_dictionary_exceptions(self) -> None:
        uniform = bytes(range(256)) * 16
        result = screen.screen(uniform, "fp4", codecs=("palette2", "rans"))
        self.assertEqual(min(result["candidates"], key=lambda c: c["container_bytes"])["id"], "raw")
        for candidate in result["candidates"][1:]:
            self.assertEqual(candidate["raw_fallback_tiles"], candidate["tiles"])
        repeated = (bytes(range(16)) * 100 + b"\xff" * 16) * 3 + b"tail!"
        packed, result = codec.encode_container(repeated, "scale-u8", "vector16", 256)
        self.assertGreater(result["saving_pct"], 80)
        self.assertGreater(result["dictionary_vector_coverage"], .9)
        self.assertEqual(codec.decode_container(packed), repeated)
        with self.assertRaises(ValueError):
            codec.vector_decode(b"\xfe", 16, bytes(range(16)), 16)
        table = bytes(range(8))
        data = table + b"\xff" * 8 + table + b"\xfe" * 8 + b"end"
        direct = codec.vector_encode(data, table, 8, True)
        self.assertEqual(codec.vector_decode(direct, len(data), table, 8, True), data)
        self.assertEqual(struct.unpack_from("<4H", direct), (0, 0x8000, 0, 0x8001))
        broken = direct[:2] + struct.pack("<H", 0x8001) + direct[4:]
        with self.assertRaises(ValueError):
            codec.vector_decode(broken, len(data), table, 8, True)
        data = (table + b"\xff" * 8) * 70 + b"end"
        warped = codec.vector_encode(data, table, 8, warp=True)
        self.assertEqual(codec.vector_decode(warped, len(data), table, 8, warp=True), data)
        broken = warped[:2] + struct.pack("<H", 15) + warped[4:]
        with self.assertRaises(ValueError):
            codec.vector_decode(broken, len(data), table, 8, warp=True)


class ReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def shard(self, dtype: str = "F8_E4M3", extra: int = 0) -> Path:
        data = bytes(range(256)) * 512
        header = json.dumps({"weight": {"dtype": dtype, "shape": [512, 256],
                                        "data_offsets": [0, len(data) + extra]}}).encode()
        path = self.root / "model.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + data)
        return path

    def test_bounded_disjoint_windows_and_dtype(self) -> None:
        path = self.shard()
        data, report = screen.read_sample(path, "weight", "fp8-e4m3", 16384, 2048, 4)
        self.assertEqual(len(data), 16384)
        self.assertEqual(report["tensor_bytes"], 131072)
        windows = report["windows"]
        self.assertTrue(all(w["offset"] % 2048 == 0 for w in windows))
        self.assertTrue(all(a["offset"] + a["bytes"] <= b["offset"] for a, b in zip(windows, windows[1:])))
        with self.assertRaises(ValueError):
            screen.read_sample(path, "weight", "fp8-e5m2", 16384, 2048, 4)
        with self.assertRaises(ValueError):
            screen.read_sample(path, None, "fp8-e4m3", 16384, 2048, 4)
        with self.assertRaises(ValueError):
            screen.read_sample(path, "weight", "fp8-e4m3", 1024, 2048, 4)

    def test_invalid_offsets_and_shard_paths(self) -> None:
        path = self.shard(extra=1)
        with self.assertRaises(ValueError):
            screen.read_sample(path, "weight", "fp8-e4m3", 16384, 2048, 4)
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "../model.safetensors"}}))
        with self.assertRaises(ValueError):
            screen.read_sample(self.root, "weight", "fp8-e4m3", 16384, 2048, 4)

    def test_small_raw_and_cli_preserve_existing_report(self) -> None:
        source = self.root / "raw.bin"
        source.write_bytes(b"\x7f" * 99)
        raw, metadata = screen.read_sample(source, None, "scale-u8", 8192, 2048, 4)
        self.assertEqual(raw, source.read_bytes())
        self.assertTrue(metadata["entire_tensor_sampled"])
        report = self.root / "result.json"
        command = [sys.executable, str(Path(screen.__file__)), "--source", str(source),
                   "--kind", "scale-u8", "--codecs", "vector16", "--report", str(report)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        document = json.loads(report.read_text())
        self.assertEqual(document["selection"]["recommended_codec"], "raw")
        self.assertEqual(document["selection"]["status"], "gpu_measurements_required")
        original = report.read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(report.read_bytes(), original)


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [{"id": "raw", "container_bytes": 1000},
                           {"id": "fast", "container_bytes": 900, "container_sha256": "fast-hash"},
                           {"id": "small", "container_bytes": 700, "container_sha256": "small-hash"}]
        self.evidence = {
            "schema_version": 1, "sample_sha256": "input", "layout": "runtime",
            "hardware": "test fixture, not a measurement", "workload": "paired synthetic test",
            "scope": "kernel", "baseline": {"samples_us": [100] * 10, "resident_bytes": 1000},
            "candidates": {
                "fast": {"samples_us": [99] * 10, "resident_bytes": 900,
                         "container_sha256": "fast-hash", "exact_gpu_roundtrip": True,
                         "output_check_passed": True},
                "small": {"samples_us": [107] * 10, "resident_bytes": 700,
                          "container_sha256": "small-hash", "exact_gpu_roundtrip": True,
                          "output_check_passed": True},
            },
        }

    def choose(self, goal: str, evidence: dict | None, **overrides: float) -> dict:
        return policy.select(self.candidates, policy.Policy(goal, **overrides), evidence, "input", "runtime")

    def test_goals_are_limits_not_predictions(self) -> None:
        self.assertEqual(self.choose("speed", self.evidence)["recommended_codec"], "fast")
        self.assertEqual(self.choose("balanced", self.evidence)["recommended_codec"], "fast")
        self.assertEqual(self.choose("memory", self.evidence)["recommended_codec"], "small")
        self.assertEqual(self.choose("memory", self.evidence, max_slowdown_pct=2)["recommended_codec"], "fast")
        self.assertEqual(self.choose("balanced", None)["recommended_codec"], "raw")
        self.assertEqual(self.choose("memory", None)["size_winner"], "small")
        self.assertEqual(self.choose("memory", None)["measurements"], [])

    def test_tail_latency_and_gpu_correctness_gate(self) -> None:
        self.evidence["candidates"]["fast"]["samples_us"][-1] = 200
        self.evidence["candidates"]["small"]["exact_gpu_roundtrip"] = False
        self.assertEqual(self.choose("balanced", self.evidence)["recommended_codec"], "raw")
        self.assertNotIn("small", self.choose("memory", self.evidence)["pareto_candidates"])
        self.evidence["candidates"]["small"]["exact_gpu_roundtrip"] = True
        self.evidence["candidates"]["small"]["output_check_passed"] = False
        self.assertEqual(self.choose("memory", self.evidence)["recommended_codec"], "raw")

    def test_mismatched_or_invalid_evidence_rejected(self) -> None:
        for key, value in (("sample_sha256", "wrong"), ("layout", "checkpoint"), ("hardware", "")):
            bad = deepcopy(self.evidence)
            bad[key] = value
            with self.assertRaises(ValueError):
                self.choose("balanced", bad)
        for key, value in (("container_sha256", "wrong"), ("samples_us", [1]),
                           ("samples_us", [float("nan")] * 10), ("resident_bytes", 0)):
            bad = deepcopy(self.evidence)
            bad["candidates"]["fast"][key] = value
            with self.assertRaises(ValueError):
                self.choose("balanced", bad)
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                policy.Policy("balanced", max_slowdown_pct=value)

    def test_analytical_budget_is_labelled_and_uses_bytes(self) -> None:
        result = policy.read_time_budget(1048576, 524288, [100, 200])
        self.assertEqual(result["ideal_saved_read_us_for_sample"], {"low": 2.62144, "high": 5.24288})
        self.assertEqual(result["basis"], "analytical_memory_read_budget_not_predicted_latency")
        self.assertEqual(policy.read_time_budget(100, 200, [100])["ideal_saved_read_us_for_sample"]["high"], 0)
        with self.assertRaises(ValueError):
            policy.read_time_budget(100, 50, [0])

    def test_cache_hint_does_not_invent_an_observed_hit_rate(self) -> None:
        hint = policy.cache_reuse_hint(100, 100, 110, 3)
        self.assertAlmostEqual(hint["required_hit_rate_pct"], 70)
        self.assertEqual(hint["basis"], "linear_hit_miss_cost_model_not_measured_reuse")
        self.assertEqual(policy.cache_reuse_hint(100, 100, 110, 0)["required_hit_rate_pct"], 100)
        self.assertEqual(policy.cache_reuse_hint(100, 100, 102, 3)["required_hit_rate_pct"], 0)
        self.assertIsNone(policy.cache_reuse_hint(100, 105, 110, 3)["required_hit_rate_pct"])
        with self.assertRaises(ValueError):
            policy.cache_reuse_hint(0, 100, 110, 3)


if __name__ == "__main__":
    unittest.main()
