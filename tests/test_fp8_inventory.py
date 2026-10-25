"""Checkpoint accounting must distinguish target layers, MTP and explicit overrides."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_fp8_inventory import headers, inventory


class InventoryTests(unittest.TestCase):
    def test_target_boundary_and_group_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [f"model.language_model.layers.{i}.mlp.gate_proj" for i in range(4)]
            config = {"text_config": {"num_hidden_layers": 3}, "quantization_config": {
                "quant_method": "compressed-tensors", "ignore": ["re:.*layers\\.2\\..*"],
                "config_groups": {
                    "fp8": {"format": "mxfp8-quantized", "targets": ["Linear"],
                            "weights": {"num_bits": 8, "type": "float", "group_size": 32}},
                    "override": {"format": "mxfp4-pack-quantized", "targets": [names[1]],
                                 "weights": {"num_bits": 4, "type": "float", "group_size": 32}}}}}
            metadata, offset = {}, 0
            for name in names:
                for suffix, dtype, shape, size in (("weight", "F8_E4M3", [256, 128], 32768),
                                                   ("weight_scale", "U8", [256, 4], 1024)):
                    metadata[name + "." + suffix] = {"dtype": dtype, "shape": shape,
                                                      "data_offsets": [offset, offset + size]}
                    offset += size
            shard = "model.safetensors"
            raw = json.dumps(metadata).encode()
            (root / shard).write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
            (root / "config.json").write_text(json.dumps(config))
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": dict.fromkeys(metadata, shard)}))
            report = inventory(root, (6, 7.3))
            self.assertEqual(report["totals"]["declared_group32_mxfp8"]["tensor_count"], 2)
            target = report["totals"]["mlp_family_tp2_candidate"]
            self.assertEqual(target["tensor_count"], 1)
            self.assertEqual(target["weight_bytes_all_tp_ranks"], 32768)
            self.assertEqual(target["unchanged_scale_bytes"], 1024)
            self.assertEqual(target["estimated_saved_bytes_range"], [32768 * .06, 32768 * .073])

    def test_rejects_paths_and_out_of_file_tensor_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                headers(root, {"weight": "../model.safetensors"})
            raw = json.dumps({"weight": {"dtype": "F8_E4M3", "shape": [64, 128],
                                         "data_offsets": [0, 8192]}}).encode()
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
            with self.assertRaisesRegex(ValueError, "span"):
                headers(root, {"weight": "model.safetensors"})


if __name__ == "__main__":
    unittest.main()
