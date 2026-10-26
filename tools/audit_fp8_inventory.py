"""Count stored group-32 FP8 opportunities without loading checkpoint payloads.

Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
This inventories the source checkpoint, not dynamically converted runtime
weights or a drafter. Estimates apply an explicitly supplied range of savings;
they are not measured compression ratios or deployment allocation deltas.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def headers(source: Path, index: dict) -> tuple[dict, dict]:
    """Read indexed safetensors headers and reject unsafe paths or byte spans."""
    tensors, fingerprints = {}, {}
    for shard in sorted(set(index.values())):
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError("Unsafe checkpoint shard path")
        path = source / shard
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= min(32 * 2**20, path.stat().st_size - 8):
                raise ValueError("Invalid safetensors header length")
            raw = stream.read(length)
        fingerprints[shard] = hashlib.sha256(raw).hexdigest()
        remaining = path.stat().st_size - length - 8
        for name, spec in json.loads(raw).items():
            if name == "__metadata__":
                continue
            if index.get(name) != shard or name in tensors:
                raise ValueError("Header/index mismatch")
            a, b = spec["data_offsets"]
            shape = spec["shape"]
            if (type(a) is not int or type(b) is not int or not 0 <= a <= b <= remaining
                    or not isinstance(shape, list)
                    or any(type(d) is not int or d < 0 for d in shape)):
                raise ValueError("Invalid tensor span or shape")
            tensors[name] = {**spec, "bytes": b - a, "shard": shard}
    if set(tensors) != set(index):
        raise ValueError("Index has missing tensor headers")
    return tensors, fingerprints


def matches(pattern: str, name: str) -> bool:
    """Resolve explicit module names and declared regular expressions."""
    return bool(re.match(pattern[3:], name)) if pattern.startswith("re:") else pattern == name


def inventory(source: Path, saving_range: tuple[float, float]) -> dict:
    """Separate compatible source data from the MLP family exercised by the harness."""
    config_bytes = (source / "config.json").read_bytes()
    index_bytes = (source / "model.safetensors.index.json").read_bytes()
    config = json.loads(config_bytes)
    quant = config["quantization_config"]
    if quant.get("quant_method") != "compressed-tensors":
        raise ValueError("Require a declared compressed-tensors checkpoint")
    groups = quant["config_groups"]
    layer_count = config.get("text_config", config).get("num_hidden_layers")
    if type(layer_count) is not int or layer_count <= 0:
        raise ValueError("Require the target model's declared layer count")
    tensors, fingerprints = headers(source, json.loads(index_bytes)["weight_map"])
    rows = []
    by_dtype = Counter()
    all_bytes = sum(t["bytes"] for t in tensors.values())
    for name, tensor in sorted(tensors.items()):
        dtype = tensor["dtype"]
        if dtype not in ("F8_E4M3", "F8_E5M2", "F8_E4M3FN"):
            continue
        if tensor["bytes"] != math.prod(tensor["shape"]):
            raise ValueError("FP8 shape/byte mismatch")
        by_dtype[dtype] += tensor["bytes"]
        if not name.endswith(".weight") or len(tensor["shape"]) != 2:
            continue
        module = name.removesuffix(".weight")
        explicit = [g for g in groups.values()
                    if any(matches(t, module) for t in g["targets"] if t != "Linear")]
        candidates = explicit or [g for g in groups.values() if "Linear" in g["targets"]]
        ignored = any(matches(p, module) for p in quant.get("ignore", []))
        compatible = not ignored and len(candidates) == 1
        g = candidates[0] if len(candidates) == 1 else {}
        w = g.get("weights", {})
        compatible &= (g.get("format") == "mxfp8-quantized"
                       and w.get("num_bits") == 8 and w.get("group_size") == 32
                       and w.get("type") == "float" and dtype != "F8_E5M2")
        n, k = tensor["shape"]
        scale = tensors.get(module + ".weight_scale", {})
        compatible &= (k % 32 == 0 and scale.get("dtype") == "U8"
                       and scale.get("shape") == [n, k // 32]
                       and scale.get("bytes") == n * k // 32)
        mlp = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.mlp\."
                           r"(shared_experts\.)?(gate_proj|up_proj|down_proj)", module)
        target_layer = bool(mlp and int(mlp[1]) < layer_count)
        rank_shape = [n, k // 2] if mlp and mlp[3] == "down_proj" else [n // 2, k]
        rn, rk = rank_shape
        native = (n % 2 == 0 and k % 2 == 0 and rn >= 64 and rk >= 64
                  and rn % 64 == 0 and rk % 64 == 0 and (rn % 128 == 0 or rk % 128 == 0)
                  and rn * rk <= 2**31 - 1)
        rows.append({"name": name, "dtype": dtype, "shape": [n, k],
                     "weight_bytes": tensor["bytes"], "scale_bytes": scale.get("bytes", 0),
                     "declared_group32_mxfp8": bool(compatible),
                     "mlp_family_tp2_candidate": bool(compatible and target_layer and native),
                     "target_layer": target_layer,
                     "mlp_kind": ("shared" if mlp[2] else "dense") if mlp else None,
                     "projection": mlp[3] if mlp else None})
    sums = {}
    for field in ("declared_group32_mxfp8", "mlp_family_tp2_candidate"):
        selected = [r for r in rows if r[field]]
        size = sum(r["weight_bytes"] for r in selected)
        sums[field] = {"tensor_count": len(selected), "weight_bytes_all_tp_ranks": size,
                       "unchanged_scale_bytes": sum(r["scale_bytes"] for r in selected),
                       "estimated_saved_bytes_range": [size * p / 100 for p in saving_range]}
    return {"scope": "checkpoint headers; data compatibility is not runtime integration eligibility",
            "source": str(source), "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(), "header_sha256": fingerprints,
            "all_checkpoint_tensor_bytes": all_bytes, "stored_fp8_bytes_by_dtype": dict(by_dtype),
            "target_layer_count": layer_count,
            "assumed_saving_pct_range": saving_range, "totals": sums, "weights": rows,
            "limits": ["Saving ranges are estimates, not a complete runtime compression audit.",
                       "TP sharding divides these source bytes; do not multiply the total by TP2.",
                       "Metadata cannot prove activation dtype, runtime backend or scale contents.",
                       "Runtime BF16-to-MXFP8 conversions and a separate drafter are excluded.",
                       "Stored next-token prediction layers are counted in source totals but excluded from target MLP candidates.",
                       "No GPU allocation, timing, KV-cache or model-quality result."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--saving-range", type=float, nargs=2, default=(6.0, 7.3))
    args = parser.parse_args()
    if not 0 <= args.saving_range[0] <= args.saving_range[1] <= 100:
        parser.error("Require 0 <= minimum <= maximum <= 100")
    if args.report.exists():
        parser.error("Preserve previous inventory; choose a fresh report")
    result = inventory(args.source, tuple(args.saving_range))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "totals": result["totals"]}))


if __name__ == "__main__":
    main()
