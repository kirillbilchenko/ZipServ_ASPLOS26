"""Screen exact FP4/FP8 compression and explain memory/latency tradeoffs, without CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from compression_policy import Policy, read_time_budget, select
from low_precision_codec import CODECS, KINDS, decode_container, encode_container, entropy, split_fields

DTYPES = {"fp4": {"U8"}, "bytes": {"U8"}, "scale-u8": {"U8", "F8_E4M3", "F8_E8M0"},
          "fp8-e4m3": {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"},
          "fp8-e5m2": {"F8_E5M2", "F8_E5M2FNUZ"}}


def read_sample(source: Path, tensor: str | None, kind: str, sample_bytes: int,
                tile_bytes: int, windows: int) -> tuple[bytes, dict]:
    """Read bounded, disjoint, tile-aligned windows with validated safetensors offsets."""
    if (kind not in KINDS or not 64 <= tile_bytes <= 16384 or tile_bytes % 32
            or windows < 1 or sample_bytes < tile_bytes * windows):
        raise ValueError("Sample budget must fit all windows; tiles must be 64..16384, divisible by 32")
    metadata = {"source": str(source), "tensor": tensor, "metadata_sha256": {}}
    path = source
    if source.is_dir():
        if tensor is None:
            raise ValueError("A checkpoint directory requires --tensor")
        index = source / "model.safetensors.index.json"
        if index.exists():
            content = index.read_bytes()
            metadata["metadata_sha256"][index.name] = hashlib.sha256(content).hexdigest()
            shard = json.loads(content)["weight_map"][tensor]
            if not isinstance(shard, str) or Path(shard).name != shard:
                raise ValueError("Unsafe checkpoint shard path")
            path = source / shard
        else:
            path = source / "model.safetensors"
        config = source / "config.json"
        if config.exists():
            metadata["metadata_sha256"][config.name] = hashlib.sha256(config.read_bytes()).hexdigest()
    elif source.suffix == ".safetensors" and tensor is None:
        raise ValueError("A safetensors file requires --tensor")
    with path.open("rb") as stream:
        file_bytes = path.stat().st_size
        start, size = 0, file_bytes
        if tensor is not None:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= min(32 * 1024**2, file_bytes - 8):
                raise ValueError("Invalid safetensors header length")
            header = stream.read(header_size)
            spec = json.loads(header)[tensor]
            offsets, shape = spec["data_offsets"], spec["shape"]
            if (spec["dtype"] not in DTYPES[kind] or not isinstance(offsets, list)
                    or len(offsets) != 2 or any(type(v) is not int or v < 0 for v in offsets)
                    or not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)):
                raise ValueError("Tensor dtype/shape/offsets do not match the explicitly selected kind")
            size = offsets[1] - offsets[0]
            start = 8 + header_size + offsets[0]
            if size <= 0 or size != math.prod(shape) or start + size > file_bytes:
                raise ValueError("Tensor byte interval is inconsistent or exceeds the shard")
            metadata.update(dtype=spec["dtype"], shape=shape,
                            shard_header_sha256=hashlib.sha256(header).hexdigest())
        if size <= 0:
            raise ValueError("Cannot screen an empty tensor")
        window = min(size, sample_bytes // windows // tile_bytes * tile_bytes)
        count = min(windows, max(1, size // window))
        positions = [0] if count == 1 else [((size - window) * i // (count - 1))
                                          // tile_bytes * tile_bytes for i in range(count)]
        parts = []
        for position in positions:
            stream.seek(start + position)
            part = stream.read(window)
            if len(part) != window:
                raise ValueError("Truncated sample")
            parts.append(part)
    raw = b"".join(parts)
    metadata.update(shard=str(path), tensor_bytes=size, sampled_bytes=len(raw),
                    windows=[{"offset": p, "bytes": window} for p in positions],
                    entire_tensor_sampled=len(raw) == size,
                    sample_sha256=hashlib.sha256(raw).hexdigest())
    return raw, metadata


def screen(raw: bytes, kind: str, tile_bytes: int = 2048, streams: tuple[int, ...] = (1, 32),
           codecs: tuple[str, ...] = CODECS, bandwidth_gbps: list[float] | None = None) -> dict:
    """Compare real serialized sizes, validating every candidate's full reconstruction."""
    if not raw or kind not in KINDS or not streams or set(streams) - {1, 32} or set(codecs) - set(CODECS):
        raise ValueError("Invalid kind, codecs, streams or empty sample")
    symbols, _, _, _ = split_fields(raw, kind)
    cases = [{"id": "raw", "codec": "raw", "container_bytes": len(raw), "saving_pct": 0.0,
              "exact_roundtrip": True, "container_sha256": hashlib.sha256(raw).hexdigest()}]
    for codec in codecs:
        for lanes in streams if codec in ("rans", "huffman") else (1,):
            packed, result = encode_container(raw, kind, codec, tile_bytes, lanes)
            if decode_container(packed) != raw:
                raise RuntimeError(f"Exact reconstruction failed: {codec}/{lanes}")
            result.update(id=f"{codec}-s{lanes}", exact_roundtrip=True)
            if bandwidth_gbps:
                result["performance_hint"] = read_time_budget(len(raw), len(packed), bandwidth_gbps)
            cases.append(result)
    return {"kind": kind, "raw_bytes": len(raw), "symbol_entropy_bits": entropy(symbols),
            "entropy_field": ("exponent" if kind.startswith("fp8") else "code" if kind == "fp4"
                              else "byte" if kind == "bytes" else "scale"),
            "candidates": cases}


def main() -> None:
    """Write a reproducible report and concise console hints without changing weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Raw bytes, safetensors shard or checkpoint directory")
    parser.add_argument("--tensor", help="Exact safetensors key; no tensor-name format inference")
    parser.add_argument("--kind", choices=KINDS, required=True, help="Physical representation of the input")
    parser.add_argument("--layout", choices=("checkpoint", "runtime"), default="checkpoint",
                        help="Declared source order; the screen never silently permutes bytes")
    parser.add_argument("--sample-bytes", type=int, default=65536)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--tile-bytes", type=int, default=2048)
    parser.add_argument("--streams", default="1,32")
    parser.add_argument("--codecs", default=",".join(CODECS))
    parser.add_argument("--goal", choices=("speed", "balanced", "memory"), default="balanced")
    parser.add_argument("--max-slowdown-pct", type=float)
    parser.add_argument("--min-saving-pct", type=float)
    parser.add_argument("--bandwidth-gbps", help="Comma-separated assumed effective GB/s; budget only, not predicted slowdown")
    parser.add_argument("--timings", type=Path, help="Matched GPU measurement JSON; required to recommend a runtime codec")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report exists; use a new path to preserve evidence")
    try:
        policy = Policy(args.goal, args.max_slowdown_pct, args.min_saving_pct)
        streams = tuple(int(s) for s in args.streams.split(","))
        codecs = tuple(args.codecs.split(","))
        bandwidths = [float(b) for b in args.bandwidth_gbps.split(",")] if args.bandwidth_gbps else None
        if bandwidths:
            read_time_budget(1, 0, bandwidths)
        raw, source = read_sample(args.source, args.tensor, args.kind, args.sample_bytes,
                                  args.tile_bytes, args.windows)
        result = screen(raw, args.kind, args.tile_bytes, streams, codecs, bandwidths)
        evidence = json.loads(args.timings.read_text()) if args.timings else None
        decision = select(result["candidates"], policy, evidence, source["sample_sha256"], args.layout)
    except (ValueError, KeyError, OSError, struct.error) as error:
        parser.error(str(error))
    provenance = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ("screen_low_precision.py", "low_precision_codec.py", "compression_policy.py")}
    provenance["low_precision_codec.py"] = hashlib.sha256(
        Path(encode_container.__code__.co_filename).read_bytes()).hexdigest()
    report = {
        "schema_version": 1, "source": source, "layout": args.layout, **result,
        "selection": decision, "tool_sha256": provenance,
        "timing_evidence_sha256": hashlib.sha256(args.timings.read_bytes()).hexdigest() if args.timings else None,
        "quality": {"sample_weight_bits": "exact", "inference_outputs": "not_tested"},
        "limits": ["Sample-level reference-container accounting, not a whole-model savings forecast.",
                   "Codebooks are fitted to this sample; full tensors and held-out data need separate checks.",
                   "Headers, codebooks, 64-bit offsets, tags, stream states/lengths and padding are included.",
                   "GPU lookup tables, scratch, caches and allocator rounding are not measured by this CPU screen.",
                   "A dictionary's vector coverage is not a GPU cache-hit rate or evidence of temporal reuse.",
                   "Checkpoint windows are not the native kernel layout; --layout runtime requires a real runtime dump.",
                   "No GPU decoder, serving switch, new quantization or inference quality result is provided."],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"{args.kind}: {len(raw):,} sampled bytes of {source['tensor_bytes']:,}; {args.layout} order")
    print("Candidate             Stored bytes   Saving   Exact")
    for case in sorted(result["candidates"], key=lambda c: c["container_bytes"]):
        print(f"{case['id']:<22}{case['container_bytes']:>12,} {case['saving_pct']:>7.2f}%   yes")
    print(f"Goal: {policy.goal}; slowdown limit {policy.max_slowdown_pct:g}%; minimum saving {policy.min_saving_pct:g}%.")
    print(f"Runtime recommendation: {decision['recommended_codec']} ({decision['status']}).")
    if evidence is None:
        print("GPU slowdown and quality effects: unmeasured. Policy limits are not predictions.")
    else:
        print(f"Measured scope: {decision['scope']}; {decision['hardware']}; {decision['workload']}.")
        print("Candidate             Resident saving   Median slowdown   P95 slowdown   Passes gates")
        for measured in decision["measurements"]:
            print(f"{measured['id']:<22}{measured['saving_pct']:>14.2f}%"
                  f"{measured['median_slowdown_pct']:>17.2f}%{measured['p95_slowdown_pct']:>14.2f}%"
                  f"   {'yes' if measured['within_limits'] else 'no'}")
        print("Sample percentiles apply to this workload; they are not confidence bounds or other-workload forecasts.")
    if bandwidths:
        best = next(c for c in result["candidates"] if c["id"] == decision["size_winner"])
        if "performance_hint" in best:
            hint = best["performance_hint"]["ideal_saved_read_us_per_input_mib"]
            print(f"Best size candidate's ideal read-time budget: {hint['low']:.3f}..{hint['high']:.3f} us/input MiB.")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
