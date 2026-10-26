"""One NVTX-marked FP8 operation for external Nsight memory-traffic measurement.

Profile each arm in a fresh process with the same source and pressure write.
No timing result from profiler instrumentation is substituted for the paired
unprofiled benchmark. Use the profiler's application replay if multiple passes
are required, so its independent kernel replay does not alter inter-kernel reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from bench_fp8_palette import Encoded, digest, gemv, load_library, source_fixture, tile_order


def main():
    """Warm an arm, pressure the cache, then mark exactly one measured invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--arm", choices=("stock", "separate", "fused", "raw_fused",
                                         "fused_tiled", "raw_tiled", "fused_range16", "fused_range32"),
                        required=True)
    args = parser.parse_args()
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as marlin
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(174)
    torch.cuda.set_per_process_memory_fraction(768 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    weight, scale, provenance = source_fixture(args.source, 0, 0, "gate_up")
    n, k = weight.shape
    layer = torch.nn.Module()
    layer.input_size_per_partition, layer.output_size_per_partition = k, n
    layer.register_parameter("weight", torch.nn.Parameter(weight.cuda(), requires_grad=False))
    layer.register_parameter("weight_scale", torch.nn.Parameter(scale.cuda(), requires_grad=False))
    marlin.prepare_mxfp8_layer_for_marlin(layer)
    x = torch.randn((8, k), dtype=torch.bfloat16, device="cuda") * .125
    lib = load_library(args.library)
    expected = marlin.apply_mxfp8_marlin_linear(x, layer.weight, layer.weight_scale, None, n, k)
    outputs = []
    if args.arm == "stock":
        def run():
            outputs[:] = [marlin.apply_mxfp8_marlin_linear(x, layer.weight, layer.weight_scale, None, n, k)]
    elif args.arm == "separate":
        encoded = Encoded(lib, layer.weight.view(torch.uint8).cpu().numpy().tobytes(), 3)
        decoded = torch.empty_like(layer.weight)
        decode = encoded.bind(decoded.view(torch.uint8).reshape(-1))
        def run():
            decode()
            outputs[:] = [marlin.apply_mxfp8_marlin_linear(x, decoded, layer.weight_scale, None, n, k)]
    else:
        scales = scale.cuda()
        codes = weight.cuda().view(torch.uint8)
        output = torch.empty_like(expected)
        tiled = args.arm.endswith("_tiled") or args.arm.startswith("fused_range")
        codec = int(args.arm.removeprefix("fused_range")) if args.arm.startswith("fused_range") else 3
        raw = weight.view(torch.uint8).numpy().tobytes()
        if tiled:
            import numpy as np
            raw = tile_order(raw, n, k)
            codes = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).cuda()
        if args.arm.startswith("fused"):
            encoded = Encoded(lib, raw, codec)
            run = gemv(lib, encoded.data, scales, x, output, codec, tensor_cores=True, tiled=tiled)
        else:
            run = gemv(lib, codes, scales, x, output, tensor_cores=True, tiled=tiled)
        outputs[:] = [output]
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    error = float((outputs[0].float() - expected.float()).norm() / expected.float().norm())
    if not bool(torch.isfinite(outputs[0]).all()) or error > .01:
        raise RuntimeError("Profiler fixture output failed the component check")
    pressure = torch.zeros(128 * 2**20, dtype=torch.uint8, device="cuda")
    pressure.add_(1)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("zipserv_fp8_traffic")
    run()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(json.dumps({"arm": args.arm, "shape_mnk": [8, n, k], "source": provenance,
        "runtime_weight_sha256": digest(layer.weight), "input_sha256": digest(x),
        "relative_l2_to_stock": error, "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "pressure_bytes": pressure.numel(), "pressure_is_guaranteed_eviction": False,
        "encoded_bytes": len(encoded.data) if args.arm.startswith("fused") or args.arm == "separate"
        else None}), flush=True)


if __name__ == "__main__":
    main()
