# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
# Adapted from MxWave runtime verification.
"""Bounded GPU verification of the external ZipServ bridge and graph/compile contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from zipserv_vllm.runtime import _linear, compress, decompress, linear


def main() -> None:
    """Verify complete BF16 bit coverage and several layer/batch/Split-K combinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    torch.manual_seed(42)
    report: dict[str, object] = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
    }
    # Includes all NaN payloads, infinities, subnormals, and signed zeros.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16).reshape(256, 256)
    edge = compress(bits.view(torch.bfloat16)).to("cuda")
    report["all_bf16_bits_exact"] = bool(
        torch.equal(decompress(edge).cpu().view(torch.uint16), bits)
    )
    assert report["all_bf16_bits_exact"], "BF16 bitwise round trip failed"
    del edge
    cases: list[dict[str, object]] = []
    # Changing exception densities exercises the old static-shared-memory sizing bug.
    for distribution in ("normal", "wide", "normal_again"):
        weight = torch.randn(128, 512)
        if distribution == "wide":
            weight *= torch.exp2(torch.randint(-15, 15, weight.shape).float())
        weight = weight.to(torch.bfloat16)
        packed = compress(weight).to("cuda")
        exact = torch.equal(
            decompress(packed).cpu().view(torch.uint16), weight.view(torch.uint16)
        )
        assert exact, f"{distribution}: decompression differs"
        dense = weight.cuda()
        for tokens in (0, 1, 3, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 128):
            x = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
            for split_k in (1, 2, 4, 8):
                actual = linear(x, packed, split_k=split_k)
                expected = torch.nn.functional.linear(x, dense)
                relative_l2 = float(
                    (actual.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-30)
                )
                # The wide case is a decoding/shared-memory test; it can overflow BF16 GEMM.
                if distribution != "wide":
                    assert relative_l2 < 0.01, (
                        distribution,
                        tokens,
                        split_k,
                        relative_l2,
                    )
                assert actual.shape == expected.shape
                cases.append(
                    {
                        "distribution": distribution,
                        "tokens": tokens,
                        "split_k": split_k,
                        "relative_l2": relative_l2,
                    }
                )
        x = torch.randn(2, 3, 512, device="cuda", dtype=torch.bfloat16)
        assert linear(x, packed).shape == (2, 3, 128)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            streamed = linear(x, packed)
        torch.cuda.current_stream().wait_stream(stream)
        assert torch.equal(streamed, linear(x, packed))
        if distribution == "normal":
            report["custom_op_contract"] = torch.library.opcheck(
                _linear,
                (
                    x,
                    list(packed.buffers),
                    packed.m,
                    packed.k,
                    packed.max_common,
                    packed.max_rare,
                    packed.start_exp,
                    1,
                ),
                test_utils=("test_schema", "test_faketensor"),
            )
            # Invalid calls must fail before native launch/allocation.
            for data, current, split in (
                (x.float(), packed, 1),
                (x, packed, 3),
                (x, replace(packed, m=2**28 + 64), 1),
            ):
                try:
                    linear(data, current, split_k=split)
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        "Invalid dtype, Split-K, or output indexing was accepted"
                    )
            report["invalid_calls_rejected"] = True
            # Warm first, then capture and replay the allocation and fused decode/GEMM.
            static_x = torch.randn(9, 512, device="cuda", dtype=torch.bfloat16)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    linear(static_x, packed)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_y = linear(static_x, packed)
            graph.replay()
            assert torch.equal(graph_y, linear(static_x, packed))
            report["cuda_graph_pass"] = True
            compiled = torch.compile(
                lambda data, current=packed: linear(data, current),
                backend="eager",
                fullgraph=True,
                dynamic=True,
            )
            for tokens in (3, 17, 65):
                data = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
                assert torch.equal(compiled(data), linear(data, packed))
            report["dynamo_fullgraph_pass"] = True
            report["normal_weight_bytes"] = weight.numel() * 2
            report["normal_packed_bytes"] = packed.storage_bytes
    torch.cuda.synchronize()
    report["cases"] = cases
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "complete"
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))


if __name__ == "__main__":
    main()
