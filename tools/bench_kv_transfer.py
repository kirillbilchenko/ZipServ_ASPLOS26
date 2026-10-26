# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Two-rank NCCL transfer screen, including SplitZip encode/framing/decode.

This measures a serial point-to-point path with a receiver-ready acknowledgement.
It is NOT an integrated disaggregated serving benchmark or compressed all-reduce.
Use a separately reviewed upstream SplitZip artifact and real captured BF16 data.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
import torch.distributed as dist


def set_table(codec, values):
    codec.enc_lut = torch.zeros(256, dtype=torch.uint8, device="cuda")
    codec.enc_lut_marked = torch.full((256,), 16, dtype=torch.uint8, device="cuda")
    codec.common_lut = torch.zeros(256, dtype=torch.uint8, device="cuda")
    codec.dec_lut = torch.tensor(values, dtype=torch.uint8, device="cuda")
    for code, exponent in enumerate(values):
        codec.enc_lut[exponent] = code
        codec.enc_lut_marked[exponent] = code
        codec.common_lut[exponent] = 1


def pack_wire(codec, encoded):
    return torch.cat([codec.dec_lut, encoded.pk, encoded.sm,
                      encoded.chunk_id.view(torch.uint8), encoded.local_pos.view(torch.uint8), encoded.esc_val])


def unpack_wire(module, codec, body, n, n_esc):
    expected = 16 + (n + 1) // 2 + n + 7 * n_esc
    if body.numel() != expected:
        raise ValueError("Inconsistent received payload")
    pos = 0
    parts = []
    # Torch views require aligned offsets; receive into independent typed tensors
    # when needed. The copies, if any, are charged to receive/decode time.
    for count, dtype in ((16, torch.uint8), ((n + 1) // 2, torch.uint8),
                         (n, torch.uint8), (n_esc * 4, torch.int32),
                         (n_esc * 2, torch.uint16), (n_esc, torch.uint8)):
        part = body[pos:pos + count]
        if pos % {torch.uint8: 1, torch.uint16: 2, torch.int32: 4}[dtype]:
            part = part.clone()
        parts.append(part.view(dtype))
        pos += count
    table, pk, sm, chunks, offsets, escapes = parts
    codec.dec_lut = table.clone()  # Receive storage will be reused by the next arm.
    scratch = torch.empty(0, device="cuda", dtype=torch.int32)
    return module.ChunkLocalGPUEncoded(pk, sm, scratch, scratch, chunks, offsets, escapes, n, n_esc, 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--init-method", required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=15)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    raw_bytes = args.sample.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != metadata["sha256"]:
        raise ValueError("Sample checksum mismatch")
    spec = importlib.util.spec_from_file_location("splitzip_reference", args.reference / "codec_gpu.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.cuda.set_device(0)
    codec = module.ChunkLocalSplitZipGPU(device="cuda", chunk_size=1024)
    set_table(codec, metadata["calibration_table"])
    sample = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.int16).copy()).view(torch.bfloat16).cuda()
    receive_buffer = torch.empty(len(raw_bytes) + 64, dtype=torch.uint8, device="cuda")
    recv_header = torch.empty(4, dtype=torch.int64, device="cuda")
    ack = torch.zeros(1, dtype=torch.int32, device="cuda")
    dist.init_process_group("nccl", init_method=args.init_method, rank=args.rank, world_size=2,
                            timeout=timedelta(seconds=120), device_id=torch.device("cuda", 0))
    results = []
    for size in (32 << 10, 2 << 20, 16 << 20):
        if size > len(raw_bytes):
            continue
        raw = sample[:size // 2]
        # Compile/allocate before timed runs on both peers; keep tested bytes fixed.
        initial = codec.encode(raw)
        wire_bytes = initial.compressed_bytes + 16 + 32
        assert torch.equal(codec.decode(initial).view(torch.int16), raw.view(torch.int16))
        del initial
        for sender in (0, 1):
            timings = {"raw": [], "compressed": []}
            for iteration in range(args.rounds + 3):
                order = ("raw", "compressed") if iteration % 2 == 0 else ("compressed", "raw")
                for arm in order:
                    dist.barrier()
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    if args.rank == sender:
                        if arm == "compressed":
                            encoded = codec.encode(raw)
                            body = pack_wire(codec, encoded)
                            if body.numel() >= raw.numel() * 2:
                                raise RuntimeError("Screen expects a beneficial BF16 frame; raw fallback required")
                            fields = [raw.numel(), encoded.n_esc, body.numel(), 1]
                        else:
                            body = raw.view(torch.uint8)
                            fields = [raw.numel(), 0, body.numel(), 0]
                        header = torch.tensor(fields, dtype=torch.int64, device="cuda")
                        dist.send(header, dst=1 - sender)
                        dist.send(body, dst=1 - sender)
                        dist.recv(ack, src=1 - sender)
                        torch.cuda.synchronize()
                        elapsed_ms = (time.perf_counter() - begin) * 1000
                        if iteration >= 3:
                            timings[arm].append(elapsed_ms)
                        if arm == "compressed":
                            del encoded
                        del body, header
                    else:
                        dist.recv(recv_header, src=sender)
                        n, n_esc, length, encoded_flag = recv_header.cpu().tolist()
                        if n != raw.numel() or not 0 <= n_esc <= n or not 0 < length <= receive_buffer.numel():
                            raise ValueError("Invalid incoming frame")
                        body = receive_buffer[:length]
                        dist.recv(body, src=sender)
                        if encoded_flag:
                            received = unpack_wire(module, codec, body, n, n_esc)
                            decoded = codec.decode(received)
                        else:
                            decoded = body.view(torch.bfloat16)
                        torch.cuda.synchronize()
                        dist.send(ack, dst=sender)
                        # Correctness check after acknowledgement, outside timed path.
                        if not torch.equal(decoded.view(torch.int16), raw.view(torch.int16)):
                            raise AssertionError("Received KV differs bit-for-bit")
                        if encoded_flag:
                            del received
                        del body, decoded
            if args.rank == sender:
                ratios = [c / r for r, c in zip(timings["raw"], timings["compressed"])]
                row = {"sender_rank": sender, "raw_bytes": size, "compressed_wire_bytes": wire_bytes,
                       "raw_median_ms": statistics.median(timings["raw"]),
                       "compressed_median_ms": statistics.median(timings["compressed"]),
                       "paired_ratio_median": statistics.median(ratios),
                       "samples_ms": timings, "bit_exact": True}
                results.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
    report = {"schema": 1, "rank": args.rank, "sample": metadata,
              "upstream_commit": (args.reference / "revision.txt").read_text().strip(),
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "scope": "Serial encode + frame + NCCL point-to-point + decode + receiver-ready acknowledgement; no overlap or serving integration",
              "paired_rotating_order": True, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
