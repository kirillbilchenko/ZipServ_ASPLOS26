# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Paired serial/preallocated/pipelined raw and SplitZip RDMA paths.

The source and complete reconstructed destination remain allocated. Scratch is
bounded by two staging buffers. Encoder count synchronization, metadata transfer,
host work, pipeline fill/drain and receiver-ready acknowledgement are timed.
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

from bench_kv_transfer import pack_wire, set_table
from kv_transfer_buffers import BatchCounts, Stage, decode_into, encode_into, encode_precounted, validate_header


def load_reference(path):
    if (path / "revision.txt").read_text().strip() != "42c743118a29db3270ae6a921a4a4eecb20cad04":
        raise ValueError("Unexpected upstream revision")
    spec = importlib.util.spec_from_file_location("splitzip_reference", path / "codec_gpu.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gpu_checks(module, codec, sample):
    checked = []
    # Exercise raw fallback with every BF16 word, and the coded decoder against
    # the unmodified upstream encoder (independent encoder/decoder cross-check).
    exhaustive = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    original = codec.encode(exhaustive)
    wire = pack_wire(codec, original)
    output = torch.empty_like(exhaustive)
    decode_into(module, wire, [65536, original.n_esc, wire.numel(), 1], output)
    if not torch.equal(output.view(torch.int16), exhaustive.view(torch.int16)):
        raise AssertionError("Preallocated decoder failed exhaustive BF16 check")
    stage = Stage(codec, 65536)
    for label, source in (("all_bf16_words", exhaustive), ("real_kv", sample[:65536]),
                          ("all_bf16_words_again", exhaustive), ("real_kv_again", sample[65536:131072])):
        _, body, fields = encode_into(module, codec, source, stage)
        decode_into(module, body, fields, output)
        if not torch.equal(output.view(torch.int16), source.view(torch.int16)):
            raise AssertionError("Reused stage failed bitwise reconstruction")
        checked.append({"case": label, "coded": bool(fields[3]), "escapes": fields[1]})
    counts = BatchCounts(131072, 65536)
    for label, source in (("batch_real", sample[:131072]),
                          ("batch_mixed", torch.cat([exhaustive, sample[:65536]]))):
        rows = counts.prepare(module, codec, source, stage.wire.numel())
        for i, fields in enumerate(rows):
            raw = source[i * 65536:(i + 1) * 65536]
            body = encode_precounted(module, codec, raw, stage, counts.starts[i], fields)
            decode_into(module, body, fields, output)
            if not torch.equal(output.view(torch.int16), raw.view(torch.int16)):
                raise AssertionError("Batched counts/scatter disagrees with original bytes")
        checked.append({"case": label, "coded_frames": sum(row[3] for row in rows)})
    torch.cuda.synchronize()
    return checked


class Pipeline:
    def __init__(self, module, codec, sample, chunk_bytes, control_group=None):
        if sample.numel() * 2 % chunk_bytes or chunk_bytes % 16:
            raise ValueError("Sample must divide into aligned equal chunks")
        self.module, self.codec, self.sample = module, codec, sample
        self.control_group = control_group
        self.chunk_words = chunk_bytes // 2
        self.chunks = sample.numel() // self.chunk_words
        self.stages = [Stage(codec, self.chunk_words) for _ in range(2)]
        self.batch = BatchCounts(sample.numel(), self.chunk_words)
        self.output = torch.empty_like(sample)
        self.encode_stream = torch.cuda.Stream()
        self.net_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        self.ack = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.whole_header = torch.tensor([sample.numel(), 0, sample.numel() * 2, 0], dtype=torch.int64, device="cuda")
        self.whole_recv_header = torch.empty_like(self.whole_header)
        self.cached_bodies = []
        self.cache_build_ms = []
        self.cache_device_bytes = 0
        torch.cuda.synchronize()

    def prepare_cache(self):
        """Cache immutable encoded bytes; charge preparation separately from hits.

        This is a repeated-transfer experiment, not a vLLM prefix-cache plugin.
        Callers must invalidate the cache when the original source bits change.
        """
        self.cached_bodies = []
        self.cached_header = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.cuda.stream(self.encode_stream):
            self.cached_rows = self.batch.prepare(self.module, self.codec, self.sample, self.stages[0].wire.numel())
            self.cached_header = self.batch.header.clone()
            for i, fields in enumerate(self.cached_rows):
                raw = self.sample[i * self.chunk_words:(i + 1) * self.chunk_words]
                body = encode_precounted(self.module, self.codec, raw, self.stages[0], self.batch.starts[i], fields)
                # Raw fallbacks can reference the immutable source. Encoded
                # bodies must own their storage before this stage is reused.
                self.cached_bodies.append(body.clone() if fields[3] else body)
        self.encode_stream.synchronize()
        self.cache_build_ms.append((time.perf_counter() - started) * 1000)
        self.cache_device_bytes = self.cached_header.numel() * self.cached_header.element_size() + sum(
            body.numel() for body, fields in zip(self.cached_bodies, self.cached_rows) if fields[3])

    def run(self, rank, sender, arm):
        peer = 1 - sender
        works = []
        wire_bytes = 0
        fallback_chunks = 0
        is_raw = arm in ("raw_whole", "raw_direct", "raw_chunked", "raw_batched")
        hostmeta = arm == "compressed_hostmeta"
        cached = arm == "compressed_cached"
        batched = arm in ("raw_batched", "compressed_batched", "compressed_hostmeta", "compressed_cached")
        if hostmeta and self.control_group is None:
            raise ValueError("Host metadata needs a pre-established CPU control group")
        if cached and not self.cached_bodies:
            raise ValueError("Encoded cache must be built and charged before measuring hits")
        serial = arm == "compressed_serial"
        if rank != sender:
            # Every destination bit starts wrong, so a missing write cannot pass
            # by reusing the previous trial's identical reconstructed sample.
            torch.bitwise_not(self.sample.view(torch.int16), out=self.output.view(torch.int16))
        dist.barrier()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if arm in ("raw_whole", "raw_direct"):
            if rank == sender:
                if arm == "raw_whole":
                    dist.send(self.whole_header, dst=peer)
                dist.send(self.sample.view(torch.uint8), dst=peer)
                dist.recv(self.ack, src=peer)
            else:
                if arm == "raw_whole":
                    dist.recv(self.whole_recv_header, src=sender)
                    validate_header(self.whole_recv_header.cpu().tolist(), self.sample.numel(), self.sample.numel() * 2)
                dist.recv(self.output.view(torch.uint8), src=sender)
                torch.cuda.synchronize()
                dist.send(self.ack, dst=sender)
            wire_bytes = self.sample.numel() * 2 + (32 if arm == "raw_whole" else 0)
        elif rank == sender:
            if batched:
                with torch.cuda.stream(self.encode_stream):
                    if cached:
                        rows, batch_header = self.cached_rows, self.cached_header
                    elif is_raw:
                        rows = [[self.chunk_words, 0, self.chunk_words * 2, 0]] * self.chunks
                        batch_header = self.batch.raw_header
                    else:
                        rows = self.batch.prepare(self.module, self.codec, self.sample, self.stages[0].wire.numel(),
                                                  device_header=not hostmeta)
                        batch_header = self.batch.header
                    self.stages[0].ready.record()
                if hostmeta:
                    # Counts are already on the CPU. Do not send them back to
                    # the GPU just to receive and read them on the peer's CPU.
                    dist.send(self.batch.host, dst=peer, group=self.control_group)
                else:
                    with torch.cuda.stream(self.net_stream):
                        self.net_stream.wait_event(self.stages[0].ready)
                        work = dist.isend(batch_header, dst=peer)
                        work.block_current_stream()
                        works.append(work)
            for i in range(self.chunks):
                stage = self.stages[i % 2]
                raw = self.sample[i * self.chunk_words:(i + 1) * self.chunk_words]
                with torch.cuda.stream(self.encode_stream):
                    if i >= 2 and not cached:
                        self.encode_stream.wait_event(stage.sent)
                    if batched:
                        fields = rows[i]
                        body = (self.cached_bodies[i] if cached else raw.view(torch.uint8) if is_raw else
                                encode_precounted(self.module, self.codec, raw, stage, self.batch.starts[i], fields))
                        fallback_chunks += int(not is_raw and not fields[3])
                    elif is_raw:
                        header, body = stage.raw_header, raw.view(torch.uint8)
                        fields = (raw.numel(), 0, body.numel(), 0)
                    else:
                        header, body, fields = encode_into(self.module, self.codec, raw, stage)
                        fallback_chunks += 1 - fields[3]
                    stage.ready.record()
                with torch.cuda.stream(self.net_stream):
                    self.net_stream.wait_event(stage.ready)
                    for tensor in ((body,) if batched else (header, body)):
                        work = dist.isend(tensor, dst=peer)
                        work.block_current_stream()
                        works.append(work)
                    stage.sent.record()
                wire_bytes += 32 + fields[2]
                if serial:
                    stage.sent.synchronize()
            with torch.cuda.stream(self.net_stream):
                done = dist.irecv(self.ack, src=peer)
                done.block_current_stream()
                works.append(done)
            self.net_stream.synchronize()
        else:
            if batched:
                if hostmeta:
                    dist.recv(self.batch.host, src=sender, group=self.control_group)
                    rows = [validate_header(row, self.chunk_words, self.stages[0].wire.numel())
                            for row in self.batch.host.tolist()]
                else:
                    with torch.cuda.stream(self.net_stream):
                        work = dist.irecv(self.batch.header, src=sender)
                        work.block_current_stream()
                        works.append(work)
                        rows = [validate_header(row, self.chunk_words, self.stages[0].wire.numel())
                                for row in self.batch.header.cpu().tolist()]
            for i in range(self.chunks):
                stage = self.stages[i % 2]
                destination = self.output[i * self.chunk_words:(i + 1) * self.chunk_words]
                with torch.cuda.stream(self.net_stream):
                    if batched:
                        fields = rows[i]
                    else:
                        header = dist.irecv(stage.header, src=sender)
                        header.block_current_stream()
                        works.append(header)
                        fields = validate_header(stage.header.cpu().tolist(), self.chunk_words, stage.wire.numel())
                    n, escapes, length, coded = fields
                    if i >= 2:
                        self.net_stream.wait_event(stage.decoded)
                    body = stage.wire[:length] if coded else destination.view(torch.uint8)
                    incoming = dist.irecv(body, src=sender)
                    incoming.block_current_stream()
                    works.append(incoming)
                    stage.received.record()
                with torch.cuda.stream(self.decode_stream):
                    self.decode_stream.wait_event(stage.received)
                    if coded:
                        decode_into(self.module, body, fields, destination)
                    stage.decoded.record()
                wire_bytes += 32 + length
                fallback_chunks += (not coded and not is_raw)
                if serial:
                    stage.decoded.synchronize()
            self.decode_stream.synchronize()
            dist.send(self.ack, dst=sender)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        extra = torch.cuda.max_memory_allocated() - before
        # Work failures and bit correctness are checked outside the measured path.
        for work in works:
            work.wait()
        if rank != sender and not torch.equal(self.output.view(torch.int16), self.sample.view(torch.int16)):
            raise AssertionError("Pipelined destination differs bit-for-bit")
        return {"ms": elapsed, "wire_bytes": wire_bytes, "fallback_chunks": int(fallback_chunks),
                "peak_added_bytes": extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--init-method")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks-mib", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--sample-mib", type=int, help="Use this many MiB from the pinned sample prefix")
    parser.add_argument("--arms", nargs="+", default=["raw_whole", "raw_direct", "raw_batched", "compressed_pipeline", "compressed_batched"],
                        choices=["raw_whole", "raw_direct", "raw_chunked", "raw_batched", "compressed_serial", "compressed_pipeline", "compressed_batched", "compressed_hostmeta", "compressed_cached"])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if "raw_whole" not in args.arms or len(set(args.arms)) != len(args.arms):
        parser.error("Need one raw_whole reference and unique arms")
    metadata = json.loads(args.metadata.read_text())
    raw = args.sample.read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("Sample checksum mismatch")
    if args.sample_mib is not None:
        if not 1 <= args.sample_mib <= len(raw) // (1 << 20):
            parser.error("sample-mib exceeds the pinned sample")
        raw = raw[:args.sample_mib << 20]
    module = load_reference(args.reference)
    codec = module.ChunkLocalSplitZipGPU(device="cuda", chunk_size=1024)
    set_table(codec, metadata["calibration_table"])
    sample = torch.from_numpy(np.frombuffer(raw, dtype=np.int16).copy()).view(torch.bfloat16).cuda()
    checked = gpu_checks(module, codec, sample)
    if args.self_test:
        exhaustive = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
        source = torch.cat([exhaustive, sample[:65536]])
        pipe = Pipeline(module, codec, source, 65536 * 2)
        pipe.prepare_cache()
        torch.bitwise_not(source.view(torch.int16), out=pipe.output.view(torch.int16))
        for i, (fields, body) in enumerate(zip(pipe.cached_rows, pipe.cached_bodies)):
            decode_into(module, body, fields, pipe.output[i * 65536:(i + 1) * 65536])
        if not torch.equal(source.view(torch.int16), pipe.output.view(torch.int16)):
            raise AssertionError("Cached coded/raw frames did not reconstruct every bit")
        checked.append({"case": "cached_mixed", "coded_frames": sum(row[3] for row in pipe.cached_rows),
                        "cache_device_bytes": pipe.cache_device_bytes})
        print(json.dumps({"gpu_checks": checked, "bit_exact": True}), flush=True)
        return
    if not args.init_method:
        parser.error("--init-method required for two-rank test")
    dist.init_process_group("nccl", init_method=args.init_method, rank=args.rank, world_size=2,
                            timeout=timedelta(seconds=120), device_id=torch.device("cuda", 0))
    control_group = (dist.new_group(ranks=[0, 1], backend="gloo", timeout=timedelta(seconds=120))
                     if "compressed_hostmeta" in args.arms else None)
    results = []
    if args.profile:
        torch.cuda.profiler.start()
    for mib in args.chunks_mib:
        pipe = Pipeline(module, codec, sample, mib << 20, control_group)
        if "compressed_cached" in args.arms:
            # Retain first-build cost and three warm-allocator rebuilds. Do not
            # mix preparation with the explicitly labelled cache-hit timings.
            for _ in range(4):
                pipe.prepare_cache()
        for sender in (0, 1):
            names = args.arms
            timings = {name: [] for name in names}
            for i in range(args.rounds + 3):
                order = names[i % len(names):] + names[:i % len(names)]
                for name in order:
                    with torch.cuda.nvtx.range(f"{name}/chunk{mib}/sender{sender}/round{i}"):
                        value = pipe.run(args.rank, sender, name)
                    if i >= 3:
                        timings[name].append(value)
            if args.rank == sender:
                medians = {name: statistics.median(v["ms"] for v in samples) for name, samples in timings.items()}
                ratios = {name: statistics.median(c["ms"] / r["ms"] for c, r in zip(samples, timings["raw_whole"]))
                          for name, samples in timings.items()}
                row = {"sender": sender, "raw_bytes": len(raw), "chunk_mib": mib,
                       "stage_device_bytes": sum(s.device_bytes for s in pipe.stages),
                       "batch_count_device_bytes": pipe.batch.device_bytes,
                       "output_bytes": pipe.output.numel() * 2,
                       "cache_build_ms": pipe.cache_build_ms,
                       "cache_device_bytes": pipe.cache_device_bytes,
                       "median_ms": medians, "paired_ratio_vs_raw_whole": ratios,
                       "samples": timings, "bit_exact": True}
                results.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
        del pipe
    if args.profile:
        torch.cuda.profiler.stop()
    report = {"schema": 1, "rank": args.rank, "torch": torch.__version__,
              "nccl": torch.cuda.nccl.version(), "gpu": torch.cuda.get_device_name(0),
              "profiled": args.profile, "metadata": metadata, "gpu_checks": checked,
              "sample_prefix_bytes": len(raw),
              "destination_poisoned_before_each_trial": True,
              "raw_direct_contract": "Raw size agreed before timing; no redundant size frame; receiver-ready acknowledgement retained",
              "wire_bytes_scope": "Data plus codec/size headers; common four-byte acknowledgement excluded",
              "cache_contract": "Explicit compressed_cached arm only; immutable source; preparation charged separately; extra encoded GPU storage; no eviction or serving integration",
              "scope": "Preallocated, bounded double-buffered point-to-point; full reconstruction and acknowledgement included; no serving integration",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256(Path(__file__).with_name("kv_transfer_buffers.py").read_bytes()).hexdigest(),
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    if control_group is not None:
        dist.destroy_process_group(control_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
