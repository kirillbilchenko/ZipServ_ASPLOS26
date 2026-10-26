# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Bounded transfer buffers around the separately supplied SplitZip GPU kernels.

Research adapter, not a serving plugin. The kernel artifact must be pinned to
42c743118a29db3270ae6a921a4a4eecb20cad04. Encoding still reads the escape count
back to the CPU; this cost is intentionally included in transfer measurements.
"""
from __future__ import annotations


def validate_header(fields, expected_words: int, capacity_bytes: int):
    if len(fields) != 4 or any(type(v) is not int for v in fields):
        raise ValueError("Header requires four integers")
    n, escapes, length, coded = fields
    if n != expected_words or n < 8 or n % 8 or not 0 <= escapes <= n or coded not in (0, 1):
        raise ValueError("Invalid shape, escape count or mode")
    expected = 16 + n // 2 + n + 7 * escapes if coded else n * 2
    if length != expected or length > capacity_bytes or (not coded and escapes):
        raise ValueError("Invalid payload length or raw header")
    return n, escapes, length, coded


class Stage:
    def __init__(self, codec, words: int):
        import torch
        if words < 32 or words % 8:
            raise ValueError("This transfer adapter requires at least 32 words, in multiples of eight")
        self.words = words
        self.wire = torch.empty(words * 2, dtype=torch.uint8, device="cuda")
        self.wire[:16].copy_(codec.dec_lut)
        self.pk = self.wire[16:16 + words // 2]
        self.sm = self.wire[16 + words // 2:16 + words * 3 // 2]
        self.escape_start = 16 + words * 3 // 2
        self.escape_capacity = (self.wire.numel() - self.escape_start) // 7
        self.counts = torch.empty((words + 1023) // 1024, dtype=torch.int32, device="cuda")
        self.offsets = torch.empty_like(self.counts, dtype=torch.int64)
        self.starts = torch.empty_like(self.offsets)
        self.host_header = torch.empty(4, dtype=torch.int64, pin_memory=True)
        self.host_fields = self.host_header.numpy()
        self.header = torch.empty(4, dtype=torch.int64, device="cuda")
        self.raw_header = torch.tensor([words, 0, words * 2, 0], dtype=torch.int64, device="cuda")
        self.ready = torch.cuda.Event()
        self.sent = torch.cuda.Event()
        self.received = torch.cuda.Event()
        self.decoded = torch.cuda.Event()

    @property
    def device_bytes(self):
        return sum(t.numel() * t.element_size() for t in (
            self.wire, self.counts, self.offsets, self.starts, self.header, self.raw_header))


def encode_into(module, codec, raw, stage: Stage):
    import torch
    n = raw.numel()
    if raw.dtype != torch.bfloat16 or not raw.is_contiguous() or n != stage.words or raw.data_ptr() % 8:
        raise ValueError("Expected aligned contiguous BF16 matching the stage")
    module._enc_4bit_quad64_count2_chunks[((stage.counts.numel() + 1) // 2,)](
        raw.view(torch.int64), codec.enc_lut_marked, stage.pk.view(torch.int16),
        stage.sm.view(torch.int32), stage.counts, n, stage.counts.numel(),
        BLOCK_QUADS=512, num_warps=4)
    torch.cumsum(stage.counts, dim=0, dtype=torch.int64, out=stage.offsets)
    torch.sub(stage.offsets, stage.counts, out=stage.starts)
    escapes = int(stage.offsets[-1].item())
    length = stage.escape_start + 7 * escapes
    if escapes > stage.escape_capacity or length >= raw.numel() * 2:
        # Raw fallback before any scatter can exceed bounded escape storage.
        fields = (n, 0, raw.numel() * 2, 0)
        body = raw.view(torch.uint8)
    else:
        pos = stage.escape_start
        chunks = stage.wire[pos:pos + escapes * 4].view(torch.int32)
        pos += escapes * 4
        offsets = stage.wire[pos:pos + escapes * 2].view(torch.uint16)
        pos += escapes * 2
        values = stage.wire[pos:pos + escapes]
        if escapes:
            module._write_escapes_chunk[(stage.counts.numel(),)](
                raw.view(torch.int16), codec.common_lut, stage.starts, chunks,
                offsets, values, n, CHUNK=1024)
        fields = (n, escapes, length, 1)
        body = stage.wire[:length]
    stage.host_fields[:] = fields
    stage.header.copy_(stage.host_header, non_blocking=True)
    return stage.header, body, fields


def decode_into(module, body, fields, output):
    import torch
    n, escapes, length, coded = validate_header(list(fields), output.numel(), body.numel())
    if output.dtype != torch.bfloat16 or not output.is_contiguous() or output.data_ptr() % 8:
        raise ValueError("Expected aligned contiguous BF16 output")
    if not coded:
        output.view(torch.uint8).copy_(body)
        return
    table = body[:16]
    pk = body[16:16 + n // 2]
    sm = body[16 + n // 2:16 + n * 3 // 2]
    module._dec_4bit_quad64[((n // 4 + 511) // 512,)](
        pk.view(torch.int16), sm.view(torch.int32), table,
        output.view(torch.int64), n // 4, BLOCK=512, num_warps=4)
    if escapes:
        pos = 16 + n * 3 // 2
        chunks = body[pos:pos + escapes * 4].view(torch.int32)
        pos += escapes * 4
        offsets = body[pos:pos + escapes * 2].view(torch.uint16)
        pos += escapes * 2
        values = body[pos:pos + escapes]
        module._fix_escapes_local_linear[((escapes + 127) // 128,)](
            chunks, offsets, values, sm, output.view(torch.int16), escapes,
            CHUNK=1024, BLOCK_ESC=128)


class BatchCounts:
    """One extra input pass and one CPU size read for an entire ready batch."""
    def __init__(self, sample_words: int, chunk_words: int):
        import torch
        if sample_words <= 0 or chunk_words <= 0 or sample_words % chunk_words or chunk_words % 1024:
            raise ValueError("Batch counting requires complete 1024-word chunks")
        self.frames = sample_words // chunk_words
        self.chunk_words = chunk_words
        self.counts = torch.empty(sample_words // 1024, dtype=torch.int32, device="cuda")
        shape = (self.frames, chunk_words // 1024)
        self.offsets = torch.empty(shape, dtype=torch.int64, device="cuda")
        self.starts = torch.empty_like(self.offsets)
        self.host = torch.empty((self.frames, 4), dtype=torch.int64, pin_memory=True)
        self.fields = self.host.numpy()
        self.header = torch.empty_like(self.host, device="cuda")
        self.raw_header = torch.tensor([[chunk_words, 0, chunk_words * 2, 0]] * self.frames,
                                       dtype=torch.int64, device="cuda")

    @property
    def device_bytes(self):
        return sum(t.numel() * t.element_size() for t in (
            self.counts, self.offsets, self.starts, self.header, self.raw_header))

    def prepare(self, module, codec, sample, capacity_bytes, *, device_header=True):
        import torch
        module._count_escapes_chunk[(self.counts.numel(),)](
            sample.view(torch.int16), codec.common_lut, self.counts,
            sample.numel(), CHUNK=1024)
        counts = self.counts.view(self.frames, -1)
        torch.cumsum(counts, dim=1, dtype=torch.int64, out=self.offsets)
        torch.sub(self.offsets, counts, out=self.starts)
        # This is the only GPU -> CPU size synchronization in this path.
        totals = self.offsets[:, -1].cpu().tolist()
        rows = []
        for escapes in totals:
            length = 16 + self.chunk_words * 3 // 2 + 7 * escapes
            if length < capacity_bytes:
                rows.append([self.chunk_words, escapes, length, 1])
            else:
                rows.append([self.chunk_words, 0, self.chunk_words * 2, 0])
        self.fields[:] = rows
        if device_header:
            self.header.copy_(self.host, non_blocking=True)
        return rows


def encode_precounted(module, codec, raw, stage: Stage, starts, fields):
    import torch
    n, escapes, length, coded = validate_header(list(fields), raw.numel(), stage.wire.numel())
    if not coded:
        return raw.view(torch.uint8)
    module._enc_4bit_quad64[((n // 4 + 511) // 512,)](
        raw.view(torch.int64), codec.enc_lut, stage.pk.view(torch.int16),
        stage.sm.view(torch.int32), n // 4, BLOCK=512, num_warps=4)
    if escapes:
        pos = stage.escape_start
        chunks = stage.wire[pos:pos + escapes * 4].view(torch.int32)
        pos += escapes * 4
        offsets = stage.wire[pos:pos + escapes * 2].view(torch.uint16)
        pos += escapes * 2
        values = stage.wire[pos:pos + escapes]
        module._write_escapes_chunk[(stage.counts.numel(),)](
            raw.view(torch.int16), codec.common_lut, starts, chunks, offsets,
            values, n, CHUNK=1024)
    return stage.wire[:length]
