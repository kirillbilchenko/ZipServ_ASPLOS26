# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Optional eager-mode GPU scale cache with a strict arena budget and shared scratch."""

from __future__ import annotations

import ctypes
import hashlib
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .low_precision_codec import CODECS, HEADER, align, decode_container, vector_width
from .scale_cache_policy import CachePolicy


class EncodedScale:
    """An immutable validated GPU dictionary, with no permanently decoded owner."""

    def __init__(self, library, blob: bytes, base: int, device, descriptor=None):
        restored = decode_container(blob)
        _, _, codec_id, streams, _, tile, table_size, size = HEADER.unpack_from(blob)
        codec = CODECS[codec_id]
        if (codec not in ("vector4-warp", "vector4-direct", "vector8-warp", "vector8-direct")
                or streams != 1 or tile not in (1024, 2048) or size % tile):
            raise ValueError("Require supported vectors and complete CUDA decoder tiles")
        if type(base) is not int or not 0 <= base <= 240 or base % 16:
            raise ValueError("Aligned nibble scales require a valid aligned E8M0 base")
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("Use an explicit CUDA device index")
        self.library, self.size = library, size
        self.input = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).to(self.device)
        self.descriptor = (torch.tensor([0, base], dtype=torch.int64, device=self.device)
                           if descriptor is None else descriptor)
        if (self.descriptor.dtype != torch.int64 or self.descriptor.device != self.device
                or not self.descriptor.is_contiguous() or self.descriptor.numel() != 2):
            raise ValueError("Consumer descriptor must contain two contiguous CUDA int64 values")
        if int(self.descriptor[1].item()) != base:
            raise ValueError("Descriptor base does not match the encoded source")
        self.pointer = None
        tiles = size // tile
        meta = 64 + align(table_size, 64)
        data = meta + align((tiles + 1) * 8 + tiles * 5, 64)
        self.parameters = (len(blob), size, meta, data, tile, vector_width(codec), int(codec.endswith("-warp")))
        self.raw_sha256 = hashlib.sha256(restored).hexdigest()
        self.container_sha256 = hashlib.sha256(blob).hexdigest()
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(self.device))

    def decode(self, output: torch.Tensor) -> None:
        """Enqueue exact reconstruction into caller-owned, aligned CUDA byte storage."""
        if (output.device != self.device or output.dtype != torch.uint8 or not output.is_contiguous()
                or output.numel() != self.size or output.data_ptr() % 4):
            raise ValueError("Invalid decoded output buffer")
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.ready)
        self.input.record_stream(stream)
        output.record_stream(stream)
        status = self.library.zipserv_dictionary_decode(
            self.input.data_ptr(), output.data_ptr(), *self.parameters, stream.cuda_stream)
        if status:
            raise RuntimeError(f"Dictionary CUDA launch failed: {status}")

    def bind_pointer(self, pointer: int) -> None:
        """Update the stable native descriptor only when its backing location changes."""
        if pointer != self.pointer:
            self.descriptor[0].fill_(pointer)
            self.pointer = pointer


def load_dictionary_library(path: Path):
    """Load the explicit decoder library without touching the existing GEMM implementation."""
    library = ctypes.CDLL(str(path.resolve()))
    library.zipserv_dictionary_decode.argtypes = (
        [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p])
    library.zipserv_dictionary_decode.restype = ctypes.c_int
    return library


class DecodedScaleCache:
    """Bounded arena plus one scratch buffer; scoped borrows fence all GPU consumers.

    Sources are immutable and registered before first use. Eager execution is
    required: Python cache decisions cannot be replayed by a CUDA graph. A
    single event serializes reuse across streams, including cache hits. Consumers
    must enqueue all reads on the current stream before leaving borrow().
    """

    def __init__(self, capacity_bytes: int, enabled: bool = True, policy: str = "retain",
                 device: str = "cuda:0"):
        self.policy = CachePolicy(capacity_bytes, enabled, policy)
        self.requested_capacity = capacity_bytes
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("Use an explicit CUDA device index")
        self.sources: dict[str, EncodedScale] = {}
        self.arena = self.workspace = None
        self._lock = threading.RLock()
        self._leased = self._faulted = self._sealed = False
        self._last = torch.cuda.Event()
        self._last_recorded = False

    def register(self, key: str, source: EncodedScale, verify: bool = True) -> None:
        """Validate the full GPU reconstruction, retaining only encoded data afterward."""
        with self._lock:
            if self._sealed or source.device != self.device or key in self.sources:
                raise ValueError("Register unique same-device sources before first use")
            self.policy.register(key, source.size)
            if verify:
                output = torch.empty(source.size, dtype=torch.uint8, device=self.device)
                source.decode(output)
                if hashlib.sha256(output.cpu().numpy()).hexdigest() != source.raw_sha256:
                    raise RuntimeError("GPU dictionary source validation failed")
            self.sources[key] = source

    def _check_eager(self):
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError("Cache must be used on its owning CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Dynamic scale caching requires --enforce-eager; CUDA graph capture is unsupported")
        if self._faulted:
            raise RuntimeError("Cache is invalid after a failed consumer; rebuild it before reuse")

    def _seal(self):
        if not self._sealed:
            if not self.sources:
                raise RuntimeError("Cannot use an empty scale cache")
            self.workspace = torch.empty(max(s.size for s in self.sources.values()),
                                         dtype=torch.uint8, device=self.device)
            self.arena = torch.empty(self.policy.capacity, dtype=torch.uint8, device=self.device)
            self._sealed = True

    @contextmanager
    def borrow(self, key: str):
        """Yield a native descriptor, pinning its decoded storage through the queued consumer."""
        with self._lock:
            self._check_eager()
            if self._leased:
                raise RuntimeError("Nested cache borrows are unsupported")
            source = self.sources[key]
            self._seal()
            stream = torch.cuda.current_stream(self.device)
            if self._last_recorded:
                stream.wait_event(self._last)
            lookup = self.policy.lookup(key)
            output = (self.workspace[:source.size] if lookup.offset is None
                      else self.arena[lookup.offset:lookup.offset + source.size])
            output.record_stream(stream)
            source.descriptor.record_stream(stream)
            self._leased = True
            try:
                if not lookup.hit:
                    source.decode(output)
                source.bind_pointer(output.data_ptr())
                yield source.descriptor
            except BaseException:
                self._faulted = True
                raise
            finally:
                self._last.record(stream)
                self._last_recorded = True
                self._leased = False

    def configure(self, enabled: bool, capacity_bytes: int | None = None,
                  policy: str | None = None) -> dict:
        """Switch on/off at an idle worker boundary, finish reads, and invalidate residents."""
        with self._lock:
            self._check_eager()
            if self._leased:
                raise RuntimeError("Cannot reconfigure an active borrow")
            capacity = self.requested_capacity if capacity_bytes is None else capacity_bytes
            replacement = CachePolicy(capacity, enabled, policy or self.policy.policy)
            for key, source in self.sources.items():
                replacement.register(key, source.size)
            if self._last_recorded:
                self._last.synchronize()
            self.policy, self.requested_capacity = replacement, capacity
            self.arena = None  # Release the old arena before allocating the replacement.
            if self._sealed:
                try:
                    self.arena = torch.empty(replacement.capacity, dtype=torch.uint8, device=self.device)
                except BaseException:
                    self._faulted = True
                    raise
            for source in self.sources.values():
                source.pointer = None
            return self.report()

    def reset_stats(self) -> None:
        """Reset host lookup counters without evicting cached values."""
        with self._lock:
            if self._leased:
                raise RuntimeError("Cannot reset an active borrow")
            self.policy.reset_stats()

    def report(self) -> dict:
        """Count compressed backing, arena, scratch and descriptors separately."""
        with self._lock:
            encoded = sum(s.input.numel() for s in self.sources.values())
            descriptors = sum(s.descriptor.numel() * s.descriptor.element_size()
                              for s in self.sources.values())
            arena = self.arena.numel() if self.arena is not None else 0
            workspace = self.workspace.numel() if self.workspace is not None else 0
            return {**self.policy.report(), "granularity": "whole_scale_tensor", "cuda_graphs": False,
                    "compressed_bytes": encoded, "decoded_cache_bytes": arena,
                    "decode_workspace_bytes": workspace, "descriptor_bytes": descriptors,
                    "retained_payload_bytes": encoded + arena + workspace + descriptors,
                    "uncached_native_scale_bytes": sum(s.size for s in self.sources.values()) + descriptors,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(self.device),
                    "faulted": self._faulted}
