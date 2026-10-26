# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Bounded decoded-scale admission and eviction, independent of Torch and CUDA."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Lookup:
    """A hit/admission in the arena, or a miss using the separate workspace."""

    offset: int | None
    hit: bool
    evicted: tuple[str, ...] = ()


class CachePolicy:
    """First-fit arena with retained admissions or LRU replacement; sizes include padding."""

    def __init__(self, capacity: int, enabled: bool = True, policy: str = "retain") -> None:
        if type(capacity) is not int or capacity < 0 or capacity % 256:
            raise ValueError("Cache capacity must be a nonnegative multiple of 256 bytes")
        if type(enabled) is not bool or policy not in ("retain", "lru"):
            raise ValueError("Require a boolean enable flag and retain/lru policy")
        self.capacity = capacity if enabled else 0
        self.enabled, self.policy = enabled, policy
        self.entries: OrderedDict[str, tuple[int, int, int]] = OrderedDict()
        self.free = [(0, self.capacity)] if self.capacity else []
        self.sizes: dict[str, int] = {}
        self.counters = {"lookups": 0, "hits": 0, "misses": 0, "admissions": 0,
                         "evictions": 0, "bypasses": 0, "decoded_bytes": 0}

    def register(self, key: str, size: int) -> None:
        """Bind an immutable source key to its exact decoded size."""
        if not isinstance(key, str) or not key or key in self.sizes:
            raise ValueError("Cache keys must be unique nonempty strings")
        if type(size) is not int or size <= 0:
            raise ValueError("Decoded size must be positive")
        self.sizes[key] = size

    def _release(self, offset: int, size: int) -> None:
        blocks = sorted([*self.free, (offset, size)])
        self.free = []
        for start, length in blocks:
            if self.free and self.free[-1][0] + self.free[-1][1] == start:
                a, b = self.free[-1]
                self.free[-1] = (a, b + length)
            else:
                self.free.append((start, length))

    def lookup(self, key: str) -> Lookup:
        """Acquire a logical location; the caller serializes GPU uses and completes decoding."""
        size = self.sizes[key]  # Invalid keys must not alter policy or counters.
        self.counters["lookups"] += 1
        if key in self.entries:
            self.entries.move_to_end(key)
            self.counters["hits"] += 1
            return Lookup(self.entries[key][0], True)
        self.counters["misses"] += 1
        self.counters["decoded_bytes"] += size
        needed = (size + 255) // 256 * 256
        evicted = []
        if needed <= self.capacity:
            while True:
                slot = next((i for i, (_, length) in enumerate(self.free) if length >= needed), None)
                if slot is not None:
                    start, length = self.free.pop(slot)
                    if length > needed:
                        self.free.insert(slot, (start + needed, length - needed))
                    self.entries[key] = (start, needed, size)
                    self.counters["admissions"] += 1
                    return Lookup(start, False, tuple(evicted))
                if self.policy != "lru" or not self.entries:
                    break
                victim, (start, length, _) = self.entries.popitem(last=False)
                self._release(start, length)
                evicted.append(victim)
                self.counters["evictions"] += 1
        self.counters["bypasses"] += 1
        return Lookup(None, False, tuple(evicted))

    def clear(self, reset_stats: bool = True) -> None:
        """Invalidate all residents; callers must first finish outstanding GPU consumers."""
        self.entries.clear()
        self.free = [(0, self.capacity)] if self.capacity else []
        if reset_stats:
            self.reset_stats()

    def reset_stats(self) -> None:
        """Reset counters without manufacturing cold misses or losing residents."""
        for name in self.counters:
            self.counters[name] = 0

    def report(self) -> dict:
        """Return capacity, occupied bytes and observed lookups; never infer routing reuse."""
        return {"enabled": self.enabled, "policy": self.policy, **self.counters,
                "capacity_bytes": self.capacity,
                "occupied_bytes": sum(v[1] for v in self.entries.values()),
                "resident_entries": len(self.entries), "registered_entries": len(self.sizes),
                "hit_rate_pct": 100 * self.counters["hits"] / self.counters["lookups"]
                if self.counters["lookups"] else None}
