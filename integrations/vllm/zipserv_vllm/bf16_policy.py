# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""CPU-only planning for the BF16 native indexing and workspace contracts."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isqrt

INT32_MAX = 2**31 - 1
GRID_Y_MAX = 65535


@dataclass(frozen=True)
class DispatchSettings:
    """Opt-in hypotheses for measurement, not a calibrated performance policy."""

    mode: str = "fixed"
    large_batch_tokens: int = 768
    workspace_bytes: int = 64 * 2**20
    partial_dtype: str = "bf16"
    decode_workspace_bytes: int = 256 * 2**20

    def __post_init__(self):
        if self.mode not in ("fixed", "tokens", "shape"):
            raise ValueError("dispatch must be fixed, tokens or shape")
        if self.partial_dtype not in ("bf16", "fp32"):
            raise ValueError("partial_dtype must be bf16 or fp32")
        if type(self.large_batch_tokens) is not int or self.large_batch_tokens < 1:
            raise ValueError("large_batch_tokens must be a positive integer")
        if type(self.workspace_bytes) is not int or self.workspace_bytes < 0:
            raise ValueError("workspace_bytes must be a nonnegative integer")
        if type(self.decode_workspace_bytes) is not int or self.decode_workspace_bytes < 0:
            raise ValueError("decode_workspace_bytes must be a nonnegative integer")


def decode_budget_rejection(m: int, k: int, settings: DispatchSettings) -> str | None:
    if settings.mode != "fixed" and m * k * 2 > settings.decode_workspace_bytes:
        return (f"Decoded BF16 weight requires {m * k * 2} bytes, exceeding the "
                f"decode_workspace_bytes budget of {settings.decode_workspace_bytes}")
    return None


@dataclass(frozen=True)
class SplitCandidate:
    split_k: int
    blocks: int
    workspace_bytes: int
    score: float


def token_tile(tokens: int) -> int:
    return 8 if tokens <= 8 else min(128, (tokens + 15) // 16 * 16)


@lru_cache(maxsize=4096)
def split_candidates(m: int, tokens: int, k: int, sms: int,
                     workspace_bytes: int, partial_bytes: int = 2) -> tuple[SplitCandidate, ...]:
    """Score every admissible divisor, including non-power-of-two splits.

    The score estimates waves of K-tile work plus reduction traffic. Two tile
    units model pipeline startup per wave; reduction traffic is normalized to
    one 64x64 BF16 weight tile per SM. These constants are hypotheses: compare
    the chosen split with the measured best candidate before promoting it.
    """
    if type(sms) is not int or sms < 1 or partial_bytes not in (2, 4):
        raise ValueError("Require positive SM count and 2- or 4-byte partials")
    if type(workspace_bytes) is not int or workspace_bytes < 0:
        raise ValueError("workspace_bytes must be a nonnegative integer")
    if gemm_rejection(m, tokens, k, 1):
        return ()
    tiles = k // 64
    divisors = set()
    for factor in range(1, isqrt(tiles) + 1):
        if tiles % factor == 0:
            divisors.update((factor, tiles // factor))
    tile_n = token_tile(tokens)
    token_blocks = max(1, (tokens + tile_n - 1) // tile_n)
    rows = []
    for split in sorted(divisors):
        scratch = m * tokens * split * partial_bytes if split > 1 else 0
        if scratch > workspace_bytes or gemm_rejection(m, tokens, k, split):
            continue
        blocks = (m // 64) * token_blocks * split
        waves = (blocks + sms - 1) // sms
        core = waves * (tiles // split + 2) * (tile_n / 8)
        reduction = (2 * scratch + m * tokens * 2) / (8192 * sms) if split > 1 else 0
        rows.append(SplitCandidate(split, blocks, scratch, core + reduction))
    return tuple(rows)


@lru_cache(maxsize=4096)
def select_split(m: int, tokens: int, k: int, sms: int,
                 settings: DispatchSettings, fixed_split: int = 1) -> int | None:
    """None requests a temporary decode plus stock GEMM outside graph capture."""
    if settings.mode == "fixed":
        return fixed_split  # The native forward guard remains authoritative.
    if tokens > settings.large_batch_tokens:
        return None
    candidates = split_candidates(m, tokens, k, sms, settings.workspace_bytes,
                                  4 if settings.partial_dtype == "fp32" else 2)
    if not candidates:
        return None
    if settings.mode == "tokens":
        desired = 8 if tokens <= 32 else 4 if tokens <= 128 else 1
        return max(c.split_k for c in candidates if c.split_k <= desired)
    return min(candidates, key=lambda c: (c.score, c.workspace_bytes, c.split_k)).split_k


def gemm_rejection(m: int, tokens: int, k: int, split_k: int) -> str | None:
    """Mirror the native limits using unbounded host arithmetic, before allocating."""
    if any(type(value) is not int for value in (m, tokens, k, split_k)):
        return "GEMM dimensions and Split-K must be integers"
    if min(m, k) <= 0 or m % 64 or k % 64 or tokens < 0:
        return "Require positive 64-aligned weight dimensions and nonnegative tokens"
    if split_k < 1 or (k // 64) % split_k:
        return "Split-K must divide the number of 64-column weight tiles"
    if max(m * tokens * split_k, k * tokens, m * k) > INT32_MAX:
        return "Output, activations or Split-K workspace exceed native int32 indexing"
    if (m // 64) * split_k > GRID_Y_MAX:
        return "Split-K exceeds CUDA grid.y"
    return None


def startup_rejection(m: int, k: int, split_k: int, max_tokens: int | None) -> str | None:
    """Keep a dense layer when its configured maximum cannot use the native path."""
    if type(max_tokens) is not int or max_tokens <= 0:
        return "A positive scheduler max_num_batched_tokens is required for compression planning"
    reason = gemm_rejection(m, max_tokens, k, split_k)
    if reason:
        return reason
    if m * k > INT32_MAX - 4096:
        return "Weight packing exceeds the native int32 offset bound"
    return None
