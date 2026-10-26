"""Select compression experiments using explicit savings/latency budgets.

CPU compression ratios never qualify a GPU runtime. Timing evidence must bind
the exact input, encoded bytes, layout, hardware and workload being compared.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics

DEFAULTS = {"speed": (0.0, 0.0), "balanced": (3.0, 5.0), "memory": (10.0, 1.0)}


@dataclass(frozen=True)
class Policy:
    """User limits; the defaults are policy choices, not predicted slowdowns."""

    goal: str = "balanced"
    max_slowdown_pct: float | None = None
    min_saving_pct: float | None = None

    def __post_init__(self) -> None:
        if self.goal not in DEFAULTS:
            raise ValueError("Unknown compression goal")
        for field, default in zip(("max_slowdown_pct", "min_saving_pct"), DEFAULTS[self.goal]):
            value = getattr(self, field)
            object.__setattr__(self, field, default if value is None else value)
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Policy limits must be finite and nonnegative")
        if self.min_saving_pct >= 100:
            raise ValueError("Minimum saving must be below 100 percent")


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated sample percentile, not a confidence bound."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def validate_samples(record: dict) -> list[float]:
    """Require repeated positive latency measurements and actual resident bytes."""
    samples = record.get("samples_us")
    resident = record.get("resident_bytes")
    if (not isinstance(samples, list) or len(samples) < 5
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or x <= 0 for x in samples)
            or type(resident) is not int or resident <= 0):
        raise ValueError("Timing records need >=5 positive samples_us and positive resident_bytes")
    return samples


def select(candidates: list[dict], policy: Policy, evidence: dict | None,
           sample_sha256: str, layout: str) -> dict:
    """Apply budgets to matched measurements; keep raw when evidence is absent."""
    result = {
        "policy": asdict(policy), "limits_are_predictions": False,
        "recommended_codec": "raw", "status": "gpu_measurements_required",
        "size_winner": min(candidates, key=lambda c: c["container_bytes"])["id"],
        "measurements": [], "pareto_candidates": [],
    }
    if evidence is None:
        return result
    if (evidence.get("schema_version") != 1 or evidence.get("sample_sha256") != sample_sha256
            or evidence.get("layout") != layout
            or evidence.get("scope") not in ("kernel", "end_to_end")
            or not isinstance(evidence.get("hardware"), str) or not evidence["hardware"].strip()
            or not isinstance(evidence.get("workload"), str) or not evidence["workload"].strip()):
        raise ValueError("Timing evidence must match sample/layout and declare hardware/workload/scope")
    baseline = evidence["baseline"]
    baseline_samples = validate_samples(baseline)
    supplied = evidence.get("candidates", {})
    known = {c["id"]: c for c in candidates if c["id"] != "raw"}
    if not isinstance(supplied, dict) or set(supplied) - set(known):
        raise ValueError("Timing evidence contains unknown candidates")
    for name, record in supplied.items():
        samples = validate_samples(record)
        if (record.get("container_sha256") != known[name]["container_sha256"]
                or len(samples) != len(baseline_samples)):
            raise ValueError("Timing evidence must bind encoded bytes and paired sample counts")
        ratios = [100 * (c / b - 1) for b, c in zip(baseline_samples, samples)]
        saving = 100 * (1 - record["resident_bytes"] / baseline["resident_bytes"])
        row = {
            "id": name, "median_slowdown_pct": statistics.median(ratios),
            "p95_slowdown_pct": percentile(ratios, .95),
            "p10_slowdown_pct": percentile(ratios, .10),
            "baseline_median_us": statistics.median(baseline_samples),
            "candidate_median_us": statistics.median(samples),
            "resident_bytes": record["resident_bytes"], "saving_pct": saving,
            "exact_gpu_roundtrip": record.get("exact_gpu_roundtrip") is True,
            "output_check_passed": record.get("output_check_passed") is True,
        }
        row["within_limits"] = (row["exact_gpu_roundtrip"] and row["output_check_passed"]
                                and saving >= policy.min_saving_pct
                                and row["p95_slowdown_pct"] <= policy.max_slowdown_pct)
        result["measurements"].append(row)
    raw = {"id": "raw", "p95_slowdown_pct": 0.0, "resident_bytes": baseline["resident_bytes"]}
    points = [raw] + [m for m in result["measurements"]
                     if m["exact_gpu_roundtrip"] and m["output_check_passed"]]
    for point in points:
        dominated = any(other["resident_bytes"] <= point["resident_bytes"]
                        and other["p95_slowdown_pct"] <= point["p95_slowdown_pct"]
                        and (other["resident_bytes"] < point["resident_bytes"]
                             or other["p95_slowdown_pct"] < point["p95_slowdown_pct"])
                        for other in points)
        if not dominated:
            result["pareto_candidates"].append(point["id"])
    eligible = [m for m in result["measurements"] if m["within_limits"]]
    if eligible:
        keys = ("p95_slowdown_pct", "resident_bytes") if policy.goal == "speed" \
            else ("resident_bytes", "p95_slowdown_pct")
        result["recommended_codec"] = min(eligible, key=lambda m: tuple(m[k] for k in keys))["id"]
    result.update(status="matched_measurements", scope=evidence["scope"],
                  hardware=evidence["hardware"], workload=evidence["workload"])
    return result


def read_time_budget(raw_bytes: int, stored_bytes: int, bandwidth_gbps: list[float]) -> dict:
    """Estimate only the ideal memory-read saving under user-supplied bandwidths."""
    if raw_bytes <= 0 or not bandwidth_gbps or any(not math.isfinite(b) or b <= 0
                                                 for b in bandwidth_gbps):
        raise ValueError("Bandwidth assumptions must be finite and positive")
    saved = max(0, raw_bytes - stored_bytes)
    times = sorted(saved / (b * 1000) for b in bandwidth_gbps)
    return {
        "basis": "analytical_memory_read_budget_not_predicted_latency",
        "assumed_bandwidth_gbps": bandwidth_gbps,
        "ideal_saved_read_us_for_sample": {"low": times[0], "high": times[-1]},
        "ideal_saved_read_us_per_input_mib": {
            "low": times[0] * 1048576 / raw_bytes, "high": times[-1] * 1048576 / raw_bytes},
        "assumptions": ["One read of each original or encoded byte; identical achieved bandwidth.",
                        "No extra lookup traffic, scratch writes, launch overhead or occupancy cost.",
                        "This budget applies only to exposed decoding cost in a bandwidth-bound operation.",
                        "It is not a measured GPU result or a whole-model slowdown estimate."],
    }


def cache_reuse_hint(baseline_us: float, hit_us: float, miss_us: float,
                     max_slowdown_pct: float) -> dict:
    """Estimate required reuse from measured hit/miss costs, not an observed hit rate."""
    if (any(not math.isfinite(v) or v <= 0 for v in (baseline_us, hit_us, miss_us))
            or not math.isfinite(max_slowdown_pct) or max_slowdown_pct < 0):
        raise ValueError("Require positive finite latencies and a nonnegative slowdown limit")
    allowed = baseline_us * (1 + max_slowdown_pct / 100)
    if miss_us <= allowed:
        hit_fraction = 0.0
    elif hit_us > allowed or hit_us >= miss_us:
        hit_fraction = None
    else:
        hit_fraction = min(1.0, max(0.0, (miss_us - allowed) / (miss_us - hit_us)))
    return {
        "basis": "linear_hit_miss_cost_model_not_measured_reuse",
        "max_slowdown_pct": max_slowdown_pct,
        "required_hit_rate_pct": None if hit_fraction is None else hit_fraction * 100,
        "miss_slowdown_pct": 100 * (miss_us / baseline_us - 1),
        "hit_slowdown_pct": 100 * (hit_us / baseline_us - 1),
        "assumptions": ["Representative matched-workload hit and miss latencies.",
                        "Average cost h*hit + (1-h)*miss; no overlap or additional cache-management cost.",
                        "A real routing trace, capacity measurement and policy benchmark are still required.",
                        "A null hit-rate requirement means even the measured hit cost misses this limit."],
    }
