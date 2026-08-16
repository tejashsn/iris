"""Authoritative pixel-difference metric definitions for IRIS.

All comparison statistics are defined here. Other modules must import from this
module rather than re-deriving formulas.

Metrics measure closeness to a reference image, never visual quality
improvement. See README for interpretation guidance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MetricDict = dict[str, Any]


def compute_metrics(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float,
) -> MetricDict:
    """Compute pixel-level drift metrics between two RGB float32 images in [0, 255].

    Operates per-sample (each channel of each pixel). ``threshold`` defines
    ``pct_over_t`` only and is echoed for provenance; it is not a pass/fail cut.
    """
    ref = _as_rgb_float32(reference)
    cur = _as_rgb_float32(current)

    if ref.shape != cur.shape:
        raise ValueError(
            f"Shape mismatch: reference {ref.shape} vs current {cur.shape}"
        )

    sample_count = int(ref.size)
    result: MetricDict = {
        "sample_count": sample_count,
        "threshold": float(threshold),
    }

    if np.array_equal(ref, cur):
        result.update(_bitwise_identical_metrics(sample_count, threshold))
        return result

    abs_diff = np.abs(cur - ref)
    flat = abs_diff.reshape(-1)

    # Accumulate in float64. Summing millions of float32 samples in float32 loses
    # significant digits in exactly the 0.3-0.6 regime this tool measures, and it
    # would also diverge from the browser tool, where all arithmetic is float64.
    mean_abs = float(np.mean(flat, dtype=np.float64))
    p99_9 = percentile_float64(flat, 99.9)
    max_abs = float(np.max(flat))
    pct_over_t = float(np.mean(flat > threshold, dtype=np.float64) * 100.0)

    per_channel_mean = {
        "R": float(np.mean(abs_diff[..., 0], dtype=np.float64)),
        "G": float(np.mean(abs_diff[..., 1], dtype=np.float64)),
        "B": float(np.mean(abs_diff[..., 2], dtype=np.float64)),
    }

    similarity_pct = 100.0 * (1.0 - mean_abs / 255.0)
    within_t_pct = 100.0 - pct_over_t

    result.update(
        {
            "bitwise_identical": False,
            "mean_abs": mean_abs,
            "p99_9": p99_9,
            "max_abs": max_abs,
            "pct_over_t": pct_over_t,
            "per_channel_mean": per_channel_mean,
            "similarity_pct": similarity_pct,
            "within_t_pct": within_t_pct,
        }
    )
    return result


def compute_closer_to_reference_pct(
    reference_vs_baseline: MetricDict,
    reference_vs_current: MetricDict,
) -> float | None:
    """Three-way relative closeness: positive means current drifted less than baseline.

    Returns ``None`` when the baseline is bitwise-identical to the reference so
    callers avoid dividing by zero.
    """
    if reference_vs_baseline.get("bitwise_identical"):
        return None

    baseline_mean = reference_vs_baseline["mean_abs"]
    current_mean = reference_vs_current["mean_abs"]

    if baseline_mean == 0.0:
        if current_mean == 0.0:
            return 0.0
        return None

    return ((baseline_mean - current_mean) / baseline_mean) * 100.0


def interpret_closer_to_reference_pct(value: float | None) -> str:
    """Human-readable interpretation for three-way relative closeness."""
    if value is None:
        return (
            "closer_to_reference_pct is undefined because baseline is "
            "bitwise-identical to reference; no baseline drift to compare against."
        )
    if value > 0:
        return (
            f"Current is {value:.2f}% closer to reference than baseline "
            "(less pixel drift relative to reference)."
        )
    if value < 0:
        return (
            f"Current is {abs(value):.2f}% farther from reference than baseline "
            "(more pixel drift relative to reference)."
        )
    return "Current and baseline have equal mean absolute drift from reference."


def percentile_float64(values: np.ndarray, percent: float) -> float:
    """Linear-interpolated percentile with the blend computed in float64.

    ``np.percentile`` interpolates in the input dtype, so a float32 array yields a
    float32 result. Partitioning around the two bracketing ranks and blending in
    float64 keeps full precision without allocating a float64 copy of the array,
    and matches the browser tool, where all arithmetic is float64.
    """
    n = values.size
    if n == 0:
        return 0.0

    rank = (percent / 100.0) * (n - 1)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))

    if upper >= n:
        return float(np.max(values))
    if lower == upper:
        return float(np.partition(values, lower)[lower])

    partitioned = np.partition(values, (lower, upper))
    weight = rank - lower
    return float(partitioned[lower]) * (1.0 - weight) + float(partitioned[upper]) * weight


def _bitwise_identical_metrics(sample_count: int, threshold: float) -> MetricDict:
    return {
        "bitwise_identical": True,
        "mean_abs": 0.0,
        "p99_9": 0.0,
        "max_abs": 0.0,
        "pct_over_t": 0.0,
        "per_channel_mean": {"R": 0.0, "G": 0.0, "B": 0.0},
        "similarity_pct": 100.0,
        "within_t_pct": 100.0,
        "sample_count": sample_count,
        "threshold": float(threshold),
    }


def _as_rgb_float32(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {arr.shape}")
    return arr
