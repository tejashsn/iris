"""Comparison orchestration for pairwise and three-way modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from iris.metrics import (
    compute_closer_to_reference_pct,
    compute_metrics,
    interpret_closer_to_reference_pct,
)
from iris.regions import DEFAULT_BLOCK_SIZE, analyze_blocks
from iris.sanity import any_sanity_failed, check_image_sanity

ComparisonResult = dict[str, Any]


@dataclass
class GateLimits:
    """Optional gate limits supplied by the caller. No defaults ship in IRIS."""

    max_mean_abs: float | None = None
    max_p99_9: float | None = None
    max_max_abs: float | None = None
    max_pct_over_t: float | None = None
    min_similarity_pct: float | None = None

    def is_configured(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_mean_abs,
                self.max_p99_9,
                self.max_max_abs,
                self.max_pct_over_t,
                self.min_similarity_pct,
            )
        )


def compare_pairwise(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    threshold: float,
    reference_label: str = "reference",
    current_label: str = "current",
    limits: GateLimits | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> ComparisonResult:
    sanity = [
        check_image_sanity(reference, reference_label),
        check_image_sanity(current, current_label),
    ]
    metrics = compute_metrics(reference, current, threshold)
    gate = evaluate_gate(metrics, limits, any_sanity_failed(sanity))

    return {
        "mode": "pairwise",
        "labels": {
            "reference": reference_label,
            "current": current_label,
        },
        "sanity": sanity,
        "metrics": metrics,
        "regions": analyze_blocks(reference, current, block_size=block_size),
        "gate": gate,
    }


def compare_three_way(
    reference: np.ndarray,
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    threshold: float,
    reference_label: str = "reference",
    baseline_label: str = "baseline",
    current_label: str = "current",
    limits: GateLimits | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> ComparisonResult:
    sanity = [
        check_image_sanity(reference, reference_label),
        check_image_sanity(baseline, baseline_label),
        check_image_sanity(current, current_label),
    ]

    reference_vs_baseline = compute_metrics(reference, baseline, threshold)
    reference_vs_current = compute_metrics(reference, current, threshold)
    baseline_vs_current = compute_metrics(baseline, current, threshold)

    closer = compute_closer_to_reference_pct(
        reference_vs_baseline,
        reference_vs_current,
    )

    gate = evaluate_gate(
        reference_vs_current,
        limits,
        any_sanity_failed(sanity),
    )

    return {
        "mode": "three_way",
        "labels": {
            "reference": reference_label,
            "baseline": baseline_label,
            "current": current_label,
        },
        "sanity": sanity,
        "reference_vs_baseline": reference_vs_baseline,
        "reference_vs_current": reference_vs_current,
        "baseline_vs_current": baseline_vs_current,
        "regions": analyze_blocks(reference, current, block_size=block_size),
        "closer_to_reference_pct": closer,
        "closer_to_reference_interpretation": interpret_closer_to_reference_pct(closer),
        "gate": gate,
    }


def evaluate_gate(
    metrics: dict[str, Any],
    limits: GateLimits | None,
    sanity_failed: bool,
) -> dict[str, Any]:
    if limits is None or not limits.is_configured():
        return {
            "verdict": "REPORT_ONLY",
            "passed": None,
            "checks": [],
            "notes": [
                "No gate limits configured; metrics reported without PASS/FAIL verdict."
            ],
        }

    if sanity_failed:
        return {
            "verdict": "FAIL",
            "passed": False,
            "checks": [],
            "notes": ["Sanity check failure forces FAIL when gate limits are configured."],
        }

    checks: list[dict[str, Any]] = []

    def _check(name: str, actual: float, limit: float, comparator: str) -> None:
        if comparator == "max":
            passed = actual <= limit
        else:
            passed = actual >= limit
        checks.append(
            {
                "name": name,
                "actual": actual,
                "limit": limit,
                "comparator": comparator,
                "passed": passed,
            }
        )

    if limits.max_mean_abs is not None:
        _check("max_mean_abs", metrics["mean_abs"], limits.max_mean_abs, "max")
    if limits.max_p99_9 is not None:
        _check("max_p99_9", metrics["p99_9"], limits.max_p99_9, "max")
    if limits.max_max_abs is not None:
        _check("max_max_abs", metrics["max_abs"], limits.max_max_abs, "max")
    if limits.max_pct_over_t is not None:
        _check("max_pct_over_t", metrics["pct_over_t"], limits.max_pct_over_t, "max")
    if limits.min_similarity_pct is not None:
        _check(
            "min_similarity_pct",
            metrics["similarity_pct"],
            limits.min_similarity_pct,
            "min",
        )

    passed = all(item["passed"] for item in checks)
    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "checks": checks,
        "notes": [],
    }
