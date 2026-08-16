"""Sanity checks run before any statistic is computed."""

from __future__ import annotations

from typing import Any

import numpy as np

SanityDict = dict[str, Any]


def check_image_sanity(image: np.ndarray, label: str) -> SanityDict:
    """Detect frames that would pollute drift distributions."""
    arr = np.asarray(image, dtype=np.float32)
    flat = arr.reshape(-1)

    has_nan = bool(np.isnan(flat).any())
    has_inf = bool(np.isinf(flat).any())

    if flat.size == 0:
        is_uniform = True
        frame_mean = 0.0
        frame_max = 0.0
    else:
        is_uniform = bool(np.all(flat == flat[0]))
        frame_mean = float(np.mean(flat, dtype=np.float64))
        frame_max = float(np.max(flat))

    is_near_black = frame_mean < 1.0 and frame_max < 2.0

    issues: list[str] = []
    if has_nan:
        issues.append("NaN values detected")
    if has_inf:
        issues.append("Inf values detected")
    if is_uniform:
        issues.append("uniform frame (all samples equal)")
    if is_near_black:
        issues.append("near-black frame (mean < 1.0 and max < 2.0)")

    return {
        "label": label,
        "passed": len(issues) == 0,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_uniform": is_uniform,
        "is_near_black": is_near_black,
        "frame_mean": frame_mean,
        "frame_max": frame_max,
        "issues": issues,
    }


def any_sanity_failed(results: list[SanityDict]) -> bool:
    return any(not item["passed"] for item in results)
