"""Block-level localisation of pixel drift.

A frame-wide mean says how much drift there is; it cannot say where. This module
partitions the frame into a block grid and ranks blocks by their share of the
total absolute difference, which is what turns a diff map into an actionable
coordinate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_BLOCK_SIZE = 32
DEFAULT_TOP_N = 5

# Share of total absolute difference used to describe how concentrated drift is.
# This drives reporting only; no verdict reads it.
CONCENTRATION_TARGET_PCT = 90.0

# Wording heuristic: above this fraction of blocks, the difference is described as
# broad rather than localised. Phrasing only; no verdict reads it.
BROAD_SPREAD_BLOCK_FRACTION = 0.5


def analyze_blocks(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Rank frame blocks by their contribution to total absolute difference."""
    ref = np.asarray(reference, dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32)

    if ref.shape != cur.shape:
        raise ValueError(f"Shape mismatch: reference {ref.shape} vs current {cur.shape}")

    abs_diff = np.abs(cur - ref)
    magnitude = np.max(abs_diff, axis=2) if abs_diff.ndim == 3 else abs_diff
    height, width = magnitude.shape

    block_size = max(1, min(block_size, height, width))
    # float64 accumulation, matching iris.metrics and the browser tool.
    total = float(np.sum(magnitude, dtype=np.float64))

    blocks: list[dict[str, Any]] = []
    for y0 in range(0, height, block_size):
        for x0 in range(0, width, block_size):
            y1 = min(y0 + block_size, height)
            x1 = min(x0 + block_size, width)
            tile = magnitude[y0:y1, x0:x1]
            tile_sum = float(np.sum(tile, dtype=np.float64))
            blocks.append(
                {
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "mean_abs": float(np.mean(tile, dtype=np.float64)),
                    "max_abs": float(np.max(tile)),
                    "sum_abs": tile_sum,
                    "contribution_pct": (tile_sum / total * 100.0) if total > 0 else 0.0,
                }
            )

    ranked = sorted(blocks, key=lambda item: item["sum_abs"], reverse=True)

    cumulative = 0.0
    blocks_for_target = 0
    for block in ranked:
        if cumulative >= CONCENTRATION_TARGET_PCT:
            break
        cumulative += block["contribution_pct"]
        blocks_for_target += 1

    return {
        "block_size": block_size,
        "block_count": len(blocks),
        "grid": {
            "rows": (height + block_size - 1) // block_size,
            "cols": (width + block_size - 1) // block_size,
        },
        "frame_shape": {"height": int(height), "width": int(width)},
        "total_abs": total,
        "concentration_target_pct": CONCENTRATION_TARGET_PCT,
        "blocks_for_target_pct": blocks_for_target,
        "worst_blocks": ranked[:top_n],
    }


def describe_concentration(analysis: dict[str, Any]) -> str:
    """Describe where the difference sits, leading with the actionable coordinate."""
    if analysis["total_abs"] <= 0.0:
        return "No absolute difference to localise; the frames are identical."

    top = analysis["worst_blocks"]
    worst = top[0]
    top_share = sum(block["contribution_pct"] for block in top)
    location = f"x={worst['x0']}-{worst['x1']}, y={worst['y0']}-{worst['y1']}"

    sentence = (
        f"The {len(top)} worst blocks account for {top_share:.1f}% of the total absolute "
        f"difference, and the single largest contributor is {location} at "
        f"{worst['contribution_pct']:.1f}%."
    )

    needed = analysis["blocks_for_target_pct"]
    total_blocks = analysis["block_count"]
    target = analysis["concentration_target_pct"]

    if total_blocks and needed > total_blocks * BROAD_SPREAD_BLOCK_FRACTION:
        sentence += (
            f" Reaching {target:.0f}% of the total needs {needed} of {total_blocks} blocks, "
            "so a broad low-level difference covers the frame alongside that hot spot."
        )
    else:
        sentence += (
            f" Only {needed} of {total_blocks} blocks are needed to reach {target:.0f}% of "
            "the total, so the difference is genuinely localised."
        )
    return sentence
