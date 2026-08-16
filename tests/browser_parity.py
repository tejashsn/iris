"""Browser-parity reimplementation of IRIS metric computation."""

from __future__ import annotations

import numpy as np


def browser_load_rgba_to_rgb(data: np.ndarray, *, normalized: bool = False) -> np.ndarray:
    """Mirror browser image decoding: strip alpha, float32 RGB in 0-255."""
    arr = np.asarray(data, dtype=np.float32)
    if normalized:
        arr = arr * 255.0

    if arr.ndim == 2:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    return arr.astype(np.float32)


def _browser_percentile(values: np.ndarray, p: float) -> float:
    """Match iris-triage.html linear percentile interpolation."""
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    if n == 0:
        return 0.0
    rank = (p / 100.0) * (n - 1)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    weight = rank - lower
    if upper >= n:
        return float(sorted_vals[-1])
    # Cast before blending: JS reads Float32Array elements as float64 numbers.
    return float(sorted_vals[lower]) * (1.0 - weight) + float(sorted_vals[upper]) * weight


def browser_compute_metrics_loop(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float,
) -> dict:
    """Explicit reimplementation of iris-triage.html computeMetrics()."""
    ref = browser_load_rgba_to_rgb(reference)
    cur = browser_load_rgba_to_rgb(current)

    sample_count = int(ref.size)
    if np.array_equal(ref, cur):
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

    # JavaScript numbers are float64, so every accumulator here is float64 too.
    abs_diff = np.abs(cur - ref)
    flat = abs_diff.reshape(-1)
    mean_abs = float(np.sum(flat, dtype=np.float64) / flat.size)
    p99_9 = _browser_percentile(flat, 99.9)
    max_abs = float(np.max(flat))
    over = float(np.sum(flat > threshold, dtype=np.float64))
    pct_over_t = (over / flat.size) * 100.0

    per_channel_mean = {
        "R": float(np.sum(abs_diff[..., 0], dtype=np.float64) / abs_diff[..., 0].size),
        "G": float(np.sum(abs_diff[..., 1], dtype=np.float64) / abs_diff[..., 1].size),
        "B": float(np.sum(abs_diff[..., 2], dtype=np.float64) / abs_diff[..., 2].size),
    }
    similarity_pct = 100.0 * (1.0 - mean_abs / 255.0)
    within_t_pct = 100.0 - pct_over_t

    return {
        "bitwise_identical": False,
        "mean_abs": mean_abs,
        "p99_9": p99_9,
        "max_abs": max_abs,
        "pct_over_t": pct_over_t,
        "per_channel_mean": per_channel_mean,
        "similarity_pct": similarity_pct,
        "within_t_pct": within_t_pct,
        "sample_count": sample_count,
        "threshold": float(threshold),
    }


def browser_sanity(image: np.ndarray) -> dict:
    """Reimplementation of iris-triage.html checkSanity()."""
    arr = browser_load_rgba_to_rgb(image)
    flat = arr.reshape(-1)
    has_nan = bool(np.isnan(flat).any())
    has_inf = bool(np.isinf(flat).any())
    is_uniform = bool(np.all(flat == flat[0])) if flat.size else True
    frame_mean = float(np.sum(flat, dtype=np.float64) / flat.size) if flat.size else 0.0
    frame_max = float(np.max(flat)) if flat.size else 0.0
    is_near_black = frame_mean < 1.0 and frame_max < 2.0
    issues = []
    if has_nan:
        issues.append("NaN values detected")
    if has_inf:
        issues.append("Inf values detected")
    if is_uniform:
        issues.append("uniform frame (all samples equal)")
    if is_near_black:
        issues.append("near-black frame (mean < 1.0 and max < 2.0)")
    return {
        "passed": len(issues) == 0,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_uniform": is_uniform,
        "is_near_black": is_near_black,
        "frame_mean": frame_mean,
        "frame_max": frame_max,
        "issues": issues,
    }


def browser_closer_to_reference_pct(
    reference_vs_baseline: dict,
    reference_vs_current: dict,
) -> float | None:
    if reference_vs_baseline.get("bitwise_identical"):
        return None
    baseline_mean = reference_vs_baseline["mean_abs"]
    current_mean = reference_vs_current["mean_abs"]
    if baseline_mean == 0.0:
        if current_mean == 0.0:
            return 0.0
        return None
    return ((baseline_mean - current_mean) / baseline_mean) * 100.0


def browser_magnitude_map(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Reimplementation of iris-triage.html magnitudeMap()."""
    ref = browser_load_rgba_to_rgb(reference)
    cur = browser_load_rgba_to_rgb(current)
    height, width = ref.shape[:2]
    out = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            out[y, x] = max(
                abs(cur[y, x, 0] - ref[y, x, 0]),
                abs(cur[y, x, 1] - ref[y, x, 1]),
                abs(cur[y, x, 2] - ref[y, x, 2]),
            )
    return out


def browser_analyze_blocks(
    reference: np.ndarray,
    current: np.ndarray,
    block_size: int,
    top_n: int = 5,
) -> dict:
    """Explicit reimplementation of iris-triage.html analyzeBlocks()."""
    magnitude = browser_magnitude_map(reference, current)
    height, width = magnitude.shape
    block_size = max(1, min(block_size, width, height))

    total = 0.0
    for y in range(height):
        for x in range(width):
            total += float(magnitude[y, x])

    blocks = []
    for y0 in range(0, height, block_size):
        for x0 in range(0, width, block_size):
            y1 = min(y0 + block_size, height)
            x1 = min(x0 + block_size, width)
            tile_sum = 0.0
            tile_max = 0.0
            count = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    value = float(magnitude[y, x])
                    tile_sum += value
                    if value > tile_max:
                        tile_max = value
                    count += 1
            blocks.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "mean_abs": tile_sum / count,
                    "max_abs": tile_max,
                    "sum_abs": tile_sum,
                    "contribution_pct": (tile_sum / total * 100.0) if total > 0 else 0.0,
                }
            )

    ranked = sorted(blocks, key=lambda item: item["sum_abs"], reverse=True)

    cumulative = 0.0
    needed = 0
    for block in ranked:
        if cumulative >= 90.0:
            break
        cumulative += block["contribution_pct"]
        needed += 1

    return {
        "block_size": block_size,
        "block_count": len(blocks),
        "grid": {
            "rows": -(-height // block_size),
            "cols": -(-width // block_size),
        },
        "frame_shape": {"height": height, "width": width},
        "total_abs": total,
        "blocks_for_target_pct": needed,
        "worst_blocks": ranked[:top_n],
    }


def browser_max_pool(magnitude: np.ndarray, factor: int) -> np.ndarray:
    """Reimplementation of iris-triage.html maxPool()."""
    height, width = magnitude.shape
    out_h = -(-height // factor)
    out_w = -(-width // factor)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for y in range(out_h):
        for x in range(out_w):
            peak = 0.0
            for dy in range(factor):
                sy = y * factor + dy
                if sy >= height:
                    break
                for dx in range(factor):
                    sx = x * factor + dx
                    if sx >= width:
                        break
                    value = float(magnitude[sy, sx])
                    if value > peak:
                        peak = value
            out[y, x] = peak
    return out


def browser_colormap_lut() -> np.ndarray:
    """Reimplementation of the iris-triage.html COLORMAP_LUT builder."""
    control = [
        (0.00, (0, 0, 4)),
        (0.25, (87, 16, 110)),
        (0.50, (188, 55, 84)),
        (0.75, (249, 142, 9)),
        (1.00, (252, 255, 164)),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        lo, hi = control[0], control[-1]
        for k in range(len(control) - 1):
            if control[k][0] <= t <= control[k + 1][0]:
                lo, hi = control[k], control[k + 1]
                break
        span = hi[0] - lo[0]
        weight = (t - lo[0]) / span if span > 0 else 0.0
        for c in range(3):
            lut[i, c] = round(lo[1][c] + (hi[1][c] - lo[1][c]) * weight)
    return lut
