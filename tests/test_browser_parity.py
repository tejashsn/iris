"""Browser/Python parity.

The standalone iris-triage.html reimplements the metric, sanity, region and
colormap logic in JavaScript so it can run from file:// with no install. These
tests reimplement the browser's loops in Python and assert they agree with the
library, so the two entry points cannot silently drift apart.
"""

import numpy as np
import pytest

from iris.heatmap import _COLORMAP_LUT, _max_pool
from iris.metrics import compute_closer_to_reference_pct, compute_metrics
from iris.regions import analyze_blocks
from iris.sanity import check_image_sanity
from tests.browser_parity import (
    browser_analyze_blocks,
    browser_closer_to_reference_pct,
    browser_colormap_lut,
    browser_compute_metrics_loop,
    browser_load_rgba_to_rgb,
    browser_magnitude_map,
    browser_max_pool,
    browser_sanity,
)

TOLERANCE = 1e-9

_METRIC_FIELDS = (
    "mean_abs",
    "p99_9",
    "max_abs",
    "pct_over_t",
    "similarity_pct",
    "within_t_pct",
)


def _pair(seed: int, size: int = 48, scale: float = 2.0):
    rng = np.random.default_rng(seed)
    reference = rng.uniform(0, 255, size=(size, size, 3)).astype(np.float32)
    current = reference + rng.uniform(-scale, scale, size=reference.shape).astype(np.float32)
    return reference, np.clip(current, 0, 255).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_parity(seed):
    reference, current = _pair(seed)
    library = compute_metrics(reference, current, 1.5)
    browser = browser_compute_metrics_loop(reference, current, 1.5)

    for field in _METRIC_FIELDS:
        assert library[field] == pytest.approx(browser[field], abs=TOLERANCE)
    for channel in ("R", "G", "B"):
        assert library["per_channel_mean"][channel] == pytest.approx(
            browser["per_channel_mean"][channel], abs=TOLERANCE
        )
    assert library["sample_count"] == browser["sample_count"]


def test_metrics_parity_strips_alpha():
    reference, current = _pair(3)
    rgba = np.dstack([current, np.full(current.shape[:2], 255, dtype=np.float32)])

    library = compute_metrics(reference, current, 1.0)
    browser = browser_compute_metrics_loop(
        browser_load_rgba_to_rgb(reference), browser_load_rgba_to_rgb(rgba), 1.0
    )

    for field in _METRIC_FIELDS:
        assert library[field] == pytest.approx(browser[field], abs=TOLERANCE)


def test_bitwise_identical_parity():
    reference, _ = _pair(4)
    library = compute_metrics(reference, reference.copy(), 1.0)
    browser = browser_compute_metrics_loop(reference, reference.copy(), 1.0)

    assert library["bitwise_identical"] is browser["bitwise_identical"] is True
    for field in _METRIC_FIELDS:
        assert library[field] == pytest.approx(browser[field], abs=TOLERANCE)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: np.full((8, 8, 3), 12.0, dtype=np.float32),
        lambda: np.full((8, 8, 3), 0.5, dtype=np.float32),
        lambda: np.full((8, 8, 3), np.nan, dtype=np.float32),
    ],
)
def test_sanity_parity(builder):
    image = builder()
    library = check_image_sanity(image, "x")
    browser = browser_sanity(image)

    assert library["passed"] == browser["passed"]
    assert library["has_nan"] == browser["has_nan"]
    assert library["is_uniform"] == browser["is_uniform"]
    assert library["is_near_black"] == browser["is_near_black"]
    assert library["issues"] == browser["issues"]


def test_closer_to_reference_pct_parity():
    reference = np.full((16, 16, 3), 100.0, dtype=np.float32)
    baseline = reference + 8.0
    current = reference + 3.0

    ref_vs_base = compute_metrics(reference, baseline, 1.0)
    ref_vs_cur = compute_metrics(reference, current, 1.0)

    assert compute_closer_to_reference_pct(ref_vs_base, ref_vs_cur) == pytest.approx(
        browser_closer_to_reference_pct(ref_vs_base, ref_vs_cur), abs=TOLERANCE
    )


def test_magnitude_map_parity():
    reference, current = _pair(5, size=24)
    library = np.max(np.abs(current - reference), axis=2)
    browser = browser_magnitude_map(reference, current)

    assert np.allclose(library, browser, atol=TOLERANCE)


@pytest.mark.parametrize("block_size", [8, 16])
def test_region_analysis_parity(block_size):
    reference, current = _pair(6, size=48)
    current[8:24, 8:24, :] = 255.0

    library = analyze_blocks(reference, current, block_size=block_size)
    browser = browser_analyze_blocks(reference, current, block_size)

    assert library["block_count"] == browser["block_count"]
    assert library["grid"] == browser["grid"]
    assert library["blocks_for_target_pct"] == browser["blocks_for_target_pct"]
    assert library["total_abs"] == pytest.approx(browser["total_abs"], rel=1e-6)

    for lib_block, browser_block in zip(library["worst_blocks"], browser["worst_blocks"]):
        assert (lib_block["x0"], lib_block["y0"]) == (browser_block["x0"], browser_block["y0"])
        assert lib_block["contribution_pct"] == pytest.approx(
            browser_block["contribution_pct"], abs=1e-6
        )
        assert lib_block["max_abs"] == pytest.approx(browser_block["max_abs"], abs=TOLERANCE)


def test_region_analysis_parity_on_partial_blocks():
    reference, current = _pair(7, size=20)
    library = analyze_blocks(reference, current, block_size=8)
    browser = browser_analyze_blocks(reference, current, 8)

    assert library["grid"] == browser["grid"]
    assert library["block_count"] == browser["block_count"]
    assert [(b["x1"], b["y1"]) for b in library["worst_blocks"]] == [
        (b["x1"], b["y1"]) for b in browser["worst_blocks"]
    ]


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_max_pool_parity(factor):
    rng = np.random.default_rng(8)
    magnitude = rng.uniform(0, 255, size=(30, 26)).astype(np.float32)

    library = _max_pool(magnitude, factor)
    browser = browser_max_pool(magnitude, factor)

    assert library.shape == browser.shape
    assert np.allclose(library, browser, atol=TOLERANCE)


def test_colormap_lut_parity():
    assert np.array_equal(_COLORMAP_LUT, browser_colormap_lut())
