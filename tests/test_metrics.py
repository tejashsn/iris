"""Tests for authoritative metric definitions."""

import numpy as np
import pytest

from iris.compare import GateLimits, compare_pairwise, compare_three_way
from iris.metrics import compute_closer_to_reference_pct, compute_metrics
from tests.conftest import solid_image


def _textured_base(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(20.0, 200.0, size=(64, 64, 3)).astype(np.float32)


def test_bitwise_identical_short_circuit():
    image = solid_image(42.0)
    metrics = compute_metrics(image, image.copy(), threshold=1.0)

    assert metrics["bitwise_identical"] is True
    assert metrics["mean_abs"] == 0.0
    assert metrics["p99_9"] == 0.0
    assert metrics["max_abs"] == 0.0
    assert metrics["pct_over_t"] == 0.0
    assert metrics["similarity_pct"] == 100.0
    assert metrics["within_t_pct"] == 100.0


def test_localized_patch_tail_vs_mean():
    reference = solid_image(255.0)
    current = reference.copy()
    current[:32, :32, :] = 0.0
    metrics = compute_metrics(reference, current, threshold=1.0)

    assert metrics["pct_over_t"] == pytest.approx(1.5625)
    assert metrics["mean_abs"] < 5.0
    assert metrics["max_abs"] == pytest.approx(255.0)


def test_per_channel_imbalance_detected():
    reference = solid_image(100.0)
    current = reference.copy()
    current[..., 0] += 5.0

    metrics = compute_metrics(reference, current, threshold=1.0)
    pcm = metrics["per_channel_mean"]

    assert pcm["R"] > pcm["G"]
    assert pcm["G"] == pytest.approx(0.0)
    assert pcm["B"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("baseline_mean", "current_mean", "expected"),
    [
        (10.0, 5.0, 50.0),
        (10.0, 15.0, -50.0),
        (10.0, 10.0, 0.0),
    ],
)
def test_closer_to_reference_pct_values(baseline_mean, current_mean, expected):
    reference = solid_image(100.0)
    baseline = reference + baseline_mean
    current = reference + current_mean

    ref_vs_base = compute_metrics(reference, baseline, threshold=1.0)
    ref_vs_cur = compute_metrics(reference, current, threshold=1.0)

    assert compute_closer_to_reference_pct(ref_vs_base, ref_vs_cur) == pytest.approx(expected)


def test_closer_to_reference_pct_none_when_baseline_identical():
    reference = solid_image(50.0)
    current = reference + 2.0

    ref_vs_base = compute_metrics(reference, reference.copy(), threshold=1.0)
    ref_vs_cur = compute_metrics(reference, current, threshold=1.0)

    assert compute_closer_to_reference_pct(ref_vs_base, ref_vs_cur) is None


def test_sanity_uniform_near_black_and_nan():
    from iris.sanity import check_image_sanity

    uniform = check_image_sanity(solid_image(12.0), "uniform")
    near_black = check_image_sanity(solid_image(0.5), "near_black")
    nan_frame = solid_image(10.0)
    nan_frame[0, 0, 0] = np.nan
    nan_result = check_image_sanity(nan_frame, "nan")

    assert uniform["passed"] is False
    assert near_black["passed"] is False
    assert nan_result["passed"] is False


def test_pairwise_report_only_without_limits():
    reference = _textured_base(3)
    current = reference + 0.4

    result = compare_pairwise(reference, current, threshold=1.0, limits=GateLimits())
    assert result["gate"]["verdict"] == "REPORT_ONLY"
    assert result["gate"]["passed"] is None


def test_pairwise_pass_and_fail_with_limits():
    reference = _textured_base(1)
    healthy = reference + 0.4
    broken = reference + 20.0

    limits = GateLimits(max_mean_abs=1.0)
    healthy_result = compare_pairwise(reference, healthy, threshold=1.0, limits=limits)
    broken_result = compare_pairwise(reference, broken, threshold=1.0, limits=limits)

    assert healthy_result["gate"]["verdict"] == "PASS"
    assert broken_result["gate"]["verdict"] == "FAIL"


def test_sanity_forces_fail_when_limits_configured():
    reference = solid_image(0.5)
    current = solid_image(0.6)
    limits = GateLimits(max_mean_abs=1.0)

    result = compare_pairwise(reference, current, threshold=1.0, limits=limits)
    assert result["gate"]["verdict"] == "FAIL"


def test_three_way_emits_all_blocks_and_relative_pct():
    reference = _textured_base(2)
    baseline = reference + 10.0
    current = reference + 5.0

    result = compare_three_way(
        reference,
        baseline,
        current,
        threshold=1.0,
        limits=GateLimits(max_mean_abs=8.0),
    )

    assert "reference_vs_baseline" in result
    assert "reference_vs_current" in result
    assert "baseline_vs_current" in result
    assert result["closer_to_reference_pct"] == pytest.approx(50.0)
    assert result["gate"]["verdict"] == "PASS"


def test_max_abs_reaches_full_scale_for_localized_fault():
    reference = np.zeros((64, 64, 3), dtype=np.float32)
    reference[:] = 10.0
    current = reference.copy()
    current[0, 0, :] = 255.0

    metrics = compute_metrics(reference, current, threshold=1.0)

    assert metrics["max_abs"] == pytest.approx(245.0)
    assert metrics["mean_abs"] < 0.1
