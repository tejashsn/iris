"""Tests for block-level drift localisation."""

import numpy as np
import pytest

from iris.regions import analyze_blocks, describe_concentration


def test_localized_patch_is_isolated_to_one_block():
    reference = np.full((256, 256, 3), 128.0, dtype=np.float32)
    current = reference.copy()
    current[64:96, 32:64, :] = 0.0

    analysis = analyze_blocks(reference, current, block_size=32)

    assert analysis["grid"] == {"rows": 8, "cols": 8}
    assert analysis["block_count"] == 64
    assert analysis["blocks_for_target_pct"] == 1

    worst = analysis["worst_blocks"][0]
    assert (worst["x0"], worst["y0"]) == (32, 64)
    assert worst["contribution_pct"] == pytest.approx(100.0)


def test_uniform_shift_spreads_across_all_blocks():
    reference = np.full((128, 128, 3), 100.0, dtype=np.float32)
    current = reference + 1.0

    analysis = analyze_blocks(reference, current, block_size=32)

    assert analysis["block_count"] == 16
    assert analysis["blocks_for_target_pct"] == 15
    for block in analysis["worst_blocks"]:
        assert block["contribution_pct"] == pytest.approx(100.0 / 16)


def test_identical_frames_have_no_contribution():
    reference = np.full((64, 64, 3), 50.0, dtype=np.float32)
    analysis = analyze_blocks(reference, reference.copy(), block_size=16)

    assert analysis["total_abs"] == 0.0
    assert "identical" in describe_concentration(analysis)


def test_non_square_frame_and_partial_blocks():
    reference = np.zeros((70, 50, 3), dtype=np.float32)
    current = reference.copy()
    current[68:70, 48:50, :] = 255.0

    analysis = analyze_blocks(reference, current, block_size=32)

    assert analysis["grid"] == {"rows": 3, "cols": 2}
    worst = analysis["worst_blocks"][0]
    assert worst["x1"] == 50
    assert worst["y1"] == 70


def test_describe_concentration_reports_coordinates():
    reference = np.full((128, 128, 3), 10.0, dtype=np.float32)
    current = reference.copy()
    current[0:32, 96:128, :] = 200.0

    analysis = analyze_blocks(reference, current, block_size=32)
    description = describe_concentration(analysis)

    assert "x=96-128" in description
    assert "y=0-32" in description
    assert "genuinely localised" in description


def test_describe_concentration_calls_out_broad_spread():
    """A frame-wide noise floor plus one hot patch is broad, not localised.

    Summed over 256 blocks the low-level floor outweighs the patch, so the wording
    must not claim the difference is localised.
    """
    rng = np.random.default_rng(0)
    reference = rng.uniform(20, 200, size=(512, 512, 3)).astype(np.float32)
    current = reference + rng.normal(0, 0.55, reference.shape).astype(np.float32)
    current[0:32, 0:32, :] = 255.0

    description = describe_concentration(analyze_blocks(reference, current, block_size=32))

    assert "broad low-level difference" in description
    assert "genuinely localised" not in description
