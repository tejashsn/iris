"""Tests for readable difference visualisations."""

import numpy as np
from PIL import Image

from iris.heatmap import (
    write_diff_heatmap,
    write_heatmap_overview,
    write_worst_region_crop,
)
from iris.regions import analyze_blocks


def _frames(size: int = 512, patch: int = 32):
    reference = np.full((size, size, 3), 100.0, dtype=np.float32)
    current = reference.copy()
    current[:patch, :patch, :] = 255.0
    return reference, current


def test_full_resolution_map_preserves_dimensions(tmp_path):
    reference, current = _frames(128)
    path = write_diff_heatmap(reference, current, tmp_path / "raw.png", gain=8.0)

    with Image.open(path) as img:
        assert img.size == (128, 128)
        assert img.mode == "L"


def test_overview_is_downscaled_and_reports_its_scale(tmp_path):
    reference, current = _frames(2048)
    info = write_heatmap_overview(
        reference, current, tmp_path / "overview.png", gain=8.0, max_dim=512
    )

    assert info["downscale_factor"] == 4
    assert info["downscale_method"] == "max-pool"
    assert info["rendered_shape"]["width"] == 512
    assert info["saturation_abs_diff"] == 255.0 / 8.0
    assert "absolute difference" in info["legend"]


def test_overview_max_pooling_keeps_small_hot_patch_visible(tmp_path):
    reference, current = _frames(1024, patch=8)
    info = write_heatmap_overview(
        reference, current, tmp_path / "overview.png", gain=1.0, max_dim=128
    )

    with Image.open(info["path"]) as img:
        pixels = np.asarray(img.convert("RGB"))

    # The 8x8 patch survives an 8x reduction because pooling takes the max.
    assert pixels.max() > 200


def test_overview_adds_legend_rows_below_the_body(tmp_path):
    reference, current = _frames(256)
    info = write_heatmap_overview(
        reference, current, tmp_path / "overview.png", gain=8.0, max_dim=256, threshold=1.0
    )

    with Image.open(info["path"]) as img:
        assert img.height > info["rendered_shape"]["height"]
    assert "threshold 1" in info["legend"] or "threshold" not in info["legend"]


def test_worst_region_crop_is_a_three_panel_triptych(tmp_path):
    reference, current = _frames(256)
    analysis = analyze_blocks(reference, current, block_size=32)
    worst = analysis["worst_blocks"][0]

    info = write_worst_region_crop(
        reference, current, worst, tmp_path / "crop.png", gain=8.0, max_dim=128
    )

    assert info["region"] == {"x0": 0, "y0": 0, "x1": 32, "y1": 32}
    assert info["zoom"] == 4
    assert len(info["panels"]) == 3

    with Image.open(info["path"]) as img:
        width, height = img.size
    assert width > height
    assert width >= 32 * 4 * 3


def test_shape_mismatch_is_rejected(tmp_path):
    reference = np.zeros((16, 16, 3), dtype=np.float32)
    current = np.zeros((16, 8, 3), dtype=np.float32)

    try:
        write_diff_heatmap(reference, current, tmp_path / "x.png")
    except ValueError as exc:
        assert "Shape mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched shapes")
