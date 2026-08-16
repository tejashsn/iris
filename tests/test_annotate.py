"""Tests for the detection overlay renderer."""

import numpy as np
from PIL import Image

from iris.annotate import write_detection_overlay


def _detection():
    return {
        "backend": "fake",
        "available": True,
        "image_shape": {"width": 128, "height": 96},
        "elements": [
            {
                "element": "sun",
                "present": True,
                "confidence": 0.42,
                "boxes": [{"x0": 8, "y0": 8, "x1": 40, "y1": 40, "score": 0.42}],
            },
            {"element": "trees", "present": False, "confidence": 0.0, "boxes": []},
        ],
        "missing": ["trees"],
    }


def test_overlay_is_wider_than_image_for_the_legend(tmp_path):
    image = np.full((96, 128, 3), 120, dtype=np.float32)
    info = write_detection_overlay(image, _detection(), tmp_path / "overlay.png")

    with Image.open(info["path"]) as img:
        assert img.size[0] > 128  # legend column added on the right
        assert img.size[1] >= 96

    assert info["present"] == ["sun"]
    assert info["missing"] == ["trees"]


def test_overlay_draws_box_pixels_for_detected_element(tmp_path):
    image = np.zeros((96, 128, 3), dtype=np.float32)
    info = write_detection_overlay(image, _detection(), tmp_path / "overlay.png")

    with Image.open(info["path"]) as img:
        pixels = np.asarray(img.convert("RGB"))

    box_region = pixels[8:41, 8:41]
    assert box_region.max() > 0  # a coloured box was drawn on the black frame


def test_overlay_handles_no_boxes(tmp_path):
    image = np.full((32, 32, 3), 60, dtype=np.float32)
    detection = {
        "backend": "fake",
        "elements": [{"element": "cat", "present": False, "confidence": 0.0, "boxes": []}],
        "missing": ["cat"],
    }
    info = write_detection_overlay(image, detection, tmp_path / "overlay.png")
    assert info["present"] == []
    assert info["missing"] == ["cat"]
