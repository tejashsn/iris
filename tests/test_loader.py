"""Tests for image loading."""

import numpy as np
import pytest
from PIL import Image

from iris.loader import load_image


def test_load_png(tmp_path):
    arr = np.full((8, 8, 3), 127, dtype=np.uint8)
    path = tmp_path / "sample.png"
    Image.fromarray(arr, mode="RGB").save(path)

    loaded = load_image(path)
    assert loaded.shape == (8, 8, 3)
    assert loaded.dtype == np.float32
    assert loaded[0, 0, 0] == pytest.approx(127.0)


def test_load_npy_normalized_2d(tmp_path):
    arr = np.full((4, 4), 0.5, dtype=np.float32)
    path = tmp_path / "sample.npy"
    np.save(path, arr)

    loaded = load_image(path)
    assert loaded.shape == (4, 4, 3)
    assert loaded[0, 0, 0] == pytest.approx(127.5)


def test_load_npy_rgba_drops_alpha(tmp_path):
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., :3] = 1.0
    path = tmp_path / "sample.npy"
    np.save(path, rgba)

    loaded = load_image(path)
    assert loaded.shape == (2, 2, 3)
