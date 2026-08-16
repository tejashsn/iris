"""Fixtures and helpers for IRIS tests."""

from __future__ import annotations

import numpy as np


def solid_image(value: float, height: int = 256, width: int = 256) -> np.ndarray:
    arr = np.full((height, width, 3), value, dtype=np.float32)
    return arr


def localized_patch_image(
    background: float = 128.0,
    patch_value: float = 0.0,
    patch_size: int = 32,
    frame_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    reference = solid_image(background, frame_size, frame_size)
    current = reference.copy()
    current[:patch_size, :patch_size, :] = patch_value
    return reference, current
