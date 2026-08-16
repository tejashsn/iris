"""Image loading for PNG, JPEG, and NumPy arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as float32 HxWx3 RGB in [0, 255]."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return _load_npy(path)
    if suffix in {".png", ".jpg", ".jpeg"}:
        return _load_raster(path)
    raise ValueError(f"Unsupported image format: {suffix}")


def _load_raster(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)
    return arr


def _load_npy(path: Path) -> np.ndarray:
    raw = np.load(path)

    if raw.ndim == 2:
        raw = raw[..., np.newaxis]
    elif raw.ndim == 4:
        raw = raw[0]

    if raw.ndim != 3:
        raise ValueError(f"Expected 2D/3D/4D array in {path}, got shape {raw.shape}")

    arr = np.asarray(raw, dtype=np.float32)

    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0

    channels = arr.shape[2]
    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif channels == 3:
        pass
    elif channels == 4:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported channel count {channels} in {path}")

    return arr
