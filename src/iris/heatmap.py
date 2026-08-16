"""Absolute-difference visualisations built for readability.

A full-resolution amplified difference map is rarely useful on its own: at
benchmark resolutions it is a large, mostly-dark image with no scale. This
module produces three artifacts instead.

``write_diff_heatmap``      raw full-resolution map, unchanged provenance artifact
``write_heatmap_overview``  downscaled, colour-mapped, with a printed value legend
``write_worst_region_crop`` zoomed reference / current / difference triptych

Downscaling uses max-pooling rather than averaging so a small hot patch survives
the resize. Averaging would let a 32x32 fault vanish inside a 2048x2048 frame,
which is exactly the failure mode these metrics exist to catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_GAIN = 8.0
DEFAULT_MAX_DIM = 768
DEFAULT_CROP_MAX_DIM = 256

_LEGEND_HEIGHT = 34
_LABEL_HEIGHT = 16
_GUTTER = 6

# Inferno-style control points, interpolated into a 256-entry lookup table.
# A perceptual ramp is used so intensity ordering survives greyscale printing.
_COLORMAP_CONTROL_POINTS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, (0, 0, 4)),
    (0.25, (87, 16, 110)),
    (0.50, (188, 55, 84)),
    (0.75, (249, 142, 9)),
    (1.00, (252, 255, 164)),
)


def write_diff_heatmap(
    reference: np.ndarray,
    current: np.ndarray,
    output_path: str | Path,
    *,
    gain: float = DEFAULT_GAIN,
) -> Path:
    """Write a fixed-gain absolute-difference heatmap PNG at full resolution.

    Pixel intensity is ``clip(gain * abs(current - reference), 0, 255)``.
    Colour intensity is not severity; gain amplifies sub-threshold drift for
    visual triage only.
    """
    magnitude = _magnitude(reference, current)
    scaled = np.clip(magnitude * gain, 0.0, 255.0).astype(np.uint8)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scaled, mode="L").save(output)
    return output


def write_heatmap_overview(
    reference: np.ndarray,
    current: np.ndarray,
    output_path: str | Path,
    *,
    gain: float = DEFAULT_GAIN,
    max_dim: int = DEFAULT_MAX_DIM,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Write a downscaled, colour-mapped difference map with a value legend.

    Returns a dict describing the rendering so the report can state exactly what
    the picture means, including the absolute difference that saturates the ramp.
    """
    magnitude = _magnitude(reference, current)
    height, width = magnitude.shape

    factor = max(1, int(np.ceil(max(height, width) / float(max_dim))))
    pooled = _max_pool(magnitude, factor) if factor > 1 else magnitude

    scaled = np.clip(pooled * gain, 0.0, 255.0).astype(np.uint8)
    colored = _apply_colormap(scaled)

    saturation_value = 255.0 / gain if gain > 0 else float("inf")
    canvas = _compose_with_legend(
        colored,
        gain=gain,
        saturation_value=saturation_value,
        threshold=threshold,
        pooled_factor=factor,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)

    return {
        "path": str(output),
        "gain": gain,
        "downscale_factor": factor,
        "downscale_method": "max-pool" if factor > 1 else "none",
        "source_shape": {"height": int(height), "width": int(width)},
        "rendered_shape": {"height": int(pooled.shape[0]), "width": int(pooled.shape[1])},
        "saturation_abs_diff": saturation_value,
        "legend": (
            f"Colour ramp spans 0 to {saturation_value:.2f} absolute difference "
            f"(0-255 scale) at gain {gain:g}; values above saturate."
        ),
    }


def write_worst_region_crop(
    reference: np.ndarray,
    current: np.ndarray,
    block: dict[str, Any],
    output_path: str | Path,
    *,
    gain: float = DEFAULT_GAIN,
    max_dim: int = DEFAULT_CROP_MAX_DIM,
    labels: tuple[str, str, str] = ("reference", "current", "difference"),
) -> dict[str, Any]:
    """Write a zoomed reference / current / difference triptych for one block.

    This is the artifact that shows *what* changed rather than only where.
    """
    ref = np.asarray(reference, dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32)

    y0, y1 = int(block["y0"]), int(block["y1"])
    x0, x1 = int(block["x0"]), int(block["x1"])

    ref_crop = ref[y0:y1, x0:x1]
    cur_crop = cur[y0:y1, x0:x1]
    diff_crop = np.max(np.abs(cur_crop - ref_crop), axis=2)

    ref_img = Image.fromarray(np.clip(ref_crop, 0, 255).astype(np.uint8), mode="RGB")
    cur_img = Image.fromarray(np.clip(cur_crop, 0, 255).astype(np.uint8), mode="RGB")
    diff_img = Image.fromarray(
        _apply_colormap(np.clip(diff_crop * gain, 0.0, 255.0).astype(np.uint8)),
        mode="RGB",
    )

    crop_h, crop_w = diff_crop.shape
    zoom = max(1, int(np.floor(max_dim / max(crop_h, crop_w))))
    size = (crop_w * zoom, crop_h * zoom)
    panels = [
        img.resize(size, Image.NEAREST) for img in (ref_img, cur_img, diff_img)
    ]

    canvas = Image.new(
        "RGB",
        (size[0] * 3 + _GUTTER * 2, size[1] + _LABEL_HEIGHT),
        (18, 22, 28),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * (size[0] + _GUTTER)
        canvas.paste(panel, (x, _LABEL_HEIGHT))
        draw.text((x + 2, 3), label, fill=(230, 237, 243))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)

    return {
        "path": str(output),
        "region": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "zoom": zoom,
        "gain": gain,
        "panels": list(labels),
    }


def _magnitude(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32)

    if ref.shape != cur.shape:
        raise ValueError(f"Shape mismatch: reference {ref.shape} vs current {cur.shape}")

    abs_diff = np.abs(cur - ref)
    return np.max(abs_diff, axis=2) if abs_diff.ndim == 3 else abs_diff


def _max_pool(array: np.ndarray, factor: int) -> np.ndarray:
    height, width = array.shape
    pad_h = (-height) % factor
    pad_w = (-width) % factor
    if pad_h or pad_w:
        array = np.pad(array, ((0, pad_h), (0, pad_w)), mode="edge")
    pooled = array.reshape(
        array.shape[0] // factor, factor, array.shape[1] // factor, factor
    )
    return pooled.max(axis=(1, 3))


def _build_colormap_lut() -> np.ndarray:
    positions = np.array([point[0] for point in _COLORMAP_CONTROL_POINTS])
    colors = np.array([point[1] for point in _COLORMAP_CONTROL_POINTS], dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.uint8)
    for channel in range(3):
        # Round rather than truncate so the ramp matches the browser tool exactly.
        interpolated = np.interp(ramp, positions, colors[:, channel])
        lut[:, channel] = np.rint(interpolated).astype(np.uint8)
    return lut


_COLORMAP_LUT = _build_colormap_lut()


def _apply_colormap(gray: np.ndarray) -> np.ndarray:
    return _COLORMAP_LUT[gray]


def _compose_with_legend(
    colored: np.ndarray,
    *,
    gain: float,
    saturation_value: float,
    threshold: float | None,
    pooled_factor: int,
) -> Image.Image:
    body = Image.fromarray(colored, mode="RGB")
    width = max(body.width, 320)
    canvas = Image.new("RGB", (width, body.height + _LEGEND_HEIGHT), (18, 22, 28))
    canvas.paste(body, ((width - body.width) // 2, 0))

    bar_height = 10
    bar_y = body.height + 6
    ramp = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (bar_height, 1))
    canvas.paste(Image.fromarray(_apply_colormap(ramp), mode="RGB"), (0, bar_y))

    draw = ImageDraw.Draw(canvas)
    caption = f"0.00    absolute difference (gain {gain:g})    {saturation_value:.2f}+"
    if threshold is not None:
        caption += f"   |   threshold {threshold:g}"
    if pooled_factor > 1:
        caption += f"   |   max-pooled {pooled_factor}x"
    draw.text((3, bar_y + bar_height + 3), caption, fill=(200, 209, 217))
    return canvas
