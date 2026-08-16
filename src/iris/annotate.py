"""Render a detection overlay: the image with boxes plus a present/absent side list.

This is the artifact from the mockup — the picture with dashed-style boxes around
detected objects and a legend column listing each prompt element as detected or
missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# Distinct box colours cycled per element, roughly matching the mockup palette.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (240, 173, 45),   # amber
    (61, 139, 253),   # blue
    (63, 185, 80),    # green
    (163, 113, 247),  # purple
    (240, 105, 105),  # red
    (54, 200, 200),   # teal
)

_PRESENT_COLOR = (63, 185, 80)
_MISSING_COLOR = (139, 148, 158)
_PANEL_BG = (18, 22, 28)
_TEXT = (230, 237, 243)
_LEGEND_WIDTH = 240
_ROW_HEIGHT = 30
_PAD = 12


def write_detection_overlay(
    image: np.ndarray,
    detection: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Draw detected boxes on the image and a present/absent legend beside it."""
    base = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
    elements = detection.get("elements", [])

    legend_height = _PAD * 2 + _ROW_HEIGHT * (len(elements) + 1)
    canvas_height = max(base.height, legend_height)
    canvas = Image.new("RGB", (base.width + _LEGEND_WIDTH, canvas_height), _PANEL_BG)
    canvas.paste(base, (0, 0))

    draw = ImageDraw.Draw(canvas)

    for index, element in enumerate(elements):
        color = _PALETTE[index % len(_PALETTE)]
        for box in element.get("boxes", []):
            _draw_box(draw, box, color)
            label = f"{element['element']} {box['score']:.2f}"
            _draw_label(draw, box["x0"], box["y0"], label, color)

    _draw_legend(draw, base.width, elements)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)

    return {
        "path": str(output),
        "backend": detection.get("backend"),
        "element_count": len(elements),
        "present": [e["element"] for e in elements if e["present"]],
        "missing": [e["element"] for e in elements if not e["present"]],
    }


def _draw_box(draw: ImageDraw.ImageDraw, box: dict[str, float], color) -> None:
    x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)


def _draw_label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color) -> None:
    ty = max(0, y - 14)
    pad = 2
    try:
        bbox = draw.textbbox((x, ty), text)
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=color)
    except Exception:
        pass
    draw.text((x, ty), text, fill=(10, 12, 16))


def _draw_legend(draw: ImageDraw.ImageDraw, x_offset: int, elements: list[dict[str, Any]]) -> None:
    x = x_offset + _PAD
    y = _PAD
    draw.text((x, y), "Prompt elements", fill=_TEXT)
    y += _ROW_HEIGHT

    for index, element in enumerate(elements):
        swatch = _PALETTE[index % len(_PALETTE)]
        present = element["present"]
        draw.rectangle([x, y + 4, x + 14, y + 18], outline=swatch, width=2)
        mark = "detected" if present else "MISSING"
        mark_color = _PRESENT_COLOR if present else _MISSING_COLOR
        name = element["element"]
        conf = element.get("confidence", 0.0)
        draw.text((x + 22, y), name, fill=_TEXT)
        detail = f"{mark} ({conf:.2f})" if present else mark
        draw.text((x + 22, y + 13), detail, fill=mark_color)
        y += _ROW_HEIGHT
