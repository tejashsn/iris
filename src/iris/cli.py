"""CLI entry point for IRIS comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from iris.annotate import write_detection_overlay
from iris.compare import GateLimits, compare_pairwise, compare_three_way
from iris.detect import (
    DEFAULT_DETECT_THRESHOLD,
    available_detect_backends,
    evaluate_prompt_elements,
)
from iris.heatmap import (
    DEFAULT_CROP_MAX_DIM,
    DEFAULT_GAIN,
    DEFAULT_MAX_DIM,
    write_diff_heatmap,
    write_heatmap_overview,
    write_worst_region_crop,
)
from iris.loader import load_image
from iris.metadata import collect_metadata
from iris.provenance import diff_provenance, extract_provenance, resolve_prompt
from iris.regions import DEFAULT_BLOCK_SIZE
from iris.report import build_report, render_text_summary, write_json_report, write_text_report
from iris.semantic import available_backends, evaluate_prompt_adherence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iris-compare",
        description=(
            "Measure pixel-level drift between ComfyUI benchmark images. "
            "Metrics report closeness to a reference, not visual quality."
        ),
    )
    parser.add_argument("--reference", required=True, help="Reference image path")
    parser.add_argument("--current", required=True, help="Current image path")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline image path (enables three-way mode)",
    )
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--current-label", default="current")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Sample-difference threshold for pct_over_t (required; not a pass/fail cut)",
    )
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument(
        "--text-out",
        default=None,
        help="Optional text summary output path (defaults to --out with .txt suffix)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Block size for drift localisation (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--no-band-context",
        action="store_true",
        help="Omit prior-study drift bands from the narrative summary",
    )

    visuals = parser.add_argument_group("visuals")
    visuals.add_argument(
        "--heatmap",
        default=None,
        help="Optional full-resolution heatmap PNG path for the reference/current diff",
    )
    visuals.add_argument(
        "--heatmap-dir",
        default=None,
        help=(
            "Directory for readable visuals: colour-mapped overview with legend, "
            "worst-region crops, and per-pair maps in three-way mode"
        ),
    )
    visuals.add_argument(
        "--heatmap-gain",
        type=float,
        default=DEFAULT_GAIN,
        help=f"Heatmap visualization gain (default: {DEFAULT_GAIN:g})",
    )
    visuals.add_argument(
        "--heatmap-max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help=(
            "Longest edge of the downscaled overview; downscaling uses max-pooling so "
            f"small hot spots survive (default: {DEFAULT_MAX_DIM})"
        ),
    )
    visuals.add_argument(
        "--crop-max-dim",
        type=int,
        default=DEFAULT_CROP_MAX_DIM,
        help=f"Target panel size for worst-region crops (default: {DEFAULT_CROP_MAX_DIM})",
    )
    visuals.add_argument(
        "--worst-regions",
        type=int,
        default=2,
        help="Number of worst-region crops to write into --heatmap-dir (default: 2)",
    )

    semantic = parser.add_argument_group(
        "prompt adherence", "Advisory only; never contributes to PASS/FAIL"
    )
    semantic.add_argument(
        "--prompt",
        default=None,
        help="Prompt text to evaluate against; overrides embedded ComfyUI metadata",
    )
    semantic.add_argument(
        "--semantic-backend",
        default="none",
        help=f"Prompt-adherence backend. Available: {', '.join(available_backends())}",
    )
    semantic.add_argument(
        "--no-provenance",
        action="store_true",
        help="Skip reading ComfyUI generation metadata from PNG text chunks",
    )

    detect = parser.add_argument_group(
        "prompt-element detection", "Advisory only; spots which prompt objects are present"
    )
    detect.add_argument(
        "--detect-backend",
        default="none",
        help=f"Object-detection backend. Available: {', '.join(available_detect_backends())}",
    )
    detect.add_argument(
        "--detect-threshold",
        type=float,
        default=DEFAULT_DETECT_THRESHOLD,
        help=f"Detector score threshold for presence (default: {DEFAULT_DETECT_THRESHOLD})",
    )
    detect.add_argument(
        "--elements",
        default=None,
        help="Comma-separated object names to detect, overriding prompt extraction",
    )

    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Include environment metadata in the report",
    )

    gate = parser.add_argument_group("gate", "Optional pass/fail limits (caller-supplied only)")
    gate.add_argument("--max-mean-abs", type=float, default=None)
    gate.add_argument("--max-p99-9", type=float, default=None)
    gate.add_argument("--max-max-abs", type=float, default=None)
    gate.add_argument("--max-pct-over-t", type=float, default=None)
    gate.add_argument("--min-similarity-pct", type=float, default=None)
    gate.add_argument(
        "--gate-exit",
        action="store_true",
        help="Exit with code 1 when gate verdict is FAIL",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Prompts may contain non-ASCII text that legacy Windows consoles cannot encode.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    reference = load_image(args.reference)
    current = load_image(args.current)
    baseline = load_image(args.baseline) if args.baseline else None

    limits = GateLimits(
        max_mean_abs=args.max_mean_abs,
        max_p99_9=args.max_p99_9,
        max_max_abs=args.max_max_abs,
        max_pct_over_t=args.max_pct_over_t,
        min_similarity_pct=args.min_similarity_pct,
    )

    input_paths = {
        "reference": str(Path(args.reference)),
        "current": str(Path(args.current)),
    }
    if baseline is not None:
        input_paths["baseline"] = str(Path(args.baseline))

    if baseline is not None:
        comparison = compare_three_way(
            reference,
            baseline,
            current,
            threshold=args.threshold,
            reference_label=args.reference_label,
            baseline_label=args.baseline_label,
            current_label=args.current_label,
            limits=limits,
            block_size=args.block_size,
        )
    else:
        comparison = compare_pairwise(
            reference,
            current,
            threshold=args.threshold,
            reference_label=args.reference_label,
            current_label=args.current_label,
            limits=limits,
            block_size=args.block_size,
        )

    provenance, provenance_diff = _collect_provenance(args, baseline)
    prompt, prompt_source = resolve_prompt(args.prompt, provenance or {})

    images = {"reference": reference, "current": current}
    if baseline is not None:
        images["baseline"] = baseline

    semantic = evaluate_prompt_adherence(
        images,
        prompt,
        backend_name=args.semantic_backend,
        prompt_source=prompt_source,
    )

    element_list = (
        [e.strip() for e in args.elements.split(",") if e.strip()] if args.elements else None
    )
    prompt_elements = evaluate_prompt_elements(
        images,
        prompt,
        backend_name=args.detect_backend,
        prompt_source=prompt_source,
        elements=element_list,
        threshold=args.detect_threshold,
    )

    heatmaps = _write_visuals(args, reference, current, baseline, comparison)
    _write_detection_overlays(args, images, prompt_elements)
    metadata = collect_metadata() if args.metadata else None

    report = build_report(
        comparison,
        input_paths=input_paths,
        metadata=metadata,
        provenance=provenance,
        provenance_diff=provenance_diff,
        semantic=semantic,
        prompt_elements=prompt_elements,
        heatmaps=heatmaps or None,
        include_band_context=not args.no_band_context,
    )

    json_path = write_json_report(report, args.out)
    text_path = args.text_out or str(Path(args.out).with_suffix(".txt"))
    write_text_report(report, text_path)

    print(render_text_summary(report), end="")
    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote text report: {text_path}")

    if args.gate_exit and report["gate"]["verdict"] == "FAIL":
        return 1
    return 0


def _collect_provenance(
    args: argparse.Namespace,
    baseline: np.ndarray | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if args.no_provenance:
        return None, None

    entries = {
        "reference": extract_provenance(args.reference),
        "current": extract_provenance(args.current),
    }
    if baseline is not None:
        entries["baseline"] = extract_provenance(args.baseline)

    return entries, diff_provenance(entries)


def _write_detection_overlays(
    args: argparse.Namespace,
    images: dict[str, np.ndarray],
    prompt_elements: dict[str, Any],
) -> None:
    if not args.heatmap_dir or prompt_elements.get("verdict") != "ADVISORY":
        return

    out_dir = Path(args.heatmap_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays = {}
    for role, detection in prompt_elements.get("per_image", {}).items():
        if role not in images:
            continue
        overlays[role] = write_detection_overlay(
            images[role], detection, out_dir / f"detection_{role}.png"
        )
    if overlays:
        prompt_elements["overlays"] = overlays


def _write_visuals(
    args: argparse.Namespace,
    reference: np.ndarray,
    current: np.ndarray,
    baseline: np.ndarray | None,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    heatmaps: dict[str, Any] = {}

    if args.heatmap:
        heatmaps["raw"] = str(
            write_diff_heatmap(reference, current, args.heatmap, gain=args.heatmap_gain)
        )

    if not args.heatmap_dir:
        return heatmaps

    out_dir = Path(args.heatmap_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmaps["overview"] = write_heatmap_overview(
        reference,
        current,
        out_dir / "overview_reference_vs_current.png",
        gain=args.heatmap_gain,
        max_dim=args.heatmap_max_dim,
        threshold=args.threshold,
    )

    worst_blocks = (comparison.get("regions") or {}).get("worst_blocks", [])
    crops = []
    for index, block in enumerate(worst_blocks[: max(0, args.worst_regions)]):
        if block["max_abs"] <= 0:
            continue
        crops.append(
            write_worst_region_crop(
                reference,
                current,
                block,
                out_dir / f"worst_region_{index + 1}.png",
                gain=args.heatmap_gain,
                max_dim=args.crop_max_dim,
                labels=(args.reference_label, args.current_label, "difference"),
            )
        )
    if crops:
        heatmaps["worst_regions"] = crops

    if baseline is not None:
        heatmaps["reference_vs_baseline"] = str(
            write_diff_heatmap(
                reference,
                baseline,
                out_dir / "reference_vs_baseline.png",
                gain=args.heatmap_gain,
            )
        )
        heatmaps["baseline_vs_current"] = str(
            write_diff_heatmap(
                baseline,
                current,
                out_dir / "baseline_vs_current.png",
                gain=args.heatmap_gain,
            )
        )

    return heatmaps


if __name__ == "__main__":
    sys.exit(main())
