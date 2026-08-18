"""CLI entry point for IRIS comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from iris.annotate import write_detection_overlay
from iris.batch import (
    DEFAULT_PAIR_KEY_PATTERN,
    PairPaths,
    render_batch_stdout_summary,
    run_batch,
)
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

    inputs = parser.add_argument_group("inputs", "Single-image or batch directory mode")
    inputs.add_argument("--reference", default=None, help="Reference image path")
    inputs.add_argument("--current", default=None, help="Current image path")
    inputs.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline image path (enables three-way mode)",
    )
    inputs.add_argument(
        "--reference-dir",
        default=None,
        help="Directory of reference images for batch mode",
    )
    inputs.add_argument(
        "--current-dir",
        default=None,
        help="Directory of current images for batch mode",
    )
    inputs.add_argument(
        "--baseline-dir",
        default=None,
        help="Optional baseline image directory (enables three-way per pair in batch mode)",
    )
    inputs.add_argument(
        "--pair-key-pattern",
        default=DEFAULT_PAIR_KEY_PATTERN,
        help=(
            "Regex with a 'model' named group used to pair files across directories "
            f"(default: {DEFAULT_PAIR_KEY_PATTERN!r}); falls back to exact stem match"
        ),
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
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON report path (single mode) or output directory (batch mode)",
    )
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


def _validate_mode(args: argparse.Namespace) -> bool:
    batch = args.reference_dir is not None or args.current_dir is not None
    single = args.reference is not None or args.current is not None

    if batch and single:
        raise SystemExit("Cannot mix single-file paths with --reference-dir/--current-dir.")
    if batch:
        if not args.reference_dir or not args.current_dir:
            raise SystemExit("Batch mode requires both --reference-dir and --current-dir.")
        if args.baseline and not args.baseline_dir:
            raise SystemExit("Use --baseline-dir (not --baseline) in batch mode.")
        if args.baseline:
            raise SystemExit("Cannot use --baseline together with batch directory mode.")
        return True
    if not args.reference or not args.current:
        raise SystemExit(
            "Single-image mode requires --reference and --current, "
            "or use --reference-dir and --current-dir for batch mode."
        )
    if args.baseline_dir:
        raise SystemExit("Use --baseline (not --baseline-dir) in single-image mode.")
    return False


def _gate_limits(args: argparse.Namespace) -> GateLimits:
    return GateLimits(
        max_mean_abs=args.max_mean_abs,
        max_p99_9=args.max_p99_9,
        max_max_abs=args.max_max_abs,
        max_pct_over_t=args.max_pct_over_t,
        min_similarity_pct=args.min_similarity_pct,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    batch_mode = _validate_mode(args)
    if batch_mode:
        return _run_batch_mode(args)
    return _run_single_mode(args)


def _run_single_mode(args: argparse.Namespace) -> int:
    report, exit_code = _execute_comparison(
        args,
        reference_path=Path(args.reference),
        current_path=Path(args.current),
        baseline_path=Path(args.baseline) if args.baseline else None,
        json_path=Path(args.out),
        write_text=True,
    )

    print(render_text_summary(report), end="")
    print(f"Wrote JSON report: {Path(args.out)}")
    text_path = args.text_out or str(Path(args.out).with_suffix(".txt"))
    print(f"Wrote text report: {text_path}")

    if args.gate_exit and exit_code:
        return 1
    return 0


def _run_batch_mode(args: argparse.Namespace) -> int:
    output_dir = Path(args.out)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    metadata = collect_metadata() if args.metadata else None

    def run_pair(pair: PairPaths, report_path: Path) -> dict[str, Any]:
        record: dict[str, Any] = {
            "pair_key": pair.pair_key,
            "model": pair.model,
            "reference": str(pair.reference),
            "current": str(pair.current),
            "report_path": str(report_path),
            "error": None,
            "bitwise_identical": None,
            "headline": None,
            "metrics": None,
            "gate_verdict": None,
        }
        if pair.baseline is not None:
            record["baseline"] = str(pair.baseline)

        if baseline_dir is not None and pair.baseline is None:
            record["error"] = (
                f"No baseline file matched pair key {pair.pair_key!r} in {baseline_dir}"
            )
            return record

        try:
            report, exit_code = _execute_comparison(
                args,
                reference_path=pair.reference,
                current_path=pair.current,
                baseline_path=pair.baseline,
                json_path=report_path,
                write_text=False,
                metadata=metadata,
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record

        metrics = _primary_metrics_from_report(report)
        record.update(
            {
                "bitwise_identical": metrics.get("bitwise_identical"),
                "headline": report["summary"]["headline"],
                "metrics": metrics,
                "gate_verdict": report["gate"]["verdict"],
                "gate_exit": exit_code,
            }
        )
        return record

    summary, exit_code = run_batch(
        reference_dir=Path(args.reference_dir),
        current_dir=Path(args.current_dir),
        baseline_dir=baseline_dir,
        output_dir=output_dir,
        pair_key_pattern=args.pair_key_pattern,
        run_pair=run_pair,
    )

    print(render_batch_stdout_summary(summary), end="")
    print(f"Wrote batch summary JSON: {output_dir / 'batch_summary.json'}")
    print(f"Wrote batch summary CSV: {output_dir / 'batch_summary.csv'}")

    if args.gate_exit and exit_code:
        return 1
    return 0


def _primary_metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    if report["mode"] == "pairwise":
        return report["metrics"]
    return report["reference_vs_current"]


def _execute_comparison(
    args: argparse.Namespace,
    *,
    reference_path: Path,
    current_path: Path,
    baseline_path: Path | None,
    json_path: Path,
    write_text: bool,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    reference = load_image(reference_path)
    current = load_image(current_path)
    baseline = load_image(baseline_path) if baseline_path is not None else None

    limits = _gate_limits(args)
    input_paths = {
        "reference": str(reference_path),
        "current": str(current_path),
    }
    if baseline is not None:
        input_paths["baseline"] = str(baseline_path)

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

    provenance, provenance_diff = _collect_provenance(args, reference_path, current_path, baseline_path)
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

    if metadata is None and args.metadata:
        metadata = collect_metadata()

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

    write_json_report(report, json_path)
    if write_text:
        text_path = args.text_out or str(json_path.with_suffix(".txt"))
        write_text_report(report, text_path)

    exit_code = 1 if report["gate"]["verdict"] == "FAIL" else 0
    return report, exit_code


def _collect_provenance(
    args: argparse.Namespace,
    reference_path: Path,
    current_path: Path,
    baseline_path: Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if args.no_provenance:
        return None, None

    entries = {
        "reference": extract_provenance(reference_path),
        "current": extract_provenance(current_path),
    }
    if baseline_path is not None:
        entries["baseline"] = extract_provenance(baseline_path)

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
