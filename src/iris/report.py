"""JSON and text report rendering.

The text report leads with the plain-language summary. Raw metrics follow for
readers who want them, but the first thing on screen is a sentence, not a number.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iris.narrative import build_narrative, render_narrative

SCHEMA_VERSION = "2.0.0"

CLOSENESS_NOTE = (
    "All percentage fields measure closeness to the reference image, not visual "
    "quality improvement. similarity_pct = 100 * (1 - mean_abs/255); "
    "within_t_pct is the share of samples within the threshold; "
    "closer_to_reference_pct (three-way only) indicates how much closer the current "
    "image is to the reference than the baseline."
)

NARRATIVE_NOTE = (
    "The summary and findings are descriptive interpretations of the measured "
    "numbers. Wording heuristics choose phrasing only; the gate verdict is driven "
    "exclusively by limits supplied by the caller."
)


def build_report(
    comparison: dict[str, Any],
    *,
    input_paths: dict[str, str],
    metadata: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    provenance_diff: dict[str, Any] | None = None,
    semantic: dict[str, Any] | None = None,
    prompt_elements: dict[str, Any] | None = None,
    heatmaps: dict[str, Any] | None = None,
    extra_notes: list[str] | None = None,
    include_band_context: bool = True,
) -> dict[str, Any]:
    notes = [CLOSENESS_NOTE, NARRATIVE_NOTE]
    if extra_notes:
        notes.extend(extra_notes)

    if any(not item["passed"] for item in comparison["sanity"]) and not comparison["gate"].get(
        "checks"
    ):
        notes.append(
            "Sanity check issues detected; annotate report before deriving thresholds."
        )

    narrative = build_narrative(
        comparison,
        provenance_diff=provenance_diff,
        semantic=semantic,
        prompt_elements=prompt_elements,
        include_band_context=include_band_context,
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": comparison["mode"],
        "labels": comparison["labels"],
        "input_paths": input_paths,
        "summary": narrative,
        "sanity": comparison["sanity"],
        "regions": comparison.get("regions"),
        "gate": comparison["gate"],
        "notes": notes,
    }

    if comparison["mode"] == "pairwise":
        report["metrics"] = comparison["metrics"]
    else:
        report["reference_vs_baseline"] = comparison["reference_vs_baseline"]
        report["reference_vs_current"] = comparison["reference_vs_current"]
        report["baseline_vs_current"] = comparison["baseline_vs_current"]
        report["closer_to_reference_pct"] = comparison["closer_to_reference_pct"]
        report["closer_to_reference_interpretation"] = comparison[
            "closer_to_reference_interpretation"
        ]

    if provenance is not None:
        report["provenance"] = provenance
    if provenance_diff is not None:
        report["provenance_diff"] = provenance_diff
    if semantic is not None:
        report["prompt_adherence"] = semantic
    if prompt_elements is not None:
        report["prompt_elements"] = prompt_elements
    if heatmaps is not None:
        report["heatmaps"] = heatmaps
    if metadata is not None:
        report["metadata"] = metadata

    return report


def write_json_report(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def render_text_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("IRIS - Image Regression Inspection Suite")
    lines.append("=" * 72)
    lines.append("")

    verdict = report["gate"]["verdict"]
    lines.append(f"VERDICT: {verdict}")
    lines.append("")

    lines.append(render_narrative(report["summary"]))

    if "prompt_adherence" in report:
        lines.extend(_format_prompt_adherence(report["prompt_adherence"]))

    if "prompt_elements" in report:
        lines.extend(_format_prompt_elements(report["prompt_elements"]))

    if report.get("regions"):
        lines.extend(_format_regions(report["regions"]))

    if report.get("heatmaps"):
        lines.extend(_format_heatmaps(report["heatmaps"]))

    lines.append("DETAIL")
    lines.append("-" * 72)
    lines.append(f"schema_version: {report['schema_version']}")
    lines.append(f"timestamp_utc: {report['timestamp_utc']}")
    lines.append(f"mode: {report['mode']}")
    lines.append("")
    lines.append("Labels:")
    for key, value in report["labels"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Input paths:")
    for key, value in report["input_paths"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Sanity:")
    for item in report["sanity"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"  [{status}] {item['label']}: {', '.join(item['issues']) or 'ok'}")
    lines.append("")

    if report["mode"] == "pairwise":
        lines.extend(_format_metrics_block("Metrics", report["metrics"]))
    else:
        lines.extend(_format_metrics_block("reference_vs_baseline", report["reference_vs_baseline"]))
        lines.extend(_format_metrics_block("reference_vs_current", report["reference_vs_current"]))
        lines.extend(_format_metrics_block("baseline_vs_current", report["baseline_vs_current"]))
        lines.append(f"closer_to_reference_pct: {report['closer_to_reference_pct']}")
        lines.append("")

    lines.append("Gate checks:")
    if report["gate"].get("checks"):
        for check in report["gate"]["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            lines.append(
                f"  [{status}] {check['name']}: actual={check['actual']}, "
                f"limit={check['limit']}"
            )
    else:
        lines.append("  none configured")
    for note in report["gate"].get("notes", []):
        lines.append(f"  note: {note}")

    lines.append("")
    lines.append("Notes:")
    for note in report["notes"]:
        lines.append(f"  - {note}")

    return "\n".join(lines) + "\n"


def write_text_report(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_text_summary(report), encoding="utf-8")
    return output


def _format_prompt_adherence(semantic: dict[str, Any]) -> list[str]:
    lines = ["PROMPT ADHERENCE (advisory only - never affects the verdict)", "-" * 72]
    lines.append(f"backend: {semantic['backend']}   verdict: {semantic['verdict']}")
    lines.append(f"prompt source: {semantic['prompt_source']}")
    if semantic.get("prompt"):
        lines.append(f"prompt: {semantic['prompt']}")

    for role, result in semantic.get("per_image", {}).items():
        if result.get("score") is None:
            lines.append(f"  {role}: no score available")
            continue
        lines.append(f"  {role}: overall similarity {result['score']:.4f}")
        for clause in result.get("clauses", [])[:5]:
            lines.append(f"      {clause['similarity']:.4f}  {clause['clause']}")

    for note in semantic.get("notes", []):
        lines.append(f"  note: {note}")
    lines.append("")
    return lines


def _format_prompt_elements(detection: dict[str, Any]) -> list[str]:
    lines = ["PROMPT ELEMENTS (advisory only - never affects the verdict)", "-" * 72]
    lines.append(f"backend: {detection['backend']}   verdict: {detection['verdict']}")
    if detection.get("elements"):
        lines.append(f"expected: {', '.join(detection['elements'])}")

    for role, result in detection.get("per_image", {}).items():
        lines.append(f"  {role}:")
        for element in result.get("elements", []):
            mark = "[x] detected" if element["present"] else "[ ] MISSING "
            conf = f"  (conf {element['confidence']:.2f})" if element["present"] else ""
            lines.append(f"      {mark}  {element['element']}{conf}")

    overlays = detection.get("overlays", {})
    for role, overlay in overlays.items():
        lines.append(f"  overlay ({role}): {overlay['path']}")

    for note in detection.get("notes", []):
        lines.append(f"  note: {note}")
    lines.append("")
    return lines


def _format_regions(regions: dict[str, Any]) -> list[str]:
    lines = ["WHERE THE DIFFERENCE IS", "-" * 72]
    lines.append(
        f"Block grid: {regions['grid']['rows']}x{regions['grid']['cols']} "
        f"of {regions['block_size']}px blocks over a "
        f"{regions['frame_shape']['width']}x{regions['frame_shape']['height']} frame."
    )
    if regions["total_abs"] <= 0:
        lines.append("No absolute difference to localise.")
        lines.append("")
        return lines

    lines.append("")
    lines.append(f"{'region (x0-x1, y0-y1)':<32}{'share':>9}{'mean':>10}{'max':>9}")
    for block in regions["worst_blocks"]:
        region = f"x {block['x0']}-{block['x1']}, y {block['y0']}-{block['y1']}"
        lines.append(
            f"{region:<32}{block['contribution_pct']:>8.2f}%"
            f"{block['mean_abs']:>10.3f}{block['max_abs']:>9.2f}"
        )
    lines.append("")
    return lines


def _format_heatmaps(heatmaps: dict[str, Any]) -> list[str]:
    lines = ["VISUALS", "-" * 72]
    overview = heatmaps.get("overview")
    if overview:
        lines.append(f"Overview: {overview['path']}")
        lines.append(f"  {overview['legend']}")
        if overview["downscale_factor"] > 1:
            lines.append(
                f"  Downscaled {overview['downscale_factor']}x by max-pooling so small "
                "hot spots survive the resize."
            )
    for crop in heatmaps.get("worst_regions", []):
        region = crop["region"]
        lines.append(
            f"Worst-region crop: {crop['path']} "
            f"(x {region['x0']}-{region['x1']}, y {region['y0']}-{region['y1']}, "
            f"{crop['zoom']}x zoom)"
        )
    for key in ("reference_vs_current", "reference_vs_baseline", "baseline_vs_current", "raw"):
        if key in heatmaps:
            lines.append(f"Full-resolution map ({key}): {heatmaps[key]}")
    lines.append("")
    return lines


def _format_metrics_block(title: str, metrics: dict[str, Any]) -> list[str]:
    pcm = metrics["per_channel_mean"]
    lines = [
        f"{title}:",
        f"  bitwise_identical: {metrics['bitwise_identical']}",
        f"  mean_abs: {metrics['mean_abs']}",
        f"  p99_9: {metrics['p99_9']}",
        f"  max_abs: {metrics['max_abs']}",
        f"  pct_over_t: {metrics['pct_over_t']}",
        f"  per_channel_mean: R={pcm['R']}, G={pcm['G']}, B={pcm['B']}",
        f"  similarity_pct: {metrics['similarity_pct']}",
        f"  within_t_pct: {metrics['within_t_pct']}",
        f"  sample_count: {metrics['sample_count']}",
        f"  threshold: {metrics['threshold']}",
        "",
    ]
    return lines
