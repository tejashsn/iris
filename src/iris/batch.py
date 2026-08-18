"""Batch comparison across directories of benchmark images."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from iris.report import SCHEMA_VERSION

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".npy"}

# Strip a leading NN_ index and match on the model portion before __seed_.
DEFAULT_PAIR_KEY_PATTERN = r"^\d+_(?P<model>.+)__seed_.+$"

BATCH_CSV_COLUMNS = (
    "model",
    "bitwise_identical",
    "mean_abs",
    "mean_signed",
    "p99_9",
    "max_abs",
    "pct_over_t",
    "similarity_pct",
    "error",
)


@dataclass
class PairPaths:
    pair_key: str
    model: str
    reference: Path
    current: Path
    baseline: Path | None = None


@dataclass
class DirectoryIndex:
    by_key: dict[str, Path]
    duplicate_keys: set[str]


def pair_key_from_stem(stem: str, pattern: str = DEFAULT_PAIR_KEY_PATTERN) -> str:
    """Derive a cross-directory pair key from a filename stem."""
    match = re.match(pattern, stem)
    if match:
        model = match.groupdict().get("model")
        if model:
            return model
    return stem


def list_image_files(directory: Path) -> list[Path]:
    """List supported image files in a directory (non-recursive)."""
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return files


def index_directory(
    directory: Path,
    *,
    pattern: str = DEFAULT_PAIR_KEY_PATTERN,
) -> DirectoryIndex:
    """Map pair keys to files, recording ambiguous duplicate keys."""
    by_key: dict[str, Path] = {}
    duplicate_keys: set[str] = set()

    for path in list_image_files(directory):
        key = pair_key_from_stem(path.stem, pattern)
        if key in by_key:
            duplicate_keys.add(key)
        else:
            by_key[key] = path

    for key in duplicate_keys:
        by_key.pop(key, None)

    return DirectoryIndex(by_key=by_key, duplicate_keys=duplicate_keys)


def discover_pairs(
    reference_dir: Path,
    current_dir: Path,
    baseline_dir: Path | None = None,
    *,
    pattern: str = DEFAULT_PAIR_KEY_PATTERN,
) -> tuple[list[PairPaths], dict[str, list[str]]]:
    """Pair files across directories and collect unmatched names."""
    ref_index = index_directory(reference_dir, pattern=pattern)
    cur_index = index_directory(current_dir, pattern=pattern)
    base_index = (
        index_directory(baseline_dir, pattern=pattern) if baseline_dir is not None else None
    )

    unmatched: dict[str, list[str]] = {
        "reference": [],
        "current": [],
        "baseline": [],
    }

    for key in sorted(ref_index.duplicate_keys):
        unmatched["reference"].append(f"<duplicate key: {key}>")
    for key in sorted(cur_index.duplicate_keys):
        unmatched["current"].append(f"<duplicate key: {key}>")

    ref_keys = set(ref_index.by_key)
    cur_keys = set(cur_index.by_key)
    matched_keys = sorted(ref_keys & cur_keys)

    for key in sorted(ref_keys - cur_keys):
        unmatched["reference"].append(ref_index.by_key[key].name)
    for key in sorted(cur_keys - ref_keys):
        unmatched["current"].append(cur_index.by_key[key].name)

    if base_index is not None:
        base_keys = set(base_index.by_key)
        for key in sorted(base_index.duplicate_keys):
            unmatched["baseline"].append(f"<duplicate key: {key}>")
        paired_keys = set(matched_keys)
        for key in sorted(base_keys - paired_keys):
            unmatched["baseline"].append(base_index.by_key[key].name)

    pairs: list[PairPaths] = []
    for key in matched_keys:
        baseline_path = None
        if base_index is not None:
            baseline_path = base_index.by_key.get(key)
        pairs.append(
            PairPaths(
                pair_key=key,
                model=key,
                reference=ref_index.by_key[key],
                current=cur_index.by_key[key],
                baseline=baseline_path,
            )
        )

    return pairs, unmatched


def safe_report_stem(pair_key: str) -> str:
    """Filesystem-safe stem for per-pair report filenames."""
    cleaned = re.sub(r"[^\w\-.]+", "_", pair_key).strip("._")
    return cleaned or "pair"


def build_batch_summary(
    *,
    pairs: list[dict[str, Any]],
    unmatched: dict[str, list[str]],
    three_way: bool,
    reference_dir: Path,
    current_dir: Path,
    baseline_dir: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    compared = [item for item in pairs if item.get("error") is None]
    identical = [item for item in compared if item.get("bitwise_identical")]
    errors = [item for item in pairs if item.get("error") is not None]

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "batch",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "three_way": three_way,
        "input_dirs": {
            "reference": str(reference_dir),
            "current": str(current_dir),
            **({"baseline": str(baseline_dir)} if baseline_dir is not None else {}),
        },
        "output_dir": str(output_dir),
        "counts": {
            "pairs_scheduled": len(pairs),
            "pairs_compared": len(compared),
            "bitwise_identical": len(identical),
            "differ": len(compared) - len(identical),
            "errors": len(errors),
            "unmatched_reference": len(unmatched.get("reference", [])),
            "unmatched_current": len(unmatched.get("current", [])),
            "unmatched_baseline": len(unmatched.get("baseline", [])),
        },
        "unmatched": unmatched,
        "pairs": pairs,
    }


def write_batch_summary_json(summary: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_batch_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in BATCH_CSV_COLUMNS})
    return path


def csv_row_from_pair_result(result: dict[str, Any]) -> dict[str, Any]:
    """Build one CSV row from a per-pair batch result record."""
    metrics = result.get("metrics") or {}
    error = result.get("error")

    def _metric(name: str) -> Any:
        if error is not None:
            return ""
        return metrics.get(name, "")

    return {
        "model": result.get("model", ""),
        "bitwise_identical": _metric("bitwise_identical"),
        "mean_abs": _metric("mean_abs"),
        "mean_signed": _metric("mean_signed"),
        "p99_9": _metric("p99_9"),
        "max_abs": _metric("max_abs"),
        "pct_over_t": _metric("pct_over_t"),
        "similarity_pct": _metric("similarity_pct"),
        "error": error or "",
    }


def render_batch_stdout_summary(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        (
            f"Batch comparison: {counts['pairs_compared']} pairs compared, "
            f"{counts['bitwise_identical']} identical, "
            f"{counts['differ']} differ, "
            f"{counts['errors']} errors."
        )
    ]

    for side in ("reference", "current", "baseline"):
        names = summary["unmatched"].get(side) or []
        if names:
            preview = ", ".join(names[:5])
            suffix = f" (+{len(names) - 5} more)" if len(names) > 5 else ""
            lines.append(f"Unmatched {side}: {len(names)} ({preview}{suffix})")

    return "\n".join(lines) + "\n"


def run_batch(
    *,
    reference_dir: Path,
    current_dir: Path,
    baseline_dir: Path | None,
    output_dir: Path,
    pair_key_pattern: str,
    run_pair: Callable[[PairPaths, Path], dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    """Discover pairs, run ``run_pair`` for each, and write batch artifacts."""
    pairs, unmatched = discover_pairs(
        reference_dir,
        current_dir,
        baseline_dir,
        pattern=pair_key_pattern,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    exit_code = 0

    for pair in pairs:
        report_path = output_dir / f"{safe_report_stem(pair.pair_key)}.json"
        result = run_pair(pair, report_path)
        pair_results.append(result)
        csv_rows.append(csv_row_from_pair_result(result))
        if result.get("gate_verdict") == "FAIL":
            exit_code = 1

    summary = build_batch_summary(
        pairs=pair_results,
        unmatched=unmatched,
        three_way=baseline_dir is not None,
        reference_dir=reference_dir,
        current_dir=current_dir,
        baseline_dir=baseline_dir,
        output_dir=output_dir,
    )

    write_batch_summary_json(summary, output_dir / "batch_summary.json")
    write_batch_csv(csv_rows, output_dir / "batch_summary.csv")

    return summary, exit_code
