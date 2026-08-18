"""Tests for batch directory pairing and batch CLI runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from iris.batch import (
    BATCH_CSV_COLUMNS,
    discover_pairs,
    pair_key_from_stem,
    safe_report_stem,
)
from iris.cli import main


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _textured(seed: int = 0, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(20.0, 200.0, size=(size, size, 3)).astype(np.float32)


def test_pair_key_strips_index_and_seed_suffix():
    stem = "02_comfyui_sd3_medium_fp16__seed_2383469069393109076"
    assert pair_key_from_stem(stem) == "comfyui_sd3_medium_fp16"


def test_pair_key_falls_back_to_exact_stem():
    assert pair_key_from_stem("plain_reference") == "plain_reference"


def test_discover_pairs_matches_different_seed_filenames(tmp_path):
    ref_dir = tmp_path / "ref"
    cur_dir = tmp_path / "cur"
    ref_dir.mkdir()
    cur_dir.mkdir()

    base = _textured(1)
    _write_png(
        ref_dir / "02_comfyui_sd3_medium_fp16__seed_111.png",
        base,
    )
    _write_png(
        cur_dir / "05_comfyui_sd3_medium_fp16__seed_222.png",
        base + 1,
    )

    pairs, unmatched = discover_pairs(ref_dir, cur_dir)

    assert len(pairs) == 1
    assert pairs[0].pair_key == "comfyui_sd3_medium_fp16"
    assert unmatched == {"reference": [], "current": [], "baseline": []}


def test_discover_pairs_reports_unmatched_on_both_sides(tmp_path):
    ref_dir = tmp_path / "ref"
    cur_dir = tmp_path / "cur"
    ref_dir.mkdir()
    cur_dir.mkdir()

    base = _textured(2)
    _write_png(ref_dir / "01_model_a__seed_1.png", base)
    _write_png(cur_dir / "01_model_a__seed_2.png", base)
    _write_png(ref_dir / "orphan_ref.png", base)
    _write_png(cur_dir / "orphan_cur.png", base + 1)

    pairs, unmatched = discover_pairs(ref_dir, cur_dir)

    assert len(pairs) == 1
    assert unmatched["reference"] == ["orphan_ref.png"]
    assert unmatched["current"] == ["orphan_cur.png"]


def test_batch_cli_pairs_and_writes_summary(tmp_path, capsys):
    ref_dir = tmp_path / "ref"
    cur_dir = tmp_path / "cur"
    out_dir = tmp_path / "out"
    ref_dir.mkdir()
    cur_dir.mkdir()

    identical = _textured(3)
    different = _textured(4)
    _write_png(ref_dir / "01_model_a__seed_1.png", identical)
    _write_png(cur_dir / "02_model_a__seed_9.png", identical.copy())
    _write_png(ref_dir / "01_model_b__seed_1.png", different)
    _write_png(cur_dir / "02_model_b__seed_9.png", different + 2)

    code = main(
        [
            "--reference-dir", str(ref_dir),
            "--current-dir", str(cur_dir),
            "--threshold", "1.0",
            "--out", str(out_dir),
        ]
    )

    assert code == 0
    summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "batch"
    assert summary["counts"]["pairs_compared"] == 2
    assert summary["counts"]["bitwise_identical"] == 1
    assert summary["counts"]["differ"] == 1

    stdout = capsys.readouterr().out
    assert "2 pairs compared" in stdout
    assert "1 identical" in stdout
    assert "1 differ" in stdout

    assert (out_dir / f"{safe_report_stem('model_a')}.json").exists()
    assert (out_dir / f"{safe_report_stem('model_b')}.json").exists()


def test_batch_cli_mid_pair_failure_does_not_stop_run(tmp_path):
    ref_dir = tmp_path / "ref"
    cur_dir = tmp_path / "cur"
    out_dir = tmp_path / "out"
    ref_dir.mkdir()
    cur_dir.mkdir()

    good = _textured(5)
    bad_ref = _textured(6, size=32)
    bad_cur = _textured(7, size=48)

    _write_png(ref_dir / "01_model_ok__seed_1.png", good)
    _write_png(cur_dir / "01_model_ok__seed_2.png", good + 1)
    _write_png(ref_dir / "01_model_bad__seed_1.png", bad_ref)
    _write_png(cur_dir / "01_model_bad__seed_2.png", bad_cur)

    code = main(
        [
            "--reference-dir", str(ref_dir),
            "--current-dir", str(cur_dir),
            "--threshold", "1.0",
            "--out", str(out_dir),
        ]
    )

    assert code == 0
    summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["pairs_compared"] == 1
    assert summary["counts"]["errors"] == 1

    by_model = {item["model"]: item for item in summary["pairs"]}
    assert by_model["model_ok"]["error"] is None
    assert "Shape mismatch" in by_model["model_bad"]["error"]
    assert (out_dir / f"{safe_report_stem('model_ok')}.json").exists()
    assert not (out_dir / f"{safe_report_stem('model_bad')}.json").exists()


def test_batch_csv_schema(tmp_path):
    ref_dir = tmp_path / "ref"
    cur_dir = tmp_path / "cur"
    out_dir = tmp_path / "out"
    ref_dir.mkdir()
    cur_dir.mkdir()

    base = _textured(8)
    _write_png(ref_dir / "01_model_a__seed_1.png", base)
    _write_png(cur_dir / "01_model_a__seed_2.png", base + 1)
    _write_png(ref_dir / "01_model_b__seed_1.png", base)
    _write_png(cur_dir / "01_model_b__seed_2.png", base + 40)

    main(
        [
            "--reference-dir", str(ref_dir),
            "--current-dir", str(cur_dir),
            "--threshold", "1.0",
            "--out", str(out_dir),
        ]
    )

    csv_path = out_dir / "batch_summary.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(BATCH_CSV_COLUMNS)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["model"] == "model_a"
    assert rows[0]["error"] == ""
    assert rows[0]["mean_abs"] != ""
    assert rows[1]["model"] == "model_b"


def test_batch_cli_lists_unmatched_files(tmp_path, capsys):
    ref_dir = tmp_path / "ref"
    cur_dir = tmp_path / "cur"
    out_dir = tmp_path / "out"
    ref_dir.mkdir()
    cur_dir.mkdir()

    base = _textured(9)
    _write_png(ref_dir / "01_model_a__seed_1.png", base)
    _write_png(cur_dir / "01_model_a__seed_2.png", base)
    _write_png(ref_dir / "leftover_ref.png", base)

    main(
        [
            "--reference-dir", str(ref_dir),
            "--current-dir", str(cur_dir),
            "--threshold", "1.0",
            "--out", str(out_dir),
        ]
    )

    summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["unmatched"]["reference"] == ["leftover_ref.png"]

    stdout = capsys.readouterr().out
    assert "Unmatched reference: 1" in stdout
    assert "leftover_ref.png" in stdout
