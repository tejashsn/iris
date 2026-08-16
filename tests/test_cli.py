"""End-to-end CLI tests."""

import json

import numpy as np
from PIL import Image, PngImagePlugin

from iris.cli import main
from tests.test_provenance import _comfy_graph


def _write_png(path, array: np.ndarray, graph: dict | None = None) -> None:
    info = PngImagePlugin.PngInfo()
    if graph is not None:
        info.add_text("prompt", json.dumps(graph))
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB").save(
        path, pnginfo=info
    )


def _textured(seed: int = 0, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(20.0, 200.0, size=(size, size, 3))


def test_report_only_run_without_limits(tmp_path):
    base = _textured()
    _write_png(tmp_path / "ref.png", base)
    _write_png(tmp_path / "cur.png", base + 1)

    out = tmp_path / "report.json"
    code = main(
        [
            "--reference", str(tmp_path / "ref.png"),
            "--current", str(tmp_path / "cur.png"),
            "--threshold", "1.0",
            "--out", str(out),
        ]
    )

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gate"]["verdict"] == "REPORT_ONLY"
    assert report["summary"]["headline"]
    assert out.with_suffix(".txt").exists()


def test_gate_exit_returns_one_on_failure(tmp_path):
    base = _textured()
    _write_png(tmp_path / "ref.png", base)
    _write_png(tmp_path / "cur.png", base + 40)

    code = main(
        [
            "--reference", str(tmp_path / "ref.png"),
            "--current", str(tmp_path / "cur.png"),
            "--threshold", "1.0",
            "--max-mean-abs", "1.0",
            "--gate-exit",
            "--out", str(tmp_path / "report.json"),
        ]
    )

    assert code == 1


def test_provenance_mismatch_surfaced_in_report(tmp_path):
    base = _textured()
    _write_png(tmp_path / "ref.png", base, _comfy_graph(seed=1))
    _write_png(tmp_path / "cur.png", base + 1, _comfy_graph(seed=2))

    out = tmp_path / "report.json"
    main(
        [
            "--reference", str(tmp_path / "ref.png"),
            "--current", str(tmp_path / "cur.png"),
            "--threshold", "1.0",
            "--out", str(out),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["provenance_diff"]["matches"] is False
    assert report["prompt_adherence"]["prompt"] is not None
    assert "reference PNG metadata" in report["prompt_adherence"]["prompt_source"]

    findings = " ".join(report["summary"]["findings"])
    assert "NOT generated with the same configuration" in findings


def test_visuals_written_to_heatmap_dir(tmp_path):
    base = _textured(size=256)
    current = base.copy()
    current[:32, :32, :] = 255.0
    _write_png(tmp_path / "ref.png", base)
    _write_png(tmp_path / "cur.png", current)

    heatmap_dir = tmp_path / "visuals"
    out = tmp_path / "report.json"
    main(
        [
            "--reference", str(tmp_path / "ref.png"),
            "--current", str(tmp_path / "cur.png"),
            "--threshold", "1.0",
            "--heatmap-dir", str(heatmap_dir),
            "--out", str(out),
        ]
    )

    assert (heatmap_dir / "overview_reference_vs_current.png").exists()
    assert (heatmap_dir / "worst_region_1.png").exists()

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["heatmaps"]["overview"]["legend"]
    assert report["heatmaps"]["worst_regions"]


def test_three_way_run_writes_all_pair_maps(tmp_path):
    base = _textured()
    _write_png(tmp_path / "ref.png", base)
    _write_png(tmp_path / "lkg.png", base + 5)
    _write_png(tmp_path / "cur.png", base + 2)

    heatmap_dir = tmp_path / "visuals"
    out = tmp_path / "report.json"
    code = main(
        [
            "--reference", str(tmp_path / "ref.png"),
            "--baseline", str(tmp_path / "lkg.png"),
            "--current", str(tmp_path / "cur.png"),
            "--threshold", "1.0",
            "--heatmap-dir", str(heatmap_dir),
            "--out", str(out),
        ]
    )

    assert code == 0
    assert (heatmap_dir / "reference_vs_baseline.png").exists()
    assert (heatmap_dir / "baseline_vs_current.png").exists()

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["mode"] == "three_way"
    assert report["closer_to_reference_pct"] > 0


def test_no_band_context_flag_suppresses_context(tmp_path):
    base = _textured()
    _write_png(tmp_path / "ref.png", base)
    _write_png(tmp_path / "cur.png", base + 1)

    out = tmp_path / "report.json"
    main(
        [
            "--reference", str(tmp_path / "ref.png"),
            "--current", str(tmp_path / "cur.png"),
            "--threshold", "1.0",
            "--no-band-context",
            "--out", str(out),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["context"] == []


def test_detection_overlays_and_report_block(tmp_path):
    import iris.detect as detect

    class FakeDetector:
        name = "fake-cli"

        def is_available(self):
            return True

        def detect(self, image, queries, threshold):
            out = {q: [] for q in queries}
            if "sun" in out:
                out["sun"] = [{"x0": 2, "y0": 2, "x1": 20, "y1": 20, "score": 0.5}]
            return out

    detect.register_detect_backend("fake-cli", lambda: FakeDetector())

    base = _textured()
    _write_png(tmp_path / "ref.png", base)
    _write_png(tmp_path / "cur.png", base + 1)

    heatmap_dir = tmp_path / "visuals"
    out = tmp_path / "report.json"
    main(
        [
            "--reference", str(tmp_path / "ref.png"),
            "--current", str(tmp_path / "cur.png"),
            "--threshold", "1.0",
            "--prompt", "a sun and a bottle",
            "--detect-backend", "fake-cli",
            "--heatmap-dir", str(heatmap_dir),
            "--out", str(out),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    block = report["prompt_elements"]
    assert block["verdict"] == "ADVISORY"
    assert block["verdict"] not in {"PASS", "FAIL"}
    assert report["gate"]["verdict"] == "REPORT_ONLY"  # detection never changes the gate

    ref_elements = {e["element"]: e for e in block["per_image"]["reference"]["elements"]}
    assert ref_elements["sun"]["present"] is True
    assert ref_elements["bottle"]["present"] is False

    assert (heatmap_dir / "detection_reference.png").exists()
    assert (heatmap_dir / "detection_current.png").exists()


def test_prompt_override_reaches_the_report(tmp_path):
    base = _textured()
    _write_png(tmp_path / "ref.png", base)
    _write_png(tmp_path / "cur.png", base + 1)

    out = tmp_path / "report.json"
    main(
        [
            "--reference", str(tmp_path / "ref.png"),
            "--current", str(tmp_path / "cur.png"),
            "--threshold", "1.0",
            "--prompt", "a red umbrella on a wet street",
            "--out", str(out),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    adherence = report["prompt_adherence"]
    assert adherence["prompt"] == "a red umbrella on a wet street"
    assert adherence["verdict"] == "UNAVAILABLE"
    assert adherence["verdict"] not in {"PASS", "FAIL"}
