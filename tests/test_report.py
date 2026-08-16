"""Tests for report assembly and the summary-first text rendering."""

import json

import numpy as np

from iris.compare import GateLimits, compare_pairwise, compare_three_way
from iris.report import SCHEMA_VERSION, build_report, render_text_summary, write_json_report


def _textured(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(20.0, 200.0, size=(size, size, 3)).astype(np.float32)


def _report(**kwargs):
    reference = _textured(1)
    current = reference + 0.4
    comparison = compare_pairwise(reference, current, threshold=1.0, **kwargs.pop("compare", {}))
    return build_report(
        comparison,
        input_paths={"reference": "ref.png", "current": "cur.png"},
        **kwargs,
    )


def test_report_carries_schema_and_summary():
    report = _report()

    assert report["schema_version"] == SCHEMA_VERSION
    assert "headline" in report["summary"]
    assert report["summary"]["findings"]
    assert "regions" in report


def test_text_output_leads_with_verdict_and_summary():
    rendered = render_text_summary(_report())
    lines = [line for line in rendered.splitlines() if line.strip()]

    verdict_index = next(i for i, line in enumerate(lines) if line.startswith("VERDICT:"))
    summary_index = next(i for i, line in enumerate(lines) if line == "SUMMARY")
    detail_index = next(i for i, line in enumerate(lines) if line == "DETAIL")

    assert verdict_index < summary_index < detail_index


def test_text_output_still_contains_raw_metrics():
    rendered = render_text_summary(_report())

    for field in ("mean_abs", "p99_9", "max_abs", "pct_over_t", "similarity_pct"):
        assert field in rendered


def test_regions_section_lists_worst_blocks():
    reference = np.full((128, 128, 3), 100.0, dtype=np.float32)
    current = reference.copy()
    current[32:64, 64:96, :] = 0.0
    comparison = compare_pairwise(reference, current, threshold=1.0)
    report = build_report(comparison, input_paths={"reference": "r", "current": "c"})
    rendered = render_text_summary(report)

    assert "WHERE THE DIFFERENCE IS" in rendered
    assert "x 64-96, y 32-64" in rendered


def test_prompt_adherence_section_marked_advisory():
    semantic = {
        "verdict": "ADVISORY",
        "backend": "fake",
        "prompt": "a red umbrella",
        "prompt_source": "supplied via --prompt",
        "per_image": {"reference": {"score": 0.31, "clauses": []}},
        "divergence": None,
        "findings": [],
        "notes": ["advisory note"],
    }
    rendered = render_text_summary(_report(semantic=semantic))

    assert "advisory only - never affects the verdict" in rendered
    assert "0.3100" in rendered


def test_provenance_blocks_are_included():
    provenance = {"reference": {"available": True}}
    provenance_diff = {"comparable": True, "matches": True, "differences": [], "notes": []}
    report = _report(provenance=provenance, provenance_diff=provenance_diff)

    assert report["provenance"] == provenance
    assert report["provenance_diff"]["matches"] is True


def test_heatmap_section_explains_the_gain(tmp_path):
    heatmaps = {
        "overview": {
            "path": "overview.png",
            "legend": "Colour ramp spans 0 to 31.88 absolute difference",
            "downscale_factor": 4,
        },
        "worst_regions": [
            {"path": "crop.png", "region": {"x0": 0, "y0": 0, "x1": 32, "y1": 32}, "zoom": 4}
        ],
    }
    rendered = render_text_summary(_report(heatmaps=heatmaps))

    assert "VISUALS" in rendered
    assert "31.88 absolute difference" in rendered
    assert "max-pooling" in rendered


def test_three_way_report_has_all_blocks():
    reference = _textured(2)
    baseline = reference + 10.0
    current = reference + 5.0
    comparison = compare_three_way(reference, baseline, current, threshold=1.0)
    report = build_report(
        comparison,
        input_paths={"reference": "r", "baseline": "b", "current": "c"},
    )

    assert report["reference_vs_baseline"]
    assert report["reference_vs_current"]
    assert report["baseline_vs_current"]
    assert report["closer_to_reference_pct"] is not None


def test_json_report_round_trips(tmp_path):
    report = _report(compare={"limits": GateLimits(max_mean_abs=1.0)})
    path = write_json_report(report, tmp_path / "report.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["gate"]["verdict"] == "PASS"
    assert loaded["summary"]["headline"]
