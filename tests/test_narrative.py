"""Tests for the plain-language narrative layer."""

import numpy as np
import pytest

from iris.compare import GateLimits, compare_pairwise, compare_three_way
from iris.narrative import (
    OBSERVED_DIFFERENT_SEED_MEAN_ABS,
    OBSERVED_HEALTHY_MEAN_ABS,
    build_narrative,
    render_narrative,
)


def _textured(seed: int = 0, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(20.0, 200.0, size=(size, size, 3)).astype(np.float32)


def test_bitwise_identical_headline_says_so():
    reference = _textured(1)
    comparison = compare_pairwise(reference, reference.copy(), threshold=1.0)
    narrative = build_narrative(comparison)

    assert "bitwise identical" in narrative["headline"]
    assert any("Nothing to investigate" in step for step in narrative["next_steps"])


def test_headline_mentions_tail_not_just_mean():
    reference = _textured(2)
    current = reference + 0.4
    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison)

    assert "99.9th percentile" in narrative["headline"]
    assert "0-255 scale" in narrative["headline"]


def test_localized_fault_described_as_concentrated():
    reference = np.full((256, 256, 3), 128.0, dtype=np.float32)
    current = reference.copy()
    current[:32, :32, :] = 0.0

    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison)
    text = " ".join(narrative["findings"])

    assert "concentrated in a small number of samples" in text
    assert "localised fault" in text


def test_uniform_drift_described_as_spread():
    reference = _textured(3)
    current = reference + 0.5
    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison)
    text = " ".join(narrative["findings"])

    assert "distributed fairly evenly" in text


def test_single_channel_shift_called_out():
    reference = _textured(4)
    current = reference.copy()
    current[..., 0] += 3.0

    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison)
    text = " ".join(narrative["findings"])

    assert "Only the R channel differs" in text
    assert "precision or range fault" in text


def test_band_context_recognises_healthy_range():
    reference = _textured(5)
    current = reference + 0.45
    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison, include_band_context=True)
    context = " ".join(narrative["context"])

    low, high = OBSERVED_HEALTHY_MEAN_ABS
    assert f"{low}-{high}" in context
    assert "not a gate" in context


def test_band_context_recognises_different_seed_range():
    reference = _textured(6)
    current = reference + 70.0
    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison, include_band_context=True)
    context = " ".join(narrative["context"])

    low, high = OBSERVED_DIFFERENT_SEED_MEAN_ABS
    assert f"{low}-{high}" in context
    assert "compositional change" in context


def test_band_context_can_be_suppressed():
    reference = _textured(7)
    current = reference + 0.45
    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison, include_band_context=False)

    assert narrative["context"] == []


def test_report_only_is_explained_in_words():
    reference = _textured(8)
    current = reference + 0.4
    comparison = compare_pairwise(reference, current, threshold=1.0, limits=GateLimits())
    narrative = build_narrative(comparison)
    text = " ".join(narrative["findings"])

    assert "reports measurements only" in text
    assert "no threshold in this tool is legitimate" in text.lower()


def test_gate_failure_names_the_offending_limit():
    reference = _textured(9)
    current = reference + 20.0
    comparison = compare_pairwise(
        reference, current, threshold=1.0, limits=GateLimits(max_mean_abs=1.0)
    )
    narrative = build_narrative(comparison)
    text = " ".join(narrative["findings"])

    assert "FAIL against the limits you supplied" in text
    assert "max_mean_abs" in text


def test_provenance_mismatch_dominates_the_explanation():
    reference = _textured(10)
    current = _textured(11)
    comparison = compare_pairwise(reference, current, threshold=1.0)

    provenance_diff = {
        "comparable": True,
        "matches": False,
        "differences": [{"field": "seeds", "values": {}}],
        "notes": [],
    }
    narrative = build_narrative(comparison, provenance_diff=provenance_diff)
    text = " ".join(narrative["findings"])

    assert "NOT generated with the same configuration" in text
    assert "not evidence of a build regression" in text
    assert any("matched seed" in step for step in narrative["next_steps"])


def test_provenance_match_attributes_drift_to_software():
    reference = _textured(12)
    current = reference + 0.4
    comparison = compare_pairwise(reference, current, threshold=1.0)

    provenance_diff = {"comparable": True, "matches": True, "differences": [], "notes": []}
    narrative = build_narrative(comparison, provenance_diff=provenance_diff)
    text = " ".join(narrative["findings"])

    assert "comes from the software stack" in text


def test_three_way_narrative_covers_ratchet_reasoning():
    reference = _textured(13)
    baseline = reference + 10.0
    current = reference + 5.0
    comparison = compare_three_way(reference, baseline, current, threshold=1.0)
    narrative = build_narrative(comparison)
    text = " ".join(narrative["findings"])

    assert "50.00% closer to reference" in text
    assert "promoted to a new baseline" in text


def test_sanity_failure_appears_in_findings():
    reference = np.full((64, 64, 3), 0.5, dtype=np.float32)
    current = np.full((64, 64, 3), 0.6, dtype=np.float32)
    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison)
    text = " ".join(narrative["findings"])

    assert "failed a sanity check" in text
    assert "regression in its own right" in text


def test_render_narrative_has_readable_sections():
    reference = _textured(14)
    current = reference + 0.4
    comparison = compare_pairwise(reference, current, threshold=1.0)
    rendered = render_narrative(build_narrative(comparison))

    assert "SUMMARY" in rendered
    assert "WHAT THIS MEANS" in rendered
    assert "WHAT TO DO NEXT" in rendered
    assert max(len(line) for line in rendered.splitlines()) <= 74


def test_uniform_shift_described_as_directed():
    reference = np.full((64, 64, 3), 100.0, dtype=np.float32)
    current = reference + 2.0
    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison)
    text = " ".join(narrative["findings"])

    assert "uniformly brighter" in text
    assert "scaling or conversion" in text


def test_random_scatter_described_as_cancelling():
    rng = np.random.default_rng(7)
    reference = rng.uniform(50.0, 150.0, size=(64, 64, 3)).astype(np.float32)
    noise = rng.choice([-1.0, 1.0], size=reference.shape).astype(np.float32)
    current = reference + noise
    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison)
    text = " ".join(narrative["findings"])

    assert "cancel out" in text
    assert "random scatter" in text


def test_localized_same_direction_avoids_uniform_wording():
    reference = np.full((256, 256, 3), 128.0, dtype=np.float32)
    current = reference.copy()
    current[:32, :32, :] = 0.0
    comparison = compare_pairwise(reference, current, threshold=1.0)
    narrative = build_narrative(comparison)
    text = " ".join(narrative["findings"])

    assert "Every pixel that changed moved the same way" in text
    assert "uniformly" not in text.lower()
