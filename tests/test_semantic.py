"""Tests for the advisory prompt-adherence layer."""

import numpy as np
import pytest

from iris.semantic import (
    NullBackend,
    available_backends,
    evaluate_prompt_adherence,
    get_backend,
    register_backend,
    split_prompt_clauses,
)


class FakeBackend:
    """Deterministic stand-in so tests never download model weights."""

    name = "fake"

    def __init__(self, scores: dict[float, float] | None = None) -> None:
        self.scores = scores or {}

    def is_available(self) -> bool:
        return True

    def score(self, image: np.ndarray, prompt: str) -> dict:
        key = float(np.mean(image))
        base = self.scores.get(key, 0.30)
        clauses = split_prompt_clauses(prompt)
        return {
            "backend": self.name,
            "available": True,
            "score": base,
            "clauses": sorted(
                (
                    {"clause": clause, "similarity": base - 0.01 * index}
                    for index, clause in enumerate(clauses)
                ),
                key=lambda item: item["similarity"],
            ),
            "notes": [],
        }


def test_default_backend_is_registered():
    assert "none" in available_backends()
    assert "local-clip" in available_backends()


def test_unknown_backend_is_reported_not_raised_through_evaluate():
    images = {"reference": np.zeros((8, 8, 3), dtype=np.float32)}
    result = evaluate_prompt_adherence(images, "a cat", backend_name="nope")

    assert result["verdict"] == "UNAVAILABLE"
    assert any("Unknown prompt-adherence backend" in note for note in result["notes"])


def test_get_backend_raises_for_unknown_name():
    with pytest.raises(KeyError):
        get_backend("definitely-not-a-backend")


def test_null_backend_reports_unavailable_without_faking_a_score():
    backend = NullBackend()
    result = backend.score(np.zeros((4, 4, 3), dtype=np.float32), "a cat")

    assert result["available"] is False
    assert result["score"] is None


def test_missing_prompt_skips_evaluation():
    images = {"reference": np.zeros((8, 8, 3), dtype=np.float32)}
    result = evaluate_prompt_adherence(images, None, backend_name="none")

    assert result["verdict"] == "UNAVAILABLE"
    assert result["prompt"] is None
    assert any("No prompt available" in note for note in result["notes"])


def test_verdict_is_never_pass_or_fail():
    register_backend("fake-verdict", lambda: FakeBackend())
    images = {
        "reference": np.full((8, 8, 3), 10.0, dtype=np.float32),
        "current": np.full((8, 8, 3), 20.0, dtype=np.float32),
    }
    result = evaluate_prompt_adherence(
        images, "a red umbrella, neon reflections", backend_name="fake-verdict"
    )

    assert result["verdict"] == "ADVISORY"
    assert result["verdict"] not in {"PASS", "FAIL"}
    assert any("never contribute to the gate" in note for note in result["notes"])


def test_divergence_reports_reference_and_current_separately():
    register_backend(
        "fake-divergence",
        lambda: FakeBackend(scores={10.0: 0.32, 20.0: 0.25}),
    )
    images = {
        "reference": np.full((8, 8, 3), 10.0, dtype=np.float32),
        "current": np.full((8, 8, 3), 20.0, dtype=np.float32),
    }
    result = evaluate_prompt_adherence(
        images, "a red umbrella, neon reflections", backend_name="fake-divergence"
    )

    divergence = result["divergence"]
    assert divergence["reference_score"] == pytest.approx(0.32)
    assert divergence["current_score"] == pytest.approx(0.25)
    assert divergence["delta"] == pytest.approx(-0.07)
    assert "worse" in divergence["interpretation"]


def test_findings_flag_weakest_reference_element():
    register_backend("fake-findings", lambda: FakeBackend(scores={10.0: 0.30, 20.0: 0.30}))
    images = {
        "reference": np.full((8, 8, 3), 10.0, dtype=np.float32),
        "current": np.full((8, 8, 3), 20.0, dtype=np.float32),
    }
    result = evaluate_prompt_adherence(
        images, "a red umbrella, neon reflections", backend_name="fake-findings"
    )
    text = " ".join(result["findings"])

    assert "reference image" in text
    assert "may not represent the prompt" in text


def test_prompt_source_is_echoed():
    images = {"reference": np.zeros((8, 8, 3), dtype=np.float32)}
    result = evaluate_prompt_adherence(
        images, "a cat", backend_name="none", prompt_source="extracted from reference PNG metadata"
    )

    assert result["prompt_source"] == "extracted from reference PNG metadata"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("a red umbrella, neon reflections", ["a red umbrella", "neon reflections"]),
        ("a cat and a dog", ["a cat", "a dog"]),
        ("single phrase", ["single phrase"]),
    ],
)
def test_split_prompt_clauses(prompt, expected):
    assert split_prompt_clauses(prompt) == expected
