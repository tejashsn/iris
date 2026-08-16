"""Tests for advisory prompt-element detection."""

import numpy as np
import pytest

from iris.detect import (
    NullDetectionBackend,
    available_detect_backends,
    evaluate_prompt_elements,
    extract_prompt_elements,
    get_detect_backend,
    register_detect_backend,
)


class FakeDetector:
    """Deterministic detector so tests never download OWLv2 weights.

    ``present`` maps a query to a box; anything absent is reported missing.
    """

    name = "fake"

    def __init__(self, present: dict[str, dict] | None = None) -> None:
        self.present = present or {}

    def is_available(self) -> bool:
        return True

    def detect(self, image, queries, threshold):
        out = {}
        for query in queries:
            box = self.present.get(query)
            out[query] = [box] if box and box["score"] >= threshold else []
        return out


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "A bottle on a table in front of mountains with trees, sun in sky",
            ["bottle", "table", "mountains", "trees", "sun", "sky"],
        ),
        ("a red umbrella and a blue car", ["red umbrella", "blue car"]),
        ("", []),
    ],
)
def test_extract_prompt_elements(prompt, expected):
    assert extract_prompt_elements(prompt) == expected


def test_backends_registered():
    assert "none" in available_detect_backends()
    assert "owlv2" in available_detect_backends()


def test_null_backend_reports_unavailable():
    backend = NullDetectionBackend()
    assert backend.is_available() is False
    assert backend.detect(np.zeros((4, 4, 3), np.float32), ["sun"], 0.1) == {"sun": []}


def test_get_backend_raises_for_unknown():
    with pytest.raises(KeyError):
        get_detect_backend("no-such-detector")


def test_missing_prompt_skips_detection():
    result = evaluate_prompt_elements({"reference": np.zeros((8, 8, 3), np.float32)}, None)
    assert result["verdict"] == "UNAVAILABLE"
    assert any("No prompt available" in note for note in result["notes"])


def test_unavailable_backend_is_reported_not_raised():
    result = evaluate_prompt_elements(
        {"reference": np.zeros((8, 8, 3), np.float32)},
        "a sun",
        backend_name="none",
    )
    assert result["verdict"] == "UNAVAILABLE"
    assert result["elements"] == ["sun"]


def test_detection_reports_present_and_missing():
    register_detect_backend(
        "fake-present",
        lambda: FakeDetector(present={"sun": {"x0": 1, "y0": 1, "x1": 5, "y1": 5, "score": 0.4}}),
    )
    images = {"reference": np.zeros((32, 32, 3), np.float32)}
    result = evaluate_prompt_elements(
        images, "a sun and a bottle", backend_name="fake-present"
    )

    assert result["verdict"] == "ADVISORY"
    ref = result["per_image"]["reference"]
    by_name = {e["element"]: e for e in ref["elements"]}
    assert by_name["sun"]["present"] is True
    assert by_name["sun"]["confidence"] == pytest.approx(0.4)
    assert by_name["bottle"]["present"] is False
    assert ref["missing"] == ["bottle"]


def test_verdict_is_never_pass_or_fail():
    register_detect_backend("fake-verdict", lambda: FakeDetector())
    result = evaluate_prompt_elements(
        {"reference": np.zeros((8, 8, 3), np.float32)},
        "a sun",
        backend_name="fake-verdict",
    )
    assert result["verdict"] not in {"PASS", "FAIL"}
    assert any("never contribute" in note.lower() or "advisory" in note.lower() for note in result["notes"])


def test_findings_explain_missing_reference_elements():
    register_detect_backend(
        "fake-missing-ref",
        lambda: FakeDetector(present={"bottle": {"x0": 0, "y0": 0, "x1": 4, "y1": 4, "score": 0.5}}),
    )
    images = {
        "reference": np.zeros((16, 16, 3), np.float32),
        "current": np.zeros((16, 16, 3), np.float32),
    }
    result = evaluate_prompt_elements(
        images, "a bottle, trees, sun", backend_name="fake-missing-ref"
    )
    text = " ".join(result["findings"])

    assert "reference image is missing prompt elements" in text
    assert "trees" in text and "sun" in text


def test_findings_flag_element_dropped_between_reference_and_current():
    class RoleAwareDetector:
        name = "role-aware"

        def is_available(self):
            return True

        def detect(self, image, queries, threshold):
            # Sun present only when the frame is bright (reference), gone in current.
            bright = float(np.mean(image)) > 100
            out = {q: [] for q in queries}
            if bright:
                out["sun"] = [{"x0": 0, "y0": 0, "x1": 4, "y1": 4, "score": 0.6}]
            return out

    register_detect_backend("role-aware", lambda: RoleAwareDetector())
    images = {
        "reference": np.full((16, 16, 3), 200.0, np.float32),
        "current": np.full((16, 16, 3), 10.0, np.float32),
    }
    result = evaluate_prompt_elements(images, "sun", backend_name="role-aware")
    text = " ".join(result["findings"])

    assert "detected in reference but not in current" in text


def test_explicit_elements_override_prompt_extraction():
    register_detect_backend("fake-override", lambda: FakeDetector())
    result = evaluate_prompt_elements(
        {"reference": np.zeros((8, 8, 3), np.float32)},
        "a completely different prompt",
        backend_name="fake-override",
        elements=["widget", "gadget"],
    )
    assert result["elements"] == ["widget", "gadget"]
