"""Guard against the browser tool drifting away from the Python library.

The parity tests in test_browser_parity.py verify the *algorithms* agree. These
tests verify the shared *constants* embedded in iris-triage.html still match the
Python modules, which is the other way the two entry points can diverge.
"""

import re
from pathlib import Path

import pytest

from iris.narrative import (
    CHANNEL_SKEW_RATIO_HINT,
    CONCENTRATION_RATIO_HINT,
    OBSERVED_DIFFERENT_SEED_MEAN_ABS,
    OBSERVED_HEALTHY_MEAN_ABS,
)
from iris.regions import BROAD_SPREAD_BLOCK_FRACTION, CONCENTRATION_TARGET_PCT
from iris.report import SCHEMA_VERSION

HTML_PATH = Path(__file__).resolve().parents[1] / "iris-triage.html"


@pytest.fixture(scope="module")
def html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def test_html_exists_and_is_self_contained(html):
    assert HTML_PATH.exists()
    assert "<script src=" not in html
    assert "http://" not in html
    assert "https://" not in html
    assert "cdn" not in html.lower()


def test_schema_version_matches(html):
    assert f'SCHEMA_VERSION = "{SCHEMA_VERSION}"' in html


def test_observed_bands_match(html):
    low, high = OBSERVED_HEALTHY_MEAN_ABS
    assert f"OBSERVED_HEALTHY_MEAN_ABS = [{low}, {high}]" in html

    low, high = OBSERVED_DIFFERENT_SEED_MEAN_ABS
    assert f"OBSERVED_DIFFERENT_SEED_MEAN_ABS = [{low}, {high}]" in html


def test_wording_heuristics_match(html):
    assert f"CONCENTRATION_RATIO_HINT = {CONCENTRATION_RATIO_HINT}" in html
    assert f"CHANNEL_SKEW_RATIO_HINT = {CHANNEL_SKEW_RATIO_HINT}" in html
    assert f"CONCENTRATION_TARGET_PCT = {CONCENTRATION_TARGET_PCT}" in html
    assert f"BROAD_SPREAD_BLOCK_FRACTION = {BROAD_SPREAD_BLOCK_FRACTION}" in html


def test_colormap_control_points_match(html):
    from iris.heatmap import _COLORMAP_CONTROL_POINTS

    match = re.search(
        r"COLORMAP_CONTROL_POINTS = \[(.*?)\];", html, re.DOTALL
    )
    assert match, "colormap control points not found in HTML"
    block = match.group(1)

    for position, (r, g, b) in _COLORMAP_CONTROL_POINTS:
        assert f"[{position:.2f}, [{r}, {g}, {b}]]" in block


def test_no_default_gate_limits_in_html(html):
    """Gate inputs must ship empty so no numeric gate is ever a default."""
    for field in (
        "max-mean-abs",
        "max-p99-9",
        "max-max-abs",
        "max-pct-over-t",
        "min-similarity-pct",
    ):
        pattern = rf'id="{re.escape(field)}"[^>]*>'
        match = re.search(pattern, html)
        assert match, f"gate input {field} missing"
        assert "value=" not in match.group(0), f"{field} must not ship a default value"


def test_prompt_adherence_marked_advisory(html):
    assert "advisory" in html
    assert "never contribute to the gate verdict" in html
    assert 'verdict: "UNAVAILABLE"' in html


def test_element_extraction_stopwords_match(html):
    from iris.detect import _STOPWORDS

    match = re.search(r"ELEMENT_STOPWORDS = new Set\(\[(.*?)\]\)", html, re.DOTALL)
    assert match, "ELEMENT_STOPWORDS not found in HTML"
    html_words = set(re.findall(r'"([a-z\-]+)"', match.group(1)))
    assert html_words == _STOPWORDS


def test_prompt_elements_panel_present_and_advisory(html):
    assert 'id="elements-panel"' in html
    assert 'id="load-detections"' in html
    assert "Prompt-element detection is advisory only" in html


def test_closeness_note_present(html):
    assert "closeness to the reference image, not visual quality improvement" in html
