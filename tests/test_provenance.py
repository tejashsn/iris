"""Tests for Tier 0 ComfyUI metadata extraction and diffing."""

import json

import numpy as np
from PIL import Image, PngImagePlugin

from iris.provenance import diff_provenance, extract_provenance, resolve_prompt


def _comfy_graph(
    *,
    seed: int = 12345,
    positive: str = "a red umbrella on a wet street, neon reflections",
    negative: str = "blurry, low quality",
    ckpt: str = "sdxl_base_1.0.safetensors",
    sampler: str = "euler",
) -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": sampler,
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
            },
        },
    }


def _write_comfy_png(path, graph: dict | None) -> None:
    array = np.full((16, 16, 3), 120, dtype=np.uint8)
    info = PngImagePlugin.PngInfo()
    if graph is not None:
        info.add_text("prompt", json.dumps(graph))
    Image.fromarray(array, mode="RGB").save(path, pnginfo=info)


def test_extract_reads_prompt_seed_sampler_and_model(tmp_path):
    path = tmp_path / "ref.png"
    _write_comfy_png(path, _comfy_graph())

    prov = extract_provenance(path)

    assert prov["available"] is True
    assert "red umbrella" in prov["positive_prompt"]
    assert prov["negative_prompt"] == "blurry, low quality"
    assert prov["seeds"][0]["value"] == 12345
    assert "sdxl_base_1.0.safetensors" in prov["models"]
    assert prov["samplers"][0]["sampler_name"] == "euler"


def test_extract_degrades_on_missing_metadata(tmp_path):
    path = tmp_path / "plain.png"
    _write_comfy_png(path, None)

    prov = extract_provenance(path)

    assert prov["available"] is False
    assert prov["positive_prompt"] is None
    assert prov["notes"]


def test_extract_degrades_on_non_png(tmp_path):
    path = tmp_path / "sample.npy"
    np.save(path, np.zeros((4, 4, 3), dtype=np.float32))

    prov = extract_provenance(path)

    assert prov["available"] is False
    assert "only embedded in PNG" in prov["notes"][0]


def test_diff_detects_seed_mismatch(tmp_path):
    ref = tmp_path / "ref.png"
    cur = tmp_path / "cur.png"
    _write_comfy_png(ref, _comfy_graph(seed=111))
    _write_comfy_png(cur, _comfy_graph(seed=222))

    result = diff_provenance(
        {"reference": extract_provenance(ref), "current": extract_provenance(cur)}
    )

    assert result["comparable"] is True
    assert result["matches"] is False
    assert [item["field"] for item in result["differences"]] == ["seeds"]


def test_diff_reports_match_for_identical_config(tmp_path):
    ref = tmp_path / "ref.png"
    cur = tmp_path / "cur.png"
    _write_comfy_png(ref, _comfy_graph())
    _write_comfy_png(cur, _comfy_graph())

    result = diff_provenance(
        {"reference": extract_provenance(ref), "current": extract_provenance(cur)}
    )

    assert result["matches"] is True
    assert result["differences"] == []


def test_diff_detects_prompt_and_model_mismatch(tmp_path):
    ref = tmp_path / "ref.png"
    cur = tmp_path / "cur.png"
    _write_comfy_png(ref, _comfy_graph())
    _write_comfy_png(
        cur, _comfy_graph(positive="a blue umbrella", ckpt="sd15.safetensors")
    )

    result = diff_provenance(
        {"reference": extract_provenance(ref), "current": extract_provenance(cur)}
    )

    fields = {item["field"] for item in result["differences"]}
    assert fields == {"positive_prompt", "models"}


def test_diff_not_comparable_with_single_source(tmp_path):
    ref = tmp_path / "ref.png"
    cur = tmp_path / "cur.png"
    _write_comfy_png(ref, _comfy_graph())
    _write_comfy_png(cur, None)

    result = diff_provenance(
        {"reference": extract_provenance(ref), "current": extract_provenance(cur)}
    )

    assert result["comparable"] is False
    assert result["matches"] is None


def test_node_reordering_is_not_a_difference(tmp_path):
    ref = tmp_path / "ref.png"
    cur = tmp_path / "cur.png"
    graph = _comfy_graph()
    renumbered = {
        "10": graph["1"],
        "20": graph["2"],
        "30": graph["3"],
        "40": {
            **graph["4"],
            "inputs": {
                **graph["4"]["inputs"],
                "model": ["10", 0],
                "positive": ["20", 0],
                "negative": ["30", 0],
            },
        },
    }
    _write_comfy_png(ref, graph)
    _write_comfy_png(cur, renumbered)

    result = diff_provenance(
        {"reference": extract_provenance(ref), "current": extract_provenance(cur)}
    )

    assert result["matches"] is True


def test_resolve_prompt_prefers_explicit_override(tmp_path):
    path = tmp_path / "ref.png"
    _write_comfy_png(path, _comfy_graph())
    entries = {"reference": extract_provenance(path)}

    prompt, source = resolve_prompt("explicit text", entries)
    assert prompt == "explicit text"
    assert "--prompt" in source

    prompt, source = resolve_prompt(None, entries)
    assert "red umbrella" in prompt
    assert "reference" in source

    prompt, source = resolve_prompt(None, {})
    assert prompt is None
    assert "unavailable" in source
