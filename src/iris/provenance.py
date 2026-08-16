"""Tier 0 provenance: extract and diff ComfyUI generation metadata.

ComfyUI embeds the API-format prompt graph and the UI workflow as PNG ``tEXt``
chunks on every image it writes. Reading them is deterministic, needs no model
weights, and answers the cheapest useful question first: were these two images
even generated the same way?

If the seed, prompt text, sampler, or checkpoint differ between two images, the
pixel drift is explained by the harness and is not evidence of a software-stack
regression. That conclusion requires no statistics at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

_MAX_LINK_DEPTH = 16

_SEED_KEYS = ("seed", "noise_seed")
_SAMPLER_KEYS = ("sampler_name", "scheduler", "steps", "cfg", "denoise")
_MODEL_KEYS = (
    "ckpt_name",
    "unet_name",
    "vae_name",
    "lora_name",
    "clip_name",
    "model_name",
)

# Fields compared by diff_provenance. A mismatch on any of these means the two
# images did not come from the same generation configuration.
COMPARED_FIELDS = (
    "positive_prompt",
    "negative_prompt",
    "seeds",
    "samplers",
    "models",
)


def extract_provenance(path: str | Path) -> dict[str, Any]:
    """Read ComfyUI generation parameters from a PNG's text chunks.

    Returns a dict with ``available`` False when the file carries no ComfyUI
    metadata (JPEG, .npy, or a PNG written by another tool).
    """
    path = Path(path)
    result: dict[str, Any] = {
        "source": str(path),
        "available": False,
        "positive_prompt": None,
        "negative_prompt": None,
        "seeds": [],
        "samplers": [],
        "models": [],
        "notes": [],
    }

    if path.suffix.lower() != ".png":
        result["notes"].append(
            f"ComfyUI metadata is only embedded in PNG output; {path.suffix} carries none."
        )
        return result

    try:
        with Image.open(path) as img:
            text_chunks = dict(getattr(img, "text", {}) or {})
    except Exception as exc:
        result["notes"].append(f"Could not read PNG text chunks: {exc}")
        return result

    raw_prompt = text_chunks.get("prompt")
    if not raw_prompt:
        result["notes"].append(
            "No 'prompt' text chunk found; image was likely not written by ComfyUI "
            "or metadata was stripped."
        )
        return result

    try:
        graph = json.loads(raw_prompt)
    except json.JSONDecodeError as exc:
        result["notes"].append(f"'prompt' chunk is not valid JSON: {exc}")
        return result

    if not isinstance(graph, dict):
        result["notes"].append("'prompt' chunk did not decode to a node mapping.")
        return result

    result["available"] = True
    result.update(_parse_graph(graph))
    return result


def _parse_graph(graph: dict[str, Any]) -> dict[str, Any]:
    seeds: list[Any] = []
    samplers: list[dict[str, Any]] = []
    models: list[str] = []
    positive_texts: list[str] = []
    negative_texts: list[str] = []

    for node_id, node in sorted(graph.items()):
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        for key in _SEED_KEYS:
            value = inputs.get(key)
            if isinstance(value, (int, float)):
                seeds.append({"node": node_id, "key": key, "value": value})

        sampler_entry = {
            key: inputs[key]
            for key in _SAMPLER_KEYS
            if key in inputs and not _is_link(inputs[key])
        }
        if sampler_entry:
            sampler_entry["node"] = node_id
            sampler_entry["class_type"] = node.get("class_type")
            samplers.append(sampler_entry)

        for key in _MODEL_KEYS:
            value = inputs.get(key)
            if isinstance(value, str):
                models.append(value)

        if _is_link(inputs.get("positive")):
            text = _resolve_text(graph, inputs["positive"])
            if text:
                positive_texts.append(text)
        if _is_link(inputs.get("negative")):
            text = _resolve_text(graph, inputs["negative"])
            if text:
                negative_texts.append(text)

    if not positive_texts:
        # No sampler wiring found; fall back to every text encoder in the graph
        # so a prompt is still surfaced for the semantic layer.
        positive_texts = _all_encoder_texts(graph)

    return {
        "positive_prompt": _join_unique(positive_texts),
        "negative_prompt": _join_unique(negative_texts),
        "seeds": seeds,
        "samplers": samplers,
        "models": sorted(set(models)),
    }


def _all_encoder_texts(graph: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for _, node in sorted(graph.items()):
        if not isinstance(node, dict):
            continue
        if "CLIPTextEncode" not in str(node.get("class_type", "")):
            continue
        value = (node.get("inputs") or {}).get("text")
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    return texts


def _resolve_text(graph: dict[str, Any], link: Any, depth: int = 0) -> str | None:
    """Follow a [node_id, slot] link until a literal text input is reached."""
    if depth > _MAX_LINK_DEPTH or not _is_link(link):
        return None

    node = graph.get(str(link[0]))
    if not isinstance(node, dict):
        return None

    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None

    value = inputs.get("text")
    if isinstance(value, str):
        return value.strip() or None
    if _is_link(value):
        return _resolve_text(graph, value, depth + 1)

    for key in ("conditioning", "conditioning_1", "conditioning_to"):
        if _is_link(inputs.get(key)):
            resolved = _resolve_text(graph, inputs[key], depth + 1)
            if resolved:
                return resolved
    return None


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int))


def _join_unique(texts: list[str]) -> str | None:
    seen: list[str] = []
    for text in texts:
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    if not seen:
        return None
    return " | ".join(seen)


def diff_provenance(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare provenance across labelled images.

    ``entries`` maps a role name (reference/baseline/current) to the dict from
    :func:`extract_provenance`.
    """
    available = {name: data for name, data in entries.items() if data.get("available")}

    if len(available) < 2:
        return {
            "comparable": False,
            "matches": None,
            "differences": [],
            "notes": [
                "Fewer than two images carry ComfyUI metadata; generation parameters "
                "could not be compared."
            ],
        }

    differences: list[dict[str, Any]] = []
    for field in COMPARED_FIELDS:
        values = {name: _normalize(data.get(field)) for name, data in available.items()}
        distinct = {json.dumps(value, sort_keys=True) for value in values.values()}
        if len(distinct) > 1:
            differences.append({"field": field, "values": values})

    return {
        "comparable": True,
        "matches": len(differences) == 0,
        "differences": differences,
        "compared_roles": sorted(available),
        "notes": [],
    }


def _normalize(value: Any) -> Any:
    """Drop node ids so graph reordering does not read as a parameter change."""
    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(
                    {k: v for k, v in sorted(item.items()) if k != "node"}
                )
            else:
                normalized.append(item)
        return sorted(normalized, key=lambda entry: json.dumps(entry, sort_keys=True))
    return value


def resolve_prompt(
    explicit_prompt: str | None,
    provenance_entries: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    """Pick the prompt to evaluate against, preferring an explicit override.

    Returns the prompt and a short string describing where it came from.
    """
    if explicit_prompt:
        return explicit_prompt, "supplied via --prompt"

    for role in ("reference", "baseline", "current"):
        data = provenance_entries.get(role)
        if data and data.get("positive_prompt"):
            return data["positive_prompt"], f"extracted from {role} PNG metadata"

    return None, "unavailable (no --prompt and no embedded ComfyUI metadata)"
