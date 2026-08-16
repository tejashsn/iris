"""Prompt-adherence scoring — advisory only, never a gate.

Why this is a separate subsystem
--------------------------------
Prompt-adherence metrics cannot detect the drift IRIS was built to measure. At
0.3-0.6/255 the CLIP embeddings and VQA answers for a healthy build and a broken
one are identical to several decimal places. This is the same saturation problem
that disqualified SSIM: the NeurIPS 2025 meta-evaluation of compositional metrics
found VQA scores concentrate near 1.0 and that all VQA-based metrics lean on
answer-position shortcuts.

So this layer answers a different question: *is the workflow producing the right
picture at all?* It catches a wrong checkpoint, a truncated prompt, a text-encoder
precision fault, or a CLIP-skip misconfiguration — failures where the pixel
metrics report enormous drift but cannot say which element went missing.

Results land in their own namespace with verdict ``ADVISORY`` or ``UNAVAILABLE``.
Nothing here can produce PASS or FAIL, and the pixel gate never reads it.

Backends
--------
Register additional backends with :func:`register_backend`. The bundled
``local-clip`` backend is a reference implementation: it scores each clause of
the prompt separately so low-scoring clauses can be listed individually. That is
a cheaper approximation of TIFA/DSG-style decomposition and needs no VQA model.
For a full question-answering breakdown or prose explanation, register a backend
wrapping VQAScore, TIFA, or a vision-language judge.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Protocol

import numpy as np

ADVISORY_NOTE = (
    "Prompt adherence is advisory only. These scores are uncalibrated and cannot "
    "detect build-to-build pixel drift; they never contribute to the gate verdict."
)

_CLAUSE_SPLIT = re.compile(r"[,;.]|\band\b")


class PromptAdherenceBackend(Protocol):
    """Scores how well an image matches a text prompt."""

    name: str

    def is_available(self) -> bool: ...

    def score(self, image: np.ndarray, prompt: str) -> dict[str, Any]: ...


_REGISTRY: dict[str, Callable[[], PromptAdherenceBackend]] = {}


def register_backend(name: str, factory: Callable[[], PromptAdherenceBackend]) -> None:
    """Register a backend factory under ``name``."""
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str) -> PromptAdherenceBackend:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown prompt-adherence backend {name!r}. Available: {available_backends()}"
        )
    return _REGISTRY[name]()


def split_prompt_clauses(prompt: str) -> list[str]:
    """Break a prompt into checkable clauses for per-element reporting."""
    parts = [part.strip() for part in _CLAUSE_SPLIT.split(prompt)]
    return [part for part in parts if len(part) > 2]


class NullBackend:
    """Default backend: reports unavailability without pretending to score."""

    name = "none"

    def is_available(self) -> bool:
        return False

    def score(self, image: np.ndarray, prompt: str) -> dict[str, Any]:
        return {
            "backend": self.name,
            "available": False,
            "score": None,
            "clauses": [],
            "notes": [
                "No prompt-adherence backend selected. Install the semantic extra "
                "and pass --semantic-backend local-clip, or register a custom backend."
            ],
        }


class LocalClipBackend:
    """Reference backend using a locally installed CLIP model.

    Scores the whole prompt and each clause separately. Absolute CLIP similarity
    is not calibrated and should not be read as a percentage of correctness; the
    useful signal is the *difference* between reference and current, and the
    relative ranking of clauses within one image.
    """

    name = "local-clip"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._error: str | None = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._error is not None:
            return False
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
            self._model = CLIPModel.from_pretrained(self.model_name).eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            return True
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            return False

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def score(self, image: np.ndarray, prompt: str) -> dict[str, Any]:
        if not self._ensure_loaded():
            return {
                "backend": self.name,
                "available": False,
                "score": None,
                "clauses": [],
                "notes": [
                    f"CLIP backend unavailable ({self._error}). "
                    'Install with: pip install -e ".[semantic]"'
                ],
            }

        from PIL import Image as PILImage

        torch = self._torch
        pil = PILImage.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
        clauses = split_prompt_clauses(prompt)
        texts = [prompt] + [c for c in clauses if c != prompt]

        with torch.no_grad():
            inputs = self._processor(
                text=texts, images=pil, return_tensors="pt", padding=True, truncation=True
            )
            outputs = self._model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(
                dim=-1, keepdim=True
            )
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(
                dim=-1, keepdim=True
            )
            similarities = (image_embeds @ text_embeds.T).squeeze(0).tolist()

        if isinstance(similarities, float):
            similarities = [similarities]

        clause_scores = [
            {"clause": text, "similarity": float(value)}
            for text, value in zip(texts[1:], similarities[1:])
        ]
        clause_scores.sort(key=lambda item: item["similarity"])

        return {
            "backend": self.name,
            "model": self.model_name,
            "available": True,
            "score": float(similarities[0]),
            "clauses": clause_scores,
            "notes": [
                "CLIP cosine similarity is uncalibrated; compare reference against "
                "current rather than reading the absolute value."
            ],
        }


register_backend("none", NullBackend)
register_backend("local-clip", LocalClipBackend)


def evaluate_prompt_adherence(
    images: dict[str, np.ndarray],
    prompt: str | None,
    *,
    backend_name: str = "none",
    prompt_source: str = "unspecified",
) -> dict[str, Any]:
    """Score each labelled image against the prompt.

    Always returns an advisory-only block. ``images`` maps a role name
    (reference/baseline/current) to an RGB float32 array.
    """
    block: dict[str, Any] = {
        "verdict": "UNAVAILABLE",
        "backend": backend_name,
        "prompt": prompt,
        "prompt_source": prompt_source,
        "per_image": {},
        "divergence": None,
        "findings": [],
        "notes": [ADVISORY_NOTE],
    }

    if not prompt:
        block["notes"].append(
            "No prompt available, so adherence was not evaluated. Supply --prompt or "
            "use a ComfyUI PNG that still carries its metadata."
        )
        return block

    try:
        backend = get_backend(backend_name)
    except KeyError as exc:
        block["notes"].append(str(exc))
        return block

    if not backend.is_available():
        block["per_image"] = {
            role: backend.score(image, prompt) for role, image in images.items()
        }
        block["notes"].append(
            f"Backend {backend_name!r} is not usable in this environment; "
            "no adherence scores were produced."
        )
        return block

    block["per_image"] = {
        role: backend.score(image, prompt) for role, image in images.items()
    }
    block["verdict"] = "ADVISORY"
    block["divergence"] = _compute_divergence(block["per_image"])
    block["findings"] = _build_findings(block["per_image"], block["divergence"])
    return block


def _compute_divergence(per_image: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    reference = per_image.get("reference", {}).get("score")
    current = per_image.get("current", {}).get("score")
    if reference is None or current is None:
        return None

    delta = current - reference
    return {
        "reference_score": reference,
        "current_score": current,
        "delta": delta,
        "interpretation": (
            "Current and reference match the prompt equally well."
            if abs(delta) < 1e-6
            else (
                f"Current matches the prompt {'better' if delta > 0 else 'worse'} than "
                f"reference by {abs(delta):.4f} in uncalibrated similarity units."
            )
        ),
    }


def _build_findings(
    per_image: dict[str, dict[str, Any]],
    divergence: dict[str, Any] | None,
) -> list[str]:
    findings: list[str] = []

    reference = per_image.get("reference", {})
    if reference.get("clauses"):
        weakest = reference["clauses"][0]
        findings.append(
            f"In the reference image, the prompt element that matches least well is "
            f"\"{weakest['clause']}\" (similarity {weakest['similarity']:.4f}). "
            "If this is unexpectedly low, the reference itself may not represent the prompt."
        )

    current = per_image.get("current", {})
    if current.get("clauses"):
        weakest = current["clauses"][0]
        findings.append(
            f"In the current image, the weakest prompt element is "
            f"\"{weakest['clause']}\" (similarity {weakest['similarity']:.4f})."
        )

    ref_clauses = {c["clause"]: c["similarity"] for c in reference.get("clauses", [])}
    cur_clauses = {c["clause"]: c["similarity"] for c in current.get("clauses", [])}
    regressions = [
        (clause, cur_clauses[clause] - ref_clauses[clause])
        for clause in ref_clauses
        if clause in cur_clauses and cur_clauses[clause] < ref_clauses[clause]
    ]
    regressions.sort(key=lambda item: item[1])
    for clause, delta in regressions[:3]:
        findings.append(
            f"Prompt element \"{clause}\" scores {abs(delta):.4f} lower in current "
            "than in reference."
        )

    if divergence:
        findings.append(divergence["interpretation"])

    return findings
