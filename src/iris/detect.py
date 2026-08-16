"""Prompt-element object detection — advisory only, never a gate.

This answers your mockup's question: for a given prompt, which named objects are
actually present in the image, and which are missing? A missing object is a
"wrong picture" problem (wrong checkpoint, truncated prompt, dropped conditioning),
which is orthogonal to the pixel drift IRIS gates on. So detection lands in its own
advisory namespace and never contributes to PASS/FAIL.

Real bounding boxes require a model. The bundled ``owlv2`` backend is an
open-vocabulary detector that runs on CPU behind the ``[semantic]`` extra. Register
another with :func:`register_detect_backend` (GroundingDINO, a VLM judge, etc.).

The browser tool cannot run a detector offline, so the standalone HTML visualises
the detection JSON this module produces rather than recomputing it.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Protocol

import numpy as np

ADVISORY_DETECT_NOTE = (
    "Prompt-element detection is advisory only. It reports which named objects a "
    "detector found in each image and never contributes to the gate verdict."
)

DEFAULT_DETECT_THRESHOLD = 0.1

# Tokens dropped from prompt clauses when extracting checkable object phrases.
_STOPWORDS = {
    "a", "an", "the", "of", "and", "with", "at", "to", "near", "front", "top",
    "next", "in", "on", "under", "above", "below", "over", "behind", "beside",
    "by", "between", "from", "into", "onto", "is", "are", "there",
}

# Clause boundaries. Splitting on spatial prepositions turns "a bottle on a table
# in front of mountains with trees, sun in sky" into individual object phrases.
_CLAUSE_SPLIT = re.compile(
    r",|;|\.|\band\b|\bwith\b|\bin front of\b|\bon top of\b|\bnext to\b|"
    r"\bin\b|\bon\b|\bunder\b|\babove\b|\bbelow\b|\bover\b|\bbehind\b|"
    r"\bbeside\b|\bnear\b|\bby\b|\bbetween\b",
    re.IGNORECASE,
)

_WORD = re.compile(r"[a-zA-Z][a-zA-Z\-]*")


def extract_prompt_elements(prompt: str) -> list[str]:
    """Extract candidate object phrases from a prompt for presence checking.

    Heuristic and deliberately simple: split on clause/preposition boundaries,
    drop stop-words, keep the remaining phrase. Pass an explicit list to the CLI
    when you need precision. Mirrored in iris-triage.html for the checklist.
    """
    elements: list[str] = []
    for piece in _CLAUSE_SPLIT.split(prompt or ""):
        words = [w for w in _WORD.findall(piece) if w.lower() not in _STOPWORDS]
        phrase = " ".join(words).strip().lower()
        if phrase and phrase not in elements:
            elements.append(phrase)
    return elements


class ObjectDetectionBackend(Protocol):
    """Detects named object queries in an image."""

    name: str

    def is_available(self) -> bool: ...

    def detect(
        self, image: np.ndarray, queries: list[str], threshold: float
    ) -> dict[str, list[dict[str, float]]]:
        """Return a mapping of query -> list of boxes above ``threshold``.

        Each box is ``{"x0", "y0", "x1", "y1", "score"}`` in image pixel space.
        """
        ...


_REGISTRY: dict[str, Callable[[], ObjectDetectionBackend]] = {}


def register_detect_backend(name: str, factory: Callable[[], ObjectDetectionBackend]) -> None:
    _REGISTRY[name] = factory


def available_detect_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_detect_backend(name: str) -> ObjectDetectionBackend:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown detection backend {name!r}. Available: {available_detect_backends()}"
        )
    return _REGISTRY[name]()


class NullDetectionBackend:
    """Default backend: reports unavailability without inventing detections."""

    name = "none"

    def is_available(self) -> bool:
        return False

    def detect(self, image, queries, threshold):
        return {query: [] for query in queries}


class Owlv2Backend:
    """Open-vocabulary detector using google/owlv2-base-patch16-ensemble.

    Runs on CPU. Loaded lazily so importing IRIS never pulls in torch.
    """

    name = "owlv2"

    def __init__(self, model_name: str = "google/owlv2-base-patch16-ensemble") -> None:
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
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            self._torch = torch
            self._processor = Owlv2Processor.from_pretrained(self.model_name)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name).eval()
            return True
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            return False

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def detect(self, image, queries, threshold):
        if not self._ensure_loaded():
            return {query: [] for query in queries}

        from PIL import Image as PILImage

        torch = self._torch
        pil = PILImage.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
        text_queries = [f"a photo of {query}" for query in queries]

        with torch.no_grad():
            inputs = self._processor(text=[text_queries], images=pil, return_tensors="pt")
            outputs = self._model(**inputs)
            target_sizes = torch.tensor([[pil.height, pil.width]])
            results = self._processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold
            )[0]

        detections: dict[str, list[dict[str, float]]] = {query: [] for query in queries}
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            query = queries[int(label)]
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            detections[query].append(
                {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "score": float(score)}
            )
        return detections


register_detect_backend("none", NullDetectionBackend)
register_detect_backend("owlv2", Owlv2Backend)


def _detect_one(
    backend: ObjectDetectionBackend,
    image: np.ndarray,
    queries: list[str],
    threshold: float,
) -> dict[str, Any]:
    detections = backend.detect(image, queries, threshold)
    height, width = image.shape[:2]
    elements = []
    for query in queries:
        boxes = detections.get(query, [])
        confidence = max((b["score"] for b in boxes), default=0.0)
        elements.append(
            {
                "element": query,
                "present": len(boxes) > 0,
                "confidence": confidence,
                "boxes": boxes,
            }
        )
    return {
        "backend": backend.name,
        "available": True,
        "image_shape": {"width": int(width), "height": int(height)},
        "elements": elements,
        "missing": [e["element"] for e in elements if not e["present"]],
    }


def evaluate_prompt_elements(
    images: dict[str, np.ndarray],
    prompt: str | None,
    *,
    backend_name: str = "none",
    prompt_source: str = "unspecified",
    elements: list[str] | None = None,
    threshold: float = DEFAULT_DETECT_THRESHOLD,
) -> dict[str, Any]:
    """Detect prompt objects in each labelled image. Always advisory-only."""
    queries = elements if elements else (extract_prompt_elements(prompt) if prompt else [])
    block: dict[str, Any] = {
        "verdict": "UNAVAILABLE",
        "backend": backend_name,
        "prompt": prompt,
        "prompt_source": prompt_source,
        "threshold": threshold,
        "elements": queries,
        "per_image": {},
        "missing": {},
        "findings": [],
        "notes": [ADVISORY_DETECT_NOTE],
    }

    if not prompt:
        block["notes"].append(
            "No prompt available, so prompt-element detection was skipped. Supply "
            "--prompt or use a ComfyUI PNG that still carries its metadata."
        )
        return block

    if not queries:
        block["notes"].append(
            "No checkable object phrases could be extracted from the prompt; pass "
            "--elements to name them explicitly."
        )
        return block

    try:
        backend = get_detect_backend(backend_name)
    except KeyError as exc:
        block["notes"].append(str(exc))
        return block

    if not backend.is_available():
        block["notes"].append(
            f"Detection backend {backend_name!r} is not usable in this environment "
            '(install with: pip install -e ".[semantic]"). No detection was performed.'
        )
        return block

    for role, image in images.items():
        result = _detect_one(backend, image, queries, threshold)
        block["per_image"][role] = result
        block["missing"][role] = result["missing"]

    block["verdict"] = "ADVISORY"
    block["findings"] = _build_findings(block["per_image"])
    return block


def _build_findings(per_image: dict[str, dict[str, Any]]) -> list[str]:
    findings: list[str] = []

    reference = per_image.get("reference")
    if reference and reference["missing"]:
        findings.append(
            "The reference image is missing prompt elements: "
            f"{', '.join(reference['missing'])}. If those were expected, the reference "
            "may not represent the prompt, which can explain why it was flagged or "
            "chosen for inspection."
        )

    current = per_image.get("current")
    if current and current["missing"]:
        findings.append(
            "The current image is missing prompt elements: "
            f"{', '.join(current['missing'])}."
        )

    if reference and current:
        ref_present = {e["element"] for e in reference["elements"] if e["present"]}
        cur_present = {e["element"] for e in current["elements"] if e["present"]}
        dropped = sorted(ref_present - cur_present)
        gained = sorted(cur_present - ref_present)
        for element in dropped:
            findings.append(
                f"Prompt element '{element}' is detected in reference but not in current."
            )
        for element in gained:
            findings.append(
                f"Prompt element '{element}' appears in current but was not detected in "
                "reference."
            )

    return findings
