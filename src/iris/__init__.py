"""IRIS — Image Regression Inspection Suite."""

from iris.compare import compare_pairwise, compare_three_way
from iris.detect import (
    evaluate_prompt_elements,
    extract_prompt_elements,
    register_detect_backend,
)
from iris.metrics import compute_metrics
from iris.narrative import build_narrative
from iris.provenance import diff_provenance, extract_provenance
from iris.regions import analyze_blocks
from iris.semantic import evaluate_prompt_adherence, register_backend

__all__ = [
    "compute_metrics",
    "compare_pairwise",
    "compare_three_way",
    "analyze_blocks",
    "build_narrative",
    "extract_provenance",
    "diff_provenance",
    "evaluate_prompt_adherence",
    "register_backend",
    "evaluate_prompt_elements",
    "extract_prompt_elements",
    "register_detect_backend",
]

__version__ = "0.1.0"
