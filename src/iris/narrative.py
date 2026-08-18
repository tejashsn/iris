"""Plain-language interpretation of a comparison.

Everything here is descriptive. The wording heuristics below choose how a
sentence is phrased; no verdict, gate, or PASS/FAIL decision reads any of them.
The gate is driven exclusively by caller-supplied limits in :mod:`iris.compare`.
"""

from __future__ import annotations

from typing import Any

from iris.regions import describe_concentration

# Observed populations from the prior cross-GPU study, quoted as context in the
# narrative. These are NOT gates and no comparison logic reads them; they exist
# so a reader knows whether a number resembles previously measured behaviour.
OBSERVED_HEALTHY_MEAN_ABS = (0.3, 0.6)
OBSERVED_DIFFERENT_SEED_MEAN_ABS = (57.0, 84.0)

# Wording heuristics for the prose summary only.
CONCENTRATION_RATIO_HINT = 20.0
CHANNEL_SKEW_RATIO_HINT = 2.0
SIGNED_RATIO_UNIFORM_HINT = 0.8
SIGNED_RATIO_SCATTER_HINT = 0.3
CHANGED_FRACTION_UNIFORM_WORD_HINT = 50.0


def build_narrative(
    comparison: dict[str, Any],
    *,
    provenance_diff: dict[str, Any] | None = None,
    semantic: dict[str, Any] | None = None,
    prompt_elements: dict[str, Any] | None = None,
    include_band_context: bool = True,
) -> dict[str, Any]:
    """Turn a comparison result into a summary paragraph plus discrete findings."""
    metrics = _primary_metrics(comparison)
    labels = comparison["labels"]
    findings: list[str] = []

    headline = _headline(comparison, metrics, labels)
    findings.extend(_provenance_findings(provenance_diff))
    findings.extend(_sanity_findings(comparison))
    findings.extend(_distribution_findings(metrics, comparison))
    findings.extend(_signed_direction_findings(metrics))
    findings.extend(_channel_findings(metrics))
    findings.extend(_three_way_findings(comparison))
    findings.extend(_gate_findings(comparison))

    context: list[str] = []
    if include_band_context and not metrics.get("bitwise_identical"):
        context.extend(_band_context(metrics))

    if prompt_elements:
        findings.extend(prompt_elements.get("findings", []))

    if semantic:
        findings.extend(semantic.get("findings", []))

    return {
        "headline": headline,
        "findings": findings,
        "context": context,
        "next_steps": _next_steps(comparison, metrics, provenance_diff),
    }


def _primary_metrics(comparison: dict[str, Any]) -> dict[str, Any]:
    if comparison["mode"] == "pairwise":
        return comparison["metrics"]
    return comparison["reference_vs_current"]


def _headline(
    comparison: dict[str, Any],
    metrics: dict[str, Any],
    labels: dict[str, str],
) -> str:
    reference = labels.get("reference", "reference")
    current = labels.get("current", "current")

    if metrics.get("bitwise_identical"):
        return (
            f"{current} is bitwise identical to {reference}: every RGB sample matches "
            "exactly, so there is no drift to characterise."
        )

    mean_abs = metrics["mean_abs"]
    p99_9 = metrics["p99_9"]
    max_abs = metrics["max_abs"]
    pct_over_t = metrics["pct_over_t"]
    threshold = metrics["threshold"]

    return (
        f"Comparing {current} against {reference}: the average RGB sample differs by "
        f"{mean_abs:.4f} on a 0-255 scale, {pct_over_t:.4f}% of samples differ by more "
        f"than the {threshold:g} threshold, and the largest single-sample difference is "
        f"{max_abs:.2f}. The 99.9th percentile sits at {p99_9:.2f}, which is the number "
        "to watch, because a localised fault moves the tail long before it moves the mean."
    )


def _provenance_findings(provenance_diff: dict[str, Any] | None) -> list[str]:
    if not provenance_diff:
        return []

    if not provenance_diff.get("comparable"):
        return [
            "Generation parameters could not be compared because fewer than two images "
            "carry ComfyUI metadata, so a seed or prompt mismatch cannot be ruled out."
        ]

    if provenance_diff.get("matches"):
        return [
            "The embedded ComfyUI metadata matches across images: same prompt, seed, "
            "sampler settings and models. Any difference measured below therefore comes "
            "from the software stack rather than the generation request."
        ]

    fields = ", ".join(item["field"] for item in provenance_diff["differences"])
    return [
        f"The images were NOT generated with the same configuration: {fields} differ. "
        "The pixel drift below is explained by that difference and is not evidence of a "
        "build regression. Fix the harness inputs before interpreting any metric."
    ]


def _sanity_findings(comparison: dict[str, Any]) -> list[str]:
    findings = []
    for item in comparison["sanity"]:
        if not item["passed"]:
            findings.append(
                f"The {item['label']} frame failed a sanity check "
                f"({', '.join(item['issues'])}). This is a regression in its own right, "
                "and including such a frame would distort any threshold derived later."
            )
    return findings


def _distribution_findings(
    metrics: dict[str, Any],
    comparison: dict[str, Any],
) -> list[str]:
    if metrics.get("bitwise_identical"):
        return []

    findings: list[str] = []
    mean_abs = metrics["mean_abs"]
    p99_9 = metrics["p99_9"]

    if mean_abs > 0:
        ratio = p99_9 / mean_abs
        if ratio >= CONCENTRATION_RATIO_HINT:
            findings.append(
                f"The 99.9th percentile is {ratio:.0f} times the mean, so the difference "
                "is concentrated in a small number of samples rather than spread evenly "
                "across the frame. That shape is characteristic of a localised fault, "
                "not of accumulated floating-point rounding."
            )
        else:
            findings.append(
                f"The 99.9th percentile is {ratio:.1f} times the mean, so the difference "
                "is distributed fairly evenly across the frame rather than concentrated "
                "in one region."
            )

    regions = comparison.get("regions")
    if regions:
        findings.append(describe_concentration(regions))

    return findings


def _signed_direction_findings(metrics: dict[str, Any]) -> list[str]:
    if metrics.get("bitwise_identical") or metrics.get("mean_abs", 0.0) <= 0.0:
        return []

    signed_ratio = metrics["signed_ratio"]
    mean_signed = metrics["mean_signed"]
    mean_abs = metrics["mean_abs"]
    changed_fraction = metrics["pct_over_t"]

    if signed_ratio > SIGNED_RATIO_UNIFORM_HINT:
        direction = "brighter" if mean_signed > 0 else "darker"
        magnitude = abs(mean_signed)
        if changed_fraction >= CHANGED_FRACTION_UNIFORM_WORD_HINT:
            return [
                f"The changes all go one way: the frame is uniformly {direction} "
                f"by about {magnitude:.4f} on average on the 0-255 scale, which "
                "points at a scaling or conversion step rather than random scatter."
            ]
        return [
            f"Every pixel that changed moved the same way, averaging {magnitude:.4f} "
            f"{direction} where it differs, which points at a directed shift in the "
            "affected region rather than random scatter."
        ]

    if signed_ratio >= SIGNED_RATIO_SCATTER_HINT:
        return [
            "The difference is partly a uniform shift and partly scatter: signed and "
            f"absolute means diverge (mean signed {mean_signed:.4f} vs mean absolute "
            f"{mean_abs:.4f})."
        ]

    return [
        "The signed changes largely cancel out, so this looks like random scatter "
        "(rounding or precision) rather than a uniform offset: mean signed "
        f"{mean_signed:.4f} against mean absolute {mean_abs:.4f}."
    ]


def _channel_findings(metrics: dict[str, Any]) -> list[str]:
    if metrics.get("bitwise_identical"):
        return []

    per_channel = metrics["per_channel_mean"]
    values = {k: v for k, v in per_channel.items()}
    largest = max(values, key=values.get)
    smallest = min(values, key=values.get)
    high, low = values[largest], values[smallest]

    if low <= 0:
        if high > 0:
            return [
                f"Only the {largest} channel differs; {smallest} is untouched. A "
                "single-channel shift points at a precision or range fault rather than "
                "general numeric drift."
            ]
        return []

    if high / low >= CHANNEL_SKEW_RATIO_HINT:
        return [
            f"Drift is unbalanced across channels: {largest} averages {high:.4f} while "
            f"{smallest} averages {low:.4f}, a {high / low:.1f}x skew. Even drift would "
            "affect all three channels similarly, so this suggests a precision or "
            "colour-range fault."
        ]

    return [
        f"Drift is balanced across channels (R {per_channel['R']:.4f}, "
        f"G {per_channel['G']:.4f}, B {per_channel['B']:.4f}), consistent with "
        "general numeric variation rather than a channel-specific fault."
    ]


def _three_way_findings(comparison: dict[str, Any]) -> list[str]:
    if comparison["mode"] != "three_way":
        return []

    findings = [comparison["closer_to_reference_interpretation"]]
    baseline_mean = comparison["reference_vs_baseline"]["mean_abs"]
    current_mean = comparison["reference_vs_current"]["mean_abs"]
    adjacent_mean = comparison["baseline_vs_current"]["mean_abs"]

    findings.append(
        f"Measured against the reference, baseline drifts by {baseline_mean:.4f} and "
        f"current by {current_mean:.4f}; the two builds differ from each other by "
        f"{adjacent_mean:.4f}. Tracking the reference-relative numbers is what stops a "
        "regression from becoming invisible once it is promoted to a new baseline."
    )
    return findings


def _gate_findings(comparison: dict[str, Any]) -> list[str]:
    gate = comparison["gate"]
    verdict = gate["verdict"]

    if verdict == "REPORT_ONLY":
        return [
            "No gate limits were supplied, so this run reports measurements only and "
            "reaches no PASS or FAIL conclusion. That is deliberate: no threshold in "
            "this tool is legitimate until the calibration experiments are complete."
        ]

    failed = [check for check in gate.get("checks", []) if not check["passed"]]
    if verdict == "FAIL" and not gate.get("checks"):
        return [
            "The run is marked FAIL because a sanity check failed while gate limits "
            "were configured."
        ]
    if failed:
        detail = "; ".join(
            f"{check['name']} measured {check['actual']:.4f} against a limit of "
            f"{check['limit']:g}"
            for check in failed
        )
        return [f"FAIL against the limits you supplied: {detail}."]

    return [
        "PASS: every limit you supplied was satisfied. The verdict reflects your "
        "limits, not any threshold built into IRIS."
    ]


def _band_context(metrics: dict[str, Any]) -> list[str]:
    mean_abs = metrics["mean_abs"]
    healthy_low, healthy_high = OBSERVED_HEALTHY_MEAN_ABS
    seed_low, seed_high = OBSERVED_DIFFERENT_SEED_MEAN_ABS

    prefix = "Context from the prior cross-GPU study (observation, not a gate):"

    if healthy_low <= mean_abs <= healthy_high:
        return [
            f"{prefix} a mean of {mean_abs:.4f} falls inside the {healthy_low}-{healthy_high} "
            "band previously measured across GPUs on a single build, which was attributed "
            "to accumulated floating-point rounding rather than compositional change."
        ]
    if mean_abs < healthy_low:
        return [
            f"{prefix} a mean of {mean_abs:.4f} is below the {healthy_low}-{healthy_high} "
            "band previously measured across GPUs on a single build, so it is quieter than "
            "the observed healthy floor."
        ]
    if seed_low <= mean_abs <= seed_high:
        return [
            f"{prefix} a mean of {mean_abs:.4f} lands in the {seed_low}-{seed_high} band "
            "previously measured between images generated from different seeds, which "
            "indicates a wholesale compositional change rather than numeric drift. Verify "
            "the seed and prompt before investigating the software stack."
        ]
    return [
        f"{prefix} a mean of {mean_abs:.4f} sits above the {healthy_low}-{healthy_high} "
        f"healthy band and below the {seed_low}-{seed_high} different-seed band observed "
        "previously. This gap was not characterised by that study, so treat it as "
        "uncharted and investigate directly."
    ]


def _next_steps(
    comparison: dict[str, Any],
    metrics: dict[str, Any],
    provenance_diff: dict[str, Any] | None,
) -> list[str]:
    if metrics.get("bitwise_identical"):
        return [
            "Nothing to investigate for this pair. Repeated bitwise-identical runs are "
            "the evidence needed for the bitwise-identity calibration experiment."
        ]

    steps: list[str] = []

    if provenance_diff and provenance_diff.get("comparable") and not provenance_diff.get("matches"):
        steps.append(
            "Re-run with matched seed, prompt, sampler and checkpoint before drawing any "
            "conclusion from the pixel metrics."
        )

    regions = comparison.get("regions")
    if regions and regions.get("worst_blocks"):
        worst = regions["worst_blocks"][0]
        steps.append(
            f"Open the worst-region crop for x={worst['x0']}-{worst['x1']}, "
            f"y={worst['y0']}-{worst['y1']} to see what changed there, rather than reading "
            "the whole-frame difference map."
        )

    if any(not item["passed"] for item in comparison["sanity"]):
        steps.append(
            "Resolve the sanity failure first; a NaN, uniform or near-black frame makes "
            "the remaining statistics meaningless."
        )

    steps.append(
        "Compare this run against previous runs of the same workflow family before "
        "treating the numbers as a floor; per-family floors differ, and video workflows "
        "compound across frames."
    )
    return steps


def render_narrative(narrative: dict[str, Any]) -> str:
    """Render the narrative as a readable block of text."""
    lines = ["SUMMARY", "-" * 72]
    lines.extend(_wrap_paragraph(narrative["headline"]))
    lines.append("")

    if narrative["findings"]:
        lines.append("WHAT THIS MEANS")
        lines.append("-" * 72)
        for finding in narrative["findings"]:
            lines.extend(_wrap_bullet(finding))
        lines.append("")

    if narrative["context"]:
        lines.append("CONTEXT")
        lines.append("-" * 72)
        for item in narrative["context"]:
            lines.extend(_wrap_bullet(item))
        lines.append("")

    if narrative["next_steps"]:
        lines.append("WHAT TO DO NEXT")
        lines.append("-" * 72)
        for step in narrative["next_steps"]:
            lines.extend(_wrap_bullet(step))
        lines.append("")

    return "\n".join(lines)


def _wrap_paragraph(text: str, width: int = 72) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def _wrap_bullet(text: str, width: int = 72) -> list[str]:
    import textwrap

    wrapped = textwrap.wrap(text, width=width - 2) or [""]
    return [f"- {wrapped[0]}"] + [f"  {line}" for line in wrapped[1:]]
