# IRIS — Image Regression Inspection Suite

IRIS measures **pixel-level drift** between images produced by the same ComfyUI workflow on different software builds. Seeds are pinned per workflow, so the same workflow + seed + weights should yield the same composition on every run.

Healthy cross-GPU drift on one build is roughly **0.3–0.6 mean absolute difference** on a 0–255 scale (accumulated floating-point rounding, not compositional change). Different seeds produce ~57–84 mean abs with 99%+ of pixels changed. Those populations do not overlap.

IRIS is the measurement instrument for that regime.

## What IRIS does not do

- Generate images
- Modify ComfyUI
- Explain why an image changed
- Judge perceptual quality
- Claim absolute correctness
- Replace harness seed pinning

The narrative summary describes *what* the measured numbers look like and where the difference sits. It does not diagnose root cause.

## Installation

```bash
pip install -e ".[dev]"
```

Runtime dependencies are `numpy` and `Pillow`. No GPU required, no network access, nothing downloaded at runtime.

## Usage

### CLI

```bash
iris-compare \
  --reference ref.png \
  --current nightly.png \
  --threshold 1.0 \
  --heatmap-dir visuals \
  --out report.json
```

Three-way mode (reference, baseline/LKG, current/nightly):

```bash
iris-compare \
  --reference ref.png \
  --baseline lkg.png \
  --current nightly.png \
  --threshold 1.0 \
  --out report.json
```

Optional gate limits — **IRIS ships no default numeric gates**:

```bash
iris-compare \
  --reference ref.png \
  --current nightly.png \
  --threshold 1.0 \
  --max-mean-abs 0.8 \
  --max-p99-9 4.0 \
  --gate-exit \
  --out report.json
```

`--threshold` is **required** and has **no default**. It defines `pct_over_t` only and is not itself a pass/fail cut.

### Batch folder comparison (CLI)

Compare every matching PNG/JPEG across two build output folders:

```bash
python -m iris.cli \
  --reference-dir /path/to/build-A \
  --current-dir /path/to/build-B \
  --threshold 1.0 \
  --out /path/to/batch-report
```

Writes `batch_summary.json`, `batch_summary.csv`, and one JSON report per matched pair. Pairs images by filename (same workflow + seed name in both folders).

### Browser batch dashboard

Open `iris-batch.html` (or the hosted demo at `https://tejashsn.github.io/iris/iris-batch.html?demo=1`). Three ways to view results:

1. **Demo report** — pre-loaded ComfyUI build comparison (`?demo=1`)
2. **Upload** `batch_summary.json` from the CLI
3. **Compare two folders** directly in the browser (images never leave your machine)

The dashboard shows an executive summary, same/different status per workflow, all IRIS metrics in plain language, and unmatched files.

### Browser triage (single image)

Open `iris-triage.html` directly (`file://`). Drag and drop reference, optional baseline, and current images. No server, no CDN, no install, nothing leaves the machine. It strips alpha and computes on RGB samples so it matches Python exactly, and it parses ComfyUI PNG metadata in JavaScript so provenance checking works there too.

## What the report tells you

The text report leads with a verdict and a plain-language summary, then findings, then context, then what to check next. Raw metrics follow for readers who want them. A trimmed example:

```
VERDICT: REPORT_ONLY

SUMMARY
Comparing current against reference: the average RGB sample differs by
1.0060 on a 0-255 scale, 1.8698% of samples differ by more than the 1
threshold, and the largest single-sample difference is 235.00. The
99.9th percentile sits at 190.00, which is the number to watch, because
a localised fault moves the tail long before it moves the mean.

WHAT THIS MEANS
- The embedded ComfyUI metadata matches across images: same prompt,
  seed, sampler settings and models. Any difference measured below
  therefore comes from the software stack rather than the generation
  request.
- The 99.9th percentile is 189 times the mean, so the difference is
  concentrated in a small number of samples rather than spread evenly
  across the frame.
- The 5 worst blocks account for 47.5% of the total absolute difference,
  and the single largest contributor is x=288-320, y=96-128 at 25.4%.
```

## Metrics

Authoritative definitions live in `src/iris/metrics.py` only.

| Field | Definition |
| --- | --- |
| `bitwise_identical` | Exact equality short-circuit; skip all other stats |
| `mean_abs` | Mean absolute difference per RGB sample |
| `p99_9` | 99.9th percentile of absolute difference |
| `max_abs` | Maximum absolute difference |
| `pct_over_t` | Percent of samples exceeding `--threshold` |
| `per_channel_mean` | R/G/B means separately |
| `similarity_pct` | `100 * (1 - mean_abs/255)` — closeness to reference |
| `within_t_pct` | Share of samples within the threshold |
| `closer_to_reference_pct` | Three-way only: how much closer current is to reference than baseline; positive means less drift |

**Closeness, not quality:** all percentage fields measure closeness to the reference image, never visual quality improvement.

Statistics accumulate in float64 even though samples are float32. Summing millions of float32 values in float32 loses significant digits in exactly the 0.3–0.6 regime being measured, and it would diverge from the browser tool, where all arithmetic is float64.

## Pass/fail policy

| Gate limits supplied | Sanity | Verdict |
| --- | --- | --- |
| No | any | `REPORT_ONLY` |
| Yes | pass | `PASS` or `FAIL` from caller limits on `reference_vs_current` |
| Yes | fail | `FAIL` |

Prompt adherence is **never** part of this table. See below.

## Why not SSIM (or LPIPS) as a gate

At ~0.5/255 drift, SSIM saturates near 1.0 and LPIPS reads ~0 for both healthy and broken builds. Tail statistics (`p99_9`, `pct_over_t`, `max_abs`) stay discriminative where the mean does not. SSIM may be used as an optional triage artifact elsewhere, but never for gating.

## Why the mean is not the gate

A 32×32 corrupted patch in a 256×256 frame barely moves `mean_abs` but spikes `p99_9` and `pct_over_t`. Use tail metrics for localised faults and the mean as a trend line across builds. This is encoded as a unit test.

## Provenance checking (Tier 0)

ComfyUI embeds the API-format prompt graph as a PNG `tEXt` chunk on every image it writes. IRIS reads it from both images and compares prompt text, seed, sampler settings, and model names.

This is the cheapest useful check and it runs first, because it is deterministic and needs no model weights. If the seed or prompt differ, the pixel drift is explained by the harness and is not evidence of a software-stack regression — a conclusion that requires no statistics at all. Node ids are normalised away, so re-saving a workflow with renumbered nodes does not read as a parameter change.

Disable with `--no-provenance`. Degrades gracefully for JPEG, `.npy`, and PNGs whose metadata was stripped.

## Prompt adherence (advisory only)

`--prompt` and `--semantic-backend` score how well an image matches its prompt. Results land in their own namespace with verdict `ADVISORY` or `UNAVAILABLE`. **Nothing here can produce PASS or FAIL, and the gate never reads it.**

### Why it is walled off

Prompt-adherence metrics cannot detect the drift IRIS exists to measure. At 0.3–0.6/255 the CLIP embeddings and VQA answers for a healthy build and a broken one are identical to several decimal places. This is the same saturation problem that disqualified SSIM; the NeurIPS 2025 meta-evaluation of compositional metrics found VQA scores concentrate near 1.0 and that VQA-based metrics lean on answer-position shortcuts.

So this layer answers a different question: *is the workflow producing the right picture at all?* It catches a wrong checkpoint, a truncated prompt, a text-encoder precision fault, or a CLIP-skip misconfiguration — failures where the pixel metrics report enormous drift but cannot say which element went missing. It also scores the **reference** image, so an unrepresentative reference can be spotted.

### Backends

| Backend | Requires | Notes |
| --- | --- | --- |
| `none` (default) | nothing | Reports unavailability rather than faking a score |
| `local-clip` | `pip install -e ".[semantic]"` | Scores the whole prompt plus each clause separately, so weak clauses can be listed individually |

`local-clip` is a reference implementation, not the state of the art. It approximates TIFA/DSG-style decomposition without needing a VQA model. Register something stronger with:

```python
from iris.semantic import register_backend

register_backend("vqascore", lambda: MyVqaScoreBackend())
```

A backend needs `name`, `is_available()`, and `score(image, prompt)`. Reasonable choices to wrap: VQAScore, TIFA, DSG, ImageReward, HPSv2, or a vision-language judge. Absolute scores from all of these are uncalibrated — the useful signal is the difference between reference and current, and the relative ranking of clauses within one image.

## Prompt-element detection (advisory only)

`--detect-backend` answers a concrete question: for the prompt, which named objects are actually **present** in each image, and which are **missing**? It draws a labelled box around each detected object and lists every prompt element as detected (with confidence) or MISSING — so you can see at a glance that, say, the reference is missing "trees" and "sun", which is why it was flagged.

```bash
iris-compare \
  --reference ref.png \
  --current nightly.png \
  --threshold 1.0 \
  --detect-backend owlv2 \
  --heatmap-dir visuals \
  --out report.json
```

This writes `visuals/detection_reference.png` and `visuals/detection_current.png` (image + boxes + a present/absent legend), adds a `prompt_elements` block to the report, and surfaces missing elements in the narrative. Like everything semantic, it is **advisory only and never touches PASS/FAIL** — a missing object is a wrong-picture problem, orthogonal to pixel drift.

### Why it is separate from the pixel gate

Prompt-element detection cannot detect build drift, for the same reason SSIM cannot: at 0.3–0.6/255 a healthy and a broken build contain the identical objects. It catches a *different* failure — wrong checkpoint, truncated prompt, dropped conditioning — where the picture itself is wrong.

### Backends and elements

| Backend | Requires | Notes |
| --- | --- | --- |
| `none` (default) | nothing | Reports unavailability rather than inventing boxes |
| `owlv2` | `pip install -e ".[detect]"` | Open-vocabulary detector (google/owlv2-base-patch16-ensemble), real boxes, runs on CPU |

Object phrases are extracted from the prompt heuristically (`extract_prompt_elements`), or named explicitly with `--elements sun,mountain,tree,bottle`. Register a stronger detector (GroundingDINO, a VLM) with `register_detect_backend(name, factory)`; a backend needs `name`, `is_available()`, and `detect(image, queries, threshold) -> {query: [boxes]}`.

### In the browser

Real detection needs a model, which cannot run in a `file://` page with no dependencies. So the standalone HTML **visualises** detection instead of computing it: run the CLI with `--detect-backend owlv2`, then load that run's `report.json` into the "Prompt elements" panel. It draws the boxes on each image and fills the detected/missing checklist. Without a loaded report it shows the parsed prompt-element checklist in an "unknown — run detector" state and says so.

## Visuals

A full-resolution amplified difference map is a large, mostly-dark image with no scale, so IRIS writes three artifacts into `--heatmap-dir`:

**`overview_reference_vs_current.png`** — downscaled and colour-mapped with a printed legend stating the absolute difference that saturates the ramp. Downscaling uses **max-pooling, not averaging**, so a small hot patch survives the resize; averaging would let a 32×32 fault vanish inside a 2048×2048 frame, which is the exact failure mode these metrics exist to catch.

**`worst_region_N.png`** — a zoomed reference / current / difference triptych for the worst blocks. This is the artifact that shows *what* changed rather than only where.

**Block table** in the report — each frame is divided into a grid and blocks are ranked by their share of total absolute difference, with pixel coordinates. When three blocks out of sixty-four account for 98% of the difference, that sentence is worth more than any image.

Gain defaults to 8 and is stated in the legend. **Colour intensity is not severity.**

`--heatmap` still writes the raw full-resolution greyscale map for provenance.

## Sanity checks

Before any statistic, IRIS detects NaN and Inf samples, uniform frames, and near-black frames (`mean < 1.0` and `max < 2.0`). These are regressions in their own right and would otherwise pollute the distribution any future threshold is derived from. With gate limits configured, a sanity failure forces FAIL; without limits the report is annotated.

## Narrative and thresholds

The summary uses wording heuristics to decide phrasing — whether to call drift "concentrated" or "spread", for instance. These live in documented constants in `iris/narrative.py` and `iris/regions.py`, they select prose only, and **no verdict reads them**. The gate is driven exclusively by caller-supplied limits.

The summary also quotes the observed drift bands from the prior cross-GPU study as **context, explicitly labelled as observation and not a gate**, so a reader can tell whether a number resembles previously measured behaviour. Suppress with `--no-band-context`.

## Calibration experiments still outstanding

No gate value is legitimate until these are complete:

1. **Bitwise identity** — N repeats of one build on one GPU, each in a fresh process. If identical, cross-build delta belongs to the software stack and gates can be tight; if not, the reason is itself a finding.
2. **Cross-build floor** — same GPU, same pinned seed, adjacent nightlies. Include one pair where ROCm did not move and one where it did. Cross-GPU numbers bound hardware variance only and say nothing about software-stack variance.
3. **Prompt sensitivity** — 3–5 prompts per model family; the floor is texture-dependent.
4. **Fault injection** — perturb known-good output deliberately and confirm the tail metrics catch it. If injected faults land inside the healthy band, the metric selection is wrong.

Floors will need to be **per workflow family**, not global: plain Euler samplers respond differently to numeric perturbation than higher-order multistep solvers (`res_multistep`, `uni_pc`), and video workflows compound across frames.

## `.npy` support

At ~0.5/255 the measurement sits on the uint8 quantization floor, where the mean becomes a rounding-crossing rate rather than a true delta. Calibration studies must therefore use float32 tensors. Arrays with max ≤ 1.5 are treated as normalized [0,1] and scaled to 0–255; 2-D, 1-channel, 3-channel, and 4-channel (alpha dropped) inputs are all handled.

## Module layout

| Module | Responsibility |
| --- | --- |
| `metrics.py` | Authoritative metric definitions — the only place formulas live |
| `sanity.py` | Pre-statistic frame validity checks |
| `loader.py` | PNG / JPEG / `.npy` to float32 RGB in 0–255 |
| `regions.py` | Block-level localisation and concentration |
| `provenance.py` | ComfyUI PNG metadata extraction and diffing |
| `semantic.py` | Pluggable prompt-adherence backends (advisory) |
| `detect.py` | Pluggable prompt-element object detection (advisory) |
| `annotate.py` | Detection overlay: boxes + present/absent legend |
| `narrative.py` | Plain-language interpretation |
| `heatmap.py` | Overview, worst-region crops, raw maps |
| `metadata.py` | Environment capture, degrades without torch/ROCm |
| `report.py` | JSON and text assembly |
| `compare.py` | Pairwise and three-way orchestration, gate evaluation |
| `cli.py` | Argument parsing and wiring |

## Development

```bash
pip install -e ".[dev]"
pytest
```

The suite runs without a GPU and without network access. It includes browser/Python parity tests that reimplement the `iris-triage.html` computation loops in Python and assert agreement to within 1e-9, covering metrics, sanity, region analysis, max-pooling, and the colour ramp. A separate test asserts the constants embedded in the HTML still match the Python modules, since that is the other way the two entry points can drift apart.
