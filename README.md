# TactileNet-v2 — Quick pipeline overview

**Purpose:**  
TactileNet-v2 converts natural images into tactile-optimized graphics by combining automatic class detection (CLIP), ChatGPT prompt refinement (vision-capable model), and either Stable Diffusion (with LoRA adapters) or the OpenAI image API — useful for producing tactile designs for blind/low-vision users.

---

## Table of contents
- [How it works (high level)](#how-it-works-high-level)  
- [Modes (1 / 2 / 3)](#modes-1--2--3)  
  - [Mode 1 — SD + LoRA adapter](#mode-1---sd--lora-adapter)  
  - [Mode 2 — ChatGPT / OpenAI image edits](#mode-2---chatgpt--openai-image-edits)  
  - [Mode 3 — Multi-base comparison with adapter](#mode-3---multi-base-comparison-with-adapter)  
- [Inputs / Outputs](#inputs--outputs)  
- [Quick CLI examples](#quick-cli-examples)  
- [Operational notes & gotchas](#operational-notes--gotchas)  
- [Quick start checklist](#quick-start-checklist)  

---

## How it works (high level)
**Input:** a single natural image (or a JSONL dataset with `Category` + `input_image`) + optional adapters + SD base model(s).  
**Core pipeline steps (applies to all modes):**
1. Detect object class using CLIP.  
2. Refine a tactile-focused prompt with ChatGPT / vision-enabled model.  
3. Generate tactile images (mode-dependent).  
4. Save generated images and append a structured `run_log.jsonl` and a detailed `pipeline_log_mode<mode>.txt`.  

Adapters are expected in `./adapters` and must follow the naming pattern: `tactile_<class_name>.safetensors`.

---

## Modes (1 / 2 / 3)

### Mode 1 — SD + LoRA adapter
**Goal:** Produce a baseline SD img2img result and multiple adapter-enhanced variants for comparison.   
**Flow:**  
- CLIP class detection → ChatGPT prompt refinement → load SD pipeline (img2img).  
- Generate **1 baseline** image (no adapter).  
- Load class-specific LoRA adapter and generate **4 adapter-enhanced** images (different seeds).  
**Output location:** `./outputs/<image_stem>_<class>_mode1/` (contains baseline + adapter images + logs).

---

### Mode 2 — ChatGPT / OpenAI image edits
**Goal:** Use the OpenAI image/edit API to directly convert the natural image to a tactile graphic (no SD/LoRA).  
**Flow:**  
- CLIP detection → concise vision-driven prompt (ChatGPT/OpenAI) → call `images.edits` (SDK or HTTP fallback) → save returned image.  
**Output:** single image `..._mode2_chatgpt.png` and run log entry.

---

### Mode 3 — Multi-base comparison with adapter
**Goal:** Apply the same class-specific adapter across multiple SD base models to evaluate robustness and differences.  
**Flow:**  
- CLIP detection → ChatGPT refine → for each base model in `BASE_MODELS_FOR_MODE3`: generate 1 baseline, load adapter, generate N images (multiple seeds).  
**Output:** `./outputs/<image_stem>_<class>_mode3/<base_model>/` (baseline + generated images + per-base logs).

---

## Inputs / Outputs

**Required environment variables**
- `OPENAI_API_KEY` — **required** for ChatGPT / image generation (Mode 2 and prompt refinement).  
- `HF_TOKEN` — recommended for HuggingFace model downloads (when loading SD models from HF).

**Inputs**
- Single image: `--image path/to/img.jpg`  
- Or dataset: `--dataset_jsonl test_dataset.jsonl` (records with `Category` and `input_image`) + optional `--dataset_base` prefix.  
- Adapters (modes 1 & 3): `./adapters/tactile_<class>.safetensors`  
- Base SD model: local `.safetensors` or HF model id (`--base_model`)

**Outputs**
- Generated images under `./outputs/<image_stem>_<class>_mode<mode>/...`  
- `run_log.jsonl` — JSONL lines capturing `{mode, object_category, refined_prompt, prompt_used_for_generation, baseline_model}`  
- `pipeline_log_mode<mode>.txt` — detailed per-run text log

---

## Quick CLI examples

Single-image runs are supported for quick testing, and full-folder or dataset runs are also handled.

Mode 1 (local SD + adapter):
bash
python test.py --image path/to/img.jpg --mode 1 \
  --adapters ./adapters \
  --base_model /path/to/base_model.safetensors \
  --out ./outputs

Mode 2 (OpenAI image edits):
bash
export OPENAI_API_KEY="sk-..."
python test.py --image path/to/img.jpg --mode 2 --out ./outputs

Mode 3 (multi-base adapter comparison):
bash
python test.py --image path/to/img.jpg --mode 3 --adapters ./adapters --out ./outputs

---

## Dataset / Folder Runs

Instead of a single image, you can run on an entire dataset JSONL or loop through a folder of images.

Example (dataset JSONL):
bash
python test.py --mode 1 \
  --dataset_jsonl test_dataset.jsonl \
  --dataset_base /path/to/repo/root \
  --adapters ./adapters \
  --base_model /path/to/base_model.safetensors \
  --out ./outputs/dataset_run

Example (folder loop, bash script):
bash
IMG_DIR="/path/to/imgs/"
MODE=3
ADAPTERS="./adapters"
BASE_MODEL="/path/to/base_model.safetensors"
OUTDIR="./outputs"

for IMAGE_PATH in "$IMG_DIR"/*.{png,jpg,jpeg}; do
  [ -e "$IMAGE_PATH" ] || continue
  echo "Processing: $IMAGE_PATH"
  python test.py --image "$IMAGE_PATH" --mode $MODE \
    --adapters "$ADAPTERS" \
    --base_model "$BASE_MODEL" \
    --out "$OUTDIR"
done

---

## Operational notes & gotchas

- **Adapter naming is strict**: `tactile_<class>.safetensors`. If missing, modes 1/3 will abort for that image/class.  
- **CLIP detection** uses a similarity threshold — if below threshold, the pipeline exits (no guessing).  
- **Mode 2 (OpenAI)** supports certain sizes (e.g., `1024x1024`); pipeline maps unsupported sizes to sensible defaults and resizes output.  
- **`SKIP_SAFETY_CHECK`** is settable in the script — change only if you understand the implications.  
- **Device selection**: `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`. Local GPU recommended for SD runs.  
- **Logs** are append-only (`run_log.jsonl`) for easy downstream metrics and analysis.  

---

## Quick start checklist

- Export `OPENAI_API_KEY` in your environment.  
- (Optional) Export `HF_TOKEN` for HuggingFace downloads.  
- Put adapter files in `./adapters/` with names like `tactile_cat.safetensors`.  
- Run single-image tests with the example CLI for your desired mode.  
- Or run batch jobs using dataset JSONL / folder loop scripts.  
- Inspect `./outputs/` and `run_log.jsonl` for results and metadata.  

## Evaluation (ChatGPT-based, context-aware)

Purpose
- Assess how well each generated tactile graphic communicates the essential content of its reference (natural) image, using a single, context-primed ChatGPT run grounded in BANA tactile graphics principles.

Data artifacts
- Context triplets (few-shot): `context_data/t1/<Category>/img.(png|jpg)` + expert feedback/rating (and optional `context_data/t2/<Category>/img.(png|jpg)` as the “after”).
- Reference images: e.g., `test/<Category>/ref.png`.
- Generated outputs: e.g., `outputs_mode1/...`, `outputs_mode2/ref_banana_mode2/banana_mode2_chatgpt.png`, `outputs_mode3/...`.

Algorithm (high level)
1. Pair discovery
   - Recursively scan an outputs directory or a single generated image.
   - Infer `Category` and `Mode` from path names like `ref_<class>_mode<mode>/...`.
   - Resolve the corresponding reference via `ref_root + ref_pattern` (default `{class_name}/ref.png`).
   - Construct an evaluation list of pairs: (id, category, mode, reference image, generated tactile image).
2. Load few-shot context
   - Read 50+ examples from `context_data/t1` (and optional `t2`), plus `feedback.csv` (Category, Expert Feedback, Rating).
   - Each context item provides: Category, prior expert feedback/rating (e.g., “Missing landing gear”), t1 (before), and t2 (after) to demonstrate expected improvements.
3. Image preprocessing (cost-aware)
   - Resize all images to ≤ 640 px on the long side.
   - Convert to compressed JPEG; embed via base64 data URLs to reduce tokens and bandwidth.
4. Single-chat evaluation (batched)
   - Build one messages payload:
     - System prompt: defines the evaluator role, BANA-aligned criteria, allowed ratings, and strict JSON output format.
     - Context block: concise text + images for each few-shot triplet (t1 + feedback/rating + optional t2).
     - Evaluation block: for each pair, include item id, category, mode, reference (natural) image, and generated tactile image.
     - Output constraint: “Return STRICT JSON array only, one entry per item.”
   - Submit a single chat completion with a reasoning-capable, vision-enabled, cost-efficient model (default: gpt-4o-mini), temperature 0.0.
5. Parse and normalize results
   - Expect a JSON array of objects: `{id, rating, justification}`.
   - Normalize `rating` to one of: “Accept as Is”, “Accept with Minor Edits”, “Accept with Major Edits”, “Reject”.
   - Persist results to JSONL and CSV, joining metadata (category, mode, paths, timestamp).

Prompt design (core templates)
- System prompt (concise):
  - “You are an expert tactile graphics evaluator familiar with BANA tactile graphics guidelines. Evaluate how well a generated tactile graphic conveys the essential structure and features of its reference image. Prioritize: clear raised outlines, simplified but accurate proportions, distinct textures separating parts, high figure–ground clarity, reduced clutter, preservation of class-defining features, and tactile readability. Allowed ratings: ‘Accept as Is’, ‘Accept with Minor Edits’, ‘Accept with Major Edits’, ‘Reject’. Output STRICT JSON only; for each item return: {id, rating, justification (one concise sentence grounded in tactile principles/BANA/expert patterns)}.”
- Few-shot context block (per example):
  - Text: “Category: <Category>; Expert Feedback: <feedback or N/A>; Given Rating: <rating or N/A>; Before (t1):” + image(t1) [+ “After (t2):” + image(t2) if available].
- Evaluation block (per item):
  - Text: “Item id: <id>; Category: <category>; Mode: <mode>; Reference (natural) image:” + image(ref)
  - Text: “Generated tactile graphic:” + image(gen)
  - Final instruction: “Return STRICT JSON array only, no prose.”

Evaluation criteria (BANA-aligned, operationalized)
- Class-defining features are retained and unambiguous (e.g., airplane landing gear, animal ear shape, bicycle spokes vs. frame).
- Clear raised outlines for primary forms; simplified internal details that aid touch recognition.
- Distinct textures/line styles for different parts or materials; consistent coding.
- Accurate, simplified proportions; avoidance of visual clutter and overlapping confusion.
- Strong figure–ground separation; minimal distracting background.
- High-contrast partitioning (for mixed-ability use) without relying on shading that does not translate to tactile.
- Orientation, scale, and spatial relations that reduce ambiguity (e.g., limb crossings, occlusions).
- Proper omission/abstraction of non-essential details; no decorative noise that harms tactile legibility.

Flow (inputs-to-outputs perspective)
- Inputs:
  - Context: `context_data/t1`, optional `context_data/t2`, and `feedback.csv` mapping Category → Expert Feedback/Rating.
  - Pairs to evaluate: discovered automatically from outputs directories (by class/mode patterns) and matched to references via `ref_root + ref_pattern`.
  - Model: vision-capable mini-series reasoning model (default: gpt-4o-mini), temperature 0.0.
- Processing:
  - Preprocess images (resize/compress), assemble one chat with context + all evaluation pairs, request strict JSON output, parse/normalize ratings.
- ***Outputs***
  - `eval_results.jsonl` — one record per item:
    - `{id, category, mode, generated_image, ref_image, rating, justification, model, timestamp}`
  - `eval_results.csv` — same fields in tabular form for dashboards/metrics.

Cost and robustness strategies
- Single-call batching (context set once, all pairs evaluated together).
- Image downscaling and JPEG compression to reduce tokens.
- Optional context limiting (e.g., top-N examples) to fit budget while preserving class diversity.
- Deterministic pair IDs: `"<class>::<mode>::<file_stem>"` for stable traceability.
- Rating normalization and strict JSON parsing with conservative fallbacks if IDs mismatch.

Known limitations and next steps
- Extremely large batches may exceed context limits; chunking into multiple calls with shared context is the natural extension.
- Context quality governs evaluation sharpness; periodically refresh few-shot examples and expand coverage of edge cases.
- Consider light-weight rubric scoring (sub-scores for outline, texture coding, figure–ground) alongside the categorical rating for richer analytics.
