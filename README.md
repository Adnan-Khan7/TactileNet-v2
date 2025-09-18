# TactileNet-v2 — Quick pipeline overview

**Purpose (one line):**  
TactileNet converts natural images into tactile-optimized graphics by combining automatic class detection (CLIP), ChatGPT prompt refinement (vision-capable model), and either Stable Diffusion (with LoRA adapters) or the OpenAI image API — useful for producing tactile designs for blind/low-vision users.

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
**When to use:** Controlled experiments, local generation, reproducible baselines + adapter ablation.  
**Flow:**  
- CLIP class detection → ChatGPT prompt refinement → load SD pipeline (img2img).  
- Generate **1 baseline** image (no adapter).  
- Load class-specific LoRA adapter and generate **4 adapter-enhanced** images (different seeds).  
**Output location:** `./outputs/<image_stem>_<class>_mode1/` (contains baseline + adapter images + logs).

---

### Mode 2 — ChatGPT / OpenAI image edits
**Goal:** Use the OpenAI image/edit API to directly convert the natural image to a tactile graphic (no SD/LoRA).  
**When to use:** Fast prototyping or cloud-based single-model generation.  
**Flow:**  
- CLIP detection → concise vision-driven prompt (ChatGPT/OpenAI) → call `images.edits` (SDK or HTTP fallback) → save returned image.  
**Output:** single image `..._mode2_chatgpt.png` and run log entry.

---

### Mode 3 — Multi-base comparison with adapter
**Goal:** Apply the same class-specific adapter across multiple SD base models to evaluate robustness and differences.  
**When to use:** Robustness testing, ablation across base models.  
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

