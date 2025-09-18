"""
tactile_pipeline.py
Mode behaviors:
 - mode=1: Use ChatGPT (M1) to refine prompt, load adapter (M3), generate 4 images with SD v1.5 (M2).
 - mode=2: Use ChatGPT image API only (M1 generates the image directly, no SD/LoRA used).
 - mode=3: Use ChatGPT to refine prompt, apply class-specific adapter (M3) on top of N different SD base models and generate images.

Notes:
 - Set environment variables OPENAI_API_KEY and HF_TOKEN as needed.
 - Place adapters named like: tactile_<class_name>.safetensors in ADAPTERS_DIR.
"""
# imports
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import base64
from io import BytesIO
import torch
from PIL import Image
import numpy as np
import os
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from diffusers import DPMSolverMultistepScheduler
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
)
from transformers import CLIPProcessor, CLIPModel
from transformers.utils import logging as hf_logging
import warnings

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tactile_pipeline")

hf_logging.set_verbosity_warning()         # transformers to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Using a slow image processor.*")

# Sanity checks
if not os.environ.get("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable not set!")
    exit(1)

if not os.environ.get("HF_TOKEN"):
    logger.warning("HF_TOKEN environment variable not set - this may cause issues with HuggingFace models")
# ==============
# Global config
# ==============
CLASS_NAMES = [
    # we have total = 66 classes, but for brevity, we will use a subset here.
    "cat", "dog", "airplane", "car", "bicycle", "chair",
]
RUN_LOG_FILENAME = "run_log.jsonl"
OPENAI_IMAGE_MODEL = "gpt-image-1"  # version used for Mode 2 image generation
OPENAI_VISION_MODEL = "gpt-4o-mini"     # for vision-based prompt refinement
# Img2Img controls
DEFAULT_DENOISING_STRENGTH = 0.9   # lower = stick closer to source image
NEGATIVE_PROMPT = ""                # set if you want (e.g., "blurry, low quality")
USE_IMG2IMG_FOR_MODE1 = True        # because we're relying on a natural image

SKIP_SAFETY_CHECK = True  
HARDCODED_PROMPT = "Create a tactile graphic of given object."
ADAPTERS_DIR = Path("./adapters")
DEFAULT_BASE_MODEL = "/home/student/khan/image_gen_pipe/base_model/deliberate_v3.safetensors" 

# mode 3 purpose is to apply the adapter on multiple base models for comparison.
BASE_MODELS_FOR_MODE3 = [
    "/home/student/khan/image_gen_pipe/base_model/deliberate_v3.safetensors",
    "/home/student/khan/image_gen_pipe/base_model/anything-v3-1.safetensors",
    "stablediffusionapi/realistic-vision",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    
]

# Prompt template for ChatGPT refinement
# Improved template with more tactile-specific details and design principles
PROMPT_TEMPLATE = (
    "Create a tactile graphic of {object} for visually impaired users. "
    "Focus on clear raised outlines of main features, distinct textural differences for different parts, "
    "simplified but accurate proportions, and high contrast between elements. "
    "Key features to emphasize: {patterns}."
)

SEEDS = [42, 1234, 2025, 9999]
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# Create OpenAI client if key provided
client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY not set. Mode 2 (ChatGPT image generation) will fail without it.")

# CLIP - upgraded to larger model
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # used for class detection

# ==============
# Utilities
# ==============
def image_to_data_url(image_path: str) -> str:
    p = Path(image_path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def generate_image_via_openai_edit(prompt: str, size_arg: str, image_path: str, tmp_dir: Path):
    """
    Try client.images.edits (OpenAI SDK v1+). If unavailable, call the REST endpoint directly.
    Returns the response object (client SDK) or dict (HTTP JSON).
    """
    # Ensure PNG for the edits endpoint
    src_png = ensure_png_for_openai(image_path, tmp_dir)
    
    # 1) Preferred: SDK v1 method (if available in your installed version)
    try:
        if hasattr(client, "images") and hasattr(client.images, "edits"):
            return client.images.edits(
                model=OPENAI_IMAGE_MODEL,
                image=open(src_png, "rb"),
                prompt=prompt,
                size=size_arg,
                n=1,
            )
    except Exception as e:
        logger.warning(f"OpenAI client.images.edits failed; will try HTTP fallback: {e}")

    # 2) Fallback: direct HTTP call to /v1/images/edits (works regardless of SDK version)
    try:
        import httpx
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
        with open(src_png, "rb") as f:
            files = {
                "image": ("image.png", f, "image/png"),
            }
            data = {
                "model": OPENAI_IMAGE_MODEL,
                "prompt": prompt,
                "size": size_arg,
                "n": "1",
            }
            r = httpx.post(
                "https://api.openai.com/v1/images/edits",
                headers=headers,
                data=data,
                files=files,
                timeout=60.0,
            )
            r.raise_for_status()
            return r.json()
    except Exception as e_http:
        raise RuntimeError(f"OpenAI image edit failed via client and HTTP fallback: {e_http}")


def ensure_png_for_openai(image_path: str, tmp_dir: Path) -> Path:
    p = Path(image_path)
    if p.suffix.lower() == ".png":
        return p
    ensure_dir(tmp_dir)
    tmp_png = tmp_dir / (p.stem + "_openai_tmp.png")
    img = Image.open(str(p)).convert("RGB")
    img.save(tmp_png, format="PNG")
    return tmp_png

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def append_run_log(
    out_root: Path,
    mode: int,
    object_category: str,
    refined_prompt: str,
    prompt_used_for_generation: str,
    baseline_model: str,
):
    """
    Append one record to a single JSONL log file under out_root.
    Fields:
    - mode
    - object_category
    - refined_prompt
    - prompt_used_for_generation
    - baseline_model
    """
    ensure_dir(out_root)
    record = {
        "mode": mode,
        "object_category": object_category,
        "refined_prompt": refined_prompt,
        "prompt_used_for_generation": prompt_used_for_generation,
        "baseline_model": baseline_model,
    }
    log_path = out_root / RUN_LOG_FILENAME
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



def load_image(path: str, size: Optional[int] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


# ==========================
# Class detection (CLIP)
# ==========================
class ClassDetectorCLIP:
    def __init__(self, device: str = DEVICE, model_name: str = CLIP_MODEL_NAME):
        logger.info("Loading CLIP model for class detection.")
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def detect_class(self, image_path: str, candidate_classes: List[str], threshold: float = 0.15) -> Tuple[Optional[str], float]:
        image = load_image(image_path, size=224)
        inputs = self.processor(text=candidate_classes, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**{k: v for k, v in inputs.items() if k.startswith("pixel")})
            txt_emb = self.model.get_text_features(inputs["input_ids"], attention_mask=inputs["attention_mask"]) 

        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)

        sims = (txt_emb @ img_emb.T).squeeze().cpu().numpy()
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_class = candidate_classes[best_idx]
        logger.info("Class identification results:")
        logger.info(f"CLIP best class: {best_class} (sim={best_sim:.4f})")
        if best_sim < threshold:
            return None, best_sim
        return best_class, best_sim


# ==========================
# ChatGPT prompt refinement
# ==========================
def refine_prompt_with_chatgpt(class_name: str, image_path: str, template: str, mode2_designer: bool = False) -> str:
    if not client:
        logger.warning("OpenAI client not available. Falling back to template-only prompt.")
        patterns = "major contours and prominent features"
        return template.format(object=class_name, patterns=patterns)

        # Build vision input
    try:
        data_url = image_to_data_url(image_path)
    except Exception as e:
        logger.warning(f"Failed to prepare image for vision model: {e}. Falling back to text-only refinement.")
        data_url = None

    try:
        if mode2_designer:
            # Vision + single-sentence refined prompt
            system_msg = (
                "You are an expert tactile designer. Apply tactile design principles (e.g., BANA). "
                "Given an input image, produce ONE concise sentence suitable for an image-generation prompt "
                "highlighting the specific tactile-relevant features present in THIS image. No preamble."
            )
            user_text = f"Class identified: {class_name}."
            user_content = [{"type": "text", "text": user_text}]
            if data_url:
                user_content.append({"type": "image_url", "image_url": {"url": data_url}})

            resp = client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=250,
                temperature=0.2,
            )
            final_prompt = resp.choices[0].message.content.strip()
            logger.info(f"Refined prompt (vision): {final_prompt}")
            return final_prompt
        else:
            # Vision + features list -> fill template
            system_msg = (
                "You are a helpful assistant that inspects an image and identifies tactile-relevant features. "
                "Return only a short comma-separated list of 2-6 phrases (no numbering, no extra text)."
            )
            user_text = (
                f"Class identified: {class_name}\n"
                "List tactile features present in THIS image (e.g., 'rounded ears', 'long tail', 'paw pads')."
            )
            user_content = [{"type": "text", "text": user_text}]
            if data_url:
                user_content.append({"type": "image_url", "image_url": {"url": data_url}})

            resp = client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=150,
                temperature=0.2,
            )
            content = resp.choices[0].message.content.strip()
            features_line = content.splitlines()[0]
            features = features_line.replace(" and ", ",").strip()
            final_prompt = template.format(object=class_name, patterns=features)
            logger.info(f"Refined prompt from ChatGPT (vision): {final_prompt}")
            return final_prompt
    except Exception as e:
        logger.warning(f"OpenAI vision call failed: {e}. Falling back to template-only prompt.")
        patterns = "major contours and prominent features"
        return template.format(object=class_name, patterns=patterns)

# ==========================
# Stable Diffusion loader + generator
# ==========================
def load_sd_pipeline(
    model_id: str,
    device: str = DEVICE,
    use_auth_token: Optional[str] = HF_TOKEN,
    img2img: bool = False,
):
    logger.info(f"Loading SD {'img2img' if img2img else 'txt2img'} pipeline for {model_id}...")
    
    # Check if model_id is a local .safetensors file
    model_path = Path(model_id)
    if model_path.exists() and model_path.suffix == '.safetensors':
        logger.info(f"Loading model from local safetensors file: {model_id}")
        
        # Use from_single_file method for loading safetensors (available in newer diffusers versions)
        try:
            PipeClass = StableDiffusionImg2ImgPipeline if img2img else StableDiffusionPipeline
            pipe = PipeClass.from_single_file(
                model_id,
                torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            )
        except AttributeError:
            # Fall back to manual loading for older diffusers versions
            logger.warning("from_single_file not available, using manual loading")
            from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
            pipe = download_from_original_stable_diffusion_ckpt(
                checkpoint_path_or_dict=model_id,
                from_safetensors=True,
                device=device,
            )
            
            # Convert to appropriate pipeline class
            PipeClass = StableDiffusionImg2ImgPipeline if img2img else StableDiffusionPipeline
            pipe = PipeClass(**pipe.components)
    else:
        # It's a model ID, load normally
        PipeClass = StableDiffusionImg2ImgPipeline if img2img else StableDiffusionPipeline
        pipe = PipeClass.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            token=use_auth_token,
        )
        
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if SKIP_SAFETY_CHECK:
        try:
            pipe.safety_checker = None
        except Exception:
            pass
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    
    return pipe


def generate_and_save_images(
    pipeline,
    prompt: str,
    seeds: List[int],
    outdir: Path,
    basename: str,
    init_image_path: Optional[str] = None,
    denoising_strength: Optional[float] = None,
    negative_prompt: Optional[str] = None,
):
    ensure_dir(outdir)
    results = []

    # Use the pipeline's tokenizer to properly truncate the prompt
    def truncate_prompt_for_pipeline(text, pipeline):
        if text is None:
            return None
            
        # Tokenize with the pipeline's tokenizer and truncate
        inputs = pipeline.tokenizer(
            text,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Decode the truncated tokens back to text
        truncated_text = pipeline.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        
        # Check if truncation occurred
        original_tokens = pipeline.tokenizer(text, return_tensors="pt").input_ids[0]
        if len(original_tokens) > pipeline.tokenizer.model_max_length:
            logger.warning(f"Prompt truncated to {pipeline.tokenizer.model_max_length} tokens. Original length: {len(original_tokens)}")
            
        return truncated_text

    # Truncate both prompt and negative prompt
    truncated_prompt = truncate_prompt_for_pipeline(prompt, pipeline)
    truncated_negative_prompt = truncate_prompt_for_pipeline(negative_prompt, pipeline)

    # Prepare init image if provided (img2img)
    init_img = None
    if init_image_path is not None:
        init_img = load_image(init_image_path)
        init_img = init_img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

    for s in seeds:
        generator = torch.Generator(device=DEVICE).manual_seed(s)
        logger.info(f"Generating image for seed {s} ...")
        
        with torch.autocast(DEVICE if DEVICE.startswith("cuda") else "cpu"):
            if init_img is not None and isinstance(pipeline, StableDiffusionImg2ImgPipeline):
                image = pipeline(
                    prompt=truncated_prompt,
                    image=init_img,
                    strength=denoising_strength if denoising_strength is not None else DEFAULT_DENOISING_STRENGTH,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    negative_prompt=truncated_negative_prompt,
                    generator=generator,
                ).images[0]
            else:
                image = pipeline(
                    truncated_prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    negative_prompt=truncated_negative_prompt,
                    generator=generator,
                ).images[0]

        fname = outdir / f"{basename}_seed{s}.png"
        image.save(fname)
        logger.info(f"Saved {fname}")
        results.append(str(fname))
    return results


# ==========================
# Main runner
# ==========================
def run_pipeline(
    image_path: str,
    mode: int = 1,
    class_list: List[str] = CLASS_NAMES,
    adapters_dir: Path = ADAPTERS_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    base_models_for_mode3: List[str] = BASE_MODELS_FOR_MODE3,
    out_root: Path = Path("./outputs"),
    class_name_override: Optional[str] = None,
):
    # Set up file logging in the output directory
    ensure_dir(out_root)
    log_file = out_root / f"pipeline_log_mode{mode}.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("=============================================")
    logger.info("      Starting TactileNet Pipeline       ")
    logger.info("=============================================")
    logger.info(f"Image: {image_path}")
    logger.info(f"Mode: {mode}")
    if mode in (1, 3):
        logger.info(f"Adapters dir: {adapters_dir}")
        logger.info(f"Base model: {base_model}")
    elif mode == 2:
    # define once near globals if you haven’t already:
    # OPENAI_IMAGE_MODEL = "gpt-image-1"
        logger.info(f"Image model: {OPENAI_IMAGE_MODEL}")
    logger.info(f"Output dir: {out_root}")
    logger.info("=============================================")

    if class_name_override:
        class_name = class_name_override
        logger.info(f"Using provided class override: {class_name}")
    else:
        detector = ClassDetectorCLIP(device=DEVICE)
        class_name, sim = detector.detect_class(image_path, class_list)
    if class_name is None:
        logger.error("User class un-identified (below similarity threshold). Quitting.")
        return

    mode2_designer = (mode == 2)
    refined_prompt = refine_prompt_with_chatgpt(class_name, image_path, PROMPT_TEMPLATE, mode2_designer=mode2_designer)
    logger.info(f"Refined prompt: {refined_prompt}")

    # For modes that require an adapter (1 and 3), check adapter availability.
    adapter_filename = adapters_dir / f"tactile_{class_name}.safetensors"
    if mode in (1, 3) and not adapter_filename.exists():
        logger.error(f"Adapter for class {class_name} not found at {adapter_filename}. Quitting.")
        return

    outdir = out_root / f"{Path(image_path).stem}_{class_name}_mode{mode}"
    ensure_dir(outdir)

    # Mode 1: Stable Diffusion + adapter (LoRA)
    if mode == 1:
        logger.info("Running Mode 1: Stable Diffusion with adapter (img2img). Baseline: 1 image (no adapter) + 4 images (with adapter).")
       
        base_pipe = load_sd_pipeline(base_model, device=DEVICE, img2img=USE_IMG2IMG_FOR_MODE1)
        baseline = generate_and_save_images(
            base_pipe,
            refined_prompt,
            [SEEDS[0]],
            outdir,
            basename=f"{class_name}_mode1_baseline",
            init_image_path=image_path if USE_IMG2IMG_FOR_MODE1 else None,
            denoising_strength=DEFAULT_DENOISING_STRENGTH,
            negative_prompt=NEGATIVE_PROMPT,
        )
                    
        append_run_log(
            out_root=out_root,
            mode=1,
            object_category=class_name,
            refined_prompt=refined_prompt,
            prompt_used_for_generation=refined_prompt,
            baseline_model=str(base_model),
        )

        # Adapter run: 4 images (with adapter) with the refined prompt
        adapter_pipe = load_sd_pipeline(base_model, device=DEVICE, img2img=USE_IMG2IMG_FOR_MODE1)
        try:
            adapter_pipe.load_lora_weights(str(adapter_filename))
            logger.info(f"Loaded LoRA adapter from {adapter_filename}")
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter {adapter_filename}: {e}")
        
        
        generated = generate_and_save_images(
            adapter_pipe,
            refined_prompt,
            SEEDS,
            outdir,
            basename=f"{class_name}_mode1",
            init_image_path=image_path if USE_IMG2IMG_FOR_MODE1 else None,
            denoising_strength=DEFAULT_DENOISING_STRENGTH,
            negative_prompt=NEGATIVE_PROMPT,
        )
        append_run_log(
            out_root=out_root,
            mode=1,
            object_category=class_name,
            refined_prompt=refined_prompt,
            prompt_used_for_generation=refined_prompt,
            baseline_model=str(base_model),
        )

        logger.info("Mode 1 completed.")

        return baseline + generated

    # Mode 2: ChatGPT image generation only (no SD/adapter)
    elif mode == 2:
        logger.info("Running Mode 2: ChatGPT image generation")
        if not client:
            logger.error("OpenAI client not configured (OPENAI_API_KEY missing). Cannot run mode 2.")
            return []

        # Compose tactile-design prompt
        prompt = (
            "Convert this natural image into a tactile graphic for individuals with visual impairments. "
            "Follow tactile design principles (e.g., BANA). "
            "Focus on raised, smooth lines to delineate key features. "
            f"{refined_prompt}"
        )
        logger.info("Running Mode 2: ChatGPT image generation (image-to-image via edits)")
        logger.info(f"Using ChatGPT image model: {OPENAI_IMAGE_MODEL}")

        # Compatible sizes supported by the OpenAI image endpoint
        SUPPORTED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}
        requested_size = f"{WIDTH}x{HEIGHT}"

        # Map unsupported sizes to a sensible default: prefer same-aspect 1024x1024 for square requests,
        # otherwise fall back to 'auto' which lets the model decide.
        if requested_size in SUPPORTED_IMAGE_SIZES:
            size_arg = requested_size
        else:
            if WIDTH == HEIGHT:
                size_arg = "1024x1024"
            else:
                size_arg = "auto"
            logger.info(f"Requested size {requested_size} not supported by API; using {size_arg} instead.")

        try:
            # NOTE: do not pass `response_format` parameter (not supported by all clients).
            src_png = ensure_png_for_openai(image_path, outdir)
            resp = generate_image_via_openai_edit(
                prompt=prompt,
                size_arg=size_arg,
                image_path=image_path,
                tmp_dir=outdir,
            )

            # --- robustly extract returned image data ---
            if isinstance(resp, dict):
                item = resp["data"][0]
            else:
                item = resp.data[0]

            b64_image = None
            img_url = None
            if isinstance(item, dict):
                b64_image = item.get("b64_json") or item.get("b64")
                img_url = item.get("url")
            else:
                b64_image = getattr(item, "b64_json", None) or getattr(item, "b64", None)
                img_url = getattr(item, "url", None)

            if b64_image:
                img_data = base64.b64decode(b64_image)
                img = Image.open(BytesIO(img_data)).convert("RGB")
            elif img_url:
                import httpx
                r = httpx.get(img_url, follow_redirects=True, timeout=30.0)
                r.raise_for_status()
                img = Image.open(BytesIO(r.content)).convert("RGB")
            else:
                raise RuntimeError("Image response did not contain 'b64_json' or 'url' fields. Response: " + str(resp))

            # If we requested '1024x1024' but want to save at user WIDTHxHEIGHT, resize down (keeps quality)
            if size_arg == "1024x1024" and (WIDTH != 1024 or HEIGHT != 1024):
                logger.info(f"Resizing generated image from 1024x1024 to {WIDTH}x{HEIGHT} for consistency with pipeline settings.")
                img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

            out_file = outdir / f"{class_name}_mode2_chatgpt.png"
            img.save(out_file)
            logger.info(f"Saved ChatGPT-generated tactile image to {out_file}")
            logger.info(f"Mode 2 completed. Model used: ChatGPT image API")
            logger.info(f"Prompt sent to ChatGPT: {prompt}")
            append_run_log(
                out_root=out_root,
                mode=mode,
                object_category=class_name,
                refined_prompt=refined_prompt,
                prompt_used_for_generation=prompt,
                baseline_model=OPENAI_IMAGE_MODEL,
            )
            return [str(out_file)]

        except Exception as e:
            logger.error(f"ChatGPT image generation failed: {e}")
            return []

    # Mode 3: Apply adapter on multiple base models
    elif mode == 3:
        logger.info("Running Mode 3: Apply adapter on multiple base models")
        all_results = {}
        for model_id in base_models_for_mode3:
            # Create a subdirectory for this model
            model_safe_name = Path(model_id).stem
            model_outdir = outdir / model_safe_name
            ensure_dir(model_outdir)
            
            try:
                base_pipe = load_sd_pipeline(model_id, device=DEVICE, img2img=True)
            except Exception as e:
                logger.error(f"Failed to load base model {model_id}: {e}")
                continue
            
            baseline = generate_and_save_images(
                base_pipe, refined_prompt, [SEEDS[0]], model_outdir,
                basename=f"{class_name}_{Path(model_id).name}_mode3_baseline",
                init_image_path=image_path,
                denoising_strength=DEFAULT_DENOISING_STRENGTH,
                negative_prompt=NEGATIVE_PROMPT,
            )
            
            append_run_log(
                out_root=out_root,
                mode=3,
                object_category=class_name,
                refined_prompt=refined_prompt,
                prompt_used_for_generation=refined_prompt,
                baseline_model=str(model_id),
            )
            
            adapter_pipe = load_sd_pipeline(model_id, device=DEVICE, img2img=True)
            try:
                adapter_pipe.load_lora_weights(str(adapter_filename))
                logger.info(f"Loaded LoRA adapter from {adapter_filename}")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapter {adapter_filename} for {model_id}: {e}")
                continue  # Skip this model if adapter loading fails

            generated = generate_and_save_images(
                adapter_pipe, refined_prompt, SEEDS, model_outdir,
                basename=f"{class_name}_{Path(model_id).name}_mode3",
                init_image_path=image_path,
                denoising_strength=DEFAULT_DENOISING_STRENGTH,
                negative_prompt=NEGATIVE_PROMPT,
            )
            append_run_log(
                out_root=out_root,
                mode=3,
                object_category=class_name,
                refined_prompt=refined_prompt,
                prompt_used_for_generation=refined_prompt,
                baseline_model=str(model_id),
            )
            all_results[model_id] = {"baseline": baseline, "refined": generated}
        return all_results

    # Remove file handler to avoid duplicate logs in future runs
    logger.removeHandler(file_handler)
    file_handler.close()
    
    
# ==========================
# CLI
# ==========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tactile generation pipeline driver")
    parser.add_argument("--image", required=True, help="Path to natural image")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3], help="Mode (1/2/3)")
    parser.add_argument("--adapters", default=str(ADAPTERS_DIR), help="Path to adapters directory")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="Base SD model id (for mode 1/2)")
    parser.add_argument("--out", default="./outputs", help="Output directory")
    args = parser.parse_args()

    ADAPTERS_DIR = Path(args.adapters)
    run_pipeline(args.image, mode=args.mode, adapters_dir=ADAPTERS_DIR, base_model=args.base_model, out_root=Path(args.out))