"""
tactile_pipeline.py
Mode behaviors:
 - mode=1: Use ChatGPT (M1) to refine prompt, load adapter (M3), generate 4 images with SD v1.5 (M2).
 - mode=2: Use ChatGPT image API only (M1 generates the image directly, no SD/LoRA used).
 - mode=3: Use ChatGPT to refine prompt, apply class-specific adapter (M3) on top of three different SD base models and generate images.

Notes:
 - Set environment variables OPENAI_API_KEY and HF_TOKEN as needed.
 - Place adapters named like: tactile_<class_name>.safetensors in ADAPTERS_DIR.
"""
# imports
import json
import math
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import base64
from io import BytesIO
import torch
from PIL import Image
import numpy as np
import os
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from openai import OpenAI
import openai as openai_legacy 
from transformers import CLIPProcessor, CLIPModel
from safetensors.torch import load_file as safetensors_load
from compel import Compel, ReturnedEmbeddingsType


# more imports
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tactile_pipeline")

# Sanity checks
if not os.environ.get("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable not set!")
    exit(1)
else:
    logger.info("OPENAI_API_KEY is set")

if not os.environ.get("HF_TOKEN"):
    logger.warning("HF_TOKEN environment variable not set - this may cause issues with HuggingFace models")
    
    


# ==============
# Global config
# ==============
CLASS_NAMES = [
    # we have total = 66 classes, but for brevity, we will use a subset here.
    "cat", "dog", "airplane", "car", "bicycle", "chair",
]

# Img2Img controls
DEFAULT_DENOISING_STRENGTH = 0.9   # lower = stick closer to source image
NEGATIVE_PROMPT = ""                # set if you want (e.g., "blurry, low quality")
USE_IMG2IMG_FOR_MODE1 = True        # because we're relying on a natural image

SKIP_SAFETY_CHECK = True  
HARDCODED_PROMPT = "Create a tactile graphic of a frontal view of an airplane."
# HARDCODED_PROMPT = "Create a tactile graphic of a frontal view of an airplane, tailored for the visually impaired. The design should have raised, smooth lines to illustrate the plane's nose, cockpit windows, and wings spread wide, set against a plain background for clear contrast. The circular shape of the engines under each wing should be delineated with raised lines, and the cockpit windows' outline should be smoothly raised. The tires should be depicted with distinct raised textures to convey the rubbery material, contrasting with the plane's body's smoothness. The symmetry of the airplane should be emphasized with uniform raised lines to allow tactile comparison from left to right."
ADAPTERS_DIR = Path("./adapters")
# DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5" # v1.5 SD
DEFAULT_BASE_MODEL = "/home/student/khan/image_gen_pipe/base_model/deliberate_v3.safetensors" # the chosen base model of TactileNet pipeline, also based on V1.5

# # mode 3 purpose is to apply the adapter on multiple base models for comparison.
# BASE_MODELS_FOR_MODE3 = [
#     "runwayml/stable-diffusion-v1-5",
#     "/home/student/khan/image_gen_pipe/base_model/deliberate_v3.safetensors",
#     "Linaqruf/anything-v3.0",
#     "CompVis/stable-diffusion-v1-4",
#     "/home/student/khan/image_gen_pipe/base_model/anythingV3_fp16.ckpt",
# ]

# mode 3 purpose is to apply the adapter on multiple base models for comparison.
BASE_MODELS_FOR_MODE3 = [
    "/home/student/khan/image_gen_pipe/base_model/deliberate_v3.safetensors",
]

# Prompt template for ChatGPT refinement
# This template is used to generate a prompt for tactile graphics based on the identified class.
# It includes the object type and patterns to be highlighted in the tactile graphic.

PROMPT_TEMPLATE = (
    "Create a tactile graphic of a/an {object}, specifically designed for individuals "
    "with visual impairments. The graphic should feature raised, smooth lines to delineate the {patterns}, "
    "against a simplistic background"
)

try:
    CLIP_TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    logger.error(f"Failed to load CLIP tokenizer: {e}")
    CLIP_TOKENIZER = None

def truncate_prompt(prompt, max_length=77):
    if CLIP_TOKENIZER is None:
        return prompt  # fallback if tokenizer not available
    tokens = CLIP_TOKENIZER.tokenize(prompt)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        truncated_prompt = CLIP_TOKENIZER.convert_tokens_to_string(tokens)
        logger.warning(f"Prompt truncated to {max_length} tokens. Original length: {len(tokens)}")
        return truncated_prompt
    return prompt

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
    # also set legacy openai module key for backward compat if used elsewhere
    try:
        openai_legacy.api_key = OPENAI_API_KEY
    except Exception:
        pass
else:
    logger.warning("OPENAI_API_KEY not set. Mode 2 (ChatGPT image generation) will fail without it.")

# CLIP
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32" # used for class detection

# ==============
# Utilities
# ==============

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


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

    system_msg = "You are a helpful assistant that inspects an image and produces short bullet features describing the main tactile-relevant features."
    if mode2_designer:
        system_msg = (
            "You are an expert tactile designer. Take your time to apply tactile designing principles (e.g., BANA) "
            "and produce a refined prompt for a tactile graphic. Output must be a single sentence (no code) suitable for an image-generation prompt."
        )

    prompt_for_model = (
        f"Image: {Path(image_path).name}\n"
        f"Class identified: {class_name}\n"
        "Describe the primary tactile features to include (2-5 features), short phrases only.\n"
        "Examples: 'whiskers', 'rounded ears', 'long tail', 'paw pads', 'beak', 'wings'.\n"
        "Return only a comma-separated list of the features.\n"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_for_model},
            ],
            max_tokens=200,
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        features = content.splitlines()[0]
        features = features.replace(" and ", ",").strip()
        final_prompt = template.format(object=class_name, patterns=features)
        logger.info(f"Refined prompt from ChatGPT: {final_prompt}")
        return final_prompt
    except Exception as e:
        logger.warning(f"OpenAI call failed: {e}. Falling back to template-only prompt.")
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
                safety_checker=None if SKIP_SAFETY_CHECK else None,
            )
        except AttributeError:
            # Fall back to manual loading for older diffusers versions
            logger.warning("from_single_file not available, using manual loading")
            from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
            
            pipe = download_from_original_stable_diffusion_ckpt(
                checkpoint_path_or_dict=model_id,
                from_safetensors=True,
                device=device,
                load_safety_checker=not SKIP_SAFETY_CHECK,
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
):
    ensure_dir(out_root)
    detector = ClassDetectorCLIP(device=DEVICE)
    class_name, sim = detector.detect_class(image_path, class_list)
    if class_name is None:
        logger.error("User class un-identified (below similarity threshold). Quitting.")
        return

    mode2_designer = (mode == 2)
    refined_prompt = refine_prompt_with_chatgpt(class_name, image_path, PROMPT_TEMPLATE, mode2_designer=mode2_designer)

    # For modes that require an adapter (1 and 3), check adapter availability.
    adapter_filename = adapters_dir / f"tactile_{class_name}.safetensors"
    if mode in (1, 3) and not adapter_filename.exists():
        logger.error(f"Adapter for class {class_name} not found at {adapter_filename}. Quitting.")
        return

    outdir = out_root / f"{Path(image_path).stem}_{class_name}_mode{mode}"
    ensure_dir(outdir)

    # Mode 1: Stable Diffusion + adapter (LoRA)
    if mode == 1:
        # Use img2img so we can set denoising strength (we rely on the natural image)
        pipe = load_sd_pipeline(base_model, device=DEVICE, img2img=USE_IMG2IMG_FOR_MODE1)
        try:
            pipe.load_lora_weights(str(adapter_filename))
            logger.info(f"Loaded LoRA adapter from {adapter_filename}")
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter {adapter_filename}: {e}")

        generated = generate_and_save_images(
            pipe,
            refined_prompt,
            SEEDS,
            outdir,
            basename=f"{class_name}_mode1",
            init_image_path=image_path if USE_IMG2IMG_FOR_MODE1 else None,
            denoising_strength=DEFAULT_DENOISING_STRENGTH,
            negative_prompt=NEGATIVE_PROMPT,
        )
        logger.info("Generated images (refined prompt): " + ", ".join(generated))

        baseline = generate_and_save_images(
            pipe,
            HARDCODED_PROMPT,
            [SEEDS[0]],
            outdir,
            basename=f"{class_name}_mode1_baseline",
            init_image_path=image_path if USE_IMG2IMG_FOR_MODE1 else None,
            denoising_strength=DEFAULT_DENOISING_STRENGTH,
            negative_prompt=NEGATIVE_PROMPT,
        )
        logger.info("Generated baseline (hardcoded prompt): " + ", ".join(baseline))

        return generated + baseline

    # Mode 2: ChatGPT image generation only (no SD/adapter)
    elif mode == 2:
        if not client:
            logger.error("OpenAI client not configured (OPENAI_API_KEY missing). Cannot run mode 2.")
            return []

        # Compose tactile-design prompt
        prompt = (
            f"Convert this natural image into a tactile graphic for individuals with visual impairments. "
            f"Follow tactile design principles (e.g., BANA). "
            f"Focus on raised, smooth lines to delineate key features. "
            f"Class: {class_name}. "
            f"Prompt template: {PROMPT_TEMPLATE}. "
            f"Refined prompt: {refined_prompt}"
        )

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
            resp = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=size_arg,
                n=1,
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
            return [str(out_file)]

        except Exception as e:
            logger.error(f"ChatGPT image generation failed: {e}")
            return []

    # Mode 3: Apply adapter on multiple base models
    elif mode == 3:
        all_results = {}
        for model_id in base_models_for_mode3:
            try:
                pipe = load_sd_pipeline(model_id, device=DEVICE)
            except Exception as e:
                logger.error(f"Failed to load base model {model_id}: {e}")
                continue

            # --- model-specific output directory ---
            model_outdir = out_root / f"{class_name}_{Path(model_id).name}_mode3"
            model_outdir.mkdir(parents=True, exist_ok=True)

            # --- try loading adapter ---
            try:
                pipe.load_lora_weights(str(adapter_filename))
                logger.info(f"Loaded LoRA adapter from {adapter_filename}")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapter {adapter_filename} for {model_id}: {e}")

            # --- refined prompt ---
            generated = generate_and_save_images(
                pipe, refined_prompt, SEEDS, model_outdir,
                basename=f"{class_name}_{Path(model_id).name}_mode3"
            )

            # --- baseline hardcoded prompt (just 1 image, 1 seed) ---
            baseline = generate_and_save_images(
                pipe, HARDCODED_PROMPT, [SEEDS[0]], model_outdir,
                basename=f"{class_name}_{Path(model_id).name}_mode3_baseline"
            )

            all_results[model_id] = {"refined": generated, "baseline": baseline}

        logger.info(f"Mode 3 generation complete. Results: {json.dumps(all_results, indent=2)}")
        return all_results

    
    
    
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
