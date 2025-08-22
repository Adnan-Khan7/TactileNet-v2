#!/bin/bash


# Example image path (change this to your input image)
IMAGE_PATH="/home/student/khan/image_gen_pipe/imgs/airplane.png"

# Mode (1, 2, or 3)
MODE=3

# Adapters directory
ADAPTERS="./adapters"
# /home/student/khan/youssif/instruct_reward_model/train_results/30_epochs_clip+lpips/lora_adapters/unet/adapter_model.safetensors

# Base model (used in mode 1 and 2)
BASE_MODEL="runwayml/stable-diffusion-v1-5"

# Output directory
OUTDIR="./outputs"

# Call Python script
python test.py \
  --image "$IMAGE_PATH" \
  --mode $MODE \
  --adapters "$ADAPTERS" \
  --base_model "$BASE_MODEL" \
  --out "$OUTDIR"
