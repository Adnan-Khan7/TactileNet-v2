#!/bin/bash

# ============================
# Runner for TactileNet
# ============================

IMAGE_PATH="/home/student/khan/image_gen_pipe/imgs/airplane.png"

# Mode (1, 2, or 3)
MODE=3

# Adapters directory
ADAPTERS="./adapters"

# Base model (used in mode 1 and 2)
BASE_MODEL="runwayml/stable-diffusion-v1-5"

# Output directory
OUTDIR="./outputs"

echo "============================================="
echo "      Running TactileNet Pipeline       "
echo "============================================="
echo " Image:        $IMAGE_PATH"
echo " Mode:         $MODE"
echo " Adapters dir: $ADAPTERS"
echo " Base model:   $BASE_MODEL"
echo " Output dir:   $OUTDIR"
echo "============================================="

# Call Python script
python test.py \
  --image "$IMAGE_PATH" \
  --mode $MODE \
  --adapters "$ADAPTERS" \
  --base_model "$BASE_MODEL" \
  --out "$OUTDIR"

