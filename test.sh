#!/bin/bash

# ============================
# Runner for TactileNet
# ============================

IMAGE_PATH="/home/student/khan/image_gen_pipe/imgs/airplane.png"

# Mode (1, 2, or 3)
MODE=2

# Adapters directory
ADAPTERS="./adapters"

# Required for mode 1
# BASE_MODEL="/home/student/khan/image_gen_pipe/base_model/deliberate_v3.safetensors"
BASE_MODEL=""

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

