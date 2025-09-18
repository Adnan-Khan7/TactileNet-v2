#!/bin/bash

# ============================
# Runner for TactileNet
# ============================

IMG_DIR="/home/student/khan/image_gen_pipe/imgs/airplane"

# Mode (1, 2, or 3)
MODE=3

# Adapters directory
ADAPTERS="./adapters"

# Required for mode 1
# BASE_MODEL="lykon/dreamshaper-7"
# BASE_MODEL="Linaqruf/anything-v3-1"
# BASE_MODEL="runwayml/stable-diffusion-v1-5"
BASE_MODEL="stablediffusionapi/realistic-vision"

# Output directory
OUTDIR="./outputs/chatgpt_image_gen"

echo "============================================="
echo "      Running TactileNet Pipeline       "
echo "============================================="
echo " Image dir:    $IMG_DIR"
echo " Mode:         $MODE"
if [ "$MODE" -ne 2 ]; then
echo " Adapters dir: $ADAPTERS"
echo " Base model:   $BASE_MODEL"
else
echo " Image model:  gpt-image-1"
fi
echo " Output dir:   $OUTDIR"
echo "============================================="

# Loop through images in IMG_DIR
for IMAGE_PATH in "$IMG_DIR"/*.{png,jpg,jpeg}; do
  # Skip if no files match
  [ -e "$IMAGE_PATH" ] || continue

  echo "Processing: $IMAGE_PATH"

  python test.py \
    --image "$IMAGE_PATH" \
    --mode $MODE \
    --adapters "$ADAPTERS" \
    --base_model "$BASE_MODEL" \
    --out "$OUTDIR"
done
