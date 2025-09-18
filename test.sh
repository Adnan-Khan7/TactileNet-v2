#!/bin/bash

# ============================
# Runner for TactileNet
# Run: python test.py --mode 1 --dataset_jsonl /test_dataset.jsonl --dataset_base /path/to/repo/root --adapters ./adapters --base_model /path/to/base.safetensors --out ./outputs/dataset_run

# Example usage:
# mode 1: with adapters
# python test.py --mode 1 --dataset_jsonl test_dataset.jsonl --dataset_base /home/student/khan/image_gen_pipe/ --adapters ./adapters --base_model /home/student/khan/image_gen_pipe/base_model/deliberate_v3.safetensors
# ============================

IMG_DIR="/home/student/khan/image_gen_pipe/imgs/"

# Mode (1, 2, or 3)
MODE=3

# Adapters directory
ADAPTERS="./adapters"
BASE_MODEL="/home/student/khan/image_gen_pipe/base_model/deliberate_v3.safetensors"

# Output directory
OUTDIR="./outputs/"

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