#!/usr/bin/env bash
#
# Run SSTK dataset preparation: prepare (images -> MDS) then precompute (MDS -> latents).
# Set DATAROOT or edit vars below to change data paths.
#
# Usage: ./run.sh [prepare|precompute|all]
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATAROOT="${DATAROOT:-./sa1b}"
IMAGES_TXT="${DATAROOT}/image_paths.txt"
MDS_DIR="${DATAROOT}/mds"
LATENTS_DIR="${DATAROOT}/mds_latents_flux2"

run_prepare() {
    echo "=== Running SSTK prepare ==="
    python "$SCRIPT_DIR/prepare.py" \
        --images_txt "$IMAGES_TXT" \
        --local_mds_dir "${MDS_DIR}/" \
        --num_proc 16 \
        --seed 42 \
        --size 100000 \
        --min_size 512 \
        --min_aspect_ratio 0.67 \
        --max_aspect_ratio 1.33
}

run_precompute() {
    echo "=== Running precompute ==="
    python "$DATASETS_ROOT/precompute.py" \
        --datadir "${MDS_DIR}/" \
        --savedir "${LATENTS_DIR}/" \
        --num_proc 16 \
        --resolution 512 \
        --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \
        --batch_size 32 \
        --seed 42 \
        --model_dtype bfloat16 \
        --save_dtype float16 \
        --dataloader_workers 4
}

case "${1:-all}" in
    prepare)
        run_prepare
        ;;
    precompute)
        run_precompute
        ;;
    all)
        run_prepare
        run_precompute
        ;;
    *)
        echo "Usage: $0 [prepare|precompute|all]"
        exit 1
        ;;
esac
