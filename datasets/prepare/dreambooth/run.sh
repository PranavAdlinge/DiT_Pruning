#!/usr/bin/env bash
#
# Run DreamBooth dog example: prepare (dog-example -> MDS) then precompute (MDS -> latents).
# Prior preservation: GENERATE_CLASS_IMAGES=1 ./run.sh prepare
#
# Usage: ./run.sh [prepare|precompute|all]
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="${DATA_DIR:-./dreambooth}"
INSTANCE_DIR="${DATA_DIR}/instance"
CLASS_DIR="${DATA_DIR}/class"
MDS_DIR="${DATA_DIR}/mds"
LATENTS_DIR="${DATA_DIR}/mds_latents_flux2"

# Set to 1 to generate class images for prior preservation (requires GPU)
GENERATE_CLASS_IMAGES="${GENERATE_CLASS_IMAGES:-0}"

run_prepare() {
    echo "=== Running DreamBooth prepare ==="
    PREPARE_ARGS=(
        --dataset_name diffusers/dog-example
        --download_dir "$INSTANCE_DIR"
        --instance_prompt "a photo of sks dog"
        --local_mds_dir "$MDS_DIR"
        --num_proc 1
        --min_size 256
    )
    if [ "$GENERATE_CLASS_IMAGES" = "1" ]; then
        PREPARE_ARGS+=(
            --generate_class_images
            --class_prompt "a dog"
            --class_data_dir "$CLASS_DIR"
            --num_class_images 100
            --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B
            --sample_batch_size 1
            --class_image_resolution 512
            --prior_generation_precision bfloat16
        )
    fi
    python "$SCRIPT_DIR/prepare.py" "${PREPARE_ARGS[@]}"
}

run_precompute() {
    echo "=== Running precompute ==="
    python "$DATASETS_ROOT/precompute.py" \
        --datadir "$MDS_DIR" \
        --savedir "$LATENTS_DIR" \
        --num_proc 1 \
        --resolution 512 \
        --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \
        --batch_size 4 \
        --seed 42 \
        --model_dtype bfloat16 \
        --save_dtype float16 \
        --dataloader_workers 0
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
