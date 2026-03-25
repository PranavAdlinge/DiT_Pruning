#!/usr/bin/env bash
# Launch training with given config.
#
# Usage:
#   ./scripts/train.sh configs/klein4b/t2i_finetune.yaml
#   ./scripts/train.sh --config configs/klein4b/t2i_finetune.yaml --cuda_visible_devices 0,1
#   ./scripts/train.sh -c configs/klein4b/t2i_finetune.yaml -g 0,1

set -e
CONFIG=""
CUDA_VISIBLE_DEVICES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --cuda_visible_devices|-g)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        *)
            if [[ -z "$CONFIG" && "$1" != --* ]]; then
                CONFIG="$1"
            fi
            shift
            ;;
    esac
done

[[ -n "$CONFIG" ]] || { echo "Usage: $0 <config.yaml> | --config <config.yaml> [--cuda_visible_devices 0,1]"; exit 1; }

[[ -z "$CUDA_VISIBLE_DEVICES" ]] || export CUDA_VISIBLE_DEVICES
accelerate launch -m nexus.train.main --config "$CONFIG"
