#!/bin/bash
# Full pipeline on mipNeRF360, tanks&temples and deep blending, dense capture:
# for each scene, (1) train a standard 3DGS splat, (2) fit the plug-and-play
# uncertainty channel on top (Bayesian regularizer disabled, the default for dense 
# captures), (3) evaluate + visualize the learned uncertainty.
#
# Usage: bash run_dense.sh
# Override paths/GPU via env vars, e.g.:
#   DATASET_BASE_PATH=/my/data OUTPUT_BASE_PATH=/my/output CUDA=1 bash run_dense.sh

set -e

DATASET_BASE_PATH="${DATASET_BASE_PATH:-$(dirname "$0")/data}"
OUTPUT_BASE_PATH="${OUTPUT_BASE_PATH:-$(dirname "$0")/output}"
CUDA="${CUDA:-0}"

DATASETS=(mipnerf360 tandt db)

declare -A scenes
scenes["mipnerf360"]="bicycle bonsai counter flowers garden kitchen room stump treehill"
scenes["tandt"]="train truck"
scenes["db"]="drjohnson playroom"

# mipNeRF360 outdoor scenes
OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
# mipNeRF360 indoor scenes
INDOOR_SCENES=(room counter kitchen bonsai)

for DATASET_NAME in "${DATASETS[@]}"; do
    for CLASSNAME in ${scenes[$DATASET_NAME]}; do

        # use same image resolutions as original 3DGS
        if [[ "${OUTDOOR_SCENES[*]}" =~ $CLASSNAME ]]; then
            IMAGES_DIR="images_4"
        elif [[ "${INDOOR_SCENES[*]}" =~ $CLASSNAME ]]; then
            IMAGES_DIR="images_2"
        else
            IMAGES_DIR="images"
        fi

        echo "=== [$CLASSNAME] Training 3DGS ==="
        CUDA_VISIBLE_DEVICES=$CUDA python train.py \
            -s "$DATASET_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
            -i $IMAGES_DIR \
            -m "$OUTPUT_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
            --iterations 30000 \
            --eval

        echo "=== [$CLASSNAME] Fitting the uncertainty channel ==="
        CUDA_VISIBLE_DEVICES=$CUDA python train_errors.py \
            -s "$DATASET_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
            -i $IMAGES_DIR \
            -m "$OUTPUT_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
            --iterations 3000 \
            --error_sh_degree 3 \
            --lambda_reg 0.0 \
            --eval

        echo "=== [$CLASSNAME] Evaluating the uncertainty channel ==="
        python uncertainty_metrics.py \
            -i "$OUTPUT_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
            -s test \
            --plot

    done

    echo "All $DATASET_NAME scenes done. Results under $OUTPUT_BASE_PATH/$DATASET_NAME/<scene>/renders/"

    echo "=== Summary across all scenes ==="
    python collect_results.py -i "$OUTPUT_BASE_PATH/$DATASET_NAME" -s test

done

echo "All datasets done.
