#!/bin/bash
# Sparse-view example: 3DGS + uncertainty fitting with --sparseTrainingViews, sweeping the
# Bayesian regularizer strength (lambda_reg) to show its effect under sparse supervision.
#
# Usage: bash run_sparse.sh
# Override paths/GPU via env vars, e.g.:
#   DATASET_BASE_PATH=/my/data OUTPUT_BASE_PATH=/my/output CUDA=1 bash run_sparse.sh

set -e

DATASET_BASE_PATH="${DATASET_BASE_PATH:-$(dirname "$0")/data}"
OUTPUT_BASE_PATH="${OUTPUT_BASE_PATH:-$(dirname "$0")/output_sparse}"
CUDA="${CUDA:-0}"

DATASET_NAME="mipnerf360"
SCENES=(bicycle bonsai counter flowers garden kitchen room stump treehill)

# mipnerf360 outdoor scenes
OUTDOOR_SCENES=("bicycle" "flowers" "garden" "stump" "treehill")
# mipnerf360 indoor scenes
INDOOR_SCENES=("room" "counter" "kitchen" "bonsai")

for CLASSNAME in "${SCENES[@]}"; do

    # use same image resolutions as original 3DGS
    if [[ "${OUTDOOR_SCENES[*]}" =~ $CLASSNAME ]]; then
        IMAGES_DIR="images_4"
    elif [[ "${INDOOR_SCENES[*]}" =~ $CLASSNAME ]]; then
        IMAGES_DIR="images_2"
    else
        IMAGES_DIR="images"
    fi

    echo "=== [$CLASSNAME] Training 3DGS on sparse views ==="
    CUDA_VISIBLE_DEVICES=$CUDA python train.py \
        -s "$DATASET_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
        -i $IMAGES_DIR \
        -m "$OUTPUT_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
        --iterations 4000 \
        --eval \
        --sparseTrainingViews

    echo "=== [$CLASSNAME] Fitting uncertainty on sparse views, no regularization (lambda=0.0) ==="
    CUDA_VISIBLE_DEVICES=$CUDA python train_errors.py \
        -s "$DATASET_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
        -i $IMAGES_DIR \
        -m "$OUTPUT_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
        --iterations 400 \
        --error_sh_degree 3 \
        --lambda_reg 0.0 \
        --eval \
        --sparseTrainingViews \
        --uncertain_background \
        --skip_training

    echo "=== [$CLASSNAME] Evaluating uncertainty, no regularization ==="
    python uncertainty_metrics.py -i "$OUTPUT_BASE_PATH/$DATASET_NAME/$CLASSNAME" -s test \
        --unc_vmin 0.0 --unc_vmax 1.0 --plot

    # for LAMBDA_REG in 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12 10.24 20.48; do
    #     echo "=== [$CLASSNAME] Fitting uncertainty with regularization (lambda=${LAMBDA_REG}) ==="
    #     CUDA_VISIBLE_DEVICES=$CUDA python train_errors.py \
    #         -s "$DATASET_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
    #         -i $IMAGES_DIR \
    #         -m "$OUTPUT_BASE_PATH/$DATASET_NAME/$CLASSNAME" \
    #         --iterations 400 \
    #         --error_sh_degree 3 \
    #         --lambda_reg $LAMBDA_REG \
    #         --eval \
    #         --sparseTrainingViews \
    #         --uncertain_background \
    #         --save_name error+regB${LAMBDA_REG} \
    #         --renders_folder renders+regB${LAMBDA_REG} \
    #         --skip_training

    #     echo "=== [$CLASSNAME] Evaluating uncertainty with regularization (lambda=${LAMBDA_REG}) ==="
    #     python uncertainty_metrics.py -i "$OUTPUT_BASE_PATH/$DATASET_NAME/$CLASSNAME" -s test \
    #         --renders_folder renders+regB${LAMBDA_REG} --unc_vmin 0.0 --unc_vmax 1.0 --plot
    # done

done

echo "All scenes done. Results under $OUTPUT_BASE_PATH/$DATASET_NAME/<scene>/renders*/"

echo "=== Summary across all scenes (lambda_reg=0.0) ==="
python collect_results.py -i "$OUTPUT_BASE_PATH/$DATASET_NAME" -s test
