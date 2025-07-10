#!/bin/bash

# =====================================================================================
# Comprehensive Ablation Study: Scale Annealing vs. Baselines & Initializations
#
# This script systematically evaluates the impact of Scale Regularization Annealing.
#
# For each dataset, it performs:
#   1. Two BASELINE runs:
#      - An absolute baseline with original initialization.
#      - The current best model (Reg+Drop) with edge-aware initialization.
#   2. A PARAMETER SWEEP for the new annealing feature, testing it on BOTH
#      original and edge-aware initializations to find optimal settings.
#
# Author: Your Name (with assistance from AI)
# Date:   July 9, 2025
# =====================================================================================

# --- Fail on first error to ensure script integrity
set -e

# ---
# >>> 1. CENTRALIZED CONFIGURATION <<<
# ---
# Configure all experimental parameters here.

# Specify the GPU ID to use for all operations.
GPU_ID=5 # 请修改为您想用的GPU ID

# Base directory where dataset folders (e.g., walnut) are located.
BASE_DATA_DIR="/media/data2/hezhipeng/real_dataset/cone_ntrain_25_angle_360"

# Base directory where all output folders will be created.
BASE_OUTPUT_DIR="/home/hezhipeng/Workbench/r2_gaussian-main/output/ablation_final_annealing"

# List of datasets to run the experiments on.
DATASETS=("walnut" "pine" "seashell") # 您可以添加更多数据集, e.g., ("walnut" "pine" "seashell")

# --- Fixed Hyperparameters (kept constant across relevant experiments) ---
FIXED_LAMBDA_SHAPE=0.0005
FIXED_DROP_RATE_GAMMA=0.2
FIXED_EDGE_INIT_ALPHA=0.8
ITERATIONS=30000

# --- Annealing Parameter Sweep Configuration ---
# Values to test for the initial strength of scale regularization.
LAMBDA_SCALE_REG_VALUES=(0.1 0.2 0.5)
# Fixed duration for the annealing process.
FIXED_SCALE_REG_TO_ITER=15000

# ---
# >>> 2. SCRIPT EXECUTION LOGIC <<<
# ---

mkdir -p "$BASE_OUTPUT_DIR"
echo "All experiment outputs will be saved in: $BASE_OUTPUT_DIR"
echo ""

# Iterate over each dataset.
for DATASET_NAME in "${DATASETS[@]}"; do
    DATASET_PATH="${BASE_DATA_DIR}/${DATASET_NAME}"

    if [ ! -d "$DATASET_PATH" ]; then
        echo "Warning: Dataset directory '${DATASET_PATH}' not found. Skipping."
        continue
    fi

    echo "================================================="
    echo "=== Processing Dataset: ${DATASET_NAME}"
    echo "================================================="

    # Define paths to the two types of initializations
    INIT_FILE_PATH_ORIGINAL="${DATASET_PATH}/init_${DATASET_NAME}.npy"
    INIT_FILE_PATH_EDGE="${DATASET_PATH}/init_edge_aware_${FIXED_EDGE_INIT_ALPHA}.npy"

    # Check if initialization files exist
    if [ ! -f "$INIT_FILE_PATH_ORIGINAL" ]; then
        echo "ERROR: Original init file not found at ${INIT_FILE_PATH_ORIGINAL}. Skipping dataset."
        continue
    fi
     if [ ! -f "$INIT_FILE_PATH_EDGE" ]; then
        echo "ERROR: Edge-aware init file not found at ${INIT_FILE_PATH_EDGE}. You may need to generate it first. Skipping dataset."
        continue
    fi

    # --- Stage 1: Run Baseline Experiments ---
    echo ""
    echo "--- Running Baseline Experiments ---"

    # B1: Absolute Baseline (Original Init, no special techniques)
    echo "[B1/2] Running Absolute Baseline..."
    EXP_NAME_B1="${DATASET_NAME}_B1_absolute_baseline"
    OUTPUT_DIR_B1="${BASE_OUTPUT_DIR}/${EXP_NAME_B1}"
    mkdir -p "$OUTPUT_DIR_B1"
    python train.py \
        --model_path "$OUTPUT_DIR_B1" \
        --source_path "$DATASET_PATH" \
        --ply_path "$INIT_FILE_PATH_ORIGINAL" \
        --gpu_id "$GPU_ID" \
        --iterations "$ITERATIONS" \
        --lambda_shape 0.0 \
        --drop_rate_gamma 0.0 \
        --lambda_scale_reg 0.0

    # B2: Current SOTA (Edge Init + Reg + Drop)
    echo "[B2/2] Running Current SOTA Baseline..."
    EXP_NAME_B2="${DATASET_NAME}_B2_sota_baseline"
    OUTPUT_DIR_B2="${BASE_OUTPUT_DIR}/${EXP_NAME_B2}"
    mkdir -p "$OUTPUT_DIR_B2"
    python train.py \
        --model_path "$OUTPUT_DIR_B2" \
        --source_path "$DATASET_PATH" \
        --ply_path "$INIT_FILE_PATH_EDGE" \
        --gpu_id "$GPU_ID" \
        --iterations "$ITERATIONS" \
        --lambda_shape "$FIXED_LAMBDA_SHAPE" \
        --drop_rate_gamma "$FIXED_DROP_RATE_GAMMA" \
        --lambda_scale_reg 0.0

    # --- Stage 2: Run Parameter Sweep for Scale Annealing ---
    echo ""
    echo "--- Starting Parameter Sweep for Scale Annealing ---"

    for LAMBDA_REG in "${LAMBDA_SCALE_REG_VALUES[@]}"; do
        echo ""
        echo "-------------------------------------------------"
        echo "--- Testing Annealing with lambda_scale_reg = ${LAMBDA_REG}"
        echo "-------------------------------------------------"

        # O1: Ours on Original Init
        echo "[O1/2] Running with Annealing on Original Init..."
        EXP_NAME_O1="${DATASET_NAME}_O1_anneal_orig_lambda_${LAMBDA_REG}"
        OUTPUT_DIR_O1="${BASE_OUTPUT_DIR}/${EXP_NAME_O1}"
        mkdir -p "$OUTPUT_DIR_O1"
        python train.py \
            --model_path "$OUTPUT_DIR_O1" \
            --source_path "$DATASET_PATH" \
            --ply_path "$INIT_FILE_PATH_ORIGINAL" \
            --gpu_id "$GPU_ID" \
            --iterations "$ITERATIONS" \
            --lambda_shape "$FIXED_LAMBDA_SHAPE" \
            --drop_rate_gamma "$FIXED_DROP_RATE_GAMMA" \
            --lambda_scale_reg "$LAMBDA_REG" \
            --scale_reg_from_iter 0 \
            --scale_reg_to_iter "$FIXED_SCALE_REG_TO_ITER"

        # O2: Ours on Edge Init
        echo "[O2/2] Running with Annealing on Edge Init..."
        EXP_NAME_O2="${DATASET_NAME}_O2_anneal_edge_lambda_${LAMBDA_REG}"
        OUTPUT_DIR_O2="${BASE_OUTPUT_DIR}/${EXP_NAME_O2}"
        mkdir -p "$OUTPUT_DIR_O2"
        python train.py \
            --model_path "$OUTPUT_DIR_O2" \
            --source_path "$DATASET_PATH" \
            --ply_path "$INIT_FILE_PATH_EDGE" \
            --gpu_id "$GPU_ID" \
            --iterations "$ITERATIONS" \
            --lambda_shape "$FIXED_LAMBDA_SHAPE" \
            --drop_rate_gamma "$FIXED_DROP_RATE_GAMMA" \
            --lambda_scale_reg "$LAMBDA_REG" \
            --scale_reg_from_iter 0 \
            --scale_reg_to_iter "$FIXED_SCALE_REG_TO_ITER"
    done
done

echo ""
echo "========================================="
echo "=== ALL ABLATION EXPERIMENTS COMPLETED! ==="
echo "========================================="