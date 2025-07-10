#!/bin/bash

# =====================================================================================
# CVPR-Standard End-to-End Ablation Study Script for Initialization Alpha
#
# Workflow for each (Dataset, Alpha) pair:
#   1. Initialization: Calls `initialize_pcd.py` to generate a specific
#      `init_edge_aware_{alpha}.npy` point cloud.
#   2. Training: Immediately calls `train.py` using the newly generated
#      point cloud and a fixed set of optimal hyperparameters.
#   3. Logging: All outputs are saved to a uniquely named directory, e.g.,
#      `walnut_alpha_0.8/`, ensuring clean and traceable results.
#
# Author: Your Name (with assistance from AI)
# Date:   July 5, 2025
# =====================================================================================

# --- Fail on first error to ensure script integrity
set -e

# ---
# >>> 1. CENTRALIZED CONFIGURATION <<<
# ---
# Configure all experimental parameters here.

# Specify the GPU ID to use for both initialization and training.
GPU_ID=5

# Base directory where dataset folders (pine, seashell, walnut) are located.
BASE_DATA_DIR="/media/data2/hezhipeng/real_dataset/cone_ntrain_25_angle_360"

# Base directory where all output folders will be created.
BASE_OUTPUT_DIR="/home/hezhipeng/Workbench/r2_gaussian-main/output/alpha_ablation"

# List of datasets to run the experiments on.
DATASETS=("walnut" "pine" "seashell")

# List of alpha (edge_weight) values for the ablation study.
ALPHA_VALUES=(0.5 0.8 1.0)

# Fixed hyperparameters for the training phase (our best combination).
LAMBDA_SHAPE=0.0005
DROP_RATE_GAMMA=0.2

# Path to the configuration file, if needed by train.py.
# If you don't use a config file, you can leave this empty: CONFIG_FILE=""
# Based on your previous script, it seems you might have one. Let's define it.
CONFIG_FILE="" # <--- !! IMPORTANT !! Please update this path.

# ---
# >>> 2. SCRIPT EXECUTION LOGIC <<<
# ---
# The main loop that orchestrates the entire experiment.

# Create the main output directory if it doesn't exist.
mkdir -p "$BASE_OUTPUT_DIR"
echo "All experiment outputs will be saved in: $BASE_OUTPUT_DIR"
echo ""

# Iterate over each dataset.
for DATASET_NAME in "${DATASETS[@]}"; do
    DATASET_PATH="${BASE_DATA_DIR}/${DATASET_NAME}"

    # Check if the specific dataset directory exists.
    if [ ! -d "$DATASET_PATH" ]; then
        echo "Warning: Dataset directory '${DATASET_PATH}' not found. Skipping."
        continue
    fi

    echo "================================================="
    echo "=== Processing Dataset: ${DATASET_NAME}"
    echo "================================================="

    # Iterate over each alpha value for the current dataset.
    for ALPHA in "${ALPHA_VALUES[@]}"; do
        echo ""
        echo "-------------------------------------------------"
        echo "--- Starting Experiment for Alpha = ${ALPHA}"
        echo "-------------------------------------------------"

        # --- Stage 1: Initialization ---
        echo "[1/2] Generating initial point cloud..."
        INIT_FILENAME="init_edge_aware_${ALPHA}.npy"
        INIT_FILE_PATH="${DATASET_PATH}/${INIT_FILENAME}"

        python initialize_pcd.py \
            --data "$DATASET_PATH" \
            --output "$INIT_FILE_PATH" \
            --gpu_id 0 \
            --edge_sampling \
            --edge_weight "$ALPHA"

        echo "Initialization for alpha=${ALPHA} complete. File saved to: ${INIT_FILE_PATH}"

        # --- Stage 2: Training ---
        echo "[2/2] Starting training run..."
        EXP_NAME="${DATASET_NAME}_alpha_${ALPHA}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
        mkdir -p "$OUTPUT_DIR"

        python train.py \
            --model_path "$OUTPUT_DIR" \
            --source_path "$DATASET_PATH" \
            --ply_path "$INIT_FILE_PATH" \
            --gpu_id 0 \
            --lambda_shape "$LAMBDA_SHAPE" \
            --drop_rate_gamma "$DROP_RATE_GAMMA"

        echo "--- Training for Alpha = ${ALPHA} on ${DATASET_NAME} finished. ---"
        echo "Results are in: ${OUTPUT_DIR}"
        echo "-------------------------------------------------"
        echo ""
    done
done

echo ""
echo "========================================="
echo "=== ALL ABLATION EXPERIMENTS COMPLETED! ==="
echo "========================================="