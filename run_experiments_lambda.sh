#!/bin/bash

# --- 实验配置 ---
# 1. 设置你的数据集路径
SOURCE_DATA="/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/"

# 2. 设置新初始化文件的路径
#    请确保这个路径是正确的！
PLY_PATH="${SOURCE_DATA}init_edge_aware.npy"

# 3. 设置GPU ID
GPU_ID=7

# --- 检查初始化文件是否存在 ---
if [ ! -f "$PLY_PATH" ]; then
    echo "Error: Initialization file not found at $PLY_PATH"
    echo "Please generate it first using initialize_pcd.py"
    exit 1
fi

echo "Using smart initialization file: $PLY_PATH"
echo ""


# ======================================================================
# 实验一: 新初始化 + 无正则化
# ======================================================================
OUTPUT_DIR_1="output/chest_25view_edge_init_only"
LAMBDA_1=0.0

echo "======================================================================"
echo "Starting Experiment 1: Smart Initialization Only"
echo "Lambda_shape: ${LAMBDA_1}"
echo "Source data: ${SOURCE_DATA}"
echo "Output will be saved to: ${OUTPUT_DIR_1}"
echo "======================================================================"

python train.py \
  -s "${SOURCE_DATA}" \
  -m "${OUTPUT_DIR_1}" \
  --ply_path "${PLY_PATH}" \
  --lambda_shape ${LAMBDA_1} \
  --gpu_id ${GPU_ID} \
  --iterations 30000

echo "Finished Experiment 1. Results are in ${OUTPUT_DIR_1}"
echo ""
echo ""


# ======================================================================
# 实验二: 新初始化 + 最佳正则化
# ======================================================================
OUTPUT_DIR_2="output/chest_25view_edge_init_plus_reg"
LAMBDA_2=0.0005

echo "======================================================================"
echo "Starting Experiment 2: Smart Initialization + Shape Regularization"
echo "Lambda_shape: ${LAMBDA_2}"
echo "Source data: ${SOURCE_DATA}"
echo "Output will be saved to: ${OUTPUT_DIR_2}"
echo "======================================================================"

python train.py \
  -s "${SOURCE_DATA}" \
  -m "${OUTPUT_DIR_2}" \
  --ply_path "${PLY_PATH}" \
  --lambda_shape ${LAMBDA_2} \
  --gpu_id ${GPU_ID} \
  --iterations 30000

echo "Finished Experiment 2. Results are in ${OUTPUT_DIR_2}"
echo ""
echo ""


echo "All experiments completed."