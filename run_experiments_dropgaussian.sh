#!/bin/bash

# =====================================================================================
# CoR-GS 稀疏视角重建系列实验脚本
#
# 功能:
#   - 自动为不同数据集和实验配置创建独立的输出目录。
#   - 依次运行一系列对比实验，包括基线、形状正则化和DropGaussian。
#   - 所有输出（模型、日志、评估结果）都将保存在各自的实验目录中。
#
# 如何使用:
#   1. 修改下面的 "可配置变量" 部分，确保路径正确。
#   2. 在终端中给予此脚本执行权限: chmod +x run_experiments.sh
#   3. 运行脚本: ./run_experiments.sh
# =====================================================================================

# --- 可配置变量 ---

# 指定要使用的GPU ID
GPU_ID=5

# 基础数据目录 (包含所有数据集文件夹的路径)
BASE_DATA_DIR="/media/data2/hezhipeng/real_dataset/cone_ntrain_25_angle_360/"

# 基础输出目录 (所有实验结果将保存在此目录下)
BASE_OUTPUT_DIR="/home/hezhipeng/Workbench/r2_gaussian-main/output/dropgaussian25/"

# 要进行实验的数据集列表 (用空格分隔)
DATASETS=("walnut" "pine" "seashell")

# 正则化超参数
LAMBDA_SHAPE=0.0005
DROP_RATE_GAMMA=0.2

# --- 脚本主逻辑 ---

# 检查基础数据目录是否存在
if [ ! -d "$BASE_DATA_DIR" ]; then
    echo "错误: 基础数据目录 '$BASE_DATA_DIR' 不存在。"
    echo "请修改脚本中的 BASE_DATA_DIR 变量。"
    exit 1
fi

# 创建基础输出目录
mkdir -p "$BASE_OUTPUT_DIR"

# 遍历所有数据集
for DATASET_NAME in "${DATASETS[@]}"; do
    echo ""
    echo "================================================="
    echo "=== 开始处理数据集: ${DATASET_NAME}"
    echo "================================================="
    DATASET_PATH="${BASE_DATA_DIR}/${DATASET_NAME}"

    # 检查具体的数据集目录是否存在
    if [ ! -d "$DATASET_PATH" ]; then
        echo "警告: 数据集目录 '${DATASET_PATH}' 不存在，跳过此数据集。"
        continue
    fi

    # 定义两种初始化方法
    # 关联数组 (Bash 4.0+), key是描述, value是ply_path参数
    declare -A INIT_METHODS
    INIT_METHODS=(
        ["original_init"]="--ply_path ${DATASET_PATH}/init_${DATASET_NAME}.npy"
        ["edge_aware_init"]="--ply_path ${DATASET_PATH}/init_edge_aware.npy"
    )
    # 如果不使用 --ply_path，而是让代码自动寻找，可以设置为空字符串
    # INIT_METHODS["original_init"]=""

    # 遍历两种初始化方法
    for INIT_NAME in "${!INIT_METHODS[@]}"; do
        INIT_ARGS=${INIT_METHODS[$INIT_NAME]}

        # 检查初始化文件是否存在 (如果指定了路径)
        PLY_FILE_PATH=$(echo $INIT_ARGS | awk '{print $2}')
        if [[ -n "$PLY_FILE_PATH" && ! -f "$PLY_FILE_PATH" ]]; then
            echo "警告: 初始化文件 '${PLY_FILE_PATH}' 不存在，跳过 '${INIT_NAME}' 方法。"
            continue
        fi

        echo ""
        echo "-------------------------------------------------"
        echo "--- 使用初始化方法: ${INIT_NAME}"
        echo "-------------------------------------------------"

        # --- 实验 1: Baseline (无正则化) ---
        EXP_NAME="${DATASET_NAME}_${INIT_NAME}_baseline"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
        mkdir -p "$OUTPUT_DIR"
        echo "--- [1/4] 运行实验: ${EXP_NAME} ---"
        python train.py \
            --model_path "$OUTPUT_DIR" \
            --source_path "$DATASET_PATH" \
            --config "$CONFIG_FILE" \
            --gpu_id "$GPU_ID" \
            ${INIT_ARGS} \
            --lambda_shape 0.0 \
            --drop_rate_gamma 0.0
        echo "--- 实验 ${EXP_NAME} 完成 ---"


        # --- 实验 2: 仅形状正则化 (Shape Regularization) ---
        EXP_NAME="${DATASET_NAME}_${INIT_NAME}_shape_reg"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
        mkdir -p "$OUTPUT_DIR"
        echo "--- [2/4] 运行实验: ${EXP_NAME} ---"
        python train.py \
            --model_path "$OUTPUT_DIR" \
            --source_path "$DATASET_PATH" \
            --config "$CONFIG_FILE" \
            --gpu_id "$GPU_ID" \
            ${INIT_ARGS} \
            --lambda_shape ${LAMBDA_SHAPE} \
            --drop_rate_gamma 0.0
        echo "--- 实验 ${EXP_NAME} 完成 ---"


        # --- 实验 3: 仅 DropGaussian ---
        EXP_NAME="${DATASET_NAME}_${INIT_NAME}_dropgaussian"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
        mkdir -p "$OUTPUT_DIR"
        echo "--- [3/4] 运行实验: ${EXP_NAME} ---"
        python train.py \
            --model_path "$OUTPUT_DIR" \
            --source_path "$DATASET_PATH" \
            --config "$CONFIG_FILE" \
            --gpu_id "$GPU_ID" \
            ${INIT_ARGS} \
            --lambda_shape 0.0 \
            --drop_rate_gamma ${DROP_RATE_GAMMA}
        echo "--- 实验 ${EXP_NAME} 完成 ---"


        # --- 实验 4: 组合方法 (Combined) ---
        EXP_NAME="${DATASET_NAME}_${INIT_NAME}_combined"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
        mkdir -p "$OUTPUT_DIR"
        echo "--- [4/4] 运行实验: ${EXP_NAME} ---"
        python train.py \
            --model_path "$OUTPUT_DIR" \
            --source_path "$DATASET_PATH" \
            --config "$CONFIG_FILE" \
            --gpu_id "$GPU_ID" \
            ${INIT_ARGS} \
            --lambda_shape ${LAMBDA_SHAPE} \
            --drop_rate_gamma ${DROP_RATE_GAMMA}
        echo "--- 实验 ${EXP_NAME} 完成 ---"
    done
done

echo ""
echo "========================================="
echo "=== 所有实验已全部完成！ ==="
echo "========================================="