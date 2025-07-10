import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


def convert_density_to_points(input_path, output_path, threshold, max_points=0, visualize=False):
    """
    加载SAX-NeRF预测的三维密度场，并将其转换为包含XYZ坐标和密度值的点云。
    输出格式为 (N, 4) 的Numpy数组，符合R²-Gaussian的预期输入。

    Args:
        input_path (str): 输入的 image_pred.npy 文件路径。
        output_path (str): 输出的点云 .npy 文件路径。
        threshold (float): 用于筛选高密度点的归一化密度阈值 (0.0 到 1.0)。
        max_points (int): 允许的最大点数。如果为0或负数，则不进行限制。
        visualize (bool): 是否显示一个密度切片以帮助选择阈值。
    """
    print(f"--- 开始转换 (V-Final) ---")
    print(f"加载密度场: {input_path}")

    # 1. 加载数据
    try:
        density_volume = np.load(input_path)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 at {input_path}")
        return

    print(f"成功加载密度场，形状为: {density_volume.shape}")

    # 2. 归一化密度场到 [0, 1] 区间
    vol_min, vol_max = density_volume.min(), density_volume.max()
    if vol_max > vol_min:
        normalized_volume = (density_volume - vol_min) / (vol_max - vol_min)
    else:
        normalized_volume = density_volume

    print(f"密度场已归一化。")

    # 3. 可选：可视化
    if visualize:
        # ... (visualize code remains the same) ...
        pass

    # 4. 应用阈值，提取高密度点的坐标索引
    print(f"应用阈值: {threshold}")
    point_indices = np.argwhere(normalized_volume > threshold)
    num_found_points = len(point_indices)
    print(f"在阈值 {threshold} 下找到了 {num_found_points} 个候选点。")

    # 5. 可选：随机采样
    if max_points > 0 and num_found_points > max_points:
        print(f"候选点数 ({num_found_points}) 超过了设定的最大值 ({max_points})。")
        print(f"将从这些点中随机采样 {max_points} 个点...")
        random_selection_indices = np.random.choice(num_found_points, max_points, replace=False)
        point_indices = point_indices[random_selection_indices]
        print("采样完成。")

    if point_indices.shape[0] == 0:
        print(f"警告: 在当前设置下没有提取到任何点。")
        return

    # 6. 将坐标索引归一化到 [-1, 1] 的标准空间
    resolution = np.array(density_volume.shape)
    points_normalized_xyz = (point_indices / (resolution - 1.0)) * 2.0 - 1.0
    points_normalized_xyz = points_normalized_xyz[:, [2, 1, 0]]  # 调整为 (x, y, z)

    # ### 新增核心逻辑：提取对应点的密度值 ###
    # 使用原始的 (z, y, x) 索引从归一化后的密度场中提取密度值
    densities = normalized_volume[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
    # 将密度值数组的形状从 (N,) 变为 (N, 1) 以便拼接
    densities = densities.reshape(-1, 1)
    print(f"已成功提取 {len(densities)} 个点的密度值。")
    # ### 逻辑结束 ###

    # 7. 将XYZ坐标和密度值拼接成一个 (N, 4) 的数组
    output_array = np.hstack((points_normalized_xyz, densities))
    print(f"最终点云数据形状为: {output_array.shape}")

    # 8. 保存为 .npy 文件
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_path, output_array.astype(np.float32))
    print(f"点云已成功保存到: {output_path}")
    print(f"--- 转换完成 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将SAX-NeRF的密度场转换为(N, 4)的3DGS初始点云（XYZ+Density）")
    parser.add_argument('--input_path', type=str, required=True,
                        help='输入的 image_pred.npy 文件路径')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出的点云坐标 .npy 文件路径')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='归一化后的密度阈值 (0.0 到 1.0 之间)')
    parser.add_argument('--max_points', type=int, default=0,
                        help='从阈值筛选后的点中随机采样的最大点数。0或负数表示不限制。')
    parser.add_argument('--visualize', action='store_true',
                        help='添加此标志以显示一个密度切片，帮助选择阈值')

    args = parser.parse_args()
    convert_density_to_points(args.input_path, args.output_path, args.threshold, args.max_points, args.visualize)