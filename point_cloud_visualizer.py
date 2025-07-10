import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm
import argparse
import datetime
import yaml


def visualize_point_cloud(model_path, gt_volume_path=None):
    """
    可视化指定路径下的点云文件并生成GIF动画，以及生成三个方向的CT切片

    Args:
        model_path: 模型输出路径
        gt_volume_path: 真实CT体积数据路径，如果提供，会生成对照图
    """
    print(f"开始点云可视化: {model_path}")

    # 创建可视化结果保存路径
    vis_path = os.path.join(model_path, "point_cloud_visualization")
    os.makedirs(vis_path, exist_ok=True)

    # 查找最新的点云文件
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(point_cloud_dir):
        print(f"找不到点云目录: {point_cloud_dir}")
        return

    # 找到所有包含point_cloud.ply的子文件夹
    ply_folders = glob.glob(os.path.join(point_cloud_dir, "iteration_*"))
    if not ply_folders:
        print(f"找不到点云文件: {point_cloud_dir}")
        return

    # 按迭代次数排序，取最大的
    latest_folder = sorted(ply_folders, key=lambda x: int(x.split("_")[-1]))[-1]
    ply_path = os.path.join(latest_folder, "point_cloud.ply")

    if not os.path.exists(ply_path):
        print(f"找不到PLY文件: {ply_path}")
        return

    print(f"正在处理点云文件: {ply_path}")

    # 获取场景名称
    scene_name = os.path.basename(model_path.rstrip("/"))
    if not scene_name:
        scene_name = "scene"

    # 创建场景特定的可视化路径
    scene_vis_path = os.path.join(vis_path, f"{scene_name}")
    os.makedirs(scene_vis_path, exist_ok=True)

    # 加载PLY文件
    try:
        plydata = PlyData.read(ply_path)
    except Exception as e:
        print(f"读取PLY文件出错: {e}")
        return

    # 提取xyz坐标
    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1)

    # 尝试提取不透明度，如果可用
    try:
        opacities = np.asarray(plydata.elements[0]["opacity"])
        print(f"不透明度范围: {opacities.min()} - {opacities.max()}")
    except ValueError:
        print("找不到不透明度信息，使用默认值")
        opacities = np.ones(xyz.shape[0])

    # 尝试提取颜色信息，如果可用
    try:
        colors = np.stack((
            np.asarray(plydata.elements[0]["red"]),
            np.asarray(plydata.elements[0]["green"]),
            np.asarray(plydata.elements[0]["blue"])
        ), axis=1) / 255.0
        print(f"提取颜色信息, 形状: {colors.shape}")
    except ValueError:
        print("找不到颜色信息，使用灰色")
        # 使用单一颜色而不是每个点一个颜色
        colors = 'gray'

    # 尝试提取密度信息
    try:
        densities = np.asarray(plydata.elements[0]["density"])
        print(f"密度范围: {densities.min()} - {densities.max()}")
        # 根据密度值对点进行过滤，去除密度过低的点
        density_threshold = np.percentile(densities, 10)  # 过滤掉最低10%的密度值
        valid_indices = densities > density_threshold
        xyz = xyz[valid_indices]
        opacities = opacities[valid_indices]
        if isinstance(colors, np.ndarray):
            colors = colors[valid_indices]
        densities = densities[valid_indices]
        print(f"根据密度过滤后剩余点数: {xyz.shape[0]} / {len(valid_indices)}")
    except ValueError:
        print("找不到密度信息，使用所有点")
        densities = None

    # 创建图形对象
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # 计算点云中心和范围
    center = np.mean(xyz, axis=0)
    max_range = np.max(np.ptp(xyz, axis=0)) * 0.55

    # 设置轴范围为点云中心的立方体
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # 根据点数量决定点大小
    point_size = max(0.1, min(1.0, 1000.0 / np.sqrt(xyz.shape[0])))

    # 不透明度归一化处理
    if opacities.max() > opacities.min():
        alpha = (opacities - opacities.min()) / (opacities.max() - opacities.min())
    else:
        alpha = np.ones_like(opacities) * 0.5

    alpha = np.clip(alpha * 1.2, 0.1, 1.0)

    # 绘制散点图
    print(f"绘制散点图: 点数={xyz.shape[0]}, 点大小={point_size}, alpha形状={alpha.shape}")

    # 使用不透明度的标量值而不是数组
    scatter = ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        s=point_size,
        alpha=0.5,  # 使用固定不透明度避免维度不匹配
        c=colors,
        edgecolors='none'
    )

    # 取消三维网格坐标系
    ax.axis('off')
    ax.grid(False)

    # 上下翻转三维图像以匹配常规视角
    # ax.invert_zaxis()

    # 记录静态视图
    elevation, azimuth = 20, 30
    ax.view_init(elev=elevation, azim=azimuth)
    static_view_path = os.path.join(scene_vis_path, f'{scene_name}_static_view.png')
    plt.savefig(static_view_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"静态视图已保存: {static_view_path}")

    # 绘制动画序列
    proj_num = 36  # 每10度一帧，共36帧
    angle_interval = 360 / proj_num
    image_files = []

    print("生成旋转视图序列...")
    for i in tqdm(range(proj_num)):
        angle = i * angle_interval
        ax.view_init(elev=elevation, azim=angle)
        frame_path = os.path.join(scene_vis_path, f'frame_{i:03d}_elev_{elevation}_azim_{angle:.1f}.png')
        plt.savefig(frame_path, dpi=200, bbox_inches='tight', pad_inches=0)
        image_files.append(frame_path)

    # 创建GIF动画
    fps = 30
    duration = 1500 / fps  # 每帧持续时间(毫秒)
    gif_filename = os.path.join(scene_vis_path, f'{scene_name}_rotation_fps_{fps}.gif')

    print(f"正在创建GIF动画: {gif_filename}")

    # 读取第一帧以确定裁剪区域
    try:
        img = Image.open(image_files[0])
        width, height = img.size

        # 确定裁剪区域 - 自动计算以移除边距
        left_margin = int(width * 0.1)
        right_margin = int(width * 0.1)
        top_margin = int(height * 0.1)
        bottom_margin = int(height * 0.1)

        box = (
            left_margin,
            top_margin,
            width - right_margin,
            height - bottom_margin
        )

        # 裁剪并创建GIF帧
        gif_frames = []
        for filename in tqdm(image_files):
            img = Image.open(filename)
            try:
                cropped_img = img.crop(box)
                gif_frames.append(cropped_img)
            except Exception as e:
                print(f"裁剪图像失败: {e}")
                gif_frames.append(img)  # 如果裁剪失败则使用原始图像

        # 保存GIF动画
        gif_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF动画已保存: {gif_filename}")

        # 删除临时帧图像
        for file in image_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"无法删除临时文件 {file}: {e}")

    except Exception as e:
        print(f"创建GIF动画时出错: {e}")

    plt.close(fig)

    # 生成三个方向的CT切片图
    print("开始生成CT切片图...")
    generate_ct_slices(model_path, scene_vis_path)

    print(f"点云可视化完成: {scene_vis_path}")
    return scene_vis_path


def generate_ct_slices(model_path, output_path):
    """
    生成三个方向的CT切片图

    Args:
        model_path: 模型路径，包含vol_pred.npy和vol_gt.npy
        output_path: 输出路径
    """
    slices_path = os.path.join(output_path, "ct_slices")
    os.makedirs(slices_path, exist_ok=True)

    # 查找最新的评估结果
    eval_dir = os.path.join(model_path, "eval")
    if not os.path.exists(eval_dir):
        print(f"找不到评估目录: {eval_dir}")
        return

    # 找到最新的评估文件夹
    eval_folders = sorted(glob.glob(os.path.join(eval_dir, "iter_*")),
                          key=lambda x: int(x.split("_")[-1]))

    if not eval_folders:
        print(f"找不到评估结果: {eval_dir}")
        return

    latest_eval = eval_folders[-1]
    print(f"使用最新评估结果: {latest_eval}")

    # 寻找vol_pred.npy和vol_gt.npy
    vol_pred_path = os.path.join(latest_eval, "vol_pred.npy")
    vol_gt_path = os.path.join(latest_eval, "vol_gt.npy")

    # 检查文件是否存在
    if not os.path.exists(vol_pred_path):
        print(f"找不到预测体积数据: {vol_pred_path}")
        # 尝试在其他位置查找
        vol_pred_path = find_volume_file(model_path, "vol_pred.npy")
        if not vol_pred_path:
            print("无法找到vol_pred.npy文件，将跳过切片生成")
            return

    # 加载预测体积
    vol_pred = np.load(vol_pred_path)
    print(f"已加载预测体积: {vol_pred.shape}")

    # 尝试加载GT体积（如果存在）
    vol_gt = None
    if os.path.exists(vol_gt_path):
        vol_gt = np.load(vol_gt_path)
        print(f"已加载GT体积: {vol_gt.shape}")
    else:
        print(f"找不到GT体积数据: {vol_gt_path}")
        # 尝试在其他位置查找
        vol_gt_path = find_volume_file(model_path, "vol_gt.npy")
        if vol_gt_path:
            vol_gt = np.load(vol_gt_path)
            print(f"已加载GT体积: {vol_gt.shape}")

    # 生成切片位置
    shape = vol_pred.shape
    slice_positions = {
        'axial': np.linspace(shape[0] * 0.3, shape[0] * 0.7, 5).astype(int),  # Z方向5个切片
        'coronal': np.linspace(shape[1] * 0.3, shape[1] * 0.7, 5).astype(int),  # Y方向5个切片
        'sagittal': np.linspace(shape[2] * 0.3, shape[2] * 0.7, 5).astype(int)  # X方向5个切片
    }

    # 生成三个方向的CT切片图
    directions = ['axial', 'coronal', 'sagittal']
    for direction in directions:
        slice_dir_path = os.path.join(slices_path, direction)
        os.makedirs(slice_dir_path, exist_ok=True)

        for i, pos in enumerate(slice_positions[direction]):
            fig, axes = plt.subplots(1, 2 if vol_gt is not None else 1, figsize=(10, 5 if vol_gt is not None else 5))

            # 如果没有GT数据，axes可能不是数组
            if vol_gt is None:
                axes = [axes]

            # 获取预测切片
            if direction == 'axial':
                slice_pred = vol_pred[pos, :, :]
                slice_gt = vol_gt[pos, :, :] if vol_gt is not None else None
                title = f'Axial Slice (Z={pos})'
            elif direction == 'coronal':
                slice_pred = vol_pred[:, pos, :]
                slice_gt = vol_gt[:, pos, :] if vol_gt is not None else None
                title = f'Coronal Slice (Y={pos})'
            else:  # sagittal
                slice_pred = vol_pred[:, :, pos]
                slice_gt = vol_gt[:, :, pos] if vol_gt is not None else None
                title = f'Sagittal Slice (X={pos})'

            # 显示预测切片
            axes[0].imshow(slice_pred, cmap='gray')
            axes[0].set_title('Prediction: ' + title)
            axes[0].axis('off')

            # 如果有GT数据，显示对比
            if vol_gt is not None:
                axes[1].imshow(slice_gt, cmap='gray')
                axes[1].set_title('Ground Truth: ' + title)
                axes[1].axis('off')

            # 保存图像
            plt.tight_layout()
            slice_path = os.path.join(slice_dir_path, f'{direction}_slice_{i + 1}.png')
            plt.savefig(slice_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            print(f"生成切片: {slice_path}")


def find_volume_file(base_path, filename):
    """
    递归查找指定文件名的文件

    Args:
        base_path: 起始搜索路径
        filename: 要查找的文件名

    Returns:
        找到的文件路径，如果未找到则返回None
    """
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


# 添加到训练完成后的回调
def on_training_finish(model_path):
    """训练完成后执行的回调函数"""
    print("训练完成，等待vol_pred.npy生成完成...")
    # 等待vol_pred.npy生成完成
    max_wait = 30  # 最多等待30秒
    waited = 0
    vol_pred_path = None

    while waited < max_wait:
        # 尝试查找vol_pred.npy
        vol_pred_path = find_volume_file(model_path, "vol_pred.npy")
        if vol_pred_path:
            print(f"找到vol_pred.npy: {vol_pred_path}")
            break

        import time
        time.sleep(1)
        waited += 1
        print(f"等待vol_pred.npy生成... {waited}/{max_wait}秒")

    if not vol_pred_path:
        print(f"在{max_wait}秒内未找到vol_pred.npy，将尝试继续进行点云可视化")

    print("开始执行点云可视化...")
    vis_path = visualize_point_cloud(model_path)
    print(f"可视化结果保存在: {vis_path}")


# 如果直接运行此脚本，可以为已完成的训练生成可视化
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='点云可视化工具')
    parser.add_argument('--model_path', type=str, default='',
                        help='模型输出路径，为空则使用最新的输出')
    args = parser.parse_args()

    if not args.model_path:
        # 查找output目录下最新的文件夹
        output_dir = "./output/"
        if os.path.exists(output_dir):
            folders = glob.glob(os.path.join(output_dir, "*"))
            if folders:
                latest_folder = max(folders, key=os.path.getmtime)
                args.model_path = latest_folder
                print(f"使用最新的输出文件夹: {args.model_path}")
            else:
                print(f"在 {output_dir} 中找不到输出文件夹")
                exit(1)
        else:
            print(f"输出目录 {output_dir} 不存在")
            exit(1)

    visualize_point_cloud(args.model_path)