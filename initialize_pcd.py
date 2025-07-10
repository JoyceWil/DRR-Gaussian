import os
import numpy as np
import tigre.algorithms as algs
# import open3d as o3d # <--- 注释掉，因为这个脚本里没有用到o3d
import sys
import argparse
import os.path as osp
import json
import pickle
from tqdm import trange
import copy
import torch

# <--- 新增: 导入scipy库用于边缘检测
from scipy.ndimage import gaussian_filter, sobel

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry, recon_volume
from r2_gaussian.arguments import ParamGroup, ModelParams, PipelineParams
from r2_gaussian.utils.plot_utils import show_one_volume, show_two_volume
from r2_gaussian.gaussian import GaussianModel, query, initialize_gaussian
from r2_gaussian.utils.image_utils import metric_vol
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.general_utils import t2a

np.random.seed(0)


class InitParams(ParamGroup):
    def __init__(self, parser):
        self.recon_method = "fdk"
        self.n_points = 50000
        self.density_thresh = 0.05
        self.density_rescale = 0.15
        self.random_density_max = 1.0  # Parameters for random mode
        # <--- 新增: 为边缘检测添加新参数
        self.edge_sampling = False  # 是否启用边缘采样
        self.edge_sigma = 1.0  # 高斯平滑的sigma值
        self.edge_weight = 0.8  # 边缘采样与随机采样之间的权重 (1.0表示纯边缘采样)

        super().__init__(parser, "Initialization Parameters")


def init_pcd(
        projs,
        angles,
        geo,
        scanner_cfg,
        args: InitParams,
        save_path,
):
    "Initialize Gaussians."
    recon_method = args.recon_method
    n_points = args.n_points
    assert recon_method in ["random", "fdk"], "--recon_method not supported."
    if recon_method == "random":
        print(f"Initialize random point clouds.")
        sampled_positions = np.array(scanner_cfg["offOrigin"])[None, ...] + np.array(
            scanner_cfg["sVoxel"]
        )[None, ...] * (np.random.rand(n_points, 3) - 0.5)
        sampled_densities = (
                np.random.rand(
                    n_points,
                )
                * args.random_density_max
        )
    else:
        # Use traditional algorithms for initialization
        print(
            f"Initialize point clouds with the volume reconstructed from {recon_method}."
        )
        vol = recon_volume(projs, angles, copy.deepcopy(geo), recon_method)
        # show_one_volume(vol)

        density_mask = vol > args.density_thresh

        # <--- 修改: 从这里开始是新的边缘采样逻辑
        if args.edge_sampling:
            print(f"Using edge-aware sampling with sigma={args.edge_sigma} and weight={args.edge_weight}")

            # 1. 对FDK体数据进行轻微平滑以去噪
            smoothed_volume = gaussian_filter(vol, sigma=args.edge_sigma)

            # 2. 计算3D Sobel边缘强度图
            grad_x = sobel(smoothed_volume, axis=0)
            grad_y = sobel(smoothed_volume, axis=1)
            grad_z = sobel(smoothed_volume, axis=2)
            edge_map = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)

            # 3. 只保留在密度掩码内的边缘
            edge_map[~density_mask] = 0

            # 4. 准备概率分布
            # 边缘概率
            edge_probabilities = edge_map.flatten()
            if np.sum(edge_probabilities) > 0:
                edge_probabilities /= np.sum(edge_probabilities)
            else:
                print("Warning: No edges found. Edge sampling might be ineffective.")
                edge_probabilities = np.zeros_like(edge_probabilities)

            # 均匀随机概率 (用于混合)
            random_probabilities = density_mask.flatten().astype(float)
            if np.sum(random_probabilities) > 0:
                random_probabilities /= np.sum(random_probabilities)
            else:
                print("Warning: No valid voxels found based on density_thresh. Cannot perform random sampling.")
                random_probabilities = np.zeros_like(random_probabilities)

            # 5. 混合概率
            combined_probabilities = (args.edge_weight * edge_probabilities +
                                      (1 - args.edge_weight) * random_probabilities)

            if np.sum(combined_probabilities) == 0:
                raise ValueError("No valid voxels to sample from. Check density_thresh or FDK reconstruction.")

            # 归一化混合概率
            combined_probabilities /= np.sum(combined_probabilities)

            # 6. 根据混合概率进行采样
            voxel_indices_1d = np.random.choice(
                vol.size,
                size=n_points,
                replace=False,  # 如果点数不多，可以用False
                p=combined_probabilities
            )
            sampled_indices = np.column_stack(np.unravel_index(voxel_indices_1d, vol.shape))

        else:
            # 原始的随机采样逻辑
            print("Using random sampling.")
            valid_indices = np.argwhere(density_mask)
            assert (
                    valid_indices.shape[0] >= n_points
            ), "Valid voxels less than target number of sampling. Check threshold"
            sampled_indices = valid_indices[
                np.random.choice(len(valid_indices), n_points, replace=False)
            ]
        # <--- 修改结束

        offOrigin = np.array(scanner_cfg["offOrigin"])
        dVoxel = np.array(scanner_cfg["dVoxel"])
        sVoxel = np.array(scanner_cfg["sVoxel"])

        sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
        sampled_densities = vol[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]
        sampled_densities = sampled_densities * args.density_rescale

    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    np.save(save_path, out)
    print(f"Initialization saved in {save_path}.")


def main(
        args, init_args: InitParams, model_args: ModelParams, pipe_args: PipelineParams
):
    # <--- 新增: 在所有CUDA操作之前设置GPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        print(f"Using GPU: {args.gpu_id}")
    else:
        print("CUDA not available, running on CPU.")

    # Read scene
    data_path = args.data
    model_args.source_path = data_path
    scene = Scene(model_args, False)  # ! Here we scale the scene to [-1,1]^3 space.
    train_cameras = scene.getTrainCameras()
    projs_train = np.concatenate(
        [t2a(cam.original_image) for cam in train_cameras], axis=0
    )
    angles_train = np.stack([t2a(cam.angle) for cam in train_cameras], axis=0)
    scanner_cfg = scene.scanner_cfg
    geo = get_geometry(scanner_cfg)

    save_path = args.output
    if not save_path:
        save_path = osp.join(
            data_path, "init_" + osp.basename(data_path).split(".")[0] + ".npy"
        )

    # <--- 修改: 如果文件存在，先删除
    if osp.exists(save_path):
        print(f"Initialization file {save_path} exists! Deleting it first.")
        os.remove(save_path)

    os.makedirs(osp.dirname(save_path), exist_ok=True)

    init_pcd(
        projs=projs_train,
        angles=angles_train,
        geo=geo,
        scanner_cfg=scanner_cfg,
        args=init_args,
        save_path=save_path,
    )

    # Evaluate using ground truth volume (for debug only)
    if args.evaluate:
        with torch.no_grad():
            model_args.ply_path = save_path
            scale_bound = None
            volume_to_world = max(scanner_cfg["sVoxel"])
            if model_args.scale_min and model_args.scale_max:
                scale_bound = (
                        np.array([model_args.scale_min, model_args.scale_max])
                        * volume_to_world
                )
            gaussians = GaussianModel(scale_bound)
            initialize_gaussian(gaussians, model_args, None)
            vol_pred = query(
                gaussians,
                scanner_cfg["offOrigin"],
                scanner_cfg["nVoxel"],
                scanner_cfg["sVoxel"],
                pipe_args,
            )["vol"]
            vol_gt = scene.vol_gt.cuda()
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            print(f"3D PSNR for initial Gaussians: {psnr_3d}")
            # show_two_volume(vol_gt, vol_pred, title1="gt", title2="init")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate initialization parameters")
    init_parser = InitParams(parser)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--data", type=str, help="Path to data.")
    parser.add_argument("--output", default=None, type=str, help="Path to output.")
    parser.add_argument("--evaluate", default=False, action="store_true",
                        help="Add this flag to evaluate quality (given GT volume, for debug only)")
    # <--- 新增: 添加GPU选择参数
    parser.add_argument("--gpu_id", type=int, default=1, help="ID of the GPU to use.")
    # fmt: on

    args = parser.parse_args()
    main(args, init_parser.extract(args), lp.extract(args), pp.extract(args))