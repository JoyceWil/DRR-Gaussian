import sys
import numpy as np
import torch

sys.path.append("./")
from r2_gaussian.utils.loss_utils import ssim

# 添加LPIPS导入
try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips package not available. LPIPS metrics will not be computed.")

# 初始化LPIPS模型（全局变量，延迟加载）
_lpips_model = None


def get_lpips_model(net_type='alex', device=None):
    """获取LPIPS模型，如果未初始化则初始化"""
    global _lpips_model
    if _lpips_model is None and LPIPS_AVAILABLE:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _lpips_model = lpips.LPIPS(net=net_type).to(device)
    return _lpips_model


@torch.no_grad()
def compute_lpips(img1, img2, mask=None):
    """计算LPIPS感知相似度

    Args:
        img1 (torch.Tensor): 参考图像 [b, c, h, w], 值范围[0,1]
        img2 (torch.Tensor): 测试图像 [b, c, h, w], 值范围[0,1]
        mask (torch.Tensor, optional): 掩码 [b, 1, h, w]. Defaults to None.

    Returns:
        torch.Tensor: LPIPS值 [b, 1], 值越小表示感知上越相似
    """
    if not LPIPS_AVAILABLE:
        return torch.tensor([[float('nan')]])

    # 获取LPIPS模型
    device = img1.device
    lpips_model = get_lpips_model(device=device)

    # 确保输入是3通道RGB图像
    if img1.shape[1] == 1:
        img1 = img1.repeat(1, 3, 1, 1)
    if img2.shape[1] == 1:
        img2 = img2.repeat(1, 3, 1, 1)

    # LPIPS需要输入范围为[-1,1]
    img1_scaled = img1 * 2 - 1
    img2_scaled = img2 * 2 - 1

    # 如果有掩码，需要特殊处理
    if mask is not None:
        # 将掩码区域之外的部分设为全黑
        img1_masked = img1_scaled * mask
        img2_masked = img2_scaled * mask
        lpips_values = lpips_model(img1_masked, img2_masked)
    else:
        lpips_values = lpips_model(img1_scaled, img2_scaled)

    return lpips_values


def mse(img1, img2, mask=None):
    """MSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_channel = img1.shape[1]
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(1, n_channel)
        mask = torch.where(mask != 0, True, False)

        mse = torch.stack(
            [
                (((img1[i, mask[i]] - img2[i, mask[i]])) ** 2).mean(0, keepdim=True)
                for i in range(img1.shape[0])
            ],
            dim=0,
        )

    else:
        mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return mse


def rmse(img1, img2, mask=None):
    """RMSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    rmse = mse_out ** 0.5
    return rmse


@torch.no_grad()
def psnr(img1, img2, mask=None, pixel_max=1.0):
    """PSNR

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    psnr_out = 10 * torch.log10(pixel_max ** 2 / mse_out.float())
    if mask is not None:
        if torch.isinf(psnr_out).any():
            print(mse_out.mean(), psnr_out.mean())
            psnr_out = 10 * torch.log10(pixel_max ** 2 / mse_out.float())
            psnr_out = psnr_out[~torch.isinf(psnr_out)]

    return psnr_out


@torch.no_grad()
def metric_vol(img1, img2, metric="psnr", pixel_max=1.0):
    """Metrics for volume. img1 must be GT."""
    assert metric in ["psnr", "ssim", "lpips"]
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())

    if metric == "psnr":
        if pixel_max is None:
            pixel_max = img1.max()
        mse_out = torch.mean((img1 - img2) ** 2)
        psnr_out = 10 * torch.log10(pixel_max ** 2 / mse_out.float())
        return psnr_out.item(), None
    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            n_slice = img1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1 = img1[i, :, :]
                    slice2 = img2[i, :, :]
                elif axis == 1:
                    slice1 = img1[:, i, :]
                    slice2 = img2[:, i, :]
                elif axis == 2:
                    slice1 = img1[:, :, i]
                    slice2 = img2[:, :, i]
                else:
                    raise NotImplementedError
                if slice1.max() > 0:
                    result = ssim(slice1[None, None], slice2[None, None])
                    count += 1
                else:
                    result = 0
                results.append(result)
            results = torch.tensor(results)
            mean_results = torch.sum(results) / count
            ssims.append(mean_results.item())
        return float(np.mean(ssims)), ssims
    elif metric == "lpips":
        if not LPIPS_AVAILABLE:
            return float('nan'), None

        lpips_values = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            n_slice = img1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1 = img1[i, :, :]
                    slice2 = img2[i, :, :]
                elif axis == 1:
                    slice1 = img1[:, i, :]
                    slice2 = img2[:, i, :]
                elif axis == 2:
                    slice1 = img1[:, :, i]
                    slice2 = img2[:, :, i]
                else:
                    raise NotImplementedError

                if slice1.max() > 0:
                    # 标准化切片到[0,1]范围
                    norm_slice1 = slice1 / slice1.max()
                    norm_slice2 = slice2 / slice2.max()

                    # 计算LPIPS
                    result = compute_lpips(
                        norm_slice1[None, None], norm_slice2[None, None]
                    ).item()
                    count += 1
                else:
                    result = 0
                results.append(result)

            results = np.array(results)
            mean_result = np.sum(results) / max(count, 1)
            lpips_values.append(mean_result)

        return float(np.mean(lpips_values)), lpips_values


@torch.no_grad()
def metric_proj(img1, img2, metric="psnr", axis=2, pixel_max=1.0):
    """Metrics for projection

    Args:
        img1 (_type_): [x, y, z]
        img2 (_type_): [x, y, z]
        pixel_max (float, optional): _description_. Defaults to 1.0.
    """
    assert axis in [0, 1, 2, None]
    assert metric in ["psnr", "ssim", "lpips"]
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    n_slice = img1.shape[axis]

    results = []
    count = 0
    for i in range(n_slice):
        if axis == 0:
            slice1 = img1[i, :, :]
            slice2 = img2[i, :, :]
        elif axis == 1:
            slice1 = img1[:, i, :]
            slice2 = img2[:, i, :]
        elif axis == 2:
            slice1 = img1[:, :, i]
            slice2 = img2[:, :, i]
        else:
            raise NotImplementedError

        if slice1.max() > 0:
            slice1 = slice1 / slice1.max()
            slice2 = slice2 / slice2.max()

            if metric == "psnr":
                result = psnr(
                    slice1[None, None], slice2[None, None], pixel_max=pixel_max
                )
            elif metric == "ssim":
                result = ssim(slice1[None, None], slice2[None, None])
            elif metric == "lpips":
                if LPIPS_AVAILABLE:
                    result = compute_lpips(slice1[None, None], slice2[None, None]).item()
                else:
                    result = float('nan')
            else:
                raise NotImplementedError
            count += 1
        else:
            result = 0
        results.append(result)

    if count > 0:
        if metric == "lpips" and not LPIPS_AVAILABLE:
            return float('nan'), results
        results_tensor = torch.tensor(results) if metric != "lpips" else np.array(results)
        mean_results = torch.sum(results_tensor) / count if metric != "lpips" else np.sum(results_tensor) / count
        return mean_results.item(), results
    else:
        return 0, results