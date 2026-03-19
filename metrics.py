"""
metrics.py
----------
Evaluation metrics for image denoising.

Provides:
    batch_psnr_ssim(pred_t, target_t) — mean PSNR and SSIM over a batch
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity   as ssim_metric


def batch_psnr_ssim(
    pred_t:   torch.Tensor,
    target_t: torch.Tensor,
) -> tuple[float, float]:
    """
    Compute mean PSNR and SSIM over a batch of images.

    Images are expected as float tensors in [0, 1]. Computation is done on
    CPU numpy arrays via scikit-image.

    Parameters
    ----------
    pred_t   : torch.Tensor, shape (B, C, H, W)
        Predicted (denoised) images.
    target_t : torch.Tensor, shape (B, C, H, W)
        Ground-truth (clean) images.

    Returns
    -------
    mean_psnr : float — mean PSNR (dB) across the batch
    mean_ssim : float — mean SSIM across the batch
    """
    pred   = pred_t.detach().cpu().permute(0, 2, 3, 1).numpy()    # (B, H, W, C)
    target = target_t.detach().cpu().permute(0, 2, 3, 1).numpy()
    pred   = np.clip(pred,   0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    psnr_vals, ssim_vals = [], []
    for p, t in zip(pred, target):
        psnr_vals.append(psnr_metric(t, p, data_range=1.0))
        ssim_vals.append(ssim_metric(t, p, data_range=1.0, channel_axis=2))

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))
