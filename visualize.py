"""
visualize.py
------------
Plotting helpers for training diagnostics and visual result inspection.

Provides:
    show_pairs(noisy, clean)     — display clean / noisy image pairs
    plot_history(history)        — plot training curves (loss, PSNR, SSIM)
    show_test_results(...)       — display noisy / denoised / clean triplets
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity   as ssim_metric


def show_pairs(
    noisy_batch: torch.Tensor,
    clean_batch: torch.Tensor,
    n:     int = 8,
    title: str = "DataLoader samples",
) -> None:
    """
    Display side-by-side clean (top row) and noisy (bottom row) image pairs.

    Parameters
    ----------
    noisy_batch : (B, C, H, W) tensor
    clean_batch : (B, C, H, W) tensor
    n           : number of pairs to show (default 8)
    title       : figure title string
    """
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4))
    for i in range(n):
        clean_np = clean_batch[i].permute(1, 2, 0).cpu().numpy()
        noisy_np = noisy_batch[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(np.clip(clean_np, 0, 1))
        axes[1, i].imshow(np.clip(noisy_np, 0, 1))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Clean", fontsize=11)
    axes[1, 0].set_ylabel("Noisy", fontsize=11)
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_history(history: dict) -> None:
    """
    Plot training / validation curves from the history dict produced by
    the training loop.

    Six subplots are drawn:
        (1) Total loss          (4) SSIM loss (1 − SSIM)
        (2) Physics sub-terms   (5) PSNR (dB)
        (3) Data sub-terms      (6) SSIM metric

    Parameters
    ----------
    history : dict
        Keys expected (as lists indexed by epoch):
        train_loss, val_loss, train_pde, train_ic, train_bc,
        train_mse, train_l1, train_ssim, val_ssim,
        train_psnr, val_psnr.
    """
    epochs_ran = len(history["train_loss"])
    ep = range(1, epochs_ran + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1 — Total loss
    ax = axes[0, 0]
    ax.plot(ep, history["train_loss"], label="Train")
    ax.plot(ep, history["val_loss"],   label="Val", linestyle="--")
    ax.set_title("Total Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    # 2 — Physics sub-terms
    ax = axes[0, 1]
    ax.plot(ep, history["train_pde"], label="PDE (heat Δ²)")
    ax.plot(ep, history["train_ic"],  label="IC  (input anchor)")
    ax.plot(ep, history["train_bc"],  label="BC  (border)")
    ax.set_title("Physics Loss Terms (train)"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # 3 — Data sub-terms
    ax = axes[0, 2]
    ax.plot(ep, history["train_mse"], label="MSE (data)")
    ax.plot(ep, history["train_l1"],  label="L1  (sharpness)")
    ax.set_title("Data Loss Terms (train)"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # 4 — SSIM loss
    ax = axes[1, 0]
    ax.plot(ep, [1 - v for v in history["train_ssim"]], label="Train")
    ax.plot(ep, [1 - v for v in history["val_ssim"]],   label="Val", linestyle="--")
    ax.set_title("SSIM Loss (1 − SSIM) ↓"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # 5 — PSNR
    ax = axes[1, 1]
    ax.plot(ep, history["train_psnr"], label="Train")
    ax.plot(ep, history["val_psnr"],   label="Val", linestyle="--")
    ax.set_title("PSNR (dB) ↑"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # 6 — SSIM metric
    ax = axes[1, 2]
    ax.plot(ep, history["train_ssim"], label="Train")
    ax.plot(ep, history["val_ssim"],   label="Val", linestyle="--")
    ax.set_title("SSIM ↑"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def show_test_results(
    vis_noisy: list[torch.Tensor],
    vis_pred:  list[torch.Tensor],
    vis_clean: list[torch.Tensor],
    n_show:    int = 10,
) -> None:
    """
    Display test-set triplets: noisy → denoised → clean.

    Row 1: noisy input    (PSNR in red)
    Row 2: denoised output (PSNR / SSIM in green)
    Row 3: clean ground truth

    Parameters
    ----------
    vis_noisy, vis_pred, vis_clean : lists of (B, C, H, W) CPU tensors
        Collected from the first few batches of the test loop.
    n_show : int
        Number of examples to display (default 10).
    """
    v_noisy = torch.cat(vis_noisy)[:n_show]
    v_pred  = torch.cat(vis_pred) [:n_show]
    v_clean = torch.cat(vis_clean)[:n_show]

    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 1.8, 6))

    for i in range(n_show):
        noisy_np = np.clip(v_noisy[i].permute(1, 2, 0).numpy(), 0, 1)
        pred_np  = np.clip(v_pred [i].permute(1, 2, 0).numpy(), 0, 1)
        clean_np = np.clip(v_clean[i].permute(1, 2, 0).numpy(), 0, 1)

        p_in  = psnr_metric(clean_np, noisy_np, data_range=1.0)
        p_out = psnr_metric(clean_np, pred_np,  data_range=1.0)
        s_out = ssim_metric(clean_np, pred_np,  data_range=1.0, channel_axis=2)

        axes[0, i].imshow(noisy_np); axes[0, i].axis("off")
        axes[1, i].imshow(pred_np);  axes[1, i].axis("off")
        axes[2, i].imshow(clean_np); axes[2, i].axis("off")

        axes[0, i].set_title(f"{p_in:.1f} dB", fontsize=7, color="red")
        axes[1, i].set_title(
            f"{p_out:.1f} dB\nSSIM:{s_out:.2f}", fontsize=7, color="green"
        )

    axes[0, 0].set_ylabel("Noisy",    fontsize=10)
    axes[1, 0].set_ylabel("Denoised", fontsize=10)
    axes[2, 0].set_ylabel("Clean",    fontsize=10)

    fig.suptitle(
        "Test Set: Noisy → Denoised → Clean  (PSNR above each image)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()
