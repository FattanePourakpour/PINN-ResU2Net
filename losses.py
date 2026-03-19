"""
losses.py
---------
Physics-Informed composite loss for image denoising.

The loss embeds a heat-equation PDE constraint, initial and boundary conditions,
structural similarity (SSIM), and L1 sharpness alongside the standard MSE data
fidelity term.


Loss formulation
---------------------------------------
    L_total = L_data
            + λ_pde  * L_PDE
            + λ_ic   * L_IC
            + λ_bc   * L_BC
            + λ_ssim * L_SSIM
            + λ_l1   * L_L1

where each term is defined as:

    L_data  = MSE(pred, clean)                  — pixel fidelity
    L_PDE   = ||Δ pred||²                       — heat-equation steady-state prior
    L_IC    = MSE(pred, noisy)                  — initial condition  (t = 0)
    L_BC    = MSE(pred_border, noisy_border)    — Dirichlet boundary condition
    L_SSIM  = 1 − SSIM(pred, clean)            — perceptual / structural quality
    L_L1    = L1(pred, clean)                   — sharpness / outlier robustness

Default weights: λ_pde=0.05, λ_ic=0.03, λ_bc=0.02, λ_ssim=0.15, λ_l1=0.05
"""

import torch
import torch.nn.functional as F


# ── Laplacian (PDE term)
# Fixed discrete Laplacian kernel (5-point stencil):  0  1  0
#                                                      1 -4  1
#                                                      0  1  0
_LAP_KERNEL = torch.tensor(
    [[[[0., 1., 0.],
       [1., -4., 1.],
       [0., 1., 0.]]]],
    dtype=torch.float32,
)


def laplacian(img: torch.Tensor) -> torch.Tensor:
    """
    Apply the discrete 5-point Laplacian operator per channel.

    Parameters
    ----------
    img : torch.Tensor, shape (B, C, H, W)
        Input image tensor in [0, 1].

    Returns
    -------
    torch.Tensor, shape (B, C, H, W)
        Laplacian response (∆ img) used as the PDE residual.
    """
    C = img.shape[1]
    k = _LAP_KERNEL.to(img.device).expand(C, 1, 3, 3)
    return F.conv2d(img, k, padding=1, groups=C)


# ── Differentiable SSIM

def _gaussian_kernel(
    window_size: int = 11,
    sigma:       float = 1.5,
    channels:    int = 3,
) -> torch.Tensor:
    """Build a normalised 2-D Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g     /= g.sum()
    kernel = (g.unsqueeze(1) * g.unsqueeze(0))          # (W, W)
    kernel = kernel.unsqueeze(0).unsqueeze(0)            # (1, 1, W, W)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


_SSIM_WIN: torch.Tensor | None = None   # module-level cached kernel


def ssim_loss(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    window_size: int   = 11,
    C1:          float = 0.01 ** 2,
    C2:          float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Differentiable SSIM loss  =  1 − SSIM(pred, target).

    Allows backpropagation through the structural similarity metric so the
    network directly optimises the evaluation criterion.

    Parameters
    ----------
    pred        : (B, C, H, W) float tensor in [0, 1]
    target      : (B, C, H, W) float tensor in [0, 1]
    window_size : Gaussian window size (default 11)
    C1, C2      : SSIM stability constants (default 0.01², 0.03²)

    Returns
    -------
    scalar torch.Tensor — mean SSIM loss over the batch.
    """
    global _SSIM_WIN
    _B, C, _H, _W = pred.shape
    if _SSIM_WIN is None or _SSIM_WIN.shape[0] != C:
        _SSIM_WIN = _gaussian_kernel(window_size, sigma=1.5, channels=C).to(pred.device)

    pad  = window_size // 2
    mu1  = F.conv2d(pred,   _SSIM_WIN, padding=pad, groups=C)
    mu2  = F.conv2d(target, _SSIM_WIN, padding=pad, groups=C)

    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1  = F.conv2d(pred   * pred,   _SSIM_WIN, padding=pad, groups=C) - mu1_sq
    s2  = F.conv2d(target * target, _SSIM_WIN, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(pred   * target, _SSIM_WIN, padding=pad, groups=C) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    )
    return 1.0 - ssim_map.mean()


# ── Full PINN composite loss

def pinn_loss(
    pred:     torch.Tensor,
    target:   torch.Tensor,
    noisy:    torch.Tensor,
    lam_pde:  float = 0.05,
    lam_ic:   float = 0.03,
    lam_bc:   float = 0.02,
    lam_ssim: float = 0.15,
    lam_l1:   float = 0.05,
):
    """
    Six-term physics-informed composite loss.

    Parameters
    ----------
    pred     : (B, C, H, W) — network output in [0, 1]
    target   : (B, C, H, W) — clean ground-truth image in [0, 1]
    noisy    : (B, C, H, W) — corrupted input; used for IC and BC terms
    lam_pde  : weight for the heat-equation PDE term       (default 0.05)
    lam_ic   : weight for the initial-condition term       (default 0.03)
    lam_bc   : weight for the boundary-condition term      (default 0.02)
    lam_ssim : weight for the SSIM perceptual term         (default 0.15)
    lam_l1   : weight for the L1 sharpness term            (default 0.05)

    Returns
    -------
    total : scalar torch.Tensor  — weighted composite loss (backprop-ready)
    comps : dict[str, float]     — individual loss components (for logging)
    """
    # L_data — pixel-level MSE fidelity
    l_data = F.mse_loss(pred, target)

    # L_PDE — heat-equation residual: penalise spatial roughness in output
    l_pde = laplacian(pred).pow(2).mean()

    # L_IC — initial condition: denoised output should not drift far from input
    l_ic = F.mse_loss(pred, noisy)

    # L_BC — Dirichlet boundary condition: 1-pixel border of pred ≈ noisy border
    pred_b  = torch.cat(
        [pred[:, :, 0, :], pred[:, :, -1, :],
         pred[:, :, :, 0], pred[:, :, :, -1]], dim=-1,
    )
    noisy_b = torch.cat(
        [noisy[:, :, 0, :], noisy[:, :, -1, :],
         noisy[:, :, :, 0], noisy[:, :, :, -1]], dim=-1,
    )
    l_bc = F.mse_loss(pred_b, noisy_b)

    # L_SSIM — differentiable structural similarity (directly optimises metric)
    l_ssim = ssim_loss(pred, target)

    # L_L1 — sharpness and outlier robustness
    l_l1 = F.l1_loss(pred, target)

    total = (
        l_data
        + lam_pde  * l_pde
        + lam_ic   * l_ic
        + lam_bc   * l_bc
        + lam_ssim * l_ssim
        + lam_l1   * l_l1
    )

    comps = dict(
        mse=l_data.item(), pde=l_pde.item(),
        ic=l_ic.item(),    bc=l_bc.item(),
        ssim=l_ssim.item(), l1=l_l1.item(),
    )
    return total, comps
