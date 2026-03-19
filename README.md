# PINN-ResU2Net: Physics-Informed Denoising of Blood Cell Microscopy Images

A physics-informed neural network (PINN) framework for denoising blood cell microscopy images, coupling a **ResU2Net** backbone with a **six-term composite loss** that embeds heat-equation PDE constraints, initial/boundary conditions, structural similarity (SSIM), and L1 sharpness directly into training.

**Results on BloodMNIST** (σ = 25/255 Gaussian noise, 3,421 test images):

| Metric | Noisy Baseline | Model Output | Gain |
|--------|:--------------:|:------------:|:----:|
| PSNR (dB) | 20.67 | 29.46 | **+8.79 dB** |
| SSIM | 0.708 | 0.942 | **+0.234** |
| ms / image | — | 0.716 | — |

---

## Project Structure

```
pinn_bloodmnist/
│
├── pinn_resu2net_bloodmnist.ipynb  # Main experiment notebook
│
├── dataset.py      # BloodMNIST loading and NoisyDataset wrapper
├── model.py        # ResU2Net architecture (RSU blocks + outer U-Net)
├── losses.py       # Six-term PINN composite loss function
├── metrics.py      # PSNR / SSIM evaluation utilities
├── visualize.py    # Plotting helpers (pairs, history, test results)
│
├── requirements.txt
└── README.md
```

### File Descriptions

| File | Contents |
|------|----------|
| `pinn_resu2net_bloodmnist.ipynb` | End-to-end experiment: data loading → visualisation → model instantiation → training → evaluation → visual results |
| `dataset.py` | `NoisyDataset` (on-the-fly Gaussian noise wrapper) and `get_dataloaders()` factory |
| `model.py` | `cbr()` helper, `RSU` (Residual U-block), `ResU2Net` encoder-decoder |
| `losses.py` | `laplacian()`, `ssim_loss()`, `pinn_loss()` with all six terms |
| `metrics.py` | `batch_psnr_ssim()` — batch-level PSNR and SSIM via scikit-image |
| `visualize.py` | `show_pairs()`, `plot_history()`, `show_test_results()` |

---

## Method

### Dataset

[BloodMNIST](https://medmnist.com/) from the MedMNIST v2 benchmark provides 28 × 28 RGB microscopy patches of eight blood-cell morphologies (Eosinophil, Lymphocyte, Monocyte, Neutrophil, Basophil, Abnormal lymphocyte, Immature granulocyte, Erythroblast).

| Split | Images | Classes |
|-------|-------:|:-------:|
| Train | 11,959 | 8 |
| Validation | 1,712 | 8 |
| Test | 3,421 | 8 |

Additive Gaussian noise (σ = 25/255) is applied **on-the-fly** each epoch so the model always trains on fresh corruptions, providing implicit data augmentation.

---

### Architecture — ResU2Net

The backbone is a lightweight encoder-decoder whose stages are **Residual U-blocks (RSU)** — the key building block from [U²-Net](https://doi.org/10.1016/j.patcog.2020.107404), adapted here for 28 × 28 inputs.

#### RSU Block

Each RSU is itself a small nested U-Net with three explicit encoder-decoder levels plus a dilated bottleneck. The dilation=2 bottleneck (`cbr_dil2`) enlarges the effective receptive field without extra pooling, giving every stage both local and wider spatial context.

#### Outer ResU2Net

Four RSU blocks form an outer U-Net with three downsampling steps:

```
Input  (3,  28, 28)
  enc1  (2b, 28, 28) → max-pool → enc2  (4b, 14, 14)
                                    → max-pool → enc3  (8b,  7,  7)
                                                  → max-pool → bridge (8b, 4, 4)
  dec1  (b,  28, 28) ← upsample+cat ← dec2  (2b, 14, 14)
                                    ← upsample+cat ← dec3  (4b,  7,  7)
                                                  ← upsample+cat ←
Output: 1×1 conv → sigmoid  →  (3, 28, 28)  in [0, 1]
```

Skip connections between each encoder stage and its matching decoder stage ensure fine-grained morphological detail (cell boundaries, cytoplasm texture) is preserved through the bottleneck.

---

### Physics-Informed Loss

The core contribution is a six-term composite loss that couples data-driven learning with PDE-based physical priors:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{pde}}\,\mathcal{L}_{\text{PDE}} + \lambda_{\text{ic}}\,\mathcal{L}_{\text{IC}} + \lambda_{\text{bc}}\,\mathcal{L}_{\text{BC}} + \lambda_{\text{ssim}}\,\mathcal{L}_{\text{SSIM}} + \lambda_{\text{l1}}\,\mathcal{L}_{\text{L1}}$$

#### Term definitions

**Data fidelity (pixel MSE)**

$$\mathcal{L}_{\text{data}} = \|\hat{X} - X_c\|_2^2$$

Standard pixel-level reconstruction loss. The primary supervision signal.

**Heat-equation PDE prior**

$$\mathcal{L}_{\text{PDE}} = \|\Delta \hat{X}\|_2^2, \qquad \lambda_{\text{pde}} = 0.05$$

$\Delta$ is the discrete Laplacian applied via a fixed 3 × 3 convolution kernel (the 5-point stencil):

$$\Delta = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

This penalises spatial roughness inconsistent with the steady-state heat equation $\Delta u = 0$, constraining the denoised image to lie on a physically plausible diffusion trajectory. Note: as the network learns sharper cell boundaries the Laplacian naturally grows — this is expected behaviour, balanced by the small weight.

**Initial condition**

$$\mathcal{L}_{\text{IC}} = \|\hat{X} - X_N\|_2^2, \qquad \lambda_{\text{ic}} = 0.03$$

In the PINN diffusion analogy, the noisy input $X_N$ is the observed state at time $t=0$. This term prevents the network from drifting to a solution that ignores the observed measurement — a direct transcription of the physical initial condition $u(\mathbf{x}, 0) = X_N(\mathbf{x})$.

**Dirichlet boundary condition**

$$\mathcal{L}_{\text{BC}} = \|\hat{X}_{\partial\Omega} - X_{N,\partial\Omega}\|_2^2, \qquad \lambda_{\text{bc}} = 0.02$$

Enforces that the 1-pixel image border of the denoised output matches the noisy input border. This prevents the network from hallucinating content at image edges.

**Differentiable SSIM**

$$\mathcal{L}_{\text{SSIM}} = 1 - \text{SSIM}(\hat{X},\, X_c), \qquad \lambda_{\text{ssim}} = 0.15$$

The structural similarity index is made differentiable using an 11 × 11 Gaussian window (σ = 1.5, C₁ = 0.01², C₂ = 0.03²). This directly optimises the evaluation metric, counteracting the blurring tendency of pure MSE training by explicitly preserving luminance, contrast, and structural patterns.

**L1 sharpness**

$$\mathcal{L}_{\text{L1}} = \|\hat{X} - X_c\|_1, \qquad \lambda_{\text{l1}} = 0.05$$

Complements MSE: the L1 norm is less sensitive to large outlier errors (since it doesn't square them), encouraging the network to recover sharp edges rather than averaging them away.

#### Loss weight rationale

| Term | Weight | Rationale |
|------|:------:|-----------|
| $\mathcal{L}_{\text{data}}$ | 1.00 | Primary signal — dominates training |
| $\mathcal{L}_{\text{SSIM}}$ | 0.15 | Largest auxiliary weight; most impactful for perceptual quality |
| $\mathcal{L}_{\text{PDE}}$ | 0.05 | Regulariser; small enough not to over-smooth |
| $\mathcal{L}_{\text{L1}}$ | 0.05 | Sharpness auxiliary |
| $\mathcal{L}_{\text{IC}}$ | 0.03 | Soft constraint; prevents extreme drift from input |
| $\mathcal{L}_{\text{BC}}$ | 0.02 | Weakest term; boundary pixels are few |

---

### Training Protocol

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam, lr = 1e-4 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| Early stopping | Patience 50 epochs |
| Max epochs | 500 (stopped at epoch 122) |
| Batch size | 64 |
| Noise level σ | 25 / 255 (on-the-fly) |
| Model base width | 16 channels |

---

## How to Run

### 1. Clone / download

```bash
git clone https://github.com/<your-username>/pinn-bloodmnist.git
cd pinn-bloodmnist
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU recommended** — training on CPU is possible but significantly slower.  
> CUDA 11.8+ or MPS (Apple Silicon) will be detected automatically.

### 3. Run the notebook

```bash
jupyter notebook pinn_resu2net_bloodmnist.ipynb
```

Run all cells top to bottom. BloodMNIST will be downloaded automatically (~17 MB) on first run.

### 4. Use the modules directly

```python
from dataset import get_dataloaders
from model import ResU2Net
from losses import pinn_loss
from metrics import batch_psnr_ssim

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
model = ResU2Net(in_ch=3, out_ch=3, base=16).to(DEVICE)

# Single forward + loss pass
noisy, clean = next(iter(train_loader))
noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
pred = model(noisy)
loss, components = pinn_loss(pred, clean, noisy)

psnr, ssim = batch_psnr_ssim(pred, clean)
print(f"Loss: {loss.item():.4f} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
```

---

## References

1. Qin, X. et al. "U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection." *Pattern Recognition*, 106, 107404, 2020. https://doi.org/10.1016/j.patcog.2020.107404

2. Osorio Quero, C. & Crespo, M. L. "Physics-Informed Neural Network for Denoising Images Using Nonlinear PDE." *Electronics*, 15(3), 560, 2026. https://doi.org/10.3390/electronics15030560

3. Yang, J. et al. "MedMNIST v2 — A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification." *Scientific Data*, 10, 41, 2023. https://doi.org/10.1038/s41597-022-01721-8
