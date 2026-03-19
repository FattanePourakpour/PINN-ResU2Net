"""
model.py
--------
ResU2Net architecture for 28×28 RGB image denoising.

Hierarchy
---------
    cbr()       — Conv2d → BatchNorm2d → ReLU building block
    RSU         — Residual U-block: a small nested U-Net inside each stage
    ResU2Net    — Outer encoder-decoder built from RSU blocks

Key design choices
------------------
* Each RSU block has an explicit 3-level nested U structure (encoder levels 1-2,
  dilated bottleneck, symmetric decoder) plus a residual skip from the block input.
  This gives every stage both local and wide receptive fields without depth explosion.

* The outer ResU2Net performs 3 max-pool downsampling steps (28→14→7→4) and
  symmetrically upsamples back to 28×28 via bilinear interpolation + skip-cat.

* A final 1×1 convolution followed by sigmoid maps features → RGB in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building block

def cbr(in_ch: int, out_ch: int, dilation: int = 1) -> nn.Sequential:
    """Conv2d → BatchNorm2d → ReLU helper."""
    return nn.Sequential(
        nn.Conv2d(
            in_ch, out_ch, 3,
            padding=dilation, dilation=dilation, bias=False,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ── Residual U-block

class RSU(nn.Module):
    """
    Residual U-block (RSU) with a fixed 3-level nested U structure.

    Architecture
    ------------
    resin  : in_ch  → out_ch          (input projection / residual branch)
    e1     : out_ch → mid_ch          (encoder level 1, full resolution)
    e2     : mid_ch → mid_ch          (encoder level 2, after 2× pool)
    e3     : mid_ch → mid_ch          (bottleneck, dilated conv, same spatial)
    d2     : mid_ch×2 → mid_ch        (decoder level 2, bilinear upsample + cat)
    d1     : mid_ch×2 → out_ch        (decoder level 1, back to full resolution)
    output : d1_features + resin(x)   (residual addition)

    Parameters
    ----------
    in_ch  : int — input channels
    mid_ch : int — inner channel width (encoder/decoder hidden channels)
    out_ch : int — output channels (also the residual branch width)
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.resin = cbr(in_ch,      out_ch)               # residual branch
        self.e1    = cbr(out_ch,     mid_ch)                # encoder level 1
        self.e2    = cbr(mid_ch,     mid_ch)                # encoder level 2
        self.e3    = cbr(mid_ch,     mid_ch, dilation=2)    # bottleneck (dilated)
        self.d2    = cbr(mid_ch * 2, mid_ch)                # decoder level 2
        self.d1    = cbr(mid_ch * 2, out_ch)                # decoder level 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.resin(x)                                        # (B, out_ch, H, W)

        # Encode
        h1 = self.e1(res)                                          # (B, mid, H,   W)
        h2 = self.e2(F.max_pool2d(h1, 2, ceil_mode=True))         # (B, mid, H/2, W/2)
        h3 = self.e3(h2)                                           # (B, mid, H/2, W/2)

        # Decode — upsample to match skip feature sizes, then concatenate
        u2 = self.d2(torch.cat([
            F.interpolate(h3, h2.shape[2:], mode="bilinear", align_corners=False),
            h2,
        ], dim=1))
        u1 = self.d1(torch.cat([
            F.interpolate(u2, h1.shape[2:], mode="bilinear", align_corners=False),
            h1,
        ], dim=1))

        return u1 + res   # residual addition


# ── ResU2Net
class ResU2Net(nn.Module):
    """
    Lightweight ResU2Net for 28×28 RGB image denoising.

    Parameters
    ----------
    in_ch  : int — number of input channels (default 3 for RGB)
    out_ch : int — number of output channels (default 3 for RGB)
    base   : int — base channel width multiplier (default 16)
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 16):
        super().__init__()
        b = base

        # Outer encoder
        self.enc1   = RSU(in_ch,  b,     b * 2)   # 28 → 28
        self.enc2   = RSU(b * 2,  b,     b * 4)   # 14 → 14
        self.enc3   = RSU(b * 4,  b * 2, b * 8)   #  7 →  7
        self.bridge = RSU(b * 8,  b * 4, b * 8)   #  4 →  4  (bottleneck)

        # Outer decoder (input = upsample(prev) concat with encoder skip)
        self.dec3 = RSU(b * 8 + b * 8, b * 2, b * 4)
        self.dec2 = RSU(b * 4 + b * 4, b,     b * 2)
        self.dec1 = RSU(b * 2 + b * 2, b,     b)

        self.out_conv = nn.Conv2d(b, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)                                         # (B, 2b, 28, 28)
        e2 = self.enc2(F.max_pool2d(e1, 2))                       # (B, 4b, 14, 14)
        e3 = self.enc3(F.max_pool2d(e2, 2))                       # (B, 8b,  7,  7)
        bt = self.bridge(F.max_pool2d(e3, 2, ceil_mode=True))     # (B, 8b,  4,  4)

        # Decoder — upsample to match skip feature map size, then concat
        d3 = self.dec3(torch.cat([
            F.interpolate(bt, e3.shape[2:], mode="bilinear", align_corners=False), e3,
        ], dim=1))
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, e2.shape[2:], mode="bilinear", align_corners=False), e2,
        ], dim=1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, e1.shape[2:], mode="bilinear", align_corners=False), e1,
        ], dim=1))

        return torch.sigmoid(self.out_conv(d1))   # (B, out_ch, 28, 28) in [0, 1]
