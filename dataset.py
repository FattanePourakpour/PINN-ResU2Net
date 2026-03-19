"""
dataset.py
----------
BloodMNIST dataset loading and noisy wrapper.

Provides:
    NoisyDataset   — wraps a MedMNIST dataset, returning (noisy, clean) pairs
    get_dataloaders — convenience factory that downloads BloodMNIST and builds
                      train / val / test DataLoaders
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from medmnist import BloodMNIST

NOISE_SIGMA = 25 / 255.0   # default noise level (normalized)
BATCH_SIZE  = 64


class NoisyDataset(Dataset):
    """
    Wraps a MedMNIST dataset; returns (noisy_image, clean_image) pairs.

    Gaussian noise is added on-the-fly each call to __getitem__, so the
    model always sees fresh noise samples (acts as light data augmentation).

    Parameters
    ----------
    base_dataset : MedMNIST dataset
        A MedMNIST dataset object (e.g. BloodMNIST instance).
    sigma : float
        Standard deviation of additive Gaussian noise in [0, 1] range.
        Default: 25/255 ≈ 0.098.
    """

    def __init__(self, base_dataset, sigma: float = NOISE_SIGMA):
        self.ds    = base_dataset
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        img, _label = self.ds[idx]                    # img: (C, H, W) in [0, 1]
        noise       = torch.randn_like(img) * self.sigma
        noisy       = torch.clamp(img + noise, 0.0, 1.0)
        return noisy, img


def get_dataloaders(
    batch_size:  int   = BATCH_SIZE,
    sigma:       float = NOISE_SIGMA,
    num_workers: int   = 2,
    download:    bool  = True,
):
    """
    Download (if needed) BloodMNIST and return DataLoaders for all three splits.

    Parameters
    ----------
    batch_size  : int   — mini-batch size (default 64)
    sigma       : float — Gaussian noise std in [0, 1] (default 25/255)
    num_workers : int   — DataLoader worker processes (default 2)
    download    : bool  — download BloodMNIST if not cached (default True)

    Returns
    -------
    train_loader, val_loader, test_loader : torch.utils.data.DataLoader
    """
    base_transform = transforms.Compose([
        transforms.ToTensor(),   # [0, 255] uint8 → [0, 1] float32
    ])

    train_base = BloodMNIST(split="train", transform=base_transform, download=download)
    val_base   = BloodMNIST(split="val",   transform=base_transform, download=download)
    test_base  = BloodMNIST(split="test",  transform=base_transform, download=download)

    train_ds = NoisyDataset(train_base, sigma=sigma)
    val_ds   = NoisyDataset(val_base,   sigma=sigma)
    test_ds  = NoisyDataset(test_base,  sigma=sigma)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader
