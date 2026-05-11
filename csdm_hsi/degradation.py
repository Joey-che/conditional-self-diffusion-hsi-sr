from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def add_gaussian_noise_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = torch.mean(x.square())
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return x + torch.sqrt(noise_power) * torch.randn_like(x)


def gaussian_kernel2d(
    kernel_size: int,
    sigma: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    axis = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def wald_protocol_torch(
    hsi: torch.Tensor,
    ratio: int,
    blur_sigma: float = 3.4,
    kernel_size: int | None = 15,
    snr_hsi_db: float | None = None,
) -> torch.Tensor:
    """Generate an LR-HSI observation by Gaussian blur and stride sampling."""

    if hsi.ndim != 4:
        raise ValueError("hsi must have shape (B, C, H, W).")
    if ratio < 1:
        raise ValueError("ratio must be >= 1.")

    _, channels, _, _ = hsi.shape
    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * blur_sigma) + 1)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    kernel = gaussian_kernel2d(kernel_size, blur_sigma, hsi.device, hsi.dtype)
    kernel = kernel[None, None].repeat(channels, 1, 1, 1)
    blurred = F.conv2d(hsi, kernel, padding=kernel_size // 2, groups=channels)
    lr_hsi = blurred[:, :, ::ratio, ::ratio]

    if snr_hsi_db is not None:
        lr_hsi = add_gaussian_noise_snr(lr_hsi, snr_hsi_db)
    return lr_hsi
