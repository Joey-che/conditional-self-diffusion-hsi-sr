from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
import tifffile


def load_cube(path: str | Path, key: str | None = None, channel_axis: int = -1) -> np.ndarray:
    """Load an HSI/MSI/RGB array and return HWC float32 data."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        array = data[key or data.files[0]]
    elif suffix == ".mat":
        mat = sio.loadmat(path)
        if key is not None:
            array = mat[key]
        else:
            array = next(
                value
                for name, value in mat.items()
                if not name.startswith("__") and isinstance(value, np.ndarray) and value.ndim in {2, 3}
            )
    elif suffix in {".tif", ".tiff"}:
        array = tifffile.imread(path)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        array = np.asarray(Image.open(path))
    else:
        raise ValueError(f"Unsupported data format: {path.suffix}")

    return ensure_hwc(np.asarray(array), channel_axis=channel_axis).astype(np.float32)


def ensure_hwc(array: np.ndarray, channel_axis: int = -1) -> np.ndarray:
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}.")
    if channel_axis not in {-1, 0, 1, 2}:
        raise ValueError("channel_axis must be -1, 0, 1, or 2.")
    axis = array.ndim - 1 if channel_axis == -1 else channel_axis
    if axis != 2:
        array = np.moveaxis(array, axis, 2)
    return array


def normalize_cube(array: np.ndarray, mode: str = "max") -> np.ndarray:
    array = array.astype(np.float32)
    mode = mode.lower()
    if mode == "none":
        return array
    if mode == "max":
        max_value = float(array.max())
        if max_value <= 0:
            raise ValueError("Cannot max-normalize an array with non-positive maximum.")
        return array / (max_value + 1e-10)
    if mode == "minmax":
        min_value = float(array.min())
        max_value = float(array.max())
        if max_value <= min_value:
            raise ValueError("Cannot min-max normalize an array with zero range.")
        return (array - min_value) / (max_value - min_value + 1e-10)
    raise ValueError("normalize mode must be one of: none, max, minmax.")


def crop_cube(
    array: np.ndarray,
    top: int = 0,
    left: int = 0,
    height: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    h, w = array.shape[:2]
    height = h - top if height is None else height
    width = w - left if width is None else width
    bottom = top + height
    right = left + width
    if top < 0 or left < 0 or bottom > h or right > w:
        raise ValueError(
            f"Invalid crop top={top}, left={left}, height={height}, width={width} for shape {array.shape}."
        )
    return array[top:bottom, left:right, :]


def cube_to_tensor(array: np.ndarray, device: torch.device | str) -> torch.Tensor:
    return torch.from_numpy(array.transpose(2, 0, 1))[None].float().to(device)


def tensor_to_cube(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)


def generate_synthetic_cube(height: int, width: int, bands: int, seed: int = 0) -> np.ndarray:
    """Generate a small smooth HSI cube for smoke tests and examples."""

    rng = np.random.default_rng(seed)
    spatial = rng.random((height, width, 4), dtype=np.float32)
    spatial = gaussian_filter(spatial, sigma=(height / 24.0, width / 24.0, 0.0))

    wavelengths = np.linspace(0.0, 1.0, bands, dtype=np.float32)
    spectra = np.stack(
        [
            np.sin(2.0 * np.pi * wavelengths) * 0.25 + 0.5,
            np.cos(2.0 * np.pi * wavelengths) * 0.20 + 0.45,
            np.exp(-((wavelengths - 0.35) ** 2) / 0.025),
            np.exp(-((wavelengths - 0.72) ** 2) / 0.035),
        ],
        axis=0,
    ).astype(np.float32)
    cube = np.einsum("hwk,kb->hwb", spatial, spectra)
    return normalize_cube(cube, mode="minmax")


def save_rgb_preview(array: np.ndarray, path: str | Path) -> None:
    """Save a quick RGB preview from a 3D array."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = array.astype(np.float32)
    if preview.shape[-1] >= 3:
        preview = preview[..., :3]
    else:
        preview = np.repeat(preview[..., :1], 3, axis=-1)
    preview = preview - preview.min()
    preview = preview / max(float(preview.max()), 1e-8)
    image = Image.fromarray((np.clip(preview, 0.0, 1.0) * 255.0).astype(np.uint8))
    image.save(path)
