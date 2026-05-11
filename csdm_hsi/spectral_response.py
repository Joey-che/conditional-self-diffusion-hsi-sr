from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def load_srf_matrix(path: str | os.PathLike[str]) -> np.ndarray:
    """Load an SRF matrix with shape (guide_channels, hsi_bands)."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        matrix = _first_2d_array({key: data[key] for key in data.files}, path)
    elif suffix == ".mat":
        mat = sio.loadmat(path)
        matrix = _first_2d_array(mat, path)
    elif suffix in {".csv", ".txt"}:
        matrix = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(
            f"Unsupported SRF format: {path.suffix}. Use .npy, .npz, .mat, .csv, or .txt."
        )
    return normalize_srf(matrix)


def normalize_srf(srf: np.ndarray) -> np.ndarray:
    srf = np.asarray(srf, dtype=np.float32)
    if srf.ndim != 2:
        raise ValueError("SRF must be a 2D matrix with shape (guide_channels, hsi_bands).")
    denom = srf.sum(axis=1, keepdims=True)
    denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
    return srf / denom


def srf_to_tensor(srf: np.ndarray, device: torch.device | str) -> torch.Tensor:
    tensor = torch.as_tensor(normalize_srf(srf), dtype=torch.float32, device=device)
    return tensor[:, :, None, None]


def _first_2d_array(mapping: dict[str, np.ndarray], path: Path) -> np.ndarray:
    for key, value in mapping.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value
    raise KeyError(f"No 2D SRF matrix found in {path}.")
