from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    _assert_same_hwc(pred, gt)
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2, axis=(0, 1))
    return float(np.mean(10.0 * np.log10((data_range**2) / (mse + 1e-8))))


def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    _assert_same_hwc(pred, gt)
    return float(np.sqrt(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)))


def compute_sam(pred: np.ndarray, gt: np.ndarray) -> float:
    _assert_same_hwc(pred, gt)
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    dot = np.sum(pred * gt, axis=-1)
    norm = np.linalg.norm(pred, axis=-1) * np.linalg.norm(gt, axis=-1)
    angles = np.arccos(np.clip(dot / (norm + 1e-8), -1.0, 1.0))
    return float(np.mean(angles) * 180.0 / np.pi)


def compute_ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    _assert_same_hwc(pred, gt)
    scores = []
    for channel in range(pred.shape[2]):
        scores.append(
            structural_similarity(
                gt[:, :, channel],
                pred[:, :, channel],
                data_range=data_range,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        )
    return float(np.mean(scores))


def compute_ergas(pred: np.ndarray, gt: np.ndarray, ratio: int) -> float:
    _assert_same_hwc(pred, gt)
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    terms = []
    for channel in range(gt.shape[2]):
        mean = float(np.mean(gt[:, :, channel]))
        if abs(mean) < 1e-12:
            continue
        rmse = np.sqrt(np.mean((pred[:, :, channel] - gt[:, :, channel]) ** 2))
        terms.append((rmse / mean) ** 2)
    if not terms:
        return float("nan")
    return float(100.0 / ratio * np.sqrt(np.mean(terms)))


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray, ratio: int) -> dict[str, float]:
    return {
        "psnr": compute_psnr(pred, gt),
        "sam": compute_sam(pred, gt),
        "rmse": compute_rmse(pred, gt),
        "ssim": compute_ssim(pred, gt),
        "ergas": compute_ergas(pred, gt, ratio),
    }


def _assert_same_hwc(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
    if pred.ndim != 3:
        raise ValueError("Expected HWC arrays.")
