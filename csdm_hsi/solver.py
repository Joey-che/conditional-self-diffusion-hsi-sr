from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .degradation import wald_protocol_torch
from .metrics import compute_all_metrics


@dataclass
class SolverConfig:
    steps: int = 40
    iterations: int = 150
    ratio: int = 8
    learning_rate: float = 1e-3
    lr_weight: float = 2.0
    guide_weight: float = 2.0
    guidance_loss: str = "mse"
    vgg_weight: float = 0.1
    blur_sigma: float = 3.4
    kernel_size: int = 15
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    use_noise_schedule: bool = True
    use_condition: bool = True
    tv_weight: float = 0.0
    verbose: bool = True


@dataclass
class ReconstructionResult:
    final: torch.Tensor
    best: torch.Tensor
    best_step: int
    best_iteration: int
    best_loss: float
    psnr_trace: list[float]
    elapsed_sec: float


class SigmaScheduler:
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        steps: int,
        device: torch.device | str,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if beta_start >= beta_end:
            raise ValueError("beta_start must be smaller than beta_end.")
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)

    def sigma(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.steps:
            raise ValueError(f"idx must be in [0, {self.steps - 1}], got {idx}.")
        reverse_idx = self.steps - 1 - idx
        alpha = torch.cumprod(1.0 - self.betas[: reverse_idx + 1], dim=0)[-1]
        return torch.sqrt(1.0 - alpha)


class SelfDiffusionSolver:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: SolverConfig,
        device: torch.device | str,
        vgg_loss: torch.nn.Module | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(device)
        self.vgg_loss = vgg_loss

    def train(
        self,
        x_init: torch.Tensor,
        lr_hsi: torch.Tensor,
        guide: torch.Tensor,
        srf: torch.Tensor,
        x_true: torch.Tensor | None = None,
    ) -> ReconstructionResult:
        cfg = self.config
        x_hat = x_init.to(self.device)
        lr_hsi = lr_hsi.to(self.device)
        guide = guide.to(self.device)
        srf = srf.to(self.device)

        scheduler = SigmaScheduler(cfg.beta_start, cfg.beta_end, cfg.steps, self.device)
        best = x_hat.detach().clone()
        best_loss = float("inf")
        best_step = 0
        best_iteration = 0
        psnr_trace: list[float] = []
        start_time = time.time()

        iterator = range(cfg.steps)
        if cfg.verbose:
            iterator = tqdm(iterator, desc="Self-diffusion", leave=True)

        for step in iterator:
            if cfg.use_noise_schedule:
                x_t = x_hat + torch.randn_like(x_hat) * scheduler.sigma(step)
            else:
                x_t = x_hat

            x_star = x_hat
            for iteration in range(cfg.iterations):
                self.optimizer.zero_grad(set_to_none=True)
                condition = guide if cfg.use_condition else None
                x_star = self.model(x_t, condition)
                loss = self._loss(x_star, lr_hsi, guide, srf)
                loss.backward()
                self.optimizer.step()

                loss_value = float(loss.detach().cpu())
                if loss_value < best_loss:
                    best = x_star.detach().clone()
                    best_loss = loss_value
                    best_step = step + 1
                    best_iteration = iteration + 1

            x_hat = x_star.detach()

            if x_true is not None:
                psnr_trace.append(_psnr_torch(x_hat, x_true.to(self.device)))

            if cfg.verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{best_loss:.6f}", step=step + 1)

        return ReconstructionResult(
            final=x_hat.detach(),
            best=best.detach(),
            best_step=best_step,
            best_iteration=best_iteration,
            best_loss=best_loss,
            psnr_trace=psnr_trace,
            elapsed_sec=time.time() - start_time,
        )

    def _loss(
        self,
        x: torch.Tensor,
        lr_hsi: torch.Tensor,
        guide: torch.Tensor,
        srf: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        pred_lr = wald_protocol_torch(
            x,
            ratio=cfg.ratio,
            blur_sigma=cfg.blur_sigma,
            kernel_size=cfg.kernel_size,
            snr_hsi_db=None,
        )
        pred_guide = F.conv2d(x, srf, None)
        if pred_lr.shape != lr_hsi.shape:
            raise ValueError(
                f"LR-HSI shape mismatch: predicted {tuple(pred_lr.shape)} vs observed {tuple(lr_hsi.shape)}. "
                "Check --ratio and crop dimensions."
            )
        if pred_guide.shape != guide.shape:
            raise ValueError(
                f"Guidance shape mismatch: predicted {tuple(pred_guide.shape)} vs observed {tuple(guide.shape)}. "
                "Check the SRF matrix and guidance image alignment."
            )

        lr_loss = F.mse_loss(pred_lr, lr_hsi)
        guide_loss = self._guidance_loss(pred_guide, guide)
        loss = cfg.lr_weight * lr_loss + cfg.guide_weight * guide_loss

        if cfg.tv_weight > 0:
            loss = loss + cfg.tv_weight * _total_variation_2d(x)
        return loss

    def _guidance_loss(self, pred: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        mode = self.config.guidance_loss.lower()
        if mode == "mse":
            return F.mse_loss(pred, guide)

        if self.vgg_loss is None:
            raise ValueError("VGG guidance was requested but no VGG loss module was supplied.")

        if mode == "vgg":
            return self.vgg_loss(pred, guide)
        if mode == "mix":
            mse = F.mse_loss(pred, guide)
            vgg = self.vgg_loss(pred, guide)
            return (mse + self.config.vgg_weight * vgg) / (1.0 + self.config.vgg_weight)
        raise ValueError("guidance_loss must be one of: mse, vgg, mix.")


def _psnr_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt).square(), dim=(2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    return float(psnr.mean().detach().cpu())


def _total_variation_2d(x: torch.Tensor) -> torch.Tensor:
    vertical = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
    horizontal = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
    return vertical + horizontal


def numpy_metrics_for_result(
    result: ReconstructionResult,
    x_true_np: np.ndarray,
    ratio: int,
) -> dict[str, dict[str, float]]:
    from .data import tensor_to_cube

    return {
        "final": compute_all_metrics(tensor_to_cube(result.final), x_true_np, ratio),
        "best": compute_all_metrics(tensor_to_cube(result.best), x_true_np, ratio),
    }
