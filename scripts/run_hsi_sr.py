#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csdm_hsi.data import (  # noqa: E402
    crop_cube,
    cube_to_tensor,
    generate_synthetic_cube,
    load_cube,
    normalize_cube,
    save_rgb_preview,
    tensor_to_cube,
)
from csdm_hsi.degradation import add_gaussian_noise_snr, wald_protocol_torch  # noqa: E402
from csdm_hsi.losses import VGGStructureLoss  # noqa: E402
from csdm_hsi.models import ConditionalUNet  # noqa: E402
from csdm_hsi.solver import SelfDiffusionSolver, SolverConfig, numpy_metrics_for_result  # noqa: E402
from csdm_hsi.spectral_response import load_srf_matrix, srf_to_tensor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run conditional self-diffusion for hyperspectral image super-resolution."
    )
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--hr-hsi", type=Path, default=None, help="HR-HSI reference for synthetic benchmark mode.")
    parser.add_argument("--lr-hsi", type=Path, default=None, help="Observed LR-HSI for real/inference mode.")
    parser.add_argument(
        "--guide",
        "--guide-msi",
        dest="guide_msi",
        type=Path,
        default=None,
        help="Observed HR RGB/MSI guidance for real/inference mode.",
    )
    parser.add_argument("--hr-key", type=str, default=None, help="MAT/NPZ key for --hr-hsi.")
    parser.add_argument("--lr-key", type=str, default=None, help="MAT/NPZ key for --lr-hsi.")
    parser.add_argument("--guide-key", type=str, default=None, help="MAT/NPZ key for --guide.")
    parser.add_argument("--channel-axis", type=int, default=-1, help="Channel axis in loaded arrays; default assumes HWC.")
    parser.add_argument("--normalize", choices=["max", "minmax", "none"], default="max")
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--crop-height", type=int, default=None)
    parser.add_argument("--crop-width", type=int, default=None)

    parser.add_argument("--synthetic-height", type=int, default=64)
    parser.add_argument("--synthetic-width", type=int, default=64)
    parser.add_argument("--synthetic-bands", type=int, default=31)

    parser.add_argument(
        "--srf",
        type=Path,
        required=True,
        help="Path to an SRF matrix with shape (guide_channels, hsi_bands).",
    )

    parser.add_argument("--ratio", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--iter", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--use-down", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-schedule", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-activation", choices=["sigmoid", "clamp", "identity"], default="sigmoid")

    parser.add_argument("--guidance-loss", choices=["mse", "vgg", "mix"], default="mse")
    parser.add_argument("--vgg-weight", type=float, default=0.1)
    parser.add_argument("--lr-weight", type=float, default=2.0)
    parser.add_argument("--guide-weight", type=float, default=2.0)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--noisy-input", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr-snr-db", type=float, default=20.0)
    parser.add_argument("--guide-snr-db", type=float, default=30.0)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', or a torch device such as 'cuda:0'.",
    )
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "synthetic":
        x_true_np = load_or_generate_hr_hsi(args)
        x_true = cube_to_tensor(x_true_np, device)
        bands = x_true_np.shape[2]
        srf = load_srf_matrix(args.srf)
        validate_srf_shape(srf, bands)
        srf_tensor = srf_to_tensor(srf, device)
        guide = F.conv2d(x_true, srf_tensor, None)
        lr_hsi = wald_protocol_torch(
            x_true,
            ratio=args.ratio,
            blur_sigma=3.4,
            kernel_size=15,
            snr_hsi_db=args.lr_snr_db if args.noisy_input else None,
        )
        if args.noisy_input:
            guide = add_gaussian_noise_snr(guide, args.guide_snr_db)
    else:
        if args.lr_hsi is None or args.guide_msi is None:
            raise ValueError("--lr-hsi and --guide are required in real mode.")
        lr_np = normalize_cube(load_cube(args.lr_hsi, key=args.lr_key, channel_axis=args.channel_axis), args.normalize)
        guide_np = normalize_cube(load_cube(args.guide_msi, key=args.guide_key, channel_axis=args.channel_axis), args.normalize)
        bands = lr_np.shape[2]
        srf = load_srf_matrix(args.srf)
        validate_srf_shape(srf, bands)
        srf_tensor = srf_to_tensor(srf, device)
        lr_hsi = cube_to_tensor(lr_np, device)
        guide = cube_to_tensor(guide_np, device)
        if guide.shape[1] != srf.shape[0]:
            raise ValueError(
                f"Guidance channels ({guide.shape[1]}) do not match SRF rows ({srf.shape[0]})."
            )
        x_true_np = None

    guide_channels = 0 if not args.use_condition else int(srf.shape[0])
    hsi_channels = bands
    height, width = guide.shape[-2:]
    x_init = torch.randn((1, hsi_channels, height, width), device=device)

    model = ConditionalUNet(
        in_channels=hsi_channels,
        out_channels=hsi_channels,
        guide_channels=guide_channels,
        width=args.width,
        depth=args.depth,
        use_down=args.use_down,
        output_activation=args.output_activation,
    ).to(device)
    init_weights(model, seed=args.seed)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    vgg_loss = None
    if args.guidance_loss in {"vgg", "mix"}:
        vgg_loss = VGGStructureLoss(pretrained=True).to(device).eval()

    config = SolverConfig(
        steps=args.steps,
        iterations=args.iter,
        ratio=args.ratio,
        learning_rate=args.learning_rate,
        lr_weight=args.lr_weight,
        guide_weight=args.guide_weight,
        guidance_loss=args.guidance_loss,
        vgg_weight=args.vgg_weight,
        use_noise_schedule=args.noise_schedule,
        use_condition=args.use_condition,
        tv_weight=args.tv_weight,
        verbose=not args.quiet,
    )
    solver = SelfDiffusionSolver(model, optimizer, config, device=device, vgg_loss=vgg_loss)
    result = solver.train(
        x_init=x_init,
        lr_hsi=lr_hsi,
        guide=guide,
        srf=srf_tensor,
        x_true=cube_to_tensor(x_true_np, device) if x_true_np is not None else None,
    )

    save_outputs(args, result, guide, srf_tensor, x_true_np)


def load_or_generate_hr_hsi(args: argparse.Namespace) -> np.ndarray:
    if args.hr_hsi is None:
        cube = generate_synthetic_cube(
            args.synthetic_height,
            args.synthetic_width,
            args.synthetic_bands,
            seed=args.seed,
        )
    else:
        cube = load_cube(args.hr_hsi, key=args.hr_key, channel_axis=args.channel_axis)
        cube = crop_cube(cube, args.crop_top, args.crop_left, args.crop_height, args.crop_width)
        cube = normalize_cube(cube, args.normalize)
    return cube.astype(np.float32)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def validate_srf_shape(srf: np.ndarray, bands: int) -> None:
    if srf.shape[1] != bands:
        raise ValueError(f"SRF has {srf.shape[1]} spectral columns but the HSI has {bands} bands.")


def init_weights(model: torch.nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def _init(module: torch.nn.Module) -> None:
        name = module.__class__.__name__
        if hasattr(module, "weight") and ("Conv" in name or "Linear" in name):
            torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.constant_(module.bias.data, 0.0)
        elif "BatchNorm2d" in name:
            torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(module.bias.data, 0.0)

    model.apply(_init)


def save_outputs(
    args: argparse.Namespace,
    result,
    guide: torch.Tensor,
    srf_tensor: torch.Tensor,
    x_true_np: np.ndarray | None,
) -> None:
    final_np = tensor_to_cube(result.final)
    best_np = tensor_to_cube(result.best)
    np.save(args.output / "reconstruction_final.npy", final_np)
    np.save(args.output / "reconstruction_best.npy", best_np)
    np.save(args.output / "psnr_trace.npy", np.asarray(result.psnr_trace, dtype=np.float32))

    with torch.no_grad():
        best_projected = F.conv2d(result.best, srf_tensor, None)
    save_rgb_preview(tensor_to_cube(best_projected), args.output / "reconstruction_best_preview.png")
    save_rgb_preview(tensor_to_cube(guide), args.output / "guide_preview.png")

    summary = {
        "best_step": result.best_step,
        "best_iteration": result.best_iteration,
        "best_loss": result.best_loss,
        "elapsed_sec": result.elapsed_sec,
        "args": vars(args) | {"output": str(args.output)},
    }
    if x_true_np is not None:
        summary["metrics"] = numpy_metrics_for_result(result, x_true_np, args.ratio)
    with open(args.output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
