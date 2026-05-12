# Conditional Self-Diffusion for HSI Super-Resolution

Official implementation for:

**Unsupervised Hyperspectral Image Super-Resolution Based on a Conditional Self-Diffusion Model**

The method reconstructs a high-resolution hyperspectral image from a low-resolution HSI and a paired high-resolution RGB/MSI guidance image by optimizing an untrained conditional denoiser directly on the target observations.

Main features:

- conditional resolution-preserving U-Net denoiser;
- self-diffusion optimization loop;
- Wald degradation model for synthetic benchmarks;
- RGB/MSI spectral response projection with user-supplied SRF matrices;
- PSNR, SAM, RMSE, SSIM, and ERGAS evaluation;
- a single command-line interface for synthetic benchmarks and real LR-HSI + HR-RGB/MSI inputs.

## Install

```bash
conda create -n csdm-hsi python=3.10
conda activate csdm-hsi
pip install -r requirements.txt
```

For CUDA, install the PyTorch build matching your driver from the official PyTorch instructions, then install the remaining requirements.

## Quick Demo

Run a tiny CPU smoke test with generated synthetic data:

```bash
bash examples/run_quick_demo.sh
```

Outputs are written to `runs/quick_demo/`:

- `reconstruction_final.npy`
- `reconstruction_best.npy`
- `summary.json`
- `guide_preview.png`
- `reconstruction_best_preview.png`

## Synthetic Benchmark Mode

Use an HR-HSI cube as the reference. The script generates LR-HSI by Gaussian blur plus downsampling, and generates HR RGB/MSI guidance with the supplied SRF matrix.

```bash
python scripts/run_hsi_sr.py \
  --mode synthetic \
  --hr-hsi /path/to/PaviaU.mat \
  --hr-key paviaU \
  --channel-axis -1 \
  --normalize max \
  --crop-height 512 \
  --crop-width 256 \
  --srf /path/to/srf.npy \
  --ratio 8 \
  --steps 40 \
  --iter 150 \
  --width 64 \
  --guidance-loss mix \
  --vgg-weight 0.5 \
  --output runs/paviau_ratio8
```

For a noisy-input benchmark, add:

```bash
--noisy-input --lr-snr-db 20 --guide-snr-db 30
```

## Real/Inferred Mode

Use an observed LR-HSI and aligned HR-RGB/MSI guidance image. No HR-HSI reference is required.

```bash
python scripts/run_hsi_sr.py \
  --mode real \
  --lr-hsi /path/to/lr_hsi.npy \
  --guide /path/to/hr_rgb.png \
  --srf /path/to/srf.npy \
  --ratio 8 \
  --steps 40 \
  --iter 150 \
  --width 64 \
  --guidance-loss mse \
  --output runs/real_case
```

The LR-HSI and HR guidance should be aligned. The HR guidance spatial size determines the reconstruction size. The LR-HSI spatial size should match the reconstruction size divided by `--ratio`.

## Data Format

Supported HSI/guidance array formats:

- `.npy`, `.npz`
- `.mat`
- `.tif`, `.tiff`
- common RGB image formats for guidance (`.png`, `.jpg`, `.jpeg`, `.bmp`)

Arrays are expected in HWC layout by default. Use `--channel-axis 0` if your data is CHW.

For `.mat` or `.npz` files, pass keys with `--hr-key`, `--lr-key`, and `--guide-key`.

## Spectral Response

An SRF matrix must be provided with `--srf`. The matrix shape is:

```text
(guide_channels, hsi_bands)
```

For example, RGB guidance uses a `3 x B` matrix, while 6-channel MSI guidance uses a `6 x B` matrix. SRF files may be `.npy`, `.npz`, `.mat`, `.csv`, or `.txt`. For `.npz` and `.mat`, the first 2D array found in the file is used.

Use measured responses from your camera/sensor or a response matrix collected from the relevant source.

## Paper Settings

The paper uses a diffusion budget of `T=40`, `K=150`, `lambda_LR=2`, and `lambda_M=2`.

Recommended settings from the paper:

| Dataset group | Width | Guidance branch |
| --- | ---: | --- |
| PaviaU | 64 | mix, `--vgg-weight 0.5` |
| CAVE | 128 | mix, `--vgg-weight 0.1` |
| Harvard | 128 | clean: mix `0.1`; noisy: pure VGG |
| KSC | 128 | pure MSE |
| Botswana | 128 | clean: pure MSE; noisy: mix `0.0125` |

Use `--guidance-loss mse`, `--guidance-loss vgg`, or `--guidance-loss mix` to select the guidance branch. VGG guidance uses pretrained VGG-16 features from `torchvision`.

## Code Availability

https://anonymous.4open.science/r/conditional-self-diffusion-hsi-sr-2026

## Attribution

This implementation builds on the Self-Diffusion framework by Guanxiong Luo and Shoujin Huang. Please cite both the HSI paper and the original Self-Diffusion work when appropriate.

```bibtex
@inproceedings{luo2025selfdiffusion,
  title={Self-diffusion for Solving Inverse Problems},
  author={Guanxiong Luo and Shoujin Huang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=5g9qls1V7Q}
}
```

## License

MIT. See `LICENSE` and `NOTICE.md`.
