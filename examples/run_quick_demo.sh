#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/quick_demo
python -c "import numpy as np; srf=np.ones((3,16), dtype=np.float32); srf/=srf.sum(axis=1, keepdims=True); np.save('runs/quick_demo/srf.npy', srf)"

python scripts/run_hsi_sr.py \
  --mode synthetic \
  --synthetic-height 32 \
  --synthetic-width 32 \
  --synthetic-bands 16 \
  --srf runs/quick_demo/srf.npy \
  --ratio 4 \
  --steps 2 \
  --iter 2 \
  --width 8 \
  --depth 2 \
  --guidance-loss mse \
  --device cpu \
  --output runs/quick_demo
