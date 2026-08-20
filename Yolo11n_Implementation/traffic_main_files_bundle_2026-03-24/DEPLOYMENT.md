# Deployment Guide

This bundle contains everything needed to run inference and generate RNN-ready state vectors.

## Contents

- `weights/best.pt` (trained model)
- `configs/runtime.yaml` (runtime config, points to the local weights)
- `scripts/detect_traffic.py` (video/image inference + annotated output)
- `scripts/stream_predict_state.py` (per-second normalized state output)
- `traffic_pipeline/` (core pipeline)
- `requirements.txt`

## 1. Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install base requirements:

```bash
pip install -r requirements.txt
```

### GPU (recommended)

Install the CUDA build of PyTorch:

```bash
pip install --upgrade --force-reinstall \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

### CPU only

```bash
pip install --upgrade --force-reinstall torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
```

If you run CPU-only, change `device: "0"` to `device: "cpu"` in `configs/runtime.yaml`.

## 2. Run Inference With Annotated Video

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source /path/to/video.mp4 \
  --config configs/runtime.yaml \
  --annotated-output outputs/boxed-output.mp4 \
  --show
```

This shows a live window and saves the annotated video to `outputs/boxed-output.mp4`.

## 3. Generate RNN Input (1 JSON per second)

```bash
./.venv/bin/python scripts/stream_predict_state.py \
  --source /path/to/video.mp4 \
  --config configs/runtime.yaml \
  --process-fps 10 \
  --emit-interval-sec 1 \
  --output-jsonl outputs/state.jsonl
```

Each line in `outputs/state.jsonl` is:

```json
{"state":[...16 values...]}
```

State order (16 values):

1. Queues: `N, E, S, W`
2. Waits: `N, E, S, W`
3. Phase: `N, E, S, W`
4. Emergency: `N, E, S, W`

All values are normalized to `[0.0, 1.0]`.
