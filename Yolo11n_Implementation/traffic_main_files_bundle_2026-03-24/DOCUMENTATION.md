# Traffic Detection Pipeline - Documentation

## Overview
This project runs a vehicle and emergency detector on traffic videos, tracks objects, and emits structured JSONL output plus an optional annotated video. It uses Ultralytics YOLO (YOLO11n) as the detector and a lightweight tracking layer to estimate queues, wait times, and emergency flags.

## Environment
- Python 3.12
- GPU optional (CUDA recommended for speed)

Install dependencies:
```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Configuration
Main runtime config: `configs/runtime.example.yaml`

Key fields:
- `detector.weights_path`: path to `best.pt`
- `detector.confidence`, `detector.iou`, `detector.imgsz`
- `detector.device`: `"0"` for GPU 0, or `"cpu"`
- `vehicle_keywords`, `emergency_keywords`
- `tracking` and `signal` parameters

If you move the project or share it with others, update `weights_path` to the local path of the weights file.

## Run Inference (Video)
Example:
```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source /path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --target-fps 10 \
  --annotated-output outputs/boxed-output.mp4 \
  --output-jsonl outputs/predictions.jsonl \
  --show
```

Notes:
- `--target-fps 10` produces about 10 JSON lines per second in the output JSONL.
- `--annotated-output` writes a video with bounding boxes.
- `--show` opens a live preview window.

Optional outputs:
- `--bridge-jsonl`: intermediate per-frame signals (if needed downstream)
- `--signal-plan-jsonl`: signal decisions over time

## Output JSONL
`--output-jsonl` writes one JSON object per processed frame. The exact schema is produced by the pipeline and reflects queue estimates, wait times, and emergency flags for lanes.

## Training
Train (fresh run):
```bash
./.venv/bin/python scripts/train_detector.py \
  --data configs/all_sources_no_police_dataset.yaml \
  --weights /home/anuj/Desktop/image-processing/yolo11n.pt \
  --epochs 70 \
  --imgsz 960 \
  --batch 4 \
  --device 0 \
  --project runs/train \
  --name ambulance-priority-all-sources-no-police-960-70 \
  --patience 20
```

Resume training:
```bash
./.venv/bin/python scripts/train_detector.py \
  --data configs/all_sources_no_police_dataset.yaml \
  --weights runs/detect/runs/train/ambulance-priority-all-sources-no-police-960-70/weights/last.pt \
  --resume
```

Weights output:
`runs/detect/runs/train/ambulance-priority-all-sources-no-police-960-70/weights/best.pt`

## Troubleshooting
- YAML error: check indentation in `configs/runtime.example.yaml`.
- No GPU: set `detector.device: "cpu"` in the config.
- Low accuracy: confirm class list and dataset balance, and ensure the dataset labels match the class names in the data config.
