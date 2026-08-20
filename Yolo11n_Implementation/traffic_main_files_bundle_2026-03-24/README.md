# Traffic Pipeline

This workspace now contains a small traffic-processing project that matches the JSON shape in your image and uses the paper's queue-length logic for signal timing.

## What it does

- `scripts/detect_traffic.py` runs YOLO inference, assigns each detection to `N/E/S/W`, tracks IDs starting at `101`, marks `is_halting`, accumulates `wait_time`, and emits JSON lists like:

```json
[
  {
    "id": 101,
    "lane": "S",
    "is_halting": true,
    "wait_time": 12.5,
    "is_emergency": true
  }
]
```

- `scripts/yolo_bridge.py` converts that list into the 16-number vector described in your screenshot:
  `stopped[N,E,S,W] + wait[N,E,S,W] + green_one_hot[N,E,S,W] + emergency[N,E,S,W]`
- `traffic_pipeline/signal_control.py` implements the paper workflow:
  queue span from YOLO bounding boxes, pixel-to-meter conversion with `y = alpha * x^n`, proportional cycle allocation, and emergency-lane override.
- `scripts/train_detector.py` trains a custom traffic detector using the local dataset at `/home/anuj/Desktop/image-processing/DataSet`.
- `scripts/fit_pixel_to_meter.py` fits the paper's calibration curve from manual pixel/meter samples.

## Quick start

Run detection on a local image:

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source /home/anuj/Desktop/image-processing/DataSet/test/images/india-kerala-trivandrum-a-typical-indian-ambulance-trying-to-make-AAA9GK_jpg.rf.f1c44ae473e2424dea1ab0c15af35b58.jpg \
  --config configs/runtime.example.yaml
```

Run detection on video and emit `10` JSON outputs per second:

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --target-fps 10 \
  --output-jsonl detections.jsonl
```

Convert saved JSON lists to bridge vectors:

```bash
./.venv/bin/python scripts/yolo_bridge.py \
  --input detections.jsonl \
  --current-green-lane N
```

Stream video into a once-per-second normalized `{"state": [...]}` payload for `/predict` using `10` sampled states per second:

```bash
./.venv/bin/python scripts/stream_predict_state.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --process-fps 10 \
  --emit-interval-sec 1 \
  --post-url http://localhost:5000/predict
```

Save an annotated video while generating those per-second state payloads:

```bash
./.venv/bin/python scripts/stream_predict_state.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --process-fps 10 \
  --emit-interval-sec 1 \
  --output-jsonl outputs/state.jsonl \
  --annotated-output outputs/state-annotated.mp4
```

Validate training config:

```bash
./.venv/bin/python scripts/train_detector.py --dry-run
```

Train a custom detector from the local dataset:

```bash
./.venv/bin/python scripts/train_detector.py \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --project runs/train \
  --name traffic-paper-model
```

Fit the paper calibration curve:

```bash
./.venv/bin/python scripts/fit_pixel_to_meter.py \
  --input calibration.csv \
  --output calibration.json
```

## Notes

- The paper is based on YOLOv9. This environment has a local Ultralytics checkpoint at `/home/anuj/Desktop/image-processing/yolo11n.pt`, so the training and inference entrypoints are wired to that local asset by default. If you add a local YOLOv9 checkpoint, update `detector.weights_path` or `--weights`.
- `configs/local_dataset.yaml` fixes the dataset paths so training works against the existing local dataset layout.
- For intersection-specific accuracy, replace the default compass lane split with `lane_polygons` in `configs/runtime.example.yaml`.
