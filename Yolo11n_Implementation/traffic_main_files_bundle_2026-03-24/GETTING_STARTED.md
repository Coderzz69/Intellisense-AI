# Getting Started

This file explains how to initialize, configure, train, and run the traffic pipeline in this repository.

## 1. What this project does

The project has two main parts:

- `scripts/train_detector.py`: trains a YOLO-based detector for Indian traffic classes
- `scripts/detect_traffic.py`: runs inference on an image, video, or webcam and returns per-vehicle JSON

The detector classes are:

- `ambulance`
- `auto-rikshaw`
- `bike`
- `bus`
- `car`
- `police vehicle`
- `truck`

## 2. Requirements

You need:

- Python 3
- `pip`
- optional NVIDIA GPU with CUDA for faster training and inference

Python dependencies are listed in [requirements.txt](/home/anuj/Desktop/image-processing/traffic/requirements.txt).

## 3. Initialize the project

If `.venv` does not exist yet, create and install dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

If `.venv` already exists, just use it:

```bash
source .venv/bin/activate
```

## 4. Update local paths before running

This repository currently uses absolute local paths in its config files. If you move the project to another machine or folder, update these files first:

- [configs/local_dataset.yaml](/home/anuj/Desktop/image-processing/traffic/configs/local_dataset.yaml)
- [configs/merged_dataset.yaml](/home/anuj/Desktop/image-processing/traffic/configs/merged_dataset.yaml)
- [configs/runtime.example.yaml](/home/anuj/Desktop/image-processing/traffic/configs/runtime.example.yaml)
- [scripts/train_detector.py](/home/anuj/Desktop/image-processing/traffic/scripts/train_detector.py)
- [traffic_pipeline/config.py](/home/anuj/Desktop/image-processing/traffic/traffic_pipeline/config.py)

Important paths used today:

- base dataset: `/home/anuj/Desktop/image-processing/DataSet`
- default base weights: `/home/anuj/Desktop/image-processing/yolo11n.pt`
- current runtime weights: `/home/anuj/Desktop/image-processing/traffic/runs/train/ambulance-priority-merged/weights/best.pt`

## 5. Validate the setup

Check that training config resolves correctly:

```bash
./.venv/bin/python scripts/train_detector.py --dry-run
```

Check the detection CLI:

```bash
./.venv/bin/python scripts/detect_traffic.py --help
```

## 6. Run detection

### Image input

Use an image when you only want detection classes and lane assignment:

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source path/to/image.jpg \
  --config configs/runtime.example.yaml
```

Important:

- single images do **not** measure `is_halting` or `wait_time`
- for image runs, those fields are returned as `null`

### Video input

Use a video when you want `is_halting` and `wait_time`:

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml
```

If you want exactly `10` JSON outputs per second of video, sample at `10 FPS`:

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --target-fps 10 \
  --output-jsonl outputs/vehicles.jsonl
```

That produces one JSON line about every `0.1s` of video time.

### Webcam input

Use webcam index `0` for the default camera:

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source 0 \
  --config configs/runtime.example.yaml
```

## 7. Show boxes or save annotated output

Show a live OpenCV window:

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --target-fps 10 \
  --show
```

Save annotated image or video:

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --target-fps 10 \
  --annotated-output outputs/boxed-output.mp4
```

Show and save at the same time:

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --target-fps 10 \
  --show \
  --annotated-output outputs/boxed-output.mp4
```

You can also save the JSON outputs:

```bash
./.venv/bin/python scripts/detect_traffic.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --target-fps 10 \
  --output-jsonl outputs/vehicles.jsonl \
  --bridge-jsonl outputs/bridge.jsonl \
  --signal-plan-jsonl outputs/signal-plan.jsonl
```

## 8. How lane labels work

Lane labels are `N`, `E`, `S`, `W`.

Current behavior:

- if `lane_polygons` are configured in [configs/runtime.example.yaml](/home/anuj/Desktop/image-processing/traffic/configs/runtime.example.yaml), the vehicle `bottom_center` point is checked against those polygons by default
- if `tracking.require_lane_polygon_match: true`, any detection outside the configured polygons is ignored
- otherwise the frame is split around the image center:
  - top = `N`
  - right = `E`
  - bottom = `S`
  - left = `W`

That means these are image-region labels unless you define real lane polygons for your camera view.

For intersection control, define polygons only around the incoming lanes you want to count. This prevents cars from adjacent lanes or the opposite direction from being forced into one of your signal lanes.

## 9. How `is_halting` and `wait_time` work

These values are temporal and require multiple frames.

- `is_halting` becomes true when the tracked vehicle moves less than the configured pixel threshold between frames
- `wait_time` is how long that tracked vehicle has remained stationary

So:

- image input: `is_halting = null`, `wait_time = null`
- video/webcam input: both values are calculated frame-by-frame

## 10. Train the detector

### Train on the original dataset

```bash
./.venv/bin/python scripts/train_detector.py \
  --data configs/local_dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --project runs/train \
  --name ambulance-priority-model
```

### Train on the merged dataset

```bash
./.venv/bin/python scripts/train_detector.py \
  --data configs/merged_dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --project runs/train \
  --name ambulance-priority-merged
```

Notes:

- use `--device 0` for GPU
- use `--device cpu` if CUDA is unavailable
- Ultralytics run artifacts are saved under `runs/detect/runs/train/...` in this workspace

## 11. Prepare the merged dataset

If you want to rebuild the merged dataset from the base dataset plus downloaded Indian traffic sources:

```bash
./.venv/bin/python scripts/prepare_merged_dataset.py --overwrite
```

The merged dataset output is written to:

- [prepared_data/merged_indian_traffic](/home/anuj/Desktop/image-processing/traffic/prepared_data/merged_indian_traffic)

Its summary file is:

- [summary.json](/home/anuj/Desktop/image-processing/traffic/prepared_data/merged_indian_traffic/summary.json)

## 12. Convert detections to the 16-number bridge vector

If you have video/webcam detection JSONL output, convert it with:

```bash
./.venv/bin/python scripts/yolo_bridge.py \
  --input outputs/vehicles.jsonl \
  --current-green-lane N
```

Use video-derived JSON here. Image outputs intentionally contain `null` temporal fields.

## 13. Stream state to `/predict` once per second

If you want to process video at `10` samples per second and convert those `10` frame-level states into one normalized payload per second, use:

```bash
./.venv/bin/python scripts/stream_predict_state.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --process-fps 10 \
  --emit-interval-sec 1 \
  --post-url http://localhost:5000/predict
```

If you only want to print/save the payloads without sending HTTP requests:

```bash
./.venv/bin/python scripts/stream_predict_state.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --process-fps 10 \
  --emit-interval-sec 1 \
  --raw-state-jsonl outputs/raw-state.jsonl \
  --output-jsonl outputs/state.jsonl
```

If you also want a boxed preview window or a saved annotated video while the per-second state is being generated:

```bash
./.venv/bin/python scripts/stream_predict_state.py \
  --source path/to/video.mp4 \
  --config configs/runtime.example.yaml \
  --process-fps 10 \
  --emit-interval-sec 1 \
  --output-jsonl outputs/state.jsonl \
  --annotated-output outputs/state-annotated.mp4 \
  --show
```

Important:

- the script samples at `10 FPS` by default, keeps those 10 raw states for the current second, then emits one aggregated `{"state": [...]}` payload
- `outputs/raw-state.jsonl` can contain the intermediate 10-FPS raw bridge vectors if you pass `--raw-state-jsonl`
- `--annotated-output` saves the processed frames with boxes, labels, lane tags, emergency flags, and halt/wait overlays
- `--show` opens a live preview of those same annotations during processing
- the emitted `state` values are normalized to `0.0..1.0`
- queue values are normalized by `--queue-cap` and wait values are normalized by `--wait-cap-sec`
- the phase segment is averaged over the second, so it can be fractional if the light changed during that second
- the emergency segment is a per-second presence flag, so it stays `0.0` or `1.0`
- JSON comments like `// Queues` are not valid and should not be included in the actual POST body
- unlike `--target-fps 1`, this script can still process multiple frames for tracking accuracy and only emits the final normalized state once per second

## 14. Run tests

Run all unit tests:

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## 15. Recommended first run

If you just want to confirm the project works end-to-end:

1. Activate the virtualenv.
2. Make sure [configs/runtime.example.yaml](/home/anuj/Desktop/image-processing/traffic/configs/runtime.example.yaml) points to a valid `.pt` weights file.
3. Run detection on a short traffic video with `--show`.
4. If the boxes look correct, rerun with `--annotated-output outputs/result.mp4`.

## 16. Current runtime model

The current runtime config points to:

- [best.pt](/home/anuj/Desktop/image-processing/traffic/runs/train/ambulance-priority-indian-ambulance/weights/best.pt)

That path is configured in [configs/runtime.example.yaml](/home/anuj/Desktop/image-processing/traffic/configs/runtime.example.yaml#L2).
