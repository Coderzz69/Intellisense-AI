# IntelliSignal AI

Traffic signals that look at the road. A camera feed goes through a fine-tuned YOLO11n detector and a
per-lane tracker, and a trained DQN decides which approach gets green. An ambulance on a red approach
gets green right away. Built for the Society Security Solution (SSS) hackathon around Indian traffic:
auto-rickshaws, bikes and buses in mixed, weaving lanes.

https://github.com/user-attachments/assets/b611d67f-0316-4d26-aca9-a235e4693f03

The demo is a fresh retrain with [sumo-rl](https://github.com/LucasAlegre/sumo-rl) on this repo's own 4×4 network,
because the original SUMO environment was never committed (see the status note below). One shared DQN drives all
12 lights. Over five unseen traffic seeds it cut total stopped time by **20.9%** (420,810 → 332,913 vehicle-seconds):
44–72% better on four seeds, but on the fifth a junction jammed (+144%). Reproduce with
`traffic_rl/sumo_rl_retrain/` (`train.py`, then `evaluate.py`).

## How it works

```
camera frame
  └─ YOLO11n            6 classes: ambulance, auto-rickshaw, bike, bus, car, truck
      └─ lane tracker   queue length, halted vehicles, accumulated wait, per approach
          └─ 16 numbers  queue[4]  wait[4]  current phase[4]  emergency[4]
              └─ DQN    → NS_GREEN or EW_GREEN, held for a minimum green time
                          an ambulance on a red approach overrides the hold
```

The DQN's decision is held for `signal.minimum_green_time` before it's reconsidered, so the output is a
stable signal plan, not a light that flickers with every frame.

| Component | Detail |
|---|---|
| Detector | YOLO11n fine-tuned at 960 px for 70 epochs. Validation mAP50 **0.708**, precision 0.763, recall 0.631 |
| Controller | DQN over the 16-number lane state (`traffic/weights/rl_latest.pt`) |
| Front ends | A React + Mapbox dashboard fed over SSE, and a standalone Flask four-feed view |
| Cloud | Optional SageMaker endpoint for the detector (`sagemaker/`, `docs/aws-sagemaker.md`) |

## Run it

Python 3.10+, Node 18+, and a Mapbox token for the dashboard map.

```bash
cd Yolo11n_Implementation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
npm install
cp .env.example .env                      # set REACT_APP_MAPBOX_ACCESS_TOKEN

export TRAFFIC_DEFAULT_VIDEO_PATH=/path/to/traffic.mp4
npm run start:backend                     # analysis + SSE on :8000
npm start                                 # dashboard on :3000
```

No footage is included. `python scripts/make_sample_video.py` makes a synthetic clip that exercises the
whole chain but has no real vehicles in it, so use real traffic video to see detections.

`curl -s localhost:8000/api/config` shows the detector weights, the DQN checkpoint and the devices that
were actually loaded. The four-feed view runs on its own with `cd traffic && python app.py` (port 5001).

More detail: [`Yolo11n_Implementation/README.md`](Yolo11n_Implementation/README.md).

## The SUMO multi-agent controller (`traffic_rl/`)

The other half of the project trains signal control in simulation: a 4×4 SUMO grid with Indian driving
behaviour (lane weaving, erratic braking, mixed vehicle classes). A master controller handles city-wide
events like an ambulance green corridor, and one shared-weight DQN runs every junction.

From the original training runs, against fixed-timer signals:

| Metric | Fixed timer | Master + DQN |
|---|---|---|
| Total wait (s) | 93,563 | 18,914 (**−79.8%**) |
| Average wait per vehicle (s) | 14.7 | 6.2 (**−57.7%**) |
| Ambulance wait | depends on the queue | about 0 (green corridor) |

![Training curves](traffic_rl/plots/training_curves.png)

> **Status:** `traffic_rl/env/`, the SUMO environment those numbers came from, is missing from this
> repository. A `.gitignore` rule meant for virtualenvs matched it and it was never committed, so the
> scripts in `traffic_rl/` can't run and the table can't be reproduced from here yet.
> [`traffic_rl/MISSING_FILES.md`](traffic_rl/MISSING_FILES.md) has the details. The vision pipeline above
> uses its own DQN and is complete.

## Layout

| Path | What |
|---|---|
| `Yolo11n_Implementation/traffic/` | The pipeline: detection, tracking, lane geometry, DQN controller, weights, config |
| `Yolo11n_Implementation/backend/` | HTTP + SSE server for the dashboard |
| `Yolo11n_Implementation/src/` | React dashboard |
| `Yolo11n_Implementation/traffic_main_files_bundle_2026-03-24/` | Detector training and dataset preparation |
| `traffic_rl/` | SUMO network, master/hybrid/baseline controllers, DQN, training (environment missing, see above) |

Stack: PyTorch · Ultralytics YOLO11 · SUMO / TraCI · Flask · React · Mapbox · AWS SageMaker
