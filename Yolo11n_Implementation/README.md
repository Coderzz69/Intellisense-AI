# IntelliSignal — Vision + RL Traffic Control

A React dashboard and a Flask four-feed UI on top of one AI pipeline:
YOLO detection → lane tracking → a trained DQN that decides the signal phase.

## The pipeline

```
video frame
  └─ TrafficDetector      local YOLO weights (default) or a SageMaker endpoint
      └─ SimpleTracker    per-lane queues, halting state, accumulated wait
          └─ bridge vector  16 floats: queue[4] wait[4] phase[4] emergency[4]
              └─ RLTrafficController   trained DQN -> NS_GREEN / EW_GREEN
                                       + emergency override
```

One implementation, in `traffic/`, serves every front end:

| Path | What it is |
| --- | --- |
| `traffic/traffic_pipeline/` | Detection, tracking, lane geometry, rule-based reference cycle |
| `traffic/traffic_control/` | The DQN, and `AITrafficPipeline` which fuses the two |
| `traffic/weights/` | `detector_best.pt` (YOLO) and `rl_latest.pt` (DQN) |
| `traffic/configs/runtime.yaml` | Single config for both halves |
| `backend/` | HTTP + SSE server for the React dashboard |
| `src/` | The React dashboard |
| `traffic/app.py` | Standalone Flask four-feed UI |
| `traffic_main_files_bundle_2026-03-24/` | Training and dataset-prep scripts |

`AITrafficPipeline` holds a phase for `signal.minimum_green_time` before
reconsidering, so the output is a stable plan rather than a per-frame argmax.
An emergency vehicle on a red approach bypasses that hold.

## Setup

Python 3.10+ and Node 18+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
npm install
cp .env.example .env               # then set REACT_APP_MAPBOX_ACCESS_TOKEN
```

The repo ships no video. Either point at your own clip:

```bash
export TRAFFIC_DEFAULT_VIDEO_PATH=/path/to/traffic.mp4
```

or generate a synthetic one to check the plumbing:

```bash
python scripts/make_sample_video.py
```

(The synthetic clip exercises the whole chain but contains no real vehicles, so
detection counts stay at zero. Use real footage for meaningful numbers.)

## Run

```bash
npm run start:backend    # http://127.0.0.1:8000
npm start                # http://localhost:3000
```

Check what the backend actually loaded:

```bash
curl -s http://127.0.0.1:8000/api/config | python -m json.tool
```

The `ai` block reports the resolved detector weights, the DQN checkpoint, the
devices in use and the minimum green time.

### Standalone four-feed UI

```bash
cd traffic && python app.py       # http://127.0.0.1:5001
```

Upload one video per approach; it returns annotated feeds plus a per-second
timeline of DQN decisions and Q-values.

## Tests

```bash
npm run test:backend
```

Covers the wired chain — detections through the tracker into the real
checkpoint, the minimum-green hold, and the emergency override.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Liveness |
| `GET /api/config` | Resolved paths, AI description, defaults, lanes |
| `GET /api/stream` | SSE analysis stream |
| `GET /api/video` | Range-capable video passthrough for the preview |

`/api/stream` accepts `videoPath` (repeatable), `configPath`, `processFps`,
`emitIntervalSec`, `maxEmits` and `realtime`.

Each `analysis` event carries an `rlDecision`:

```json
{
  "action": 1,
  "actionName": "EW_GREEN",
  "decisionSource": "emergency_override",
  "qValues": [-6.857075, -5.804466],
  "laneSignals": { "N": "red", "E": "green", "S": "red", "W": "green" },
  "greenLanes": ["E", "W"],
  "phaseChanged": true,
  "phaseHeldForSec": 0.0
}
```

`metrics.laneSignals`, `metrics.rlGreenLanes` and `metrics.rlActionName` mirror
it for convenience, and each lane in `metrics.laneStats` carries its `signal`.

## Detector backends

`TRAFFIC_INFERENCE_PROVIDER` selects where boxes come from. `local` (default)
runs `traffic/weights/detector_best.pt`; `sagemaker` calls a remote endpoint and
needs `SAGEMAKER_ENDPOINT_NAME`, AWS credentials, and
`pip install -r backend/requirements-cloud.txt`. Tracking and the DQN are
identical either way — only the detector swaps. If a SageMaker endpoint name is
set and no provider is given, `sagemaker` is assumed.

Setting `TRAFFIC_S3_BUCKET` additionally archives every emitted event to S3.

## Configuration notes

`traffic/configs/runtime.yaml` drives both halves. Two things to watch:

- `vehicle_keywords` must cover every class the detector emits
  (`ambulance, auto-rikshaw, bike, bus, car, truck`) — anything not matched is
  silently dropped.
- `detector.device` and `rl.device` accept `auto`; the backend forces CPU when
  no CUDA device is available.
