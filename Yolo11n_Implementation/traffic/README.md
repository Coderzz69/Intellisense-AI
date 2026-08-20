# Traffic RL Integration

This project combines:

- the object detection and lane-tracking pipeline from `~/Desktop/image-processing/traffic`
- the trained RL traffic-signal model from `~/Desktop/traffic_rl`

The result is a single video-to-decision pipeline that:

1. detects vehicles and ambulances with YOLO
2. assigns detections to the four lanes `N/E/S/W`
3. tracks halting vehicles and accumulated wait time
4. converts those lane metrics into the 16-value RL state
5. runs the trained DQN checkpoint
6. outputs which lanes should be `green` or `red`

## Four-Feed UI

The project also includes a browser interface where you can:

- upload four video feeds labeled `North`, `East`, `South`, and `West`
- view annotated detector output for each feed
- scrub a shared timeline across all four videos
- see the RL decision as four lane lights in the center board
- inspect per-lane vehicle count, queue, wait, emergency flag, and Q-values

Run it with the environment that already has `flask`, `torch`, `opencv-python`, `PyYAML`, and `ultralytics` installed:

```bash
cd /home/anuj/Desktop/traffic
/home/anuj/Desktop/traffic_rl/.venv/bin/python app.py
```

Then open `http://127.0.0.1:5001`.

If you need a different port:

```bash
cd /home/anuj/Desktop/traffic
PORT=5002 /home/anuj/Desktop/traffic_rl/.venv/bin/python app.py
```

## Output format

Each emitted decision is a JSON object like:

```json
{
  "timestamp_sec": 12.0,
  "state": [0.02, 0.0, 0.01, 0.0, 0.04, 0.0, 0.03, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  "q_values": [0.13821, 0.22109],
  "action": 1,
  "action_name": "EW_GREEN",
  "decision_source": "emergency_override",
  "lane_signals": {
    "N": "red",
    "E": "green",
    "S": "red",
    "W": "green"
  },
  "green_lanes": ["E", "W"],
  "red_lanes": ["N", "S"],
  "vehicles": [
    {"id": 101, "lane": "E", "is_halting": true, "wait_time": 2.3, "is_emergency": true}
  ]
}
```

## Run

Use a Python environment with `torch`, `opencv-python`, `PyYAML`, and `ultralytics`.

```bash
python scripts/run_rl_controller.py \
  --source path/to/video.mp4 \
  --config configs/runtime.yaml \
  --output-jsonl outputs/decisions.jsonl \
  --annotated-output outputs/annotated.mp4
```

For a webcam:

```bash
python scripts/run_rl_controller.py \
  --source 0 \
  --config configs/runtime.yaml \
  --show
```

## Notes

- The RL checkpoint controls paired lanes, not four independent phases:
  - action `0`: `N` and `S` are green
  - action `1`: `E` and `W` are green
- The controller keeps the current `NS/EW` phase internally and encodes that into the RL state exactly the way the checkpoint expects.
- If an ambulance is detected on the currently red pair, the controller overrides the RL action and switches priority to that pair.
