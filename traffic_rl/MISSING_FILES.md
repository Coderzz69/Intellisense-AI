# Missing from this repository: `traffic_rl/env/`

## What is missing

Every entry point in `traffic_rl/` imports the SUMO environment:

```python
from env.traffic_env import TrafficEnv   # main.py, demo.py, simulate.py, scripts/test_env.py
```

`traffic_rl/env/traffic_env.py` **is not in this repository**, so none of
`main.py`, `demo.py`, `simulate.py` or `scripts/test_env.py` can run. The
trained SUMO checkpoints under `traffic_rl/checkpoints/` are absent for the same
reason.

## Why it is missing

It was never committed. `.gitignore` carried the boilerplate Python
"Environments" block, whose bare `env/` pattern — meant for virtualenvs —
matched this project's own `env/` **source** package:

```
traffic_rl/.gitignore:118:env/          -> traffic_rl/env/traffic_env.py
traffic_rl/.gitignore:160:checkpoints/  -> traffic_rl/checkpoints/*.pt
```

Both patterns have now been corrected, so once the files are restored they will
commit normally. Verify with:

```bash
git check-ignore -v traffic_rl/env/traffic_env.py   # should print nothing
```

## How to restore it

Copy `env/` and `checkpoints/` from the machine that ran the training, then
commit them. Nothing else needs to change.

This file was deliberately **not** reconstructed from its call sites. The
reward shaping, state layout and lane/TLS mapping are not recoverable from the
callers, and the headline metrics in the top-level README were produced by the
original environment — a rewritten one would run but would not be the system
those numbers describe.

## The API the restored file must provide

Reconstructed from its callers, for reference when locating the right version:

| Member | Used by | Notes |
| --- | --- | --- |
| `TrafficEnv(cfg, label="default")` | `demo.py`, `main.py` | `label` names the TraCI connection so parallel workers do not collide (see `patch_traci.py`). |
| `.reset()` | all | Returns `{tls_id: state}` for the multi-agent callers; `scripts/test_env.py` treats it as a single state. |
| `.step(actions)` | all | `(next_state, reward, done, info)`. `main.py`/`demo.py` pass and receive `{tls_id: ...}` dicts; `scripts/test_env.py` passes a scalar action. |
| `.close()` | all | Shuts the TraCI connection down. |
| `.action_dim` | `scripts/test_env.py` | `2` per `config.yaml`. |
| `.tls_ids` | `controller/master_controller.py` | Traffic-light IDs, row-major. |
| `._traci`, `._running` | `demo.py` | Live connection handle, used for `simulation.getArrivedNumber()`. |
| `.apply_indian_traffic_behavior()` | `demo.py` | Per-step lane weaving / erratic braking. |
| `.inject_emergency_vehicle()` | `demo.py` | Spawns an ambulance for the emergency scenario. |

`config.yaml` fixes the contract at `state_dim: 5`
(`[phase_progress, is_yellow, norm_queue, norm_max_wait, has_emergency]`) and
`action_dim: 2` (`0 = hold green`, `1 = switch phase`).

Note this is a **different** model from the vision pipeline's DQN in
`Yolo11n_Implementation/traffic/weights/rl_latest.pt`, which is 16-dimensional.
That one is present and working.
