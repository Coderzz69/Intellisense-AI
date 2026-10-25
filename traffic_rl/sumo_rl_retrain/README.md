# sumo-rl retrain

The original `traffic_rl/env/` and checkpoints were never committed, so this folder retrains a controller on
the same SUMO network (`../sumo/`) with [sumo-rl](https://github.com/LucasAlegre/sumo-rl).

- One shared-weight DQN (`../model/dqn.py`) for all 12 lights.
- Every 5 s each light keeps or switches its green; greens last 10 to 60 s (sumo-rl never enforces `max_green`,
  so `guard()` does). Reward is sumo-rl's diff-waiting-time.
- Observation (`FixedObservation`, 42 numbers, same slot for the same feature at every junction): phase,
  green time so far, in/out lane summaries, per-lane density and queue.
- SUMO gets the same `<processing>` options as `../sumo/intersection.sumocfg` (sublane model, collisions
  removed, checked at junctions). sumo-rl doesn't read the sumocfg, and without these options collisions
  inside junctions lock the grid even under fixed timers.
- Episodes run in parallel worker processes; the checkpoint kept is the best on validation seeds 201-205.

```bash
pip install eclipse-sumo sumo-rl torch
python sumo_rl_retrain/train.py --episodes 54 --workers 6   # ~30 min on 12 cores
python sumo_rl_retrain/evaluate.py                          # fixed timer vs DQN on test seeds 101-105
```

Result of the committed checkpoint (total stopped vehicle-seconds over one hour, every car counted):

| Seed | Fixed timer | DQN | Change |
|---|---|---|---|
| 101 | 71,050 | 38,863 | −45.3% |
| 102 | 69,999 | 24,544 | −64.9% |
| 103 | 72,581 | 21,587 | −70.3% |
| 104 | 69,379 | 22,327 | −67.8% |
| 105 | 66,346 | 25,473 | −61.6% |
| **All** | **349,355** | **132,794** | **−62.0%** |

Trips completed: 14,544 → 14,593. For reference, a hand-written max-pressure controller with the same
settings scored 19,000–48,000 per seed, so the DQN is competitive with a strong classical baseline.
