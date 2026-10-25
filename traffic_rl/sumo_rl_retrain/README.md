# sumo-rl retrain

The original `traffic_rl/env/` and checkpoints were never committed, so this folder retrains a controller on
the same SUMO network (`../sumo/`) with [sumo-rl](https://github.com/LucasAlegre/sumo-rl).

- One shared-weight DQN (`../model/dqn.py`) for all 12 lights; observations zero-padded to the largest junction.
- Every 5 s each light keeps or switches its green (min green 10 s, yellow 3 s); reward is sumo-rl's diff-waiting-time.
- Teleport after 300 s, SUMO's default, as in `../sumo/intersection.sumocfg`.

```bash
pip install eclipse-sumo sumo-rl torch
python sumo_rl_retrain/train.py --episodes 30      # ~30 min, writes checkpoint.pt
python sumo_rl_retrain/evaluate.py                 # fixed timer vs DQN on seeds 101-105
```

Result of the committed checkpoint (total stopped vehicle-seconds, every car counted):

| Seed | Fixed timer | DQN | Change |
|---|---|---|---|
| 101 | 82,519 | 26,691 | −67.7% |
| 102 | 79,439 | 33,379 | −58.0% |
| 103 | 96,232 | 26,641 | −72.3% |
| 104 | 82,613 | 201,375 | +143.8% |
| 105 | 80,007 | 44,827 | −44.0% |
| **All** | **420,810** | **332,913** | **−20.9%** |

Seed 104 shows the policy can still let a junction jam; it needs more training before it's robust.
