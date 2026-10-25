"""
sumo_rl_retrain/train.py

Retrains a signal controller on this repo's 4x4 SUMO network with the sumo-rl
library, because the original environment (traffic_rl/env/) and checkpoints
were never committed (see ../MISSING_FILES.md).

One shared-weight DQN (model/dqn.py QNetwork) drives all 12 traffic lights.
Reward: sumo-rl's diff-waiting-time.

Two things sumo-rl leaves out, both needed to stop a junction jamming:
  * max_green is stored but never enforced, so a policy could hold one green
    for minutes and starve the cross street (it did: 225 s at B2, seed 104).
    `guard()` forces the switch at max_green, in training and in evaluation.
  * the default observation has no green-time and no downstream information,
    and concatenates per-lane blocks whose positions shift between 3-way and
    4-way junctions. `FixedObservation` gives every feature a fixed slot:
    phase, green time so far, in/out lane summaries, then per-lane density and
    queue each padded to MAX_LANES.

A checkpoint is kept only when it beats the previous best on validation seeds
201-205, which are never used for training or for the final test (101-105).

    python sumo_rl_retrain/train.py --episodes 54 --workers 6   # episodes run in parallel
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.dqn import QNetwork  # noqa: E402

if "SUMO_HOME" not in os.environ:
    import sumo  # the eclipse-sumo wheel ships the binaries
    os.environ["SUMO_HOME"] = sumo.SUMO_HOME
from gymnasium import spaces  # noqa: E402
from sumo_rl import SumoEnvironment  # noqa: E402
from sumo_rl.environment.observations import ObservationFunction  # noqa: E402

SUMO_DIR = ROOT / "sumo"
MAX_LANES = 16  # most incoming lanes at any junction in sumo/network.net.xml (B1, B2, C1, C2)
VAL_SEEDS = (201, 202, 203, 204, 205)


class FixedObservation(ObservationFunction):
    """Same layout at every junction, so one network can read all of them."""

    def __call__(self) -> np.ndarray:
        ts = self.ts
        phase = [1.0 if ts.green_phase == i else 0.0 for i in range(2)]
        can_switch = float(ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time)
        green_time = min(ts.time_since_last_phase_change, ts.max_green) / ts.max_green
        dens, queue, out = ts.get_lanes_density(), ts.get_lanes_queue(), ts.get_out_lanes_density()
        summary = [max(dens, default=0), float(np.mean(dens)) if dens else 0.0,
                   max(queue, default=0), float(np.mean(queue)) if queue else 0.0,
                   max(out, default=0), float(np.mean(out)) if out else 0.0]
        pad_to = lambda v: (list(v) + [0.0] * MAX_LANES)[:MAX_LANES]
        return np.array(phase + [can_switch, green_time] + summary + pad_to(dens) + pad_to(queue), dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        n = 2 + 2 + 6 + 2 * MAX_LANES
        return spaces.Box(low=np.zeros(n, dtype=np.float32), high=np.ones(n, dtype=np.float32))


def make_env(seed: int, fixed_ts: bool = False, extra: str = "", min_green: int = 10) -> SumoEnvironment:
    # sumo-rl starts SUMO from the net and route files and never reads sumo/intersection.sumocfg,
    # so its <processing> options are repeated here. Without collision.action=remove, cars that
    # collide inside a junction under the sublane model stay wedged there and lock the grid.
    cfg = "--lateral-resolution 1.0 --collision.action remove --collision.check-junctions true --ignore-route-errors true"
    cmd = f"--additional-files {SUMO_DIR / 'vtypes.add.xml'} {cfg} {extra}".strip()
    return SumoEnvironment(
        net_file=str(SUMO_DIR / "network.net.xml"),
        route_file=str(SUMO_DIR / "routes.rou.xml"),
        num_seconds=3600,
        delta_time=5,
        yellow_time=3,
        min_green=min_green,
        max_green=60,
        sumo_seed=seed,
        # SUMO's default, as in sumo/intersection.sumocfg: a car stuck for 300 s is teleported
        # instead of deadlocking the junction for the rest of the hour (sumo-rl's default is -1)
        time_to_teleport=300,
        fixed_ts=fixed_ts,
        sumo_warnings=False,
        additional_sumo_cmd=cmd,
        add_per_agent_info=False,
        observation_class=FixedObservation,
    )


def pad(obs: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[: len(obs)] = obs
    return out


def obs_dim(env: SumoEnvironment) -> int:
    return max(env.observation_spaces(ts).shape[0] for ts in env.ts_ids)


def guard(env: SumoEnvironment, actions: dict) -> dict:
    """Force a switch once a green has run max_green seconds (sumo-rl never does)."""
    out = {}
    for ts, a in actions.items():
        sig = env.traffic_signals[ts]
        if not sig.is_yellow and a == sig.green_phase and sig.time_since_last_phase_change >= sig.max_green:
            a = (sig.green_phase + 1) % sig.num_green_phases
        out[ts] = a
    return out


def act(net: QNetwork, obs: dict, dim: int, eps: float, n_actions: int, device) -> dict:
    actions = {}
    for ts, o in obs.items():
        if random.random() < eps:
            actions[ts] = random.randrange(n_actions)
        else:
            with torch.no_grad():
                q = net(torch.as_tensor(pad(o, dim), device=device).unsqueeze(0))
            actions[ts] = int(q.argmax(1).item())
    return actions


def _run_episode(job: tuple) -> tuple:
    """One episode in a worker process: CPU copy of the policy, its own SUMO."""
    state_dict, dim, n_actions, eps, seed, collect = job
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed)
    net = QNetwork(dim, n_actions)
    net.load_state_dict(state_dict); net.eval()
    env = make_env(seed=seed)
    obs, done = env.reset(), False
    transitions, ep_reward, stopped = [], 0.0, 0
    while not done:
        actions = guard(env, act(net, obs, dim, eps, n_actions, "cpu"))
        nxt, rew, dones, info = env.step(actions)
        done = dones["__all__"]
        stopped += info["system_total_stopped"] * 5  # one decision step = 5 sim seconds
        for ts, a in actions.items():
            if ts in nxt:
                if collect:
                    transitions.append((pad(obs[ts], dim), a, rew.get(ts, 0.0), pad(nxt[ts], dim), float(done)))
                ep_reward += rew.get(ts, 0.0)
        obs = nxt
    env.close()
    return transitions, ep_reward, stopped, float(info.get("system_mean_waiting_time", 0))


def main() -> None:
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=54)
    ap.add_argument("--workers", type=int, default=6, help="SUMO simulations running at once")
    ap.add_argument("--validate-from", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "sumo_rl_retrain" / "checkpoint.pt"))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe = make_env(seed=args.seed)
    probe.reset()
    dim, n_actions = obs_dim(probe), probe.action_spaces(probe.ts_ids[0]).n
    probe.close()
    net = QNetwork(dim, n_actions).to(device)
    target = QNetwork(dim, n_actions).to(device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    loss_fn = nn.SmoothL1Loss()
    buf: deque = deque(maxlen=150_000)
    gamma, batch, updates, best = 0.99, 256, 0, None
    cpu_weights = lambda: {k: v.detach().cpu() for k, v in net.state_dict().items()}

    pool = mp.get_context("spawn").Pool(args.workers)
    done_eps = 0
    while done_eps < args.episodes:
        n = min(args.workers, args.episodes - done_eps)
        weights = cpu_weights()
        # a different traffic seed every episode, so the policy doesn't fit one draw
        jobs = [(weights, dim, n_actions, max(0.05, 1.0 - (done_eps + k) / (0.6 * args.episodes)),
                 args.seed * 1000 + done_eps + k, True) for k in range(n)]
        results = pool.map(_run_episode, jobs)
        steps = 0
        for k, (trans, ep_reward, _, mean_wait) in enumerate(results):
            buf.extend(trans)
            steps += len(trans) // len(probe.ts_ids)
            print(f"episode {done_eps + k + 1:3d}/{args.episodes}  eps {jobs[k][3]:.2f}  reward {ep_reward:10.1f}  "
                  f"mean wait (last step) {mean_wait:.1f}s", flush=True)
        done_eps += n
        # same ratio as one simulation: one gradient step per two decision steps
        for _ in range(steps // 2 if len(buf) >= 2_000 else 0):
            s_, a, r, s2, d = map(np.array, zip(*random.sample(buf, batch)))
            s_ = torch.as_tensor(s_, device=device); s2 = torch.as_tensor(s2, device=device)
            a = torch.as_tensor(a, device=device).long(); r = torch.as_tensor(r, device=device).float()
            d = torch.as_tensor(d, device=device).float()
            q = net(s_).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # double DQN target
                a2 = net(s2).argmax(1, keepdim=True)
                y = r + gamma * (1 - d) * target(s2).gather(1, a2).squeeze(1)
            loss = loss_fn(q, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            updates += 1
            if updates % 500 == 0:
                target.load_state_dict(net.state_dict())
        if done_eps >= args.validate_from:
            weights = cpu_weights()
            # validation: greedy policy (eps 0) with the guard, seeds never trained on
            score = sum(r[2] for r in pool.map(_run_episode, [(weights, dim, n_actions, 0.0, sd, False) for sd in VAL_SEEDS]))
            line = f"after {done_eps} episodes  validation {score:,} stopped vehicle-seconds"
            if best is None or score < best:
                best = score
                torch.save({"state_dict": net.state_dict(), "obs_dim": int(dim), "n_actions": int(n_actions),
                            "episode": done_eps, "validation": int(score)}, args.out)
                line += "  (saved)"
            print(line, flush=True)
    pool.close(); pool.join()


if __name__ == "__main__":
    main()
