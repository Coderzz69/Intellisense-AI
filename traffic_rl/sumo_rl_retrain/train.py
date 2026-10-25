"""
sumo_rl_retrain/train.py

Retrains a signal controller on this repo's 4x4 SUMO network with the sumo-rl
library, because the original environment (traffic_rl/env/) and checkpoints
were never committed (see ../MISSING_FILES.md).

One shared-weight DQN (model/dqn.py QNetwork) drives all 12 traffic lights.
Each light's observation is sumo-rl's default (phase one-hot, min-green flag,
lane densities, lane queues), zero-padded to the largest junction so 3-way and
4-way junctions share the same network. Reward: sumo-rl's diff-waiting-time.

    python sumo_rl_retrain/train.py --episodes 40
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
from sumo_rl import SumoEnvironment  # noqa: E402

SUMO_DIR = ROOT / "sumo"


def make_env(seed: int, fixed_ts: bool = False, extra: str = "") -> SumoEnvironment:
    cmd = f"--additional-files {SUMO_DIR / 'vtypes.add.xml'} --lateral-resolution 1.0 {extra}".strip()
    return SumoEnvironment(
        net_file=str(SUMO_DIR / "network.net.xml"),
        route_file=str(SUMO_DIR / "routes.rou.xml"),
        num_seconds=3600,
        delta_time=5,
        yellow_time=3,
        min_green=10,
        max_green=60,
        sumo_seed=seed,
        # SUMO's default, as in sumo/intersection.sumocfg: a car stuck for 300 s is teleported
        # instead of deadlocking the junction for the rest of the hour (sumo-rl's default is -1)
        time_to_teleport=300,
        fixed_ts=fixed_ts,
        sumo_warnings=False,
        additional_sumo_cmd=cmd,
        add_per_agent_info=False,
    )


def pad(obs: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[: len(obs)] = obs
    return out


def obs_dim(env: SumoEnvironment) -> int:
    return max(env.observation_spaces(ts).shape[0] for ts in env.ts_ids)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "sumo_rl_retrain" / "checkpoint.pt"))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(seed=args.seed)
    obs = env.reset()
    dim = obs_dim(env)
    n_actions = env.action_spaces(env.ts_ids[0]).n
    net = QNetwork(dim, n_actions).to(device)
    target = QNetwork(dim, n_actions).to(device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    loss_fn = nn.SmoothL1Loss()
    buf: deque = deque(maxlen=100_000)
    gamma, batch, steps = 0.99, 256, 0

    for ep in range(args.episodes):
        if ep > 0:
            # a different traffic seed every episode, so the policy doesn't fit one draw
            obs = env.reset(seed=args.seed * 1000 + ep)
        eps = max(0.05, 1.0 - ep / (0.6 * args.episodes))
        done, ep_reward = False, 0.0
        while not done:
            actions = act(net, obs, dim, eps, n_actions, device)
            nxt, rew, dones, info = env.step(actions)
            done = dones["__all__"]
            for ts, a in actions.items():
                if ts in nxt:
                    buf.append((pad(obs[ts], dim), a, rew.get(ts, 0.0), pad(nxt[ts], dim), float(done)))
                    ep_reward += rew.get(ts, 0.0)
            obs = nxt
            steps += 1
            if len(buf) >= 2_000 and steps % 2 == 0:
                s, a, r, s2, d = map(np.array, zip(*random.sample(buf, batch)))
                s = torch.as_tensor(s, device=device); s2 = torch.as_tensor(s2, device=device)
                a = torch.as_tensor(a, device=device).long(); r = torch.as_tensor(r, device=device).float()
                d = torch.as_tensor(d, device=device).float()
                q = net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # double DQN target
                    a2 = net(s2).argmax(1, keepdim=True)
                    y = r + gamma * (1 - d) * target(s2).gather(1, a2).squeeze(1)
                loss = loss_fn(q, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if steps % 500 == 0:
                target.load_state_dict(net.state_dict())
        print(f"episode {ep + 1:3d}/{args.episodes}  eps {eps:.2f}  reward {ep_reward:10.1f}  "
              f"mean wait (last step) {info.get('system_mean_waiting_time', 0):.1f}s", flush=True)
        torch.save({"state_dict": net.state_dict(), "obs_dim": int(dim), "n_actions": int(n_actions)}, args.out)
    env.close()


if __name__ == "__main__":
    main()
