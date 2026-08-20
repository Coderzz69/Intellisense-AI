"""
simulate.py — Visual simulation using a trained DQN agent.

Opens sumo-gui so you can watch the agent control the traffic light
at the intersection in real-time.

Usage examples:
    # Use latest checkpoint (default)
    python simulate.py

    # Use a specific checkpoint
    python simulate.py --checkpoint checkpoints/episode_0500.pt

    # Run 2 episodes with a 150 ms GUI step delay
    python simulate.py --episodes 2 --delay 150

    # Watch a fixed-cycle baseline instead
    python simulate.py --mode baseline
"""

import os
import sys
import argparse
import copy

import numpy as np
import torch

# ── Project imports ──────────────────────────────────────────────────────────
from config.config import CFG
from env.traffic_env import TrafficEnv
from training.train import DQNAgent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_render_cfg(base_cfg: dict, delay_ms: int) -> dict:
    """Return a copy of cfg with render=True, sumo-gui, and custom delay."""
    cfg = copy.deepcopy(base_cfg)
    cfg["simulation"]["render"]       = True
    cfg["simulation"]["sumo_binary"]  = "sumo-gui"

    # Patch viewsettings delay so the sim runs at a human-watchable speed
    view_file = os.path.join(
        os.path.dirname(os.path.abspath(cfg["simulation"]["sumo_cfg"])),
        "viewsettings.xml",
    )
    if os.path.isfile(view_file) and delay_ms != 50:
        _patch_viewsettings_delay(view_file, delay_ms)

    return cfg


def _patch_viewsettings_delay(path: str, delay_ms: int):
    """Overwrite the <delay value="..."/> in viewsettings.xml."""
    import re
    with open(path) as f:
        content = f.read()
    content = re.sub(
        r'<delay\s+value="\d+"\s*/>',
        f'<delay value="{delay_ms}"/>',
        content,
    )
    with open(path, "w") as f:
        f.write(content)


def _load_agent(cfg: dict, checkpoint: str) -> DQNAgent:
    """Load a trained DQNAgent from a checkpoint file."""
    device_str = cfg["training"].get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    agent = DQNAgent(cfg, device)
    if checkpoint and os.path.isfile(checkpoint):
        agent.load(checkpoint)
        print(f"[SIM] Loaded checkpoint : {checkpoint}")
    else:
        print(f"[WARN] Checkpoint not found: {checkpoint!r}  — using random weights.")
    agent.epsilon = 0.0 if (checkpoint and os.path.isfile(checkpoint)) else 0.3
    return agent


def _print_header():
    print()
    print("=" * 60)
    print("  Traffic RL — Live Simulation (sumo-gui)")
    print("  DQN agent controls the intersection traffic light.")
    print("  Close the sumo-gui window (or Ctrl+C) to stop early.")
    print("=" * 60)
    print(f"\n{'Step':>5}  {'SimTime':>8}  {'Action':>8}  {'Reward':>8}  "
          f"{'Wait(s)':>8}  {'Queue':>6}")
    print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation loop
# ─────────────────────────────────────────────────────────────────────────────

ACTION_LABELS = {
    0: "NS-Str+L",
    1: "EW-Str+L",
    2: "NS-Rgt",
    3: "EW-Rgt",
}


def run_simulation(cfg: dict, agent: "DQNAgent | None", episode: int):
    """
    Run one visual episode.
    agent=None  → fixed-cycle alternating baseline.
    Returns a summary dict.
    """
    env = TrafficEnv(cfg)
    state = env.reset()

    total_reward = 0.0
    total_wait   = 0.0
    total_queue  = 0.0
    step_idx     = 0
    done         = False
    fixed_action = 0   # for baseline only

    print(f"\n[Episode {episode}]")

    while not done:
        if agent is not None:
            action = agent.select_action(state)
        else:
            action       = fixed_action
            fixed_action = 1 - fixed_action

        state, reward, done, info = env.step(action)

        total_reward += reward
        total_wait   += info["total_wait"]
        total_queue  += info["avg_queue"]
        step_idx     += 1

        label = ACTION_LABELS.get(action, str(action))
        print(f"{step_idx:>5}  {info['sim_time']:>8.0f}  {label:>8}  "
              f"{reward:>8.3f}  {info['total_wait']:>8.1f}  "
              f"{info['avg_queue']:>6.3f}")

    env.close()

    summary = {
        "episode":      episode,
        "total_reward": total_reward,
        "avg_wait":     total_wait  / max(step_idx, 1),
        "avg_queue":    total_queue / max(step_idx, 1),
        "steps":        step_idx,
    }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run a trained DQN agent visually in sumo-gui.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(CFG["paths"]["checkpoint_dir"], "latest.pt"),
        help="Path to a .pt checkpoint file",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to simulate",
    )
    parser.add_argument(
        "--delay", type=int, default=50,
        help="sumo-gui step delay in milliseconds (lower = faster)",
    )
    parser.add_argument(
        "--mode", choices=["agent", "baseline"], default="agent",
        help="'agent' = trained DQN policy, 'baseline' = fixed-cycle alternating",
    )
    args = parser.parse_args()

    _print_header()

    # Build a GUI-enabled config
    render_cfg = _build_render_cfg(CFG, args.delay)

    # Optionally load agent
    if args.mode == "agent":
        agent = _load_agent(CFG, args.checkpoint)
        print(f"[SIM] Mode       : DQN agent (greedy)")
    else:
        agent = None
        print(f"[SIM] Mode       : Fixed-cycle baseline")

    print(f"[SIM] Episodes   : {args.episodes}")
    print(f"[SIM] GUI delay  : {args.delay} ms/step")

    all_summaries = []
    try:
        for ep in range(1, args.episodes + 1):
            summary = run_simulation(render_cfg, agent, episode=ep)
            all_summaries.append(summary)
            print(f"\n  ↳ Episode {ep} done | "
                  f"total_reward={summary['total_reward']:.1f}  "
                  f"avg_wait={summary['avg_wait']:.1f}s  "
                  f"avg_queue={summary['avg_queue']:.3f}")
    except KeyboardInterrupt:
        print("\n[SIM] Stopped by user.")

    if all_summaries:
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        n = len(all_summaries)
        avg = lambda key: sum(s[key] for s in all_summaries) / n
        print(f"  Episodes run   : {n}")
        print(f"  Avg reward     : {avg('total_reward'):.2f}")
        print(f"  Avg wait       : {avg('avg_wait'):.2f}s")
        print(f"  Avg queue      : {avg('avg_queue'):.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
