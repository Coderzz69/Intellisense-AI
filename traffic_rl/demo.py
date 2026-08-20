# coding: utf-8
"""
demo.py — Hackathon Demo for SSS Traffic AI
Compare: Baseline | Hybrid AI | Master Hierarchical AI

Modes:
  triple   (default) — all 3 controllers, side-by-side table
  compare  — Baseline vs Master AI
  hybrid   — Baseline vs Hybrid AI
  baseline — Baseline only
  ai       — Master AI only

Usage:
  python demo.py --headless
  python demo.py --scenario emergency --verbose
"""

import os, time, argparse, random
import numpy as np
import torch

from config.config import CFG
from env.traffic_env import TrafficEnv
from training.train import DQNAgent
from controller.master_controller import MasterController
from controller.hybrid_controller import HybridController
from controller.baseline_controller import FixedTimerController
from evaluation.metrics import MetricsTracker


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_episode(env, controller, tracker, scenario, label):
    print(f"\n[DEMO] [{label}]")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    states_dict = env.reset()
    done, step_count, emg_injected = False, 0, False
    print(f"       managing {len(states_dict)} junctions.")
    if scenario == "heavy":
        env.apply_indian_traffic_behavior()

    while not done:
        if scenario == "emergency":
            # 5% chance of spawning an ambulance at each decision step
            if random.random() < 0.05:
                print("  -> Injecting Emergency Vehicle!")
                env.inject_emergency_vehicle()

        actions = {tls: controller.select_action(s) for tls, s in states_dict.items()}
        next_s, rewards, done, info = env.step(actions)
        tracker.step(sum(rewards.values()), info)
        states_dict = next_s
        step_count += 1

    arrived = env._traci.simulation.getArrivedNumber() if env._running else 0
    tracker.end_episode(total_arrived=arrived)
    env.close()


def run_master_episode(env, master, tracker, scenario, label, verbose=False):
    print(f"\n[DEMO] [{label}]")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    states_dict = env.reset()
    done, step_count, emg_injected = False, 0, False
    tls = list(states_dict.keys())
    print(f"       supervising {len(tls)} junctions.")
    grid_cols = max(1, int(len(tls) ** 0.5) - 1)
    master.tls_adjacency = MasterController.build_grid_adjacency(tls, grid_cols)
    master.verbose = verbose
    if scenario == "heavy":
        env.apply_indian_traffic_behavior()

    while not done:
        if scenario == "emergency":
            # 5% chance of spawning an ambulance at each decision step
            if random.random() < 0.05:
                print("  -> Injecting Emergency Vehicle!")
                env.inject_emergency_vehicle()

        actions = master.select_actions(states_dict)
        next_s, rewards, done, info = env.step(actions)
        tracker.step(sum(rewards.values()), info)
        states_dict = next_s
        step_count += 1

    arrived = env._traci.simulation.getArrivedNumber() if env._running else 0
    tracker.end_episode(total_arrived=arrived)
    env.close()


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _pct(old, new, lower_better=True):
    if max(abs(old), abs(new)) < 1e-6: return 0.0
    d = (old - new) / max(abs(old), 1e-6) * 100
    return d if lower_better else -d

METRICS = [
    ("Total Wait Time (s)",    "avg_total_wait",       True),
    ("Avg Wait / Vehicle (s)", "avg_wait_per_vehicle", True),
    ("Avg Emergency Wait (s)", "avg_emerg_wait",       True),
    ("Vehicles Arrived",       "avg_vehicles_passed",  False),
]

def print_triple(b, h, m):
    W = 70
    print("=" * W)
    hdr = f"  {'Metric':<28} {'Baseline':>10} {'HybridAI':>10}  {'vs.Base':>8}  {'MasterAI':>10}  {'vs.Base':>8}"
    print(hdr)
    print("-" * W)
    for label, key, lb in METRICS:
        bv = b[key]; hv = h[key]; mv = m[key]
        ph = _pct(bv, hv, lb); pm = _pct(bv, mv, lb)
        fmt = ".0f" if key == "avg_vehicles_passed" else ".1f"
        marker_h = " *" if ph >= pm else "  "
        marker_m = " *" if pm > ph else "  "
        print(
            f"  {label:<28} {bv:>10{fmt}} {hv:>10{fmt}}  {ph:>+7.1f}%{marker_h}  "
            f"{mv:>10{fmt}}  {pm:>+7.1f}%{marker_m}"
        )
    print("=" * W)
    print("  (* = better on this metric)  '+ %' = improvement vs Baseline")
    print("=" * W)

def print_two_way(b, a, label_a):
    W = 60
    print("=" * W)
    print(f"  {'Metric':<28} {'Baseline':>10} {label_a:>10} {'vs.Base':>8}")
    print("-" * W)
    for label, key, lb in METRICS:
        bv = b[key]; av = a[key]
        pct = _pct(bv, av, lb)
        fmt = ".0f" if key == "avg_vehicles_passed" else ".1f"
        print(f"  {label:<28} {bv:>10{fmt}} {av:>10{fmt}} {pct:>+7.1f}%")
    print("=" * W)

def print_single(stats, label):
    W = 50
    print("=" * W)
    print(f"  {label}")
    print("-" * W)
    for lbl, key, _ in METRICS:
        fmt = ".0f" if key == "avg_vehicles_passed" else ".1f"
        print(f"  {lbl:<28} {stats[key]:>10{fmt}}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["normal","heavy","emergency"], default="normal")
    parser.add_argument("--mode", choices=["triple","compare","hybrid","baseline","ai"],
                        default="triple")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--verbose",  action="store_true",
                        help="Show master-agent override messages.")
    args = parser.parse_args()

    cfg = CFG.copy()
    cfg["simulation"]["render"] = not args.headless
    if cfg["simulation"]["render"]:
        cfg["simulation"]["sumo_binary"] = "sumo-gui"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")
    print(f"[INFO] Scenario: {args.scenario.upper()} | Mode: {args.mode.upper()}\n")

    agent = DQNAgent(cfg, device)
    ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], "latest.pt")
    if os.path.exists(ckpt):
        print(f"[INFO] Loaded checkpoint: {ckpt}")
        agent.load(ckpt)
        agent.epsilon = 0.0
    else:
        print("[WARN] No checkpoint found — AI acts randomly.")

    env = TrafficEnv(cfg, label="demo_env")

    baseline_ctrl = FixedTimerController(action_dim=2)
    hybrid_ctrl   = HybridController(agent=agent)
    master_ctrl   = MasterController(agent=agent, verbose=args.verbose)

    b_t = MetricsTracker()
    h_t = MetricsTracker()
    m_t = MetricsTracker()

    run_b = args.mode in ["triple", "compare", "baseline", "hybrid"]
    run_h = args.mode in ["triple", "hybrid"]
    run_m = args.mode in ["triple", "compare", "ai"]

    if run_b:
        run_episode(env, baseline_ctrl, b_t, args.scenario, "Fixed-Timer Baseline")
    if run_h:
        run_episode(env, hybrid_ctrl,   h_t, args.scenario, "Hybrid AI  (Rule + DQN per junction)")
    if run_m:
        run_master_episode(env, master_ctrl, m_t, args.scenario,
                           "Master Hierarchical AI  (Global Supervisor + DQN)",
                           verbose=args.verbose)

    title = f" RESULTS  |  Scenario: {args.scenario.upper()}  |  Mode: {args.mode.upper()} "
    print(f"\n{title}")

    if args.mode == "triple":
        print_triple(b_t.summary(), h_t.summary(), m_t.summary())
    elif args.mode == "compare":
        print_two_way(b_t.summary(), m_t.summary(), "MasterAI")
    elif args.mode == "hybrid":
        print_two_way(b_t.summary(), h_t.summary(), "HybridAI")
    elif args.mode == "baseline":
        print_single(b_t.summary(), "Fixed-Timer Baseline")
    else:  # ai
        print_single(m_t.summary(), "Master Hierarchical AI")

    if args.scenario == "emergency":
        print("  NOTE: P1 Emergency Corridor fires immediately when ambulance is detected.")
    elif args.scenario == "heavy":
        print("  NOTE: Indian traffic behaviour applied (weaving, random braking).")


if __name__ == "__main__":
    main()
