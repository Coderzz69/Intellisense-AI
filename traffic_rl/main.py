"""
main.py — Entry point for the Traffic Signal Control DQN system.

Usage:
    python main.py                   # train for configured episodes
    python main.py --mode evaluate   # evaluate latest checkpoint
    python main.py --mode train --episodes 10   # quick smoke test

The training loop:
  1. Load config
  2. Seed everything
  3. Instantiate TrafficEnv + DQNAgent
  4. Episode loop:
       reset → step → store_transition → train_step → log
  5. Decay epsilon per episode
  6. Hard target-network update every N episodes
  7. Save checkpoint every N episodes
  8. On Ctrl+C → save latest.pt + close SUMO
"""

import os
import sys
import time
import random
import signal
import argparse
import csv

import numpy as np
import torch

# ── Project imports ──────────────────────────────────────────────────────────
from config.config import CFG, load_config
from env.traffic_env import TrafficEnv
from training.train import DQNAgent


# ── Seed helper ──────────────────────────────────────────────────────────────

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Directory setup ──────────────────────────────────────────────────────────

def ensure_dirs(cfg: dict):
    for key in ("checkpoint_dir", "log_dir", "plot_dir"):
        os.makedirs(cfg["paths"][key], exist_ok=True)


# ── CSV Logger ───────────────────────────────────────────────────────────────

class CSVLogger:
    FIELDS = ["episode", "total_reward", "avg_wait", "avg_queue",
              "epsilon", "avg_loss", "steps", "wall_time_s"]

    def __init__(self, path: str):
        self._path = path
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, row: dict):
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow({k: row.get(k, 0) for k in self.FIELDS})


# ── Training ─────────────────────────────────────────────────────────────────

def train(cfg: dict, total_episodes: int = None, load_checkpoint: str = None, num_envs: int = 8):
    device_str = cfg["training"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available — falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[INFO] Device: {device}")

    print("[DEBUG] Device setup complete")
    set_seeds(cfg["seed"])
    ensure_dirs(cfg)

    import concurrent.futures

    print(f"[DEBUG] Instantiating {num_envs} Environments (Parallel)...")
    envs = [TrafficEnv(cfg, label=f"worker_{_}") for _ in range(num_envs)]
    
    print("[DEBUG] Instantiating Agent...")
    agent = DQNAgent(cfg, device)
    
    if load_checkpoint and os.path.isfile(load_checkpoint):
        print(f"[INFO] Loading checkpoint for fine-tuning: {load_checkpoint}")
        agent.load(load_checkpoint)
        if "epsilon_start" in cfg["agent"]:
             agent.epsilon = cfg["agent"]["epsilon_start"]
             
    print("[DEBUG] Setup complete, starting training...")

    total_episodes  = total_episodes or cfg["training"]["total_episodes"]
    save_freq       = cfg["training"]["save_freq"]
    target_upd_freq = cfg["agent"]["target_update_freq"]
    ckpt_dir        = cfg["paths"]["checkpoint_dir"]
    log_path        = os.path.join(cfg["paths"]["log_dir"], "training_log.csv")
    logger          = CSVLogger(log_path)

    _shutdown = {"flag": False}

    def _handle_sigint(sig, frame):
        print("\n[INFO] Ctrl+C received — saving and shutting down …")
        _shutdown["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Pre-allocate metrics for each environment worker ──────────────
    states     = [None] * num_envs
    ep_rewards = [0.0] * num_envs
    ep_losses  = [[] for _ in range(num_envs)]
    ep_waits   = [[] for _ in range(num_envs)]
    ep_queues  = [[] for _ in range(num_envs)]
    ep_steps   = [0] * num_envs
    ep_start_t = [time.time()] * num_envs

    print(f"[INFO] Starting training — {total_episodes} episodes across {num_envs} workers\n")
    print(f"{'Ep':>5}  {'Env':>3}  {'Reward':>10}  {'AvgWait':>9}  {'AvgQueue':>9}  "
          f"{'Eps':>6}  {'Loss':>8}  {'Steps':>6}")
    print("─" * 70)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_envs)

    # Initialize all envs
    def safe_reset(i):
        try:
            return envs[i].reset()
        except Exception as e:
            print(f"[WARN] env[{i}].reset() failed: {e}")
            return None

    states = list(executor.map(safe_reset, range(num_envs)))
    completed_episodes = 0

    while completed_episodes < total_episodes and not _shutdown["flag"]:
        
        # Failsafe: if all workers crash structurally, do not spin infinitely
        if all(s is None for s in states):
            print("\n[ERROR] All environment workers died unexpectedly. Halting training.")
            break
            
        # 1. Main thread uses Neural Network to pick actions for all states
        actions_dict_list = []
        for s_dict in states:
            if s_dict is not None:
                acts = {}
                for tls, st in s_dict.items():
                    acts[tls] = agent.select_action(st)
                actions_dict_list.append(acts)
            else:
                actions_dict_list.append(None)

        # 2. Worker threads run SUMO physics steps in parallel (Release Python GIL during TraCI Sockets)
        def thread_step(args):
            i, acts = args
            if states[i] is None or acts is None:
                return (None, None, True, {})
            try:
                return envs[i].step(acts)
            except Exception as e:
                print(f"[WARN] env[{i}].step() failed: {e}")
                return (None, None, True, {})

        results = list(executor.map(thread_step, [(i, actions_dict_list[i]) for i in range(num_envs)]))

        # 3. Main thread securely processes replays and trains Neural Network sequentially (No race conditions)
        for i, (next_states_dict, rewards_dict, done, info) in enumerate(results):
            if next_states_dict is None:
                continue
                
            total_env_reward = 0.0
            for tls in states[i].keys():
                s = states[i][tls]
                a = actions_dict_list[i][tls]
                r = rewards_dict[tls]
                ns = next_states_dict[tls]
                agent.store_transition(s, a, r, ns, done)
                total_env_reward += r
            
            # Perform training
            loss = agent.train_step()
            if loss > 0:
                ep_losses[i].append(loss)

            ep_rewards[i] += total_env_reward
            ep_waits[i].append(info.get("total_wait", 0.0))
            ep_queues[i].append(info.get("avg_queue", 0.0))
            ep_steps[i] += 1
            
            if done:
                completed_episodes += 1
                
                # Update agent hyperparams
                agent.decay_epsilon(completed_episodes)
                if completed_episodes % target_upd_freq == 0:
                    agent.update_target_network()
                
                avg_wait  = float(np.mean(ep_waits[i])) if ep_waits[i] else 0.0
                avg_queue = float(np.mean(ep_queues[i])) if ep_queues[i] else 0.0
                avg_loss  = float(np.mean(ep_losses[i])) if ep_losses[i] else 0.0
                wall_time = time.time() - ep_start_t[i]
                
                print(f"{completed_episodes:>5}  {i:>3}  {ep_rewards[i]:>10.1f}  {avg_wait:>9.1f}  "
                      f"{avg_queue:>9.3f}  {agent.epsilon:>6.3f}  "
                      f"{avg_loss:>8.4f}  {ep_steps[i]:>6}")
                
                logger.log({
                    "episode":      completed_episodes,
                    "total_reward": ep_rewards[i],
                    "avg_wait":     avg_wait,
                    "avg_queue":    avg_queue,
                    "epsilon":      agent.epsilon,
                    "avg_loss":     avg_loss,
                    "steps":        ep_steps[i],
                    "wall_time_s":  wall_time,
                })
                
                if completed_episodes % save_freq == 0:
                    ckpt = os.path.join(ckpt_dir, f"episode_{completed_episodes:04d}.pt")
                    agent.save(ckpt)
                
                # Reset this specific worker for a new episode
                if completed_episodes < total_episodes and not _shutdown["flag"]:
                    states[i] = safe_reset(i)
                else:
                    states[i] = None
                    
                ep_rewards[i] = 0.0
                ep_losses[i]  = []
                ep_waits[i]   = []
                ep_queues[i]  = []
                ep_steps[i]   = 0
                ep_start_t[i] = time.time()
                
            else:
                states[i] = next_states_dict

    # ── Final cleanup ────────────────────────────────────────────────────
    executor.shutdown(wait=False)
    for e in envs:
        e.close()
        
    latest = os.path.join(ckpt_dir, "latest.pt")
    agent.save(latest)
    print(f"\n[INFO] Training complete. Model saved → {latest}")
    print(f"[INFO] Log → {log_path}")
    return latest


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Traffic Signal Control DQN")
    parser.add_argument("--mode",     choices=["train", "evaluate"], default="train")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override total_episodes from config")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint for evaluation OR starting point for training")
    parser.add_argument("--config", default=None,
                        help="Path to a custom config.yaml file")
    parser.add_argument("--num_envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    args = parser.parse_args()

    # If a custom config is provided, reload it
    cfg = CFG
    if args.config:
        print(f"[INFO] Loading custom config: {args.config}")
        cfg = load_config(args.config)

    if args.mode == "train":
        train(cfg, total_episodes=args.episodes, load_checkpoint=args.checkpoint, num_envs=args.num_envs)

    elif args.mode == "evaluate":
        # Lazy import to avoid loading torch twice
        from evaluation.evaluate import evaluate
        ckpt = args.checkpoint or os.path.join(cfg["paths"]["checkpoint_dir"], "latest.pt")
        evaluate(checkpoint_path=ckpt, n_episodes=3)


if __name__ == "__main__":
    main()
