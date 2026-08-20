"""
scripts/plot_results.py
Phase 5 — Plot training curves from the CSV log.
Saves a PNG to the plots/ directory.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import matplotlib.pyplot as plt
from config.config import CFG

log_path  = os.path.join(CFG["paths"]["log_dir"], "training_log.csv")
plot_dir  = CFG["paths"]["plot_dir"]
os.makedirs(plot_dir, exist_ok=True)

if not os.path.isfile(log_path):
    print(f"[plot_results] Log file not found: {log_path}")
    sys.exit(1)

episodes, rewards, waits, queues, epsilons = [], [], [], [], []

with open(log_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row["episode"]))
        rewards.append(float(row["total_reward"]))
        waits.append(float(row["avg_wait"]))
        queues.append(float(row["avg_queue"]))
        epsilons.append(float(row["epsilon"]))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Traffic RL — Training Curves", fontsize=14)

axes[0, 0].plot(episodes, rewards, color="steelblue")
axes[0, 0].set_title("Total Reward per Episode")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Reward")

axes[0, 1].plot(episodes, waits, color="tomato")
axes[0, 1].set_title("Avg Waiting Time per Episode (s)")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Seconds")

axes[1, 0].plot(episodes, queues, color="seagreen")
axes[1, 0].set_title("Avg Queue Length per Episode (norm.)")
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("Normalised queue")

axes[1, 1].plot(episodes, epsilons, color="goldenrod")
axes[1, 1].set_title("Epsilon (exploration) Decay")
axes[1, 1].set_xlabel("Episode")
axes[1, 1].set_ylabel("Epsilon")

plt.tight_layout()
out_path = os.path.join(plot_dir, "training_curves.png")
plt.savefig(out_path, dpi=120)
print(f"[plot_results] Saved → {out_path}")
