"""
scripts/test_agent.py
Phase 3 smoke test — verifies DQNAgent with dummy data (no SUMO needed).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from config.config import CFG
from training.train import DQNAgent

print("[test_agent] Starting agent smoke test …")

device = torch.device("cpu")
agent  = DQNAgent(CFG, device)

# Push dummy transitions until buffer is large enough to train
print("  Filling replay buffer …")
for _ in range(CFG["replay_buffer"]["min_size"] + 10):
    state      = np.random.rand(CFG["environment"]["state_dim"]).astype(np.float32)
    action     = np.random.randint(0, 2)
    reward     = float(np.random.uniform(-100, 0))
    next_state = np.random.rand(CFG["environment"]["state_dim"]).astype(np.float32)
    done       = bool(np.random.rand() > 0.99)
    agent.store_transition(state, action, reward, next_state, done)

action = agent.select_action(state)
print(f"  select_action output : {action} (valid if 0 or 1)")
assert action in (0, 1), f"Unexpected action: {action}"

loss = agent.train_step()
print(f"  train_step loss      : {loss:.6f}")
assert isinstance(loss, float) and loss >= 0.0

agent.update_target_network()
print("  update_target_network: OK")

print("[test_agent] PASSED — agent works correctly.")
