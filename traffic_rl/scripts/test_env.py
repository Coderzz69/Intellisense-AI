"""
scripts/test_env.py
Phase 2 smoke test — runs 10 steps with random actions and prints shapes.
"""
import sys, os

# Add SUMO tools to path before any imports
_sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
sys.path.insert(0, os.path.join(_sumo_home, "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import CFG
from env.traffic_env import TrafficEnv
import random

print("[test_env] Starting environment smoke test …")
env = TrafficEnv(CFG)
state = env.reset()
print(f"  Initial state shape : {state.shape}")
print(f"  Initial state values: {state}")

for i in range(10):
    action = random.randrange(env.action_dim)
    next_state, reward, done, info = env.step(action)
    print(f"  Step {i+1:2d} | action={action} | reward={reward:.1f} | "
          f"done={done} | avg_queue={info['avg_queue']:.3f}")
    if done:
        break
    state = next_state

env.close()
print("[test_env] PASSED — environment works correctly.")
