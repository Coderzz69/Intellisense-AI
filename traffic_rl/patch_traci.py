import os

# --- Patch env/traffic_env.py ---
fpath = "env/traffic_env.py"
with open(fpath, "r") as f:
    text = f.read()

# 1. Update __init__
text = text.replace("def __init__(self, cfg: dict):", "def __init__(self, cfg: dict, label: str = \"default\"):\n        self.label = label")

# 2. Update traci.start and get connection
reset_old = "traci.start(sumo_cmd, port=port)"
reset_new = "traci.start(sumo_cmd, port=port, label=self.label)\n            self._traci = traci.getConnection(self.label)"
text = text.replace(reset_old, reset_new)

# 3. Replace traci references
targets = ["trafficlight", "simulationStep", "lane", "vehicle", "simulation", "close"]
for tgt in targets:
    text = text.replace(f"traci.{tgt}", f"self._traci.{tgt}")

with open(fpath, "w") as f:
    f.write(text)

# --- Patch main.py ---
main_path = "main.py"
with open(main_path, "r") as f:
    main_text = f.read()

main_text = main_text.replace("envs = [TrafficEnv(cfg) for _ in range(num_envs)]",
                              "envs = [TrafficEnv(cfg, label=f\"worker_{_}\") for _ in range(num_envs)]")

with open(main_path, "w") as f:
    f.write(main_text)

print("Patch applied successfully.")
