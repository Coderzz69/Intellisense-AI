"""
sumo_rl_retrain/evaluate.py

Runs one hour of the 4x4 network twice on the same seed, once on SUMO's fixed
timer programme and once under the retrained DQN, and reports what each run
measured from SUMO's own tripinfo output.

    python sumo_rl_retrain/evaluate.py                     # both, prints the table
    python sumo_rl_retrain/evaluate.py --record out_dir    # also logs vehicle positions for a video
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import ROOT, QNetwork, act, make_env  # noqa: E402


def run(mode: str, seed: int, ckpt: str, tripinfo: Path, record: dict | None) -> dict:
    env = make_env(seed=seed, fixed_ts=(mode == "fixed"), extra=f"--tripinfo-output {tripinfo}")
    stats = []
    win = record["window"] if record else None
    frames = []

    original_step = env._sumo_step

    def logged_step():
        original_step()
        s = env.sumo
        t = int(s.simulation.getTime())
        ids = s.vehicle.getIDList()
        stopped = sum(1 for v in ids if s.vehicle.getSpeed(v) < 0.1)
        stats.append((t, len(ids), stopped))
        if win and win[0] <= t < win[1]:
            vehicles = []
            for v in ids:
                x, y = s.vehicle.getPosition(v)
                vehicles.append([round(x), round(y), s.vehicle.getTypeID(v)[0], int(s.vehicle.getSpeed(v) < 0.1)])
            tls = {ts: s.trafficlight.getRedYellowGreenState(ts) for ts in env.ts_ids}
            frames.append({"t": t, "v": vehicles, "tls": tls, "stopped": stopped, "n": len(ids)})

    env._sumo_step = logged_step
    obs = env.reset()
    net = None
    if mode == "dqn":
        c = torch.load(ckpt, map_location="cpu")
        net = QNetwork(c["obs_dim"], c["n_actions"])
        net.load_state_dict(c["state_dict"]); net.eval()
    done = False
    while not done:
        actions = {} if net is None else act(net, obs, c["obs_dim"], 0.0, c["n_actions"], "cpu")
        obs, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    env.close()

    trips = ET.parse(tripinfo).getroot().findall("tripinfo")
    waits = np.array([float(t.get("waitingTime")) for t in trips])
    result = {
        "mode": mode,
        "completed_trips": len(trips),
        # counts every vehicle, including those still stuck when the hour ends
        "total_wait_s": sum(st[2] for st in stats),
        "tripinfo_wait_s": round(float(waits.sum())),
        "mean_wait_completed_s": round(float(waits.mean()), 1),
        "still_on_road_at_end": stats[-1][1],
    }
    if record:
        out = Path(record["dir"]); out.mkdir(parents=True, exist_ok=True)
        (out / f"{mode}_frames.json").write_text(json.dumps(frames, separators=(",", ":")))
        (out / f"{mode}_stats.json").write_text(json.dumps({"result": result, "per_second": stats}, separators=(",", ":")))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="101,102,103,104,105", help="evaluation seeds (training used 7000+)")
    ap.add_argument("--ckpt", default=str(ROOT / "sumo_rl_retrain" / "checkpoint.pt"))
    ap.add_argument("--record", default=None, help="directory to write frames for a video")
    ap.add_argument("--window", default="1200,2400", help="sim seconds to record, start,end")
    args = ap.parse_args()
    rec = None
    if args.record:
        a, b = (int(x) for x in args.window.split(","))
        rec = {"dir": args.record, "window": (a, b)}
    tmp = Path(args.record or ".")
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    for i, seed in enumerate(seeds):
        # only the first seed is recorded for the video; it was fixed before any results were seen
        r = {m: run(m, seed, args.ckpt, tmp / f"tripinfo_{m}_{seed}.xml", rec if i == 0 else None) for m in ("fixed", "dqn")}
        rows.append((seed, r))
        f, d = r["fixed"], r["dqn"]
        print(f"seed {seed}: stopped vehicle-seconds {f['total_wait_s']:,} -> {d['total_wait_s']:,} "
              f"({(d['total_wait_s'] - f['total_wait_s']) / f['total_wait_s'] * 100:+.1f}%), "
              f"trips {f['completed_trips']} -> {d['completed_trips']}", flush=True)
    F = sum(r["fixed"]["total_wait_s"] for _, r in rows); D = sum(r["dqn"]["total_wait_s"] for _, r in rows)
    print(f"all seeds: {F:,} -> {D:,} ({(D - F) / F * 100:+.1f}%)")
    Path(tmp / "summary.json").write_text(json.dumps([{"seed": s, **r} for s, r in rows], indent=1))


if __name__ == "__main__":
    main()
