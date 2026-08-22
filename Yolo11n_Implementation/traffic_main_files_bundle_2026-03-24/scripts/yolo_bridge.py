#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _ai_path  # noqa: F401,E402  (puts the canonical traffic_pipeline on sys.path)

from traffic_pipeline.models import TrackSnapshot
from traffic_pipeline.bridge import build_bridge_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the JSON vehicle list into the 16-number bridge vector."
    )
    parser.add_argument("--input", required=True, help="JSON file or JSONL file of vehicle lists.")
    parser.add_argument(
        "--current-green-lane",
        choices=["N", "E", "S", "W"],
        default=None,
        help="Optional current green lane used for the one-hot segment.",
    )
    return parser.parse_args()


def _record_to_snapshot(record) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=int(record["id"]),
        lane=str(record["lane"]),
        is_halting=bool(record["is_halting"]),
        wait_time=float(record["wait_time"]),
        is_emergency=bool(record["is_emergency"]),
        bbox=(0.0, 0.0, 0.0, 0.0),
        class_name="unknown",
        confidence=1.0,
    )


def iter_vehicle_lists(path: Path) -> Iterable[List[dict]]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return
    if content[0] == "[" and "\n[" not in content:
        data = json.loads(content)
        if data and isinstance(data[0], dict):
            yield data
            return
    for line in content.splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    for vehicle_list in iter_vehicle_lists(Path(args.input)):
        snapshots = [_record_to_snapshot(record) for record in vehicle_list]
        vector = build_bridge_vector(
            snapshots=snapshots,
            current_green_lane=args.current_green_lane,
        )
        print(json.dumps(vector))


if __name__ == "__main__":
    main()
