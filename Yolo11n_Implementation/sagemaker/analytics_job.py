from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def iter_records(input_dir: Path):
    for path in sorted(input_dir.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def build_summary(records):
    hourly = defaultdict(lambda: {"vehicle_count_sum": 0, "events": 0, "max_congestion_score": 0.0})
    for record in records:
        payload = record.get("payload", {})
        analytics = payload.get("analytics", {})
        traffic_nodes = payload.get("trafficNodes", [])
        intersection = next((node for node in traffic_nodes if node.get("id") == "intersection-core"), {})
        vehicle_count = int(intersection.get("vehicle_count", intersection.get("vehicleCount", 0)))
        observed_at = str(payload.get("observedAtUtc", ""))[:13]
        hourly_entry = hourly[observed_at]
        hourly_entry["vehicle_count_sum"] += vehicle_count
        hourly_entry["events"] += 1
        hourly_entry["max_congestion_score"] = max(
            hourly_entry["max_congestion_score"],
            float(analytics.get("congestionScore", 0.0)),
        )

    summary = []
    for hour, values in sorted(hourly.items()):
        events = values["events"] or 1
        average_vehicle_count = round(values["vehicle_count_sum"] / events, 2)
        level = "peak" if average_vehicle_count >= 15 or values["max_congestion_score"] >= 25 else "steady"
        summary.append(
            {
                "hour": hour,
                "average_vehicle_count": average_vehicle_count,
                "max_congestion_score": round(values["max_congestion_score"], 2),
                "traffic_state": level,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate traffic events for SageMaker batch analytics.")
    parser.add_argument("--input-dir", required=True, help="Directory containing JSONL analytics events.")
    parser.add_argument("--output-dir", required=True, help="Directory where the analytics summary will be written.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(iter_records(input_dir))

    output_path = output_dir / "traffic-summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
