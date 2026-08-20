from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Tuple

CAMERA_ID_ENV = "TRAFFIC_CAMERA_ID"
CENTER_LAT_ENV = "TRAFFIC_NODE_CENTER_LAT"
CENTER_LNG_ENV = "TRAFFIC_NODE_CENTER_LNG"

DEFAULT_CENTER_LAT = 17.385
DEFAULT_CENTER_LNG = 78.4867
LANE_OFFSETS: Dict[str, Tuple[float, float]] = {
    "N": (0.0072, 0.0),
    "E": (0.0, 0.0084),
    "S": (-0.0072, 0.0),
    "W": (0.0, -0.0084),
}
LANE_LABELS = {
    "N": "North Approach",
    "E": "East Approach",
    "S": "South Approach",
    "W": "West Approach",
}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _positive(value, fallback=0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return numeric if numeric >= 0 else float(fallback)


def _intensity(vehicles: float, halting: float, emergencies: float, wait_time: float) -> float:
    return round(max(1.0, vehicles + halting * 1.5 + emergencies * 2.5 + wait_time / 4.0), 2)


def build_traffic_nodes(metrics: Dict[str, object], timestamp_label: str) -> list[Dict[str, object]]:
    center_lat = _float_env(CENTER_LAT_ENV, DEFAULT_CENTER_LAT)
    center_lng = _float_env(CENTER_LNG_ENV, DEFAULT_CENTER_LNG)
    camera_id = os.environ.get(CAMERA_ID_ENV, "camera-001")
    observed_at = datetime.now(timezone.utc).isoformat()
    lane_stats = metrics.get("laneStats", {})

    total_vehicles = _positive(metrics.get("totalVehicles", 0), 0)
    halting = _positive(metrics.get("haltingVehicleCount", 0), 0)
    emergencies = _positive(metrics.get("emergencyVehicleCount", 0), 0)
    average_wait = _positive(metrics.get("averageWaitTimeSec", 0.0), 0.0)

    nodes = [
        {
            "id": "intersection-core",
            "title": "Intersection Core",
            "description": f"Aggregated congestion snapshot at {timestamp_label}.",
            "lat": center_lat,
            "lng": center_lng,
            "latitude": center_lat,
            "longitude": center_lng,
            "vehicle_count": int(total_vehicles),
            "vehicleCount": int(total_vehicles),
            "intensity": _intensity(total_vehicles, halting, emergencies, average_wait),
            "camera_id": camera_id,
            "cameraId": camera_id,
            "timestamp": observed_at,
            "source": "backend-summary",
        }
    ]

    for lane, label in LANE_LABELS.items():
        lat_offset, lng_offset = LANE_OFFSETS[lane]
        lane_entry = lane_stats.get(lane, {})
        lane_vehicles = _positive(lane_entry.get("vehicles", 0), 0)
        lane_halting = _positive(lane_entry.get("halting", 0), 0)
        lane_emergencies = _positive(lane_entry.get("emergencyVehicles", 0), 0)
        lane_wait = _positive(lane_entry.get("averageWaitTimeSec", 0.0), 0.0)

        nodes.append(
            {
                "id": f"lane-{lane.lower()}",
                "title": label,
                "description": f"Live lane snapshot at {timestamp_label}.",
                "lane": lane,
                "lat": round(center_lat + lat_offset, 6),
                "lng": round(center_lng + lng_offset, 6),
                "latitude": round(center_lat + lat_offset, 6),
                "longitude": round(center_lng + lng_offset, 6),
                "vehicle_count": int(lane_vehicles),
                "vehicleCount": int(lane_vehicles),
                "intensity": _intensity(
                    lane_vehicles,
                    lane_halting,
                    lane_emergencies,
                    lane_wait,
                ),
                "camera_id": camera_id,
                "cameraId": camera_id,
                "timestamp": observed_at,
                "source": "backend-lane",
            }
        )

    return nodes


def build_congestion_summary(metrics: Dict[str, object]) -> Dict[str, object]:
    total_vehicles = _positive(metrics.get("totalVehicles", 0), 0)
    halting = _positive(metrics.get("haltingVehicleCount", 0), 0)
    average_wait = _positive(metrics.get("averageWaitTimeSec", 0), 0.0)
    emergencies = _positive(metrics.get("emergencyVehicleCount", 0), 0)

    congestion_score = round(
        total_vehicles + halting * 1.5 + average_wait * 0.75 + emergencies * 2.5,
        2,
    )
    if congestion_score >= 35:
        level = "severe"
    elif congestion_score >= 20:
        level = "high"
    elif congestion_score >= 10:
        level = "moderate"
    else:
        level = "low"

    return {
        "congestionScore": congestion_score,
        "congestionLevel": level,
        "isPeakTraffic": total_vehicles >= 12 or halting >= 5,
    }
