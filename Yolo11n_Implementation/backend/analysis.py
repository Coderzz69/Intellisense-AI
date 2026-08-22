from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from backend.cloud.storage import S3AnalyticsSink
from backend.cloud.traffic_nodes import build_congestion_summary, build_traffic_nodes
from backend.runtime import DEFAULT_VIDEO_ENV_VAR, get_inference_provider

LANES = ("N", "E", "S", "W")


def parse_bool(raw_value: Optional[str], default: bool) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def parse_positive_float(raw_value: Optional[str], default: float, name: str) -> float:
    if raw_value in {None, ""}:
        return default
    value = float(raw_value)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def parse_non_negative_int(raw_value: Optional[str], default: int, name: str) -> int:
    if raw_value in {None, ""}:
        return default
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"{name} must be 0 or greater")
    return value


def format_timestamp(timestamp_sec: float) -> str:
    total_seconds = max(0, int(round(timestamp_sec)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def should_process_frame(
    frame_index: int,
    timestamp_sec: float,
    process_fps: float,
    next_process_timestamp: Optional[float],
) -> tuple[bool, Optional[float]]:
    scheduled_timestamp = 0.0 if next_process_timestamp is None else next_process_timestamp
    if timestamp_sec + 1e-9 < scheduled_timestamp:
        return False, scheduled_timestamp
    return True, scheduled_timestamp + (1.0 / process_fps)


def should_emit_state(
    timestamp_sec: float,
    emit_interval_sec: float,
    next_emit_timestamp: Optional[float],
) -> tuple[bool, Optional[float]]:
    scheduled_timestamp = emit_interval_sec if next_emit_timestamp is None else next_emit_timestamp
    if timestamp_sec + 1e-9 < scheduled_timestamp:
        return False, scheduled_timestamp

    while scheduled_timestamp <= timestamp_sec + 1e-9:
        scheduled_timestamp += emit_interval_sec
    return True, scheduled_timestamp


def build_video_metadata(capture: cv2.VideoCapture, video_path: Path) -> dict[str, object]:
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = round(frame_count / fps, 2) if frame_count and fps else 0.0
    return {
        "path": str(video_path),
        "fps": round(fps, 2),
        "frameCount": frame_count,
        "width": width,
        "height": height,
        "durationSec": duration_sec,
    }


def build_vehicle_details(snapshots) -> list[dict[str, object]]:
    details = []
    for snapshot in snapshots:
        details.append(
            {
                "id": snapshot.track_id,
                "lane": snapshot.lane,
                "className": snapshot.class_name,
                "confidence": round(snapshot.confidence, 3),
                "bbox": [round(float(value), 2) for value in snapshot.bbox],
                "isHalting": bool(snapshot.is_halting),
                "waitTimeSec": round(snapshot.wait_time, 2),
                "isEmergency": bool(snapshot.is_emergency),
            }
        )
    return details


def build_lane_stats(vehicle_details: list[dict[str, object]]) -> tuple[dict[str, int], dict[str, dict[str, object]]]:
    lane_distribution = {lane: 0 for lane in LANES}
    lane_stats: dict[str, dict[str, object]] = {
        lane: {
            "vehicles": 0,
            "halting": 0,
            "ambulances": 0,
            "emergencyVehicles": 0,
            "totalWaitTimeSec": 0.0,
            "averageWaitTimeSec": 0.0,
        }
        for lane in LANES
    }

    for vehicle in vehicle_details:
        lane = vehicle["lane"]
        lane_distribution[lane] = lane_distribution.get(lane, 0) + 1
        lane_entry = lane_stats.setdefault(
            lane,
            {
                "vehicles": 0,
                "halting": 0,
                "ambulances": 0,
                "emergencyVehicles": 0,
                "totalWaitTimeSec": 0.0,
                "averageWaitTimeSec": 0.0,
            },
        )
        lane_entry["vehicles"] += 1
        if vehicle["isHalting"]:
            lane_entry["halting"] += 1
        if vehicle["isEmergency"]:
            lane_entry["emergencyVehicles"] += 1
        if "ambulance" in str(vehicle["className"]).lower():
            lane_entry["ambulances"] += 1
        lane_entry["totalWaitTimeSec"] += float(vehicle["waitTimeSec"])

    for lane, lane_entry in lane_stats.items():
        count = lane_entry["vehicles"]
        total_wait = lane_entry["totalWaitTimeSec"]
        lane_entry["totalWaitTimeSec"] = round(total_wait, 2)
        lane_entry["averageWaitTimeSec"] = round(total_wait / count, 2) if count else 0.0

    return lane_distribution, lane_stats


def _lane_signals(rl_decision: Optional[dict]) -> dict[str, str]:
    """Per-lane green/red as decided by the DQN, or 'unknown' before the first decision."""
    if not rl_decision:
        return {lane: "unknown" for lane in LANES}
    signals = rl_decision.get("laneSignals") or {}
    return {lane: str(signals.get(lane, "unknown")) for lane in LANES}


def build_analysis_payload(
    result: dict[str, object],
    *,
    emitted_index: int,
    timestamp_sec: float,
) -> dict[str, object]:
    snapshots = result.get("snapshots", [])
    vehicles = list(result.get("vehicles", []))
    vehicle_details = build_vehicle_details(snapshots)
    lane_distribution, lane_stats = build_lane_stats(vehicle_details)

    total_vehicles = len(vehicle_details)
    halting_vehicles = sum(1 for vehicle in vehicle_details if vehicle["isHalting"])
    emergency_vehicles = sum(1 for vehicle in vehicle_details if vehicle["isEmergency"])
    ambulance_count = sum(
        1 for vehicle in vehicle_details if "ambulance" in str(vehicle["className"]).lower()
    )
    total_wait_time = sum(float(vehicle["waitTimeSec"]) for vehicle in vehicle_details)
    halting_wait_time = sum(
        float(vehicle["waitTimeSec"]) for vehicle in vehicle_details if vehicle["isHalting"]
    )

    metrics = {
        "totalVehicles": total_vehicles,
        "ambulanceCount": ambulance_count,
        "emergencyVehicleCount": emergency_vehicles,
        "haltingVehicleCount": halting_vehicles,
        "movingVehicleCount": max(0, total_vehicles - halting_vehicles),
        "averageWaitTimeSec": round(total_wait_time / total_vehicles, 2) if total_vehicles else 0.0,
        "averageHaltingWaitTimeSec": round(halting_wait_time / halting_vehicles, 2)
        if halting_vehicles
        else 0.0,
        "laneDistribution": lane_distribution,
        "laneStats": lane_stats,
        "currentGreenLane": result.get("current_green_lane"),
        "signalCycleTimeSec": round(
            float(result.get("signal_plan", {}).get("total_cycle_time", 0.0))
            if result.get("signal_plan")
            else 0.0,
            2,
        ),
    }

    rl_decision = result.get("rl_decision")
    lane_signals = _lane_signals(rl_decision)
    metrics["laneSignals"] = lane_signals
    metrics["rlGreenLanes"] = [lane for lane in LANES if lane_signals.get(lane) == "green"]
    metrics["rlActionName"] = None if rl_decision is None else rl_decision.get("actionName")
    metrics["rlDecisionSource"] = None if rl_decision is None else rl_decision.get("decisionSource")
    for lane, lane_entry in lane_stats.items():
        lane_entry["signal"] = lane_signals.get(lane, "unknown")
    timestamp_label = format_timestamp(timestamp_sec)
    observed_at = datetime.now(timezone.utc).isoformat()
    traffic_nodes = build_traffic_nodes(metrics, timestamp_label)
    analytics = build_congestion_summary(metrics)

    return {
        "type": "analysis",
        "emittedIndex": emitted_index,
        "timestampSec": round(timestamp_sec, 2),
        "timestampLabel": timestamp_label,
        "observedAtUtc": observed_at,
        "vehicles": vehicles,
        "vehicleDetails": vehicle_details,
        "metrics": metrics,
        "trafficNodes": traffic_nodes,
        "analytics": analytics,
        "inference": result.get("inference"),
        "inferenceProvider": result.get("inferenceProvider", get_inference_provider()),
        "signalPlan": result.get("signal_plan"),
        "rlDecision": rl_decision,
    }


def resolve_existing_path(raw_path: Optional[str], fallback: Optional[Path], kind: str) -> Path:
    candidate = Path(raw_path).expanduser() if raw_path else fallback
    if candidate is None:
        raise FileNotFoundError(
            f"{kind} path is not configured. Set it in the request or via {DEFAULT_VIDEO_ENV_VAR}."
        )
    if not candidate.exists():
        raise FileNotFoundError(f"{kind} was not found: {candidate}")
    return candidate.resolve()


def resolve_video_paths(raw_values: list[str], fallback: Optional[Path]) -> list[Path]:
    raw_paths: list[str] = []
    for raw_value in raw_values:
        if raw_value is None:
            continue
        normalized = str(raw_value).replace("\r", "\n")
        for line in normalized.split("\n"):
            for part in line.split(","):
                candidate = part.strip().strip('"').strip("'")
                if candidate:
                    raw_paths.append(candidate)

    if not raw_paths:
        if fallback is None:
            raise FileNotFoundError(
                f"At least one video path is required. Set it in the request or via {DEFAULT_VIDEO_ENV_VAR}."
            )
        return [fallback]

    return [
        resolve_existing_path(raw_path, None, f"Video path #{index + 1}")
        for index, raw_path in enumerate(raw_paths)
    ]


def build_video_summary(
    *,
    sequence_index: int,
    sequence_total: int,
    video_path: Path,
    video_metadata: dict[str, object],
    emitted_count: int,
    last_timestamp_sec: float,
    final_frame: dict[str, object] | None,
) -> dict[str, object]:
    final_metrics = final_frame.get("metrics", {}) if final_frame else {}
    analytics = final_frame.get("analytics", {}) if final_frame else {}
    return {
        "sequenceIndex": sequence_index,
        "sequenceTotal": sequence_total,
        "videoPath": str(video_path),
        "video": video_metadata,
        "emittedCount": emitted_count,
        "lastTimestampSec": round(last_timestamp_sec, 2),
        "lastTimestampLabel": format_timestamp(last_timestamp_sec),
        "finalMetrics": final_metrics,
        "analytics": analytics,
    }


def build_sequence_summary(video_summaries: list[dict[str, object]]) -> dict[str, object]:
    completed_summaries = [
        summary for summary in video_summaries if summary.get("finalMetrics")
    ]
    total_videos = len(video_summaries)
    completed_videos = len(completed_summaries)
    total_snapshots = sum(int(summary.get("emittedCount", 0)) for summary in video_summaries)
    total_vehicles = sum(
        int(summary.get("finalMetrics", {}).get("totalVehicles", 0))
        for summary in completed_summaries
    )
    total_ambulances = sum(
        int(summary.get("finalMetrics", {}).get("ambulanceCount", 0))
        for summary in completed_summaries
    )
    total_emergencies = sum(
        int(summary.get("finalMetrics", {}).get("emergencyVehicleCount", 0))
        for summary in completed_summaries
    )
    average_wait = (
        round(
            sum(
                float(summary.get("finalMetrics", {}).get("averageWaitTimeSec", 0.0))
                for summary in completed_summaries
            )
            / completed_videos,
            2,
        )
        if completed_videos
        else 0.0
    )
    average_vehicles = round(total_vehicles / completed_videos, 2) if completed_videos else 0.0
    busiest_summary = max(
        completed_summaries,
        key=lambda summary: int(summary.get("finalMetrics", {}).get("totalVehicles", 0)),
        default=None,
    )

    return {
        "totalVideos": total_videos,
        "completedVideos": completed_videos,
        "totalSnapshots": total_snapshots,
        "totalVehiclesInFinalFrames": total_vehicles,
        "totalAmbulances": total_ambulances,
        "totalEmergencyVehicles": total_emergencies,
        "averageVehiclesPerVideo": average_vehicles,
        "averageWaitTimeSec": average_wait,
        "busiestVideoPath": None if busiest_summary is None else busiest_summary.get("videoPath"),
        "busiestVideoVehicles": 0
        if busiest_summary is None
        else int(busiest_summary.get("finalMetrics", {}).get("totalVehicles", 0)),
    }


def json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def record_analytics_event(sink: Optional[S3AnalyticsSink], payload: dict[str, object]) -> None:
    if sink is None:
        return
    try:
        sink.record(payload)
    except Exception as exc:
        print(f"[traffic-dashboard] analytics sink error: {exc}")
