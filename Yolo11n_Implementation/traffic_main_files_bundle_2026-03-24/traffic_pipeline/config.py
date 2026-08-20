from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .models import LANE_ORDER

DEFAULT_BASE_WEIGHTS = "/home/anuj/Desktop/image-processing/yolo11n.pt"


@dataclass
class DetectorConfig:
    weights_path: str = DEFAULT_BASE_WEIGHTS
    confidence: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = "cpu"


@dataclass
class TrackingConfig:
    match_distance_px: float = 80.0
    halt_distance_px: float = 8.0
    max_missed_frames: int = 5
    starting_track_id: int = 101
    lane_assignment_anchor: str = "bottom_center"
    require_lane_polygon_match: bool = True


@dataclass
class CalibrationConfig:
    alpha: float = 1.0
    exponent: float = 1.0


@dataclass
class SignalConfig:
    total_cycle_time: float = 120.0
    orange_time: float = 5.0
    minimum_green_time: float = 5.0
    lane_order: List[str] = field(default_factory=lambda: list(LANE_ORDER))


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    vehicle_keywords: List[str] = field(
        default_factory=lambda: [
            "ambulance",
            "auto-rikshaw",
            "auto-rickshaw",
            "bike",
            "bicycle",
            "bus",
            "car",
            "motorcycle",
            "truck",
        ]
    )
    emergency_keywords: List[str] = field(
        default_factory=lambda: [
            "ambulance",
            "fire truck",
            "fire_engine",
            "emergency",
        ]
    )
    lane_polygons: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)


def _read_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _as_tuples(
    polygons: Optional[Dict[str, List[List[float]]]]
) -> Dict[str, List[Tuple[float, float]]]:
    if not polygons:
        return {}
    normalized: Dict[str, List[Tuple[float, float]]] = {}
    for lane, points in polygons.items():
        normalized[lane] = [(float(x), float(y)) for x, y in points]
    return normalized


def load_config(path: Optional[str] = None) -> AppConfig:
    if path is None:
        return AppConfig()

    payload = _read_yaml(Path(path))
    detector = DetectorConfig(**payload.get("detector", {}))
    tracking = TrackingConfig(**payload.get("tracking", {}))
    calibration = CalibrationConfig(**payload.get("calibration", {}))
    signal = SignalConfig(**payload.get("signal", {}))

    return AppConfig(
        detector=detector,
        tracking=tracking,
        calibration=calibration,
        signal=signal,
        vehicle_keywords=list(
            payload.get("vehicle_keywords", AppConfig().vehicle_keywords)
        ),
        emergency_keywords=list(
            payload.get("emergency_keywords", AppConfig().emergency_keywords)
        ),
        lane_polygons=_as_tuples(payload.get("lane_polygons")),
    )
