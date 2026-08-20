from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

LANE_ORDER: Tuple[str, str, str, str] = ("N", "E", "S", "W")


@dataclass(frozen=True)
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    lane: str
    is_halting: bool
    wait_time: float
    is_emergency: bool
    bbox: Tuple[float, float, float, float]
    class_name: str
    confidence: float

    def to_output_dict(self, temporal_metrics_available: bool = True) -> Dict[str, object]:
        return {
            "id": self.track_id,
            "lane": self.lane,
            "is_halting": self.is_halting if temporal_metrics_available else None,
            "wait_time": round(self.wait_time, 2) if temporal_metrics_available else None,
            "is_emergency": self.is_emergency,
        }


@dataclass(frozen=True)
class LaneMetrics:
    lane: str
    stopped_count: int
    total_wait_time: float
    emergency_present: bool
    queue_span_pixels: float
    queue_span_meters: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "lane": self.lane,
            "stopped_count": self.stopped_count,
            "total_wait_time": round(self.total_wait_time, 2),
            "emergency_present": self.emergency_present,
            "queue_span_pixels": round(self.queue_span_pixels, 2),
            "queue_span_meters": round(self.queue_span_meters, 2),
        }


@dataclass(frozen=True)
class SignalPhase:
    lane: str
    state: str
    duration: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "lane": self.lane,
            "state": self.state,
            "duration": round(self.duration, 2),
        }


@dataclass(frozen=True)
class SignalPlan:
    total_cycle_time: float
    current_lane_order: List[str]
    lane_timings: Dict[str, Dict[str, float]]
    phases: List[SignalPhase]
    lane_metrics: Dict[str, LaneMetrics]
    emergency_override_lane: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_cycle_time": round(self.total_cycle_time, 2),
            "current_lane_order": list(self.current_lane_order),
            "lane_timings": {
                lane: {
                    key: round(value, 2)
                    for key, value in timings.items()
                }
                for lane, timings in self.lane_timings.items()
            },
            "phases": [phase.to_dict() for phase in self.phases],
            "lane_metrics": {
                lane: metrics.to_dict()
                for lane, metrics in self.lane_metrics.items()
            },
            "emergency_override_lane": self.emergency_override_lane,
        }
