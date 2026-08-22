from __future__ import annotations

from typing import Dict, List, Optional

from .bridge import build_bridge_vector
from .config import AppConfig
from .detector import TrafficDetector, YoloTrafficDetector
from .models import Detection, SignalPlan
from .signal_control import PaperSignalController
from .tracker import SimpleTracker


class TrafficPipeline:
    """Detection -> tracking -> lane metrics -> rule-based signal plan.

    The detector is injected rather than constructed unconditionally so the same
    pipeline can run against local YOLO weights or a remote inference endpoint.
    Any object satisfying :class:`TrafficDetector` works.
    """

    def __init__(
        self,
        config: AppConfig,
        detector: Optional[TrafficDetector] = None,
    ) -> None:
        self.config = config
        self.detector: TrafficDetector = detector or YoloTrafficDetector(
            config.detector,
            config.vehicle_keywords,
        )
        self.tracker = SimpleTracker(config.tracking)
        self.signal_controller = PaperSignalController(
            config.signal,
            config.calibration,
        )
        self._signal_plan: Optional[SignalPlan] = None
        self._cycle_started_at = 0.0

    @property
    def provider(self) -> str:
        return getattr(self.detector, "provider", "local")

    def process_frame(
        self,
        frame,
        timestamp_sec: float,
        temporal_metrics_available: bool = True,
    ) -> Dict[str, object]:
        detections = self.detector.detect(frame)
        result = self.process_detections(
            detections=detections,
            frame_shape=frame.shape,
            timestamp_sec=timestamp_sec,
            temporal_metrics_available=temporal_metrics_available,
        )
        result["inference"] = getattr(self.detector, "inference_metadata", None)
        result["inferenceProvider"] = self.provider
        return result

    def process_detections(
        self,
        detections: List[Detection],
        frame_shape,
        timestamp_sec: float,
        temporal_metrics_available: bool = True,
    ) -> Dict[str, object]:
        snapshots = self.tracker.update(
            detections=detections,
            frame_shape=frame_shape,
            timestamp_sec=timestamp_sec,
            lane_polygons=self.config.lane_polygons,
            emergency_keywords=self.config.emergency_keywords,
        )

        if self._should_refresh_plan(timestamp_sec, snapshots):
            self._signal_plan = self.signal_controller.build_signal_plan(snapshots)
            self._cycle_started_at = timestamp_sec

        current_green_lane = None
        if self._signal_plan is not None:
            current_green_lane = self.signal_controller.active_green_lane(
                self._signal_plan,
                max(0.0, timestamp_sec - self._cycle_started_at),
            )

        return {
            "vehicles": [
                snapshot.to_output_dict(
                    temporal_metrics_available=temporal_metrics_available
                )
                for snapshot in snapshots
            ],
            "snapshots": snapshots,
            "temporal_metrics_available": temporal_metrics_available,
            "bridge_vector": build_bridge_vector(
                snapshots=snapshots,
                current_green_lane=current_green_lane,
                lane_order=self.config.signal.lane_order,
            ),
            "current_green_lane": current_green_lane,
            "signal_plan": None if self._signal_plan is None else self._signal_plan.to_dict(),
            "inferenceProvider": self.provider,
        }

    def _should_refresh_plan(self, timestamp_sec, snapshots) -> bool:
        emergency_present = any(snapshot.is_emergency for snapshot in snapshots)
        if self._signal_plan is None:
            return True
        if emergency_present:
            return True
        if self._signal_plan.emergency_override_lane is not None and not emergency_present:
            return True
        return (timestamp_sec - self._cycle_started_at) >= self._signal_plan.total_cycle_time
