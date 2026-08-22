"""End-to-end AI pipeline: video frame -> YOLO -> tracking -> DQN signal decision.

This is the single entry point the dashboards and CLIs should use. It owns the
whole chain so callers do not have to re-assemble it (and drift apart) each time:

    frame
      -> TrafficDetector      (local YOLO weights, or a remote endpoint)
      -> SimpleTracker        (per-lane queues, halting state, wait times)
      -> PaperSignalController(rule-based reference cycle plan)
      -> RLTrafficController  (trained DQN: which lane pair goes green)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from traffic_pipeline import AppConfig, TrafficDetector, TrafficPipeline, load_config

from .config import RLConfig, load_rl_config
from .controller import RLDecision, RLTrafficController


class AITrafficPipeline:
    """Detection pipeline + trained DQN controller behind one ``process_frame``.

    The DQN is only allowed to flip the phase once ``min_green_sec`` has elapsed,
    so the emitted signal is a stable plan rather than a per-frame argmax that
    flickers. Emergency overrides bypass the hold, which is the whole point of
    the override.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        detector: Optional[TrafficDetector] = None,
        app_config: Optional[AppConfig] = None,
        rl_config: Optional[RLConfig] = None,
        min_green_sec: Optional[float] = None,
    ) -> None:
        self.config_path = str(config_path)
        self.config = app_config or load_config(self.config_path)
        self.pipeline = TrafficPipeline(self.config, detector=detector)
        self.controller = RLTrafficController(rl_config or load_rl_config(self.config_path))
        self.min_green_sec = (
            float(self.config.signal.minimum_green_time)
            if min_green_sec is None
            else float(min_green_sec)
        )

        self._pending_vectors: List[List[float]] = []
        self._decision: Optional[RLDecision] = None
        self._decided_at: Optional[float] = None

    @property
    def provider(self) -> str:
        return self.pipeline.provider

    @property
    def rl_device(self) -> str:
        return str(self.controller.device)

    def describe(self) -> Dict[str, Any]:
        return {
            "configPath": self.config_path,
            "detectorWeightsPath": self.config.detector.weights_path,
            "detectorDevice": self.config.detector.device,
            "inferenceProvider": self.provider,
            "rlCheckpointPath": self.controller.config.checkpoint_path,
            "rlDevice": self.rl_device,
            "rlStateDim": self.controller.config.state_dim,
            "rlActionDim": self.controller.config.action_dim,
            "minGreenSec": self.min_green_sec,
        }

    def process_frame(
        self,
        frame,
        timestamp_sec: float,
        temporal_metrics_available: bool = True,
    ) -> Dict[str, Any]:
        result = self.pipeline.process_frame(
            frame,
            timestamp_sec=timestamp_sec,
            temporal_metrics_available=temporal_metrics_available,
        )
        return self.attach_decision(result, timestamp_sec=timestamp_sec)

    def process_detections(
        self,
        detections,
        frame_shape,
        timestamp_sec: float,
        temporal_metrics_available: bool = True,
    ) -> Dict[str, Any]:
        result = self.pipeline.process_detections(
            detections=detections,
            frame_shape=frame_shape,
            timestamp_sec=timestamp_sec,
            temporal_metrics_available=temporal_metrics_available,
        )
        return self.attach_decision(result, timestamp_sec=timestamp_sec)

    def attach_decision(self, result: Dict[str, Any], *, timestamp_sec: float) -> Dict[str, Any]:
        """Fold one pipeline result into the RL controller and annotate it."""
        snapshots = result.get("snapshots", [])
        raw_vector = self.controller.raw_bridge_from_snapshots(snapshots)
        self._pending_vectors.append(raw_vector)

        emergency_present = any(value > 0.5 for value in raw_vector[12:16])
        if self._should_decide(timestamp_sec, emergency_present):
            self._decision = self.controller.decide(self._pending_vectors)
            self._decided_at = timestamp_sec
            self._pending_vectors = []

        result["rl_decision"] = None if self._decision is None else _decision_to_dict(
            self._decision,
            decided_at_sec=self._decided_at,
            timestamp_sec=timestamp_sec,
        )
        return result

    def _should_decide(self, timestamp_sec: float, emergency_present: bool) -> bool:
        if self._decision is None or self._decided_at is None:
            return True
        if emergency_present:
            return True
        return (timestamp_sec - self._decided_at) >= self.min_green_sec


def _decision_to_dict(
    decision: RLDecision,
    *,
    decided_at_sec: Optional[float],
    timestamp_sec: float,
) -> Dict[str, Any]:
    held_for = 0.0 if decided_at_sec is None else max(0.0, timestamp_sec - decided_at_sec)
    return {
        "action": decision.action,
        "actionName": decision.action_name,
        "decisionSource": decision.decision_source,
        "qValues": decision.q_values,
        "state": decision.state,
        "laneSignals": dict(decision.lane_signals),
        "greenLanes": list(decision.green_lanes),
        "redLanes": list(decision.red_lanes),
        "previousAction": decision.current_phase_action,
        "phaseChanged": decision.action != decision.current_phase_action,
        "decidedAtSec": None if decided_at_sec is None else round(decided_at_sec, 2),
        "phaseHeldForSec": round(held_for, 2),
    }
