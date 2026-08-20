from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .config import CalibrationConfig, SignalConfig
from .geometry import lane_queue_span_pixels
from .models import LaneMetrics, SignalPhase, SignalPlan, TrackSnapshot


class PaperSignalController:
    def __init__(
        self,
        signal_config: SignalConfig,
        calibration_config: CalibrationConfig,
    ) -> None:
        self.signal_config = signal_config
        self.calibration_config = calibration_config

    def pixel_to_meter(self, pixel_length: float) -> float:
        if pixel_length <= 0.0:
            return 0.0
        alpha = self.calibration_config.alpha
        exponent = self.calibration_config.exponent
        return alpha * (pixel_length ** exponent)

    def build_signal_plan(
        self,
        snapshots: Iterable[TrackSnapshot],
    ) -> SignalPlan:
        items = list(snapshots)
        lane_order = list(self.signal_config.lane_order)
        lane_metrics = self._compute_lane_metrics(items, lane_order)
        emergency_lane = self._choose_emergency_lane(lane_metrics, lane_order)
        if emergency_lane is not None:
            return self._build_emergency_plan(lane_metrics, lane_order, emergency_lane)

        green_times = self._allocate_green_times(lane_metrics, lane_order)
        orange_time = self.signal_config.orange_time
        lane_timings: Dict[str, Dict[str, float]] = {}
        phases: List[SignalPhase] = []
        total_cycle_time = 0.0

        phase_lengths = {
            lane: green_times[lane] + orange_time
            for lane in lane_order
        }
        total_cycle_time = sum(phase_lengths.values())

        for index, lane in enumerate(lane_order):
            red_before = sum(phase_lengths[prior_lane] for prior_lane in lane_order[:index])
            red_after = sum(phase_lengths[next_lane] for next_lane in lane_order[index + 1 :])
            lane_timings[lane] = {
                "green_time": green_times[lane],
                "orange_time": orange_time,
                "red_time": red_before + red_after,
                "red_before": red_before,
                "red_after": red_after,
            }
            phases.append(SignalPhase(lane=lane, state="green", duration=green_times[lane]))
            phases.append(SignalPhase(lane=lane, state="orange", duration=orange_time))

        return SignalPlan(
            total_cycle_time=total_cycle_time,
            current_lane_order=lane_order,
            lane_timings=lane_timings,
            phases=phases,
            lane_metrics=lane_metrics,
        )

    def active_green_lane(
        self,
        signal_plan: SignalPlan,
        elapsed_sec: float,
    ) -> Optional[str]:
        if signal_plan.emergency_override_lane is not None:
            return signal_plan.emergency_override_lane

        cycle_time = signal_plan.total_cycle_time
        if cycle_time <= 0.0:
            return None
        position = elapsed_sec % cycle_time
        cursor = 0.0
        for phase in signal_plan.phases:
            cursor += phase.duration
            if position < cursor:
                if phase.state == "green":
                    return phase.lane
                return None
        return None

    def _compute_lane_metrics(
        self,
        snapshots: List[TrackSnapshot],
        lane_order: List[str],
    ) -> Dict[str, LaneMetrics]:
        lane_metrics: Dict[str, LaneMetrics] = {}
        for lane in lane_order:
            lane_snapshots = [snapshot for snapshot in snapshots if snapshot.lane == lane]
            queue_span_pixels = lane_queue_span_pixels(
                (snapshot.bbox for snapshot in lane_snapshots),
                lane,
            )
            lane_metrics[lane] = LaneMetrics(
                lane=lane,
                stopped_count=sum(1 for snapshot in lane_snapshots if snapshot.is_halting),
                total_wait_time=sum(
                    snapshot.wait_time
                    for snapshot in lane_snapshots
                    if snapshot.is_halting
                ),
                emergency_present=any(
                    snapshot.is_emergency for snapshot in lane_snapshots
                ),
                queue_span_pixels=queue_span_pixels,
                queue_span_meters=self.pixel_to_meter(queue_span_pixels),
            )
        return lane_metrics

    def _choose_emergency_lane(
        self,
        lane_metrics: Dict[str, LaneMetrics],
        lane_order: List[str],
    ) -> Optional[str]:
        emergency_lanes = [
            lane
            for lane in lane_order
            if lane_metrics[lane].emergency_present
        ]
        if not emergency_lanes:
            return None
        return max(
            emergency_lanes,
            key=lambda lane: (
                lane_metrics[lane].stopped_count,
                lane_metrics[lane].queue_span_meters,
            ),
        )

    def _build_emergency_plan(
        self,
        lane_metrics: Dict[str, LaneMetrics],
        lane_order: List[str],
        emergency_lane: str,
    ) -> SignalPlan:
        total_cycle_time = self.signal_config.total_cycle_time
        lane_timings = {
            lane: {
                "green_time": total_cycle_time if lane == emergency_lane else 0.0,
                "orange_time": 0.0,
                "red_time": 0.0 if lane == emergency_lane else total_cycle_time,
                "red_before": 0.0,
                "red_after": 0.0 if lane == emergency_lane else total_cycle_time,
            }
            for lane in lane_order
        }
        phases = [
            SignalPhase(
                lane=emergency_lane,
                state="green",
                duration=total_cycle_time,
            )
        ]
        return SignalPlan(
            total_cycle_time=total_cycle_time,
            current_lane_order=lane_order,
            lane_timings=lane_timings,
            phases=phases,
            lane_metrics=lane_metrics,
            emergency_override_lane=emergency_lane,
        )

    def _allocate_green_times(
        self,
        lane_metrics: Dict[str, LaneMetrics],
        lane_order: List[str],
    ) -> Dict[str, float]:
        lane_count = len(lane_order)
        orange_budget = self.signal_config.orange_time * lane_count
        total_green_budget = max(
            0.0, self.signal_config.total_cycle_time - orange_budget
        )
        minimum_green = min(
            self.signal_config.minimum_green_time,
            total_green_budget / lane_count if lane_count else 0.0,
        )
        minimum_budget = minimum_green * lane_count
        variable_budget = max(0.0, total_green_budget - minimum_budget)
        total_length = sum(
            lane_metrics[lane].queue_span_meters
            for lane in lane_order
        )
        if total_length <= 0.0:
            ratios = {lane: 1.0 / lane_count for lane in lane_order}
        else:
            ratios = {
                lane: lane_metrics[lane].queue_span_meters / total_length
                for lane in lane_order
            }
        green_times = {
            lane: minimum_green + (ratios[lane] * variable_budget)
            for lane in lane_order
        }
        return green_times
