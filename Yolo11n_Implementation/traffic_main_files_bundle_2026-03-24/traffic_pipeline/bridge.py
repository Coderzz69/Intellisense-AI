from __future__ import annotations

from typing import Iterable, List, Optional

from .models import LANE_ORDER, TrackSnapshot


def build_bridge_vector(
    snapshots: Iterable[TrackSnapshot],
    current_green_lane: Optional[str],
    lane_order: Iterable[str] = LANE_ORDER,
) -> List[float]:
    lane_names = list(lane_order)
    items = list(snapshots)

    stopped_counts = [
        float(
            sum(
                1
                for snapshot in items
                if snapshot.lane == lane and snapshot.is_halting
            )
        )
        for lane in lane_names
    ]
    wait_times = [
        round(
            sum(
                snapshot.wait_time
                for snapshot in items
                if snapshot.lane == lane and snapshot.is_halting
            ),
            2,
        )
        for lane in lane_names
    ]
    green_one_hot = [
        1.0 if current_green_lane == lane else 0.0
        for lane in lane_names
    ]
    emergency_presence = [
        1.0
        if any(
            snapshot.lane == lane and snapshot.is_emergency
            for snapshot in items
        )
        else 0.0
        for lane in lane_names
    ]
    return stopped_counts + wait_times + green_one_hot + emergency_presence
