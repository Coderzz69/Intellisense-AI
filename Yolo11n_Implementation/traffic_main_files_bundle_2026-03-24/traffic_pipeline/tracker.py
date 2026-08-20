from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import TrackingConfig
from .geometry import assign_lane
from .models import Detection, TrackSnapshot


def _is_emergency(class_name: str, emergency_keywords: Iterable[str]) -> bool:
    lowered = class_name.lower()
    return any(keyword.lower() in lowered for keyword in emergency_keywords)


@dataclass
class _TrackState:
    track_id: int
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    class_name: str
    confidence: float
    lane: str
    last_seen_timestamp: float
    stationary_since: Optional[float] = None
    missed_frames: int = 0


class SimpleTracker:
    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._tracks: Dict[int, _TrackState] = {}
        self._next_track_id = config.starting_track_id

    def update(
        self,
        detections: Sequence[Detection],
        frame_shape: Tuple[int, int, int],
        timestamp_sec: float,
        lane_polygons: Dict[str, List[Tuple[float, float]]],
        emergency_keywords: Iterable[str],
    ) -> List[TrackSnapshot]:
        filtered_detections, detection_lanes = self._assign_lanes(
            detections=detections,
            frame_shape=frame_shape,
            lane_polygons=lane_polygons,
        )
        matches = self._match_detections(filtered_detections)
        matched_track_ids = {track_id for track_id, _ in matches}
        matched_detection_indices = {detection_index for _, detection_index in matches}
        snapshots: List[TrackSnapshot] = []

        for track_id, detection_index in matches:
            detection = filtered_detections[detection_index]
            track = self._tracks[track_id]
            lane = detection_lanes[detection_index]
            displacement = dist(track.centroid, detection.center)
            is_halting = displacement <= self.config.halt_distance_px
            if is_halting:
                stationary_since = track.stationary_since
                if stationary_since is None:
                    stationary_since = timestamp_sec
            else:
                stationary_since = None

            updated_track = _TrackState(
                track_id=track.track_id,
                bbox=detection.bbox,
                centroid=detection.center,
                class_name=detection.class_name,
                confidence=detection.confidence,
                lane=lane,
                last_seen_timestamp=timestamp_sec,
                stationary_since=stationary_since,
                missed_frames=0,
            )
            self._tracks[track_id] = updated_track
            wait_time = 0.0 if stationary_since is None else max(
                0.0, timestamp_sec - stationary_since
            )
            snapshots.append(
                TrackSnapshot(
                    track_id=track_id,
                    lane=lane,
                    is_halting=is_halting,
                    wait_time=wait_time,
                    is_emergency=_is_emergency(detection.class_name, emergency_keywords),
                    bbox=detection.bbox,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                )
            )

        for detection_index, detection in enumerate(filtered_detections):
            if detection_index in matched_detection_indices:
                continue
            lane = detection_lanes[detection_index]
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                centroid=detection.center,
                class_name=detection.class_name,
                confidence=detection.confidence,
                lane=lane,
                last_seen_timestamp=timestamp_sec,
            )
            snapshots.append(
                TrackSnapshot(
                    track_id=track_id,
                    lane=lane,
                    is_halting=False,
                    wait_time=0.0,
                    is_emergency=_is_emergency(detection.class_name, emergency_keywords),
                    bbox=detection.bbox,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                )
            )

        to_remove: List[int] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_track_ids or track_id in {
                snapshot.track_id for snapshot in snapshots
            }:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.config.max_missed_frames:
                to_remove.append(track_id)
        for track_id in to_remove:
            del self._tracks[track_id]

        snapshots.sort(key=lambda snapshot: snapshot.track_id)
        return snapshots

    def _assign_lanes(
        self,
        detections: Sequence[Detection],
        frame_shape: Tuple[int, int, int],
        lane_polygons: Dict[str, List[Tuple[float, float]]],
    ) -> Tuple[List[Detection], List[str]]:
        filtered_detections: List[Detection] = []
        lanes: List[str] = []
        for detection in detections:
            lane = assign_lane(
                detection.bbox,
                frame_shape,
                lane_polygons,
                anchor=self.config.lane_assignment_anchor,
                require_lane_polygon_match=self.config.require_lane_polygon_match,
            )
            if lane is None:
                continue
            filtered_detections.append(detection)
            lanes.append(lane)
        return filtered_detections, lanes

    def _match_detections(
        self, detections: Sequence[Detection]
    ) -> List[Tuple[int, int]]:
        candidates: List[Tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                if detection.class_name != track.class_name:
                    continue
                distance = dist(track.centroid, detection.center)
                if distance <= self.config.match_distance_px:
                    candidates.append((distance, track_id, detection_index))

        candidates.sort(key=lambda item: item[0])
        matched_track_ids = set()
        matched_detection_indices = set()
        matches: List[Tuple[int, int]] = []

        for _, track_id, detection_index in candidates:
            if track_id in matched_track_ids or detection_index in matched_detection_indices:
                continue
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)
            matches.append((track_id, detection_index))
        return matches
