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
    halt_reference_centroid: Tuple[float, float]
    halt_candidate_since: Optional[float] = None
    stationary_since: Optional[float] = None
    emergency_until_timestamp: float = -1.0
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
        snapshot_track_ids = set()

        for track_id, detection_index in matches:
            detection = filtered_detections[detection_index]
            track = self._tracks[track_id]
            lane = detection_lanes[detection_index]
            updated_track = self._build_matched_track(
                track=track,
                detection=detection,
                lane=lane,
                timestamp_sec=timestamp_sec,
                emergency_keywords=emergency_keywords,
            )
            self._tracks[track_id] = updated_track
            snapshots.append(self._snapshot_from_track(updated_track, timestamp_sec))
            snapshot_track_ids.add(track_id)

        for detection_index, detection in enumerate(filtered_detections):
            if detection_index in matched_detection_indices:
                continue
            lane = detection_lanes[detection_index]
            track_id = self._next_track_id
            self._next_track_id += 1
            detection_is_emergency = _is_emergency(detection.class_name, emergency_keywords)
            track = _TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                centroid=detection.center,
                class_name=detection.class_name,
                confidence=detection.confidence,
                lane=lane,
                last_seen_timestamp=timestamp_sec,
                halt_reference_centroid=detection.center,
                emergency_until_timestamp=(
                    timestamp_sec + self.config.emergency_memory_sec
                    if detection_is_emergency
                    else -1.0
                ),
            )
            self._tracks[track_id] = track
            snapshots.append(self._snapshot_from_track(track, timestamp_sec))
            snapshot_track_ids.add(track_id)

        to_remove: List[int] = []
        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids or track_id in snapshot_track_ids:
                continue
            track.missed_frames += 1
            if self._should_keep_track(track, timestamp_sec):
                snapshots.append(self._snapshot_from_track(track, timestamp_sec))
                snapshot_track_ids.add(track_id)
                continue
            to_remove.append(track_id)
        for track_id in to_remove:
            del self._tracks[track_id]

        snapshots.sort(key=lambda snapshot: snapshot.track_id)
        return snapshots

    def _build_matched_track(
        self,
        track: _TrackState,
        detection: Detection,
        lane: str,
        timestamp_sec: float,
        emergency_keywords: Iterable[str],
    ) -> _TrackState:
        halt_reference_centroid, halt_candidate_since, stationary_since = (
            self._advance_halt_state(
                track=track,
                detection_center=detection.center,
                timestamp_sec=timestamp_sec,
            )
        )
        emergency_until_timestamp = track.emergency_until_timestamp
        if _is_emergency(detection.class_name, emergency_keywords):
            emergency_until_timestamp = timestamp_sec + self.config.emergency_memory_sec

        return _TrackState(
            track_id=track.track_id,
            bbox=detection.bbox,
            centroid=detection.center,
            class_name=detection.class_name,
            confidence=detection.confidence,
            lane=lane,
            last_seen_timestamp=timestamp_sec,
            halt_reference_centroid=halt_reference_centroid,
            halt_candidate_since=halt_candidate_since,
            stationary_since=stationary_since,
            emergency_until_timestamp=emergency_until_timestamp,
            missed_frames=0,
        )

    def _advance_halt_state(
        self,
        track: _TrackState,
        detection_center: Tuple[float, float],
        timestamp_sec: float,
    ) -> Tuple[Tuple[float, float], Optional[float], Optional[float]]:
        reference = track.halt_reference_centroid
        distance_from_reference = dist(reference, detection_center)

        if track.stationary_since is not None:
            if distance_from_reference <= self.config.halt_resume_distance_px:
                return reference, track.halt_candidate_since, track.stationary_since
            return detection_center, None, None

        if distance_from_reference <= self.config.halt_distance_px:
            candidate_since = track.halt_candidate_since
            if candidate_since is None:
                candidate_since = track.last_seen_timestamp
            if (timestamp_sec - candidate_since) >= self.config.halt_confirm_sec:
                return reference, candidate_since, candidate_since
            return reference, candidate_since, None

        return detection_center, None, None

    def _snapshot_from_track(
        self,
        track: _TrackState,
        timestamp_sec: float,
    ) -> TrackSnapshot:
        wait_time = 0.0 if track.stationary_since is None else max(
            0.0, timestamp_sec - track.stationary_since
        )
        return TrackSnapshot(
            track_id=track.track_id,
            lane=track.lane,
            is_halting=track.stationary_since is not None,
            wait_time=wait_time,
            is_emergency=self._track_is_emergency(track, timestamp_sec),
            bbox=track.bbox,
            class_name=track.class_name,
            confidence=track.confidence,
        )

    def _should_keep_track(
        self,
        track: _TrackState,
        timestamp_sec: float,
    ) -> bool:
        missed_duration = max(0.0, timestamp_sec - track.last_seen_timestamp)
        if (
            track.missed_frames <= self.config.max_missed_frames
            or missed_duration <= self.config.missed_track_memory_sec
        ):
            return True
        if (
            track.stationary_since is not None
            and missed_duration <= self.config.halted_track_memory_sec
        ):
            return True
        return (
            self._track_is_emergency(track, timestamp_sec)
            and missed_duration <= self.config.emergency_memory_sec
        )

    def _track_is_emergency(
        self,
        track: _TrackState,
        timestamp_sec: float,
    ) -> bool:
        return timestamp_sec <= track.emergency_until_timestamp + 1e-9

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
                distance = dist(track.centroid, detection.center)
                if distance <= self.config.match_distance_px:
                    penalty = 0.0
                    if detection.class_name != track.class_name:
                        penalty = self.config.class_mismatch_distance_penalty_px
                    candidates.append(
                        (distance + penalty, track_id, detection_index)
                    )

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
