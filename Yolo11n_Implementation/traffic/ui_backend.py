from __future__ import annotations

import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import cv2
from werkzeug.datastructures import FileStorage

from scripts.detect_traffic import render_annotations
from traffic_control import RLTrafficController, load_rl_config
from traffic_pipeline import TrafficPipeline, load_config as load_pipeline_config
from traffic_pipeline.detector import YoloTrafficDetector

PROJECT_ROOT = Path(__file__).resolve().parent
JOB_ROOT = PROJECT_ROOT / "outputs" / "ui_jobs"
ROAD_CONFIG = [
    {"road": "North", "lane": "N", "upload_key": "north_feed", "slug": "north"},
    {"road": "East", "lane": "E", "upload_key": "east_feed", "slug": "east"},
    {"road": "South", "lane": "S", "upload_key": "south_feed", "slug": "south"},
    {"road": "West", "lane": "W", "upload_key": "west_feed", "slug": "west"},
]
ZERO_RAW_VECTOR = [0.0] * 16
DEFAULT_FEED_WORKERS = max(1, int(os.environ.get("UI_FEED_WORKERS", "4")))
LIVE_PROCESS_FPS = max(1.0, float(os.environ.get("UI_PROCESS_FPS", "5")))
LIVE_EMIT_INTERVAL_SEC = max(0.25, float(os.environ.get("UI_EMIT_INTERVAL_SEC", "1")))
ANNOTATED_VIDEO_OPTIONS = [
    ("annotated.webm", "VP80"),
    ("annotated.mp4", "mp4v"),
]

JOB_EXECUTOR = ThreadPoolExecutor(max_workers=1)
JOB_LOCK = Lock()
JOB_STATE: dict[str, dict[str, Any]] = {}


def media_url(path: Path, *, cache_token: Optional[int] = None) -> str:
    relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    url = f"/media/{relative}"
    if cache_token is not None:
        url += f"?t={cache_token}"
    return url


def create_job_state(job_id: str, status: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "jobId": job_id,
        "status": status,
        "updatedAt": time.time(),
    }
    payload.update(extra)
    return payload


def update_job(job_id: str, **payload: Any) -> None:
    with JOB_LOCK:
        state = JOB_STATE.get(job_id, {})
        state.update(payload)
        state["updatedAt"] = time.time()
        JOB_STATE[job_id] = state


def get_job_state(job_id: str) -> Optional[dict[str, Any]]:
    with JOB_LOCK:
        state = JOB_STATE.get(job_id)
        return None if state is None else dict(state)


def save_uploads(files: dict[str, FileStorage], input_dir: Path) -> dict[str, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for road in ROAD_CONFIG:
        storage = files.get(road["upload_key"])
        if storage is None or not storage.filename:
            raise ValueError(f"Missing upload for {road['road']}")
        suffix = Path(storage.filename).suffix.lower() or ".mp4"
        destination = input_dir / f"{road['slug']}{suffix}"
        storage.save(destination)
        saved[road["road"]] = destination
    return saved


def _full_frame_polygon() -> list[tuple[float, float]]:
    return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _lane_config(base_config, lane: str):
    config = deepcopy(base_config)
    config.lane_polygons = {lane: _full_frame_polygon()}
    config.tracking.require_lane_polygon_match = True
    return config


def _lane_pipeline(
    config_path: str,
    lane: str,
    *,
    base_config=None,
    shared_detector: Optional[YoloTrafficDetector] = None,
) -> TrafficPipeline:
    config = _lane_config(base_config or load_pipeline_config(config_path), lane)
    return TrafficPipeline(config, detector=shared_detector)


def _should_process_frame(
    frame_index: int,
    timestamp_sec: float,
    frame_stride: int,
    process_fps: float,
    next_process_timestamp: Optional[float],
) -> tuple[bool, Optional[float]]:
    if process_fps > 0.0:
        scheduled_timestamp = 0.0 if next_process_timestamp is None else next_process_timestamp
        if timestamp_sec + 1e-9 < scheduled_timestamp:
            return False, scheduled_timestamp
        return True, scheduled_timestamp + (1.0 / process_fps)
    return frame_index % max(frame_stride, 1) == 0, next_process_timestamp


def _should_emit_state(
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


def _raw_bridge_without_phase(bridge_vector: list[float]) -> list[float]:
    return list(bridge_vector[:8]) + [0.0, 0.0, 0.0, 0.0] + list(bridge_vector[12:16])


def _reduce_feed_window(raw_vectors: list[list[float]]) -> list[float]:
    if not raw_vectors:
        return list(ZERO_RAW_VECTOR)

    sample_count = float(len(raw_vectors))
    reduced = [0.0] * 16
    for index in range(4):
        reduced[index] = round(sum(vector[index] for vector in raw_vectors) / sample_count, 4)
        reduced[index + 4] = round(sum(vector[index + 4] for vector in raw_vectors) / sample_count, 4)
        reduced[index + 12] = round(max(vector[index + 12] for vector in raw_vectors), 4)
    return reduced


def _combine_feed_vectors(vectors: list[list[float]]) -> list[float]:
    combined = [0.0] * 16
    for vector in vectors:
        for index in range(8):
            combined[index] += float(vector[index])
        for index in range(12, 16):
            combined[index] = max(combined[index], float(vector[index]))
    return [round(value, 4) for value in combined]


def _draw_road_banner(frame, road_name: str) -> None:
    cv2.rectangle(frame, (10, 10), (220, 48), (15, 20, 28), -1)
    cv2.rectangle(frame, (10, 10), (220, 48), (240, 240, 240), 1)
    cv2.putText(
        frame,
        road_name,
        (22, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )


def _build_feed_entries(feed_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    feeds = []
    for road in ROAD_CONFIG:
        summary = feed_summaries.get(road["road"])
        if summary is None:
            continue
        video_path = Path(summary["video"])
        annotated_path = Path(summary["annotated_video"])
        feeds.append(
            {
                "road": road["road"],
                "lane": road["lane"],
                "sourceVideo": media_url(video_path),
                "annotatedVideo": media_url(annotated_path),
            }
        )
    return feeds


def _open_annotated_writer(
    *,
    output_dir: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, Path]:
    last_error: Optional[str] = None
    width, height = frame_size
    for filename, fourcc_name in ANNOTATED_VIDEO_OPTIONS:
        annotated_path = output_dir / filename
        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            max(fps, 1.0),
            (width, height),
        )
        if writer.isOpened():
            return writer, annotated_path
        writer.release()
        last_error = f"{fourcc_name} -> {annotated_path.name}"

    raise RuntimeError(
        "Could not open annotated video writer. "
        f"Tried: {', '.join(f'{fourcc} ({name})' for name, fourcc in ANNOTATED_VIDEO_OPTIONS)}. "
        f"Last attempt: {last_error}"
    )


def _save_preview_image(path: Path, frame) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise RuntimeError(f"Could not write preview image: {path}")
    return path.stat().st_mtime_ns


def _estimate_video_duration(capture: cv2.VideoCapture, fps: float) -> float:
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps > 0.0 and frame_count > 0.0:
        return max(frame_count / fps, 0.0)
    return 0.0


def _read_frame_at_time(
    capture: cv2.VideoCapture,
    timestamp_sec: float,
) -> tuple[bool, Optional[object]]:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(timestamp_sec, 0.0) * 1000.0)
    ok, frame = capture.read()
    return ok, frame if ok else None


def _build_progress_feed_entries(feed_states: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    feeds = []
    for road in ROAD_CONFIG:
        state = feed_states.get(road["road"])
        if state is None:
            continue
        entry = {
            "road": road["road"],
            "lane": road["lane"],
            "status": "completed" if state.get("done") else "processing",
            "sourceVideo": media_url(state["video_path"]),
        }
        preview_path = state.get("preview_path")
        preview_token = state.get("preview_token")
        if preview_path is not None and Path(preview_path).exists():
            entry["previewImage"] = media_url(Path(preview_path), cache_token=preview_token)
        annotated_path = state.get("annotated_path")
        if annotated_path is not None and Path(annotated_path).exists():
            entry["annotatedVideo"] = media_url(Path(annotated_path))
        feeds.append(entry)
    return feeds


def process_feed_video(
    *,
    video_path: Path,
    road_name: str,
    lane: str,
    config_path: str,
    output_dir: Path,
    process_fps: float = 10.0,
    emit_interval_sec: float = 1.0,
    frame_stride: int = 1,
    max_emits: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _lane_pipeline(config_path, lane)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    annotated_path: Optional[Path] = None
    writer = None
    frame_index = -1
    next_process_timestamp: Optional[float] = None
    next_emit_timestamp: Optional[float] = None
    raw_window: list[list[float]] = []
    second_records: dict[int, dict[str, Any]] = {}
    latest_result: Optional[dict[str, Any]] = None
    emitted = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            timestamp_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp_sec <= 0.0:
                timestamp_sec = frame_index / fps

            should_process, next_process_timestamp = _should_process_frame(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                frame_stride=frame_stride,
                process_fps=process_fps,
                next_process_timestamp=next_process_timestamp,
            )
            if not should_process:
                continue

            latest_result = pipeline.process_frame(frame, timestamp_sec=timestamp_sec)
            raw_window.append(_raw_bridge_without_phase(latest_result["bridge_vector"]))

            annotated = render_annotations(frame, dict(latest_result), pipeline)
            _draw_road_banner(annotated, road_name)
            if writer is None:
                writer, annotated_path = _open_annotated_writer(
                    output_dir=output_dir,
                    fps=max(process_fps if process_fps > 0 else fps / max(frame_stride, 1), 1.0),
                    frame_size=(annotated.shape[1], annotated.shape[0]),
                )
            writer.write(annotated)

            should_emit, next_emit_timestamp = _should_emit_state(
                timestamp_sec=timestamp_sec,
                emit_interval_sec=emit_interval_sec,
                next_emit_timestamp=next_emit_timestamp,
            )
            if should_emit:
                second = int(round(timestamp_sec))
                raw_vector = _reduce_feed_window(raw_window)
                raw_window = []
                lane_index = "NESW".index(lane)
                vehicles = latest_result["vehicles"] if latest_result is not None else []
                second_records[second] = {
                    "second": second,
                    "raw_vector": raw_vector,
                    "visible_vehicle_count": len(vehicles),
                    "stopped_count": raw_vector[lane_index],
                    "wait": raw_vector[lane_index + 4],
                    "emergency_present": raw_vector[lane_index + 12] > 0.5,
                    "vehicles": vehicles,
                }
                emitted += 1
                if max_emits and emitted >= max_emits:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    summary = {
        "road": road_name,
        "lane": lane,
        "video": str(video_path.resolve()),
        "annotated_video": str((annotated_path or (output_dir / "annotated.mp4")).resolve()),
        "emitted_states": emitted,
        "seconds": second_records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _empty_feed_record(road_name: str, lane: str) -> dict[str, Any]:
    lane_index = "NESW".index(lane)
    raw_vector = [0.0] * 16
    return {
        "second": 0,
        "raw_vector": raw_vector,
        "visible_vehicle_count": 0,
        "stopped_count": raw_vector[lane_index],
        "wait": raw_vector[lane_index + 4],
        "emergency_present": False,
        "vehicles": [],
    }


def _record_from_result(
    *,
    road_name: str,
    lane: str,
    second: int,
    raw_vector: list[float],
    result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    lane_index = "NESW".index(lane)
    vehicles = [] if result is None else result.get("vehicles", [])
    return {
        "second": second,
        "raw_vector": raw_vector,
        "visible_vehicle_count": len(vehicles),
        "stopped_count": raw_vector[lane_index],
        "wait": raw_vector[lane_index + 4],
        "emergency_present": raw_vector[lane_index + 12] > 0.5,
        "vehicles": vehicles,
    }


def _build_timeline_entry(
    second: int,
    feed_records: dict[str, dict[str, Any]],
    controller: RLTrafficController,
) -> dict[str, Any]:
    vectors = []
    road_payload: dict[str, Any] = {}
    for road in ROAD_CONFIG:
        feed_record = feed_records.get(
            road["road"],
            _empty_feed_record(road["road"], road["lane"]),
        )
        vectors.append(feed_record["raw_vector"])
        road_payload[road["road"]] = {
            "stoppedCount": round(float(feed_record["stopped_count"]), 2),
            "visibleVehicleCount": int(feed_record["visible_vehicle_count"]),
            "wait": round(float(feed_record["wait"]), 2),
            "emergency": bool(feed_record["emergency_present"]),
        }

    combined_raw = _combine_feed_vectors(vectors)
    decision = controller.decide([combined_raw])
    for road in ROAD_CONFIG:
        road_payload[road["road"]]["signal"] = decision.lane_signals[road["lane"]]

    return {
        "second": second,
        "globalSignal": decision.action_name,
        "decisionSource": decision.decision_source,
        "qValues": decision.q_values,
        "lights": {
            "North": decision.lane_signals["N"],
            "East": decision.lane_signals["E"],
            "South": decision.lane_signals["S"],
            "West": decision.lane_signals["W"],
        },
        "roads": road_payload,
    }


def build_timeline(
    feed_summaries: dict[str, dict[str, Any]],
    *,
    config_path: str,
) -> list[dict[str, Any]]:
    controller = RLTrafficController(load_rl_config(config_path))
    max_second = 0
    for road in ROAD_CONFIG:
        for second in feed_summaries[road["road"]]["seconds"].keys():
            max_second = max(max_second, int(second))

    if max_second == 0:
        return []

    timeline = []
    for second in range(1, max_second + 1):
        feed_records = {}
        for road in ROAD_CONFIG:
            feed_records[road["road"]] = feed_summaries[road["road"]]["seconds"].get(
                second,
                _empty_feed_record(road["road"], road["lane"]),
            )
        timeline.append(_build_timeline_entry(second, feed_records, controller))
    return timeline


def build_manifest(job_dir: Path, feed_summaries: dict[str, dict[str, Any]], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "title": "Four-Feed Traffic RL Interface",
        "description": "Upload North, East, South, and West road videos, then view detector outputs and the combined RL light decision timeline.",
        "feeds": _build_feed_entries(feed_summaries),
        "timeline": timeline,
        "maxSecond": 0 if not timeline else timeline[-1]["second"],
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _resolve_feed_inputs(input_dir: Path) -> list[tuple[dict[str, str], Path]]:
    resolved: list[tuple[dict[str, str], Path]] = []
    for road in ROAD_CONFIG:
        input_files = sorted(input_dir.glob(f"{road['slug']}.*"))
        if not input_files:
            raise ValueError(f"Missing saved input for {road['road']}")
        resolved.append((road, input_files[0]))
    return resolved


def _process_single_feed(
    *,
    road: dict[str, str],
    video_path: Path,
    output_dir: Path,
    config_path: str,
    max_emits: int,
) -> tuple[str, dict[str, Any]]:
    summary = process_feed_video(
        video_path=video_path,
        road_name=road["road"],
        lane=road["lane"],
        config_path=config_path,
        output_dir=output_dir / road["slug"],
        max_emits=max_emits,
    )
    return road["road"], summary


def run_job(
    *,
    job_id: str,
    input_dir: Path,
    output_dir: Path,
    config_path: str,
    max_emits: int = 0,
) -> dict[str, Any]:
    feed_inputs = _resolve_feed_inputs(input_dir)
    base_config = load_pipeline_config(config_path)
    shared_detector = YoloTrafficDetector(
        base_config.detector,
        base_config.vehicle_keywords,
    )
    controller = RLTrafficController(load_rl_config(config_path))
    feed_states: dict[str, dict[str, Any]] = {}
    max_duration_sec = 0.0

    for road, video_path in feed_inputs:
        lane_output_dir = output_dir / road["slug"]
        lane_output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = _estimate_video_duration(capture, fps)
        max_duration_sec = max(max_duration_sec, duration_sec)
        feed_states[road["road"]] = {
            "road": road["road"],
            "lane": road["lane"],
            "slug": road["slug"],
            "video_path": video_path.resolve(),
            "output_dir": lane_output_dir,
            "preview_path": lane_output_dir / "preview.jpg",
            "preview_token": None,
            "capture": capture,
            "fps": fps,
            "duration_sec": duration_sec,
            "pipeline": _lane_pipeline(
                config_path,
                road["lane"],
                base_config=base_config,
                shared_detector=shared_detector,
            ),
            "writer": None,
            "annotated_path": None,
            "latest_result": None,
            "latest_raw_vector": list(ZERO_RAW_VECTOR),
            "latest_annotated_frame": None,
            "second_records": {},
            "done": False,
        }

    total_seconds = max(1, int(math.ceil(max_duration_sec))) if max_duration_sec > 0.0 else 1

    update_job(
        job_id,
        status="processing",
        message="Processing four feeds together in synchronized live mode...",
        progress={"completed": 0, "total": total_seconds},
        previewFeeds=_build_progress_feed_entries(feed_states),
    )

    timeline: list[dict[str, Any]] = []
    next_emit_timestamp: Optional[float] = None
    emit_count = 0
    current_time_sec = 0.0
    time_step_sec = 1.0 / LIVE_PROCESS_FPS

    try:
        while True:
            active_batch: list[tuple[dict[str, Any], Any]] = []
            frames = []
            for road in ROAD_CONFIG:
                state = feed_states[road["road"]]
                if state["done"]:
                    continue
                if state["duration_sec"] > 0.0 and current_time_sec > state["duration_sec"] + 1e-6:
                    state["done"] = True
                    continue

                ok, frame = _read_frame_at_time(state["capture"], current_time_sec)
                if not ok or frame is None:
                    state["done"] = True
                    continue
                active_batch.append((state, frame))
                frames.append(frame)

            if not active_batch and all(state["done"] for state in feed_states.values()):
                break

            if active_batch:
                detections_by_frame = shared_detector.detect_many(frames)
                for (state, frame), detections in zip(active_batch, detections_by_frame):
                    result = state["pipeline"].process_detections(
                        detections=detections,
                        frame_shape=frame.shape,
                        timestamp_sec=current_time_sec,
                    )
                    annotated = render_annotations(frame, dict(result), state["pipeline"])
                    _draw_road_banner(annotated, state["road"])
                    if state["writer"] is None:
                        writer, annotated_path = _open_annotated_writer(
                            output_dir=state["output_dir"],
                            fps=LIVE_PROCESS_FPS,
                            frame_size=(annotated.shape[1], annotated.shape[0]),
                        )
                        state["writer"] = writer
                        state["annotated_path"] = annotated_path
                    state["writer"].write(annotated)
                    state["latest_result"] = result
                    state["latest_raw_vector"] = _raw_bridge_without_phase(result["bridge_vector"])
                    state["latest_annotated_frame"] = annotated

            should_emit, next_emit_timestamp = _should_emit_state(
                timestamp_sec=current_time_sec,
                emit_interval_sec=LIVE_EMIT_INTERVAL_SEC,
                next_emit_timestamp=next_emit_timestamp,
            )
            if should_emit:
                second = int(round(current_time_sec))
                feed_records: dict[str, dict[str, Any]] = {}
                for road in ROAD_CONFIG:
                    state = feed_states[road["road"]]
                    if state["latest_annotated_frame"] is not None:
                        state["preview_token"] = _save_preview_image(
                            state["preview_path"],
                            state["latest_annotated_frame"],
                        )
                    feed_records[road["road"]] = _record_from_result(
                        road_name=road["road"],
                        lane=road["lane"],
                        second=second,
                        raw_vector=list(state["latest_raw_vector"]) if state["latest_result"] is not None else list(ZERO_RAW_VECTOR),
                        result=state["latest_result"],
                    )
                    state["second_records"][second] = feed_records[road["road"]]

                timeline_entry = _build_timeline_entry(second, feed_records, controller)
                timeline.append(timeline_entry)
                emit_count += 1
                update_job(
                    job_id,
                    status="processing",
                    message=f"Processing synchronized second {min(second, total_seconds)}/{total_seconds}...",
                    progress={"completed": min(second, total_seconds), "total": total_seconds},
                    previewFeeds=_build_progress_feed_entries(feed_states),
                    currentEntry=timeline_entry,
                )
                if max_emits and emit_count >= max_emits:
                    break

            current_time_sec += time_step_sec
            if max_duration_sec > 0.0 and current_time_sec > max_duration_sec + LIVE_EMIT_INTERVAL_SEC and all(
                state["done"] for state in feed_states.values()
            ):
                break

        if not timeline and any(state["latest_result"] is not None for state in feed_states.values()):
            second = 1
            feed_records = {}
            for road in ROAD_CONFIG:
                state = feed_states[road["road"]]
                if state["latest_annotated_frame"] is not None:
                    state["preview_token"] = _save_preview_image(
                        state["preview_path"],
                        state["latest_annotated_frame"],
                    )
                feed_records[road["road"]] = _record_from_result(
                    road_name=road["road"],
                    lane=road["lane"],
                    second=second,
                    raw_vector=list(state["latest_raw_vector"]) if state["latest_result"] is not None else list(ZERO_RAW_VECTOR),
                    result=state["latest_result"],
                )
                state["second_records"][second] = feed_records[road["road"]]
            timeline.append(_build_timeline_entry(second, feed_records, controller))

        feed_summaries: dict[str, dict[str, Any]] = {}
        for road in ROAD_CONFIG:
            state = feed_states[road["road"]]
            summary = {
                "road": road["road"],
                "lane": road["lane"],
                "video": str(state["video_path"]),
                "annotated_video": str((state["annotated_path"] or (state["output_dir"] / "annotated.webm")).resolve()),
                "emitted_states": len(state["second_records"]),
                "seconds": state["second_records"],
            }
            (state["output_dir"] / "summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            feed_summaries[road["road"]] = summary

        return build_manifest(output_dir, feed_summaries, timeline)
    finally:
        for state in feed_states.values():
            capture = state.get("capture")
            if capture is not None:
                capture.release()
            writer = state.get("writer")
            if writer is not None:
                writer.release()


def _execute_job(job_id: str, input_dir: Path, output_dir: Path, config_path: str) -> None:
    try:
        update_job(job_id, status="processing", message="Running vehicle detection and RL timeline...")
        manifest = run_job(
            job_id=job_id,
            input_dir=input_dir,
            output_dir=output_dir,
            config_path=config_path,
        )
        update_job(job_id, status="completed", message="Processing complete.", manifest=manifest)
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc), message="Processing failed.")


def submit_job(files: dict[str, FileStorage], config_path: Optional[str] = None) -> str:
    job_id = uuid.uuid4().hex[:10]
    job_dir = JOB_ROOT / job_id
    input_dir = job_dir / "inputs"
    output_dir = job_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_uploads(files, input_dir)
    config_path = config_path or str(PROJECT_ROOT / "configs" / "runtime.yaml")

    with JOB_LOCK:
        JOB_STATE[job_id] = create_job_state(
            job_id,
            "queued",
            message="Upload complete. Waiting for processing worker.",
        )

    JOB_EXECUTOR.submit(_execute_job, job_id, input_dir, output_dir, config_path)
    return job_id
