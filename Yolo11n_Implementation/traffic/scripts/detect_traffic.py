#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_pipeline import TrafficPipeline, load_config
from traffic_pipeline.geometry import denormalize_polygon

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW_NAME = "traffic-detection"
LANE_COLORS = {
    "N": (255, 200, 0),
    "E": (0, 200, 255),
    "S": (0, 200, 120),
    "W": (255, 120, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run traffic detection and emit the per-frame JSON list from the spec image."
    )
    parser.add_argument("--source", required=True, help="Video path, image path, or webcam index.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "runtime.yaml"),
        help="YAML runtime config.",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="Sample video at a fixed output FPS. For example, 10 emits one JSON result every 0.1s.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames. 0 means no limit.")
    parser.add_argument("--output-jsonl", help="Optional file to save the JSON vehicle lists.")
    parser.add_argument("--bridge-jsonl", help="Optional file to save the 16-number bridge vectors.")
    parser.add_argument("--signal-plan-jsonl", help="Optional file to save paper-style signal plans.")
    parser.add_argument("--show", action="store_true", help="Display annotated detections in an OpenCV window.")
    parser.add_argument(
        "--annotated-output",
        help="Optional image or video path to save annotated detections.",
    )
    return parser.parse_args()


def open_optional(path: Optional[str]) -> Optional[IO[str]]:
    if not path:
        return None
    return Path(path).open("w", encoding="utf-8")


def emit_result(
    result,
    vehicle_handle: Optional[IO[str]],
    bridge_handle: Optional[IO[str]],
    signal_handle: Optional[IO[str]],
    frame_index: int,
    timestamp_sec: float,
) -> None:
    vehicles_json = json.dumps(result["vehicles"])
    print(vehicles_json)
    if vehicle_handle is not None:
        vehicle_handle.write(vehicles_json + "\n")
    if bridge_handle is not None:
        bridge_handle.write(json.dumps(result["bridge_vector"]) + "\n")
    if signal_handle is not None:
        signal_handle.write(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "current_green_lane": result["current_green_lane"],
                    "signal_plan": result["signal_plan"],
                }
            )
            + "\n"
        )


def _snapshot_color(snapshot) -> tuple[int, int, int]:
    if snapshot.is_emergency:
        return (0, 0, 255)
    return LANE_COLORS.get(snapshot.lane, (220, 220, 220))


def _draw_lane_polygons(frame, pipeline: TrafficPipeline) -> None:
    frame_height, frame_width = frame.shape[:2]
    for lane, polygon in pipeline.config.lane_polygons.items():
        pixel_polygon = denormalize_polygon(polygon, frame_width, frame_height)
        contour = np.array(pixel_polygon, dtype=np.int32)
        color = LANE_COLORS.get(lane, (180, 180, 180))
        cv2.polylines(frame, [contour], isClosed=True, color=color, thickness=2)
        anchor_x, anchor_y = contour[0]
        cv2.putText(
            frame,
            lane,
            (int(anchor_x), int(anchor_y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_detection_roi(frame, pipeline: TrafficPipeline) -> None:
    detector = getattr(pipeline, "detector", None)
    if detector is None or not hasattr(detector, "detection_roi"):
        return

    frame_height, frame_width = frame.shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = detector.detection_roi(frame.shape)
    if (roi_x1, roi_y1, roi_x2, roi_y2) == (0, 0, frame_width, frame_height):
        return

    overlay = frame.copy()
    shade = (12, 16, 20)
    if roi_x1 > 0:
        cv2.rectangle(overlay, (0, 0), (roi_x1 - 1, frame_height - 1), shade, -1)
    if roi_x2 < frame_width:
        cv2.rectangle(overlay, (roi_x2, 0), (frame_width - 1, frame_height - 1), shade, -1)
    if roi_y1 > 0:
        cv2.rectangle(overlay, (0, 0), (frame_width - 1, roi_y1 - 1), shade, -1)
    if roi_y2 < frame_height:
        cv2.rectangle(overlay, (0, roi_y2), (frame_width - 1, frame_height - 1), shade, -1)
    cv2.addWeighted(overlay, 0.32, frame, 0.68, 0.0, frame)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2 - 1, roi_y2 - 1), (0, 255, 255), 2)
    cv2.putText(
        frame,
        "Detection ROI",
        (roi_x1 + 10, max(24, roi_y1 + 26)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def render_annotations(frame, result, pipeline: TrafficPipeline):
    annotated = frame.copy()
    snapshots = result.get("snapshots", [])
    temporal_metrics_available = result.get("temporal_metrics_available", True)

    _draw_detection_roi(annotated, pipeline)

    if pipeline.config.lane_polygons:
        _draw_lane_polygons(annotated, pipeline)

    for snapshot in snapshots:
        x1, y1, x2, y2 = (int(round(value)) for value in snapshot.bbox)
        color = _snapshot_color(snapshot)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = (
            f"ID {snapshot.track_id} {snapshot.class_name} "
            f"{snapshot.confidence:.2f} lane={snapshot.lane}"
        )
        if snapshot.is_emergency:
            label += " EMERGENCY"
        if temporal_metrics_available:
            label += f" halt={'Y' if snapshot.is_halting else 'N'} wait={snapshot.wait_time:.1f}s"
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    current_green_lane = result.get("current_green_lane") or "-"
    summary = f"green={current_green_lane} vehicles={len(snapshots)}"
    if not temporal_metrics_available:
        summary += " temporal=video-only"
    cv2.putText(
        annotated,
        summary,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def save_annotated_image(path: str, frame) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not save annotated image: {output_path}")


def open_video_writer(path: str, frame_shape, fps: float) -> cv2.VideoWriter:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_height, frame_width = frame_shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps, 1.0),
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open annotated video output: {output_path}")
    return writer


def maybe_show(frame, delay_ms: int) -> bool:
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(delay_ms) & 0xFF
    return key not in {27, ord("q")}


def should_process_video_frame(
    frame_index: int,
    timestamp_sec: float,
    frame_stride: int,
    target_fps: float,
    next_emit_timestamp: Optional[float],
) -> tuple[bool, Optional[float]]:
    if target_fps > 0.0:
        scheduled_timestamp = 0.0 if next_emit_timestamp is None else next_emit_timestamp
        if timestamp_sec + 1e-9 < scheduled_timestamp:
            return False, scheduled_timestamp
        return True, scheduled_timestamp + (1.0 / target_fps)

    return frame_index % frame_stride == 0, next_emit_timestamp


def process_image(path: Path, pipeline: TrafficPipeline, args: argparse.Namespace) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    vehicle_handle = open_optional(args.output_jsonl)
    bridge_handle = open_optional(args.bridge_jsonl)
    signal_handle = open_optional(args.signal_plan_jsonl)
    try:
        print(
            "Single images cannot measure is_halting or wait_time. "
            "Use a video path or webcam index for temporal metrics.",
            file=sys.stderr,
        )
        result = pipeline.process_frame(
            frame,
            timestamp_sec=0.0,
            temporal_metrics_available=False,
        )
        emit_result(
            result=result,
            vehicle_handle=vehicle_handle,
            bridge_handle=bridge_handle,
            signal_handle=signal_handle,
            frame_index=0,
            timestamp_sec=0.0,
        )
        annotated = None
        if args.show or args.annotated_output:
            annotated = render_annotations(frame, result, pipeline)
        if args.annotated_output and annotated is not None:
            save_annotated_image(args.annotated_output, annotated)
        if args.show and annotated is not None:
            maybe_show(annotated, 0)
    finally:
        for handle in (vehicle_handle, bridge_handle, signal_handle):
            if handle is not None:
                handle.close()
        if args.show:
            cv2.destroyAllWindows()


def process_video(source: str, pipeline: TrafficPipeline, args: argparse.Namespace) -> None:
    capture_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    vehicle_handle = open_optional(args.output_jsonl)
    bridge_handle = open_optional(args.bridge_jsonl)
    signal_handle = open_optional(args.signal_plan_jsonl)
    writer = None
    processed = 0
    frame_index = -1
    next_emit_timestamp: Optional[float] = None

    if args.target_fps < 0.0:
        raise ValueError("--target-fps must be >= 0")
    if args.target_fps > 0.0 and fps + 1e-9 < args.target_fps:
        print(
            f"Warning: source FPS ({fps:.2f}) is lower than target FPS ({args.target_fps:.2f}). "
            "Output rate will be limited by the input video.",
            file=sys.stderr,
        )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            timestamp_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp_sec <= 0.0:
                timestamp_sec = frame_index / fps

            should_process, next_emit_timestamp = should_process_video_frame(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                frame_stride=max(args.frame_stride, 1),
                target_fps=args.target_fps,
                next_emit_timestamp=next_emit_timestamp,
            )
            if not should_process:
                continue

            result = pipeline.process_frame(frame, timestamp_sec=timestamp_sec)
            emit_result(
                result=result,
                vehicle_handle=vehicle_handle,
                bridge_handle=bridge_handle,
                signal_handle=signal_handle,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )
            annotated = None
            if args.show or args.annotated_output:
                annotated = render_annotations(frame, result, pipeline)
            if args.annotated_output and annotated is not None:
                if writer is None:
                    output_fps = args.target_fps if args.target_fps > 0.0 else fps / max(args.frame_stride, 1)
                    writer = open_video_writer(args.annotated_output, annotated.shape, output_fps)
                writer.write(annotated)
            if args.show and annotated is not None and not maybe_show(annotated, 1):
                break
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        for handle in (vehicle_handle, bridge_handle, signal_handle):
            if handle is not None:
                handle.close()
        if args.show:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    pipeline = TrafficPipeline(load_config(args.config))
    source_path = Path(args.source)
    if source_path.exists() and source_path.suffix.lower() in IMAGE_SUFFIXES:
        process_image(source_path, pipeline, args)
        return
    process_video(args.source, pipeline, args)


if __name__ == "__main__":
    main()
