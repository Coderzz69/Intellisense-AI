#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Optional

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detect_traffic import maybe_show, open_video_writer, render_annotations
from traffic_control import RLTrafficController, load_rl_config
from traffic_pipeline import TrafficPipeline, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run object detection + RL signal control on video and emit lane-level green/red decisions.",
    )
    parser.add_argument("--source", required=True, help="Video path or webcam index.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "runtime.yaml"),
        help="Integrated runtime YAML.",
    )
    parser.add_argument(
        "--emit-interval-sec",
        type=float,
        default=1.0,
        help="Emit one RL decision every N seconds of video time.",
    )
    parser.add_argument(
        "--process-fps",
        type=float,
        default=10.0,
        help="Detection/tracking FPS before each emitted decision window.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Fallback frame skip when --process-fps is 0.",
    )
    parser.add_argument(
        "--max-emits",
        type=int,
        default=0,
        help="Stop after N emitted decisions. 0 means no limit.",
    )
    parser.add_argument(
        "--output-jsonl",
        help="Optional file to save one decision payload per line.",
    )
    parser.add_argument(
        "--annotated-output",
        help="Optional path to save an annotated output video.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated output live in an OpenCV window.",
    )
    return parser.parse_args()


def open_optional(path: Optional[str]) -> Optional[IO[str]]:
    if not path:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("w", encoding="utf-8")


def should_process_frame(
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


def should_emit_state(
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


def draw_decision_overlay(frame, decision) -> None:
    cv2.rectangle(frame, (10, 10), (470, 124), (24, 30, 38), -1)
    cv2.rectangle(frame, (10, 10), (470, 124), (225, 225, 225), 1)
    cv2.putText(
        frame,
        f"RL action: {decision.action_name}",
        (22, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (130, 255, 160),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"source: {decision.decision_source}",
        (22, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"signals N={decision.lane_signals['N']} E={decision.lane_signals['E']} "
        f"S={decision.lane_signals['S']} W={decision.lane_signals['W']}",
        (22, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"q={decision.q_values}",
        (22, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (175, 205, 255),
        1,
        cv2.LINE_AA,
    )


def process_video(args: argparse.Namespace) -> None:
    pipeline = TrafficPipeline(load_config(args.config))
    controller = RLTrafficController(load_rl_config(args.config))

    capture_source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    output_handle = open_optional(args.output_jsonl)
    frame_index = -1
    next_process_timestamp: Optional[float] = None
    next_emit_timestamp: Optional[float] = None
    emitted = 0
    raw_state_window: list[list[float]] = []
    latest_result = None
    latest_decision = None

    if args.emit_interval_sec <= 0.0:
        raise ValueError("--emit-interval-sec must be > 0")
    if args.process_fps < 0.0:
        raise ValueError("--process-fps must be >= 0")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            timestamp_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp_sec <= 0.0:
                timestamp_sec = frame_index / fps

            should_process, next_process_timestamp = should_process_frame(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                frame_stride=args.frame_stride,
                process_fps=args.process_fps,
                next_process_timestamp=next_process_timestamp,
            )
            if not should_process:
                continue

            latest_result = pipeline.process_frame(frame, timestamp_sec=timestamp_sec)
            raw_state_window.append(
                controller.raw_bridge_from_snapshots(latest_result["snapshots"])
            )

            should_emit, next_emit_timestamp = should_emit_state(
                timestamp_sec=timestamp_sec,
                emit_interval_sec=args.emit_interval_sec,
                next_emit_timestamp=next_emit_timestamp,
            )
            if should_emit:
                latest_decision = controller.decide(raw_state_window)
                payload = {
                    "timestamp_sec": round(timestamp_sec, 3),
                    "state": latest_decision.state,
                    "q_values": latest_decision.q_values,
                    "action": latest_decision.action,
                    "action_name": latest_decision.action_name,
                    "decision_source": latest_decision.decision_source,
                    "lane_signals": latest_decision.lane_signals,
                    "green_lanes": latest_decision.green_lanes,
                    "red_lanes": latest_decision.red_lanes,
                    "vehicles": latest_result["vehicles"],
                }
                print(json.dumps(payload))
                if output_handle is not None:
                    output_handle.write(json.dumps(payload) + "\n")
                raw_state_window.clear()
                emitted += 1
                if args.max_emits and emitted >= args.max_emits:
                    break

            if args.show or args.annotated_output:
                render_result = latest_result
                if latest_decision is not None:
                    render_result = dict(latest_result)
                    render_result["current_green_lane"] = latest_decision.action_name
                annotated = render_annotations(frame, render_result, pipeline)
                if latest_decision is not None:
                    draw_decision_overlay(annotated, latest_decision)
                if writer is None and args.annotated_output:
                    writer = open_video_writer(args.annotated_output, annotated.shape, fps)
                if writer is not None:
                    writer.write(annotated)
                if args.show and not maybe_show(annotated, max(1, int(1000 / fps))):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if output_handle is not None:
            output_handle.close()
        if args.show:
            cv2.destroyAllWindows()


def main() -> None:
    process_video(parse_args())


if __name__ == "__main__":
    main()
