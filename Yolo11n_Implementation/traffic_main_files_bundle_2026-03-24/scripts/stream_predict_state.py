#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Optional
from urllib import error, request

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _ai_path  # noqa: F401,E402  (puts the canonical traffic_pipeline on sys.path)

from scripts.detect_traffic import maybe_show, open_video_writer, render_annotations
from traffic_pipeline import TrafficPipeline, load_config
from traffic_pipeline.models import LANE_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process video and emit POST-ready state vectors once per second."
    )
    parser.add_argument("--source", required=True, help="Video path or webcam index.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "runtime.example.yaml"),
        help="YAML runtime config.",
    )
    parser.add_argument(
        "--emit-interval-sec",
        type=float,
        default=1.0,
        help="Emit one state payload every N seconds of video time.",
    )
    parser.add_argument(
        "--process-fps",
        type=float,
        default=10.0,
        help="Processing FPS for tracking before per-second aggregation. Default is 10 FPS.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Fallback frame skip when --process-fps is 0.",
    )
    parser.add_argument(
        "--output-jsonl",
        help="Optional file to save one POST body per line.",
    )
    parser.add_argument(
        "--raw-state-jsonl",
        help="Optional file to save the intermediate 10-FPS raw bridge vectors before per-second aggregation.",
    )
    parser.add_argument(
        "--queue-cap",
        type=float,
        default=5.0,
        help="Stopped-vehicle count that maps to 1.0 in the normalized queue segment.",
    )
    parser.add_argument(
        "--wait-cap-sec",
        type=float,
        default=0.0,
        help="Total wait time in seconds that maps to 1.0 in the normalized wait segment. 0 uses signal.total_cycle_time.",
    )
    parser.add_argument(
        "--post-url",
        help="Optional URL to receive POST bodies like {\"state\": [...]}",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=3.0,
        help="HTTP timeout used with --post-url.",
    )
    parser.add_argument(
        "--max-emits",
        type=int,
        default=0,
        help="Stop after N emitted state payloads. 0 means no limit.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the processed video with annotated detections.",
    )
    parser.add_argument(
        "--annotated-output",
        help="Optional path to save an annotated video of the processed frames.",
    )
    return parser.parse_args()


def open_optional(path: Optional[str]) -> Optional[IO[str]]:
    if not path:
        return None
    return Path(path).open("w", encoding="utf-8")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_state_window(
    state_vectors: list[list[float]],
    *,
    queue_cap: float,
    wait_cap_sec: float,
) -> list[float]:
    if not state_vectors:
        return [0.0] * 16
    if queue_cap <= 0.0:
        raise ValueError("queue_cap must be > 0")
    if wait_cap_sec <= 0.0:
        raise ValueError("wait_cap_sec must be > 0")

    sample_count = float(len(state_vectors))
    lane_count = len(LANE_ORDER)
    queue_segment = []
    wait_segment = []
    phase_segment = []
    emergency_segment = []

    for index in range(lane_count):
        queue_mean = sum(vector[index] for vector in state_vectors) / sample_count
        wait_mean = sum(vector[index + lane_count] for vector in state_vectors) / sample_count
        phase_mean = sum(vector[index + (2 * lane_count)] for vector in state_vectors) / sample_count
        emergency_max = max(vector[index + (3 * lane_count)] for vector in state_vectors)
        queue_segment.append(round(clamp01(queue_mean / queue_cap), 4))
        wait_segment.append(round(clamp01(wait_mean / wait_cap_sec), 4))
        phase_segment.append(round(clamp01(phase_mean), 4))
        emergency_segment.append(round(clamp01(emergency_max), 4))

    return queue_segment + wait_segment + phase_segment + emergency_segment


def build_state_payload(
    state_vectors: list[list[float]],
    *,
    queue_cap: float,
    wait_cap_sec: float,
) -> dict[str, list[float]]:
    return {
        "state": normalize_state_window(
            state_vectors,
            queue_cap=queue_cap,
            wait_cap_sec=wait_cap_sec,
        )
    }


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


def post_state(url: str, payload: dict[str, list[float]], timeout_sec: float) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=timeout_sec) as response:
        response_body = response.read().decode("utf-8", errors="replace")
        return int(response.status), response_body


def process_video(source: str, pipeline: TrafficPipeline, args: argparse.Namespace) -> None:
    capture_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output_handle = open_optional(args.output_jsonl)
    raw_state_handle = open_optional(args.raw_state_jsonl)
    writer = None
    frame_index = -1
    next_process_timestamp: Optional[float] = None
    next_emit_timestamp: Optional[float] = None
    emitted = 0
    window_state_vectors: list[list[float]] = []

    if args.emit_interval_sec <= 0.0:
        raise ValueError("--emit-interval-sec must be > 0")
    if args.process_fps < 0.0:
        raise ValueError("--process-fps must be >= 0")
    if args.queue_cap <= 0.0:
        raise ValueError("--queue-cap must be > 0")
    if args.process_fps > 0.0 and fps + 1e-9 < args.process_fps:
        print(
            f"Warning: source FPS ({fps:.2f}) is lower than process FPS ({args.process_fps:.2f}). "
            "Processing rate will be limited by the input video.",
            file=sys.stderr,
        )
    wait_cap_sec = args.wait_cap_sec
    if wait_cap_sec <= 0.0:
        wait_cap_sec = pipeline.config.signal.total_cycle_time

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

            result = pipeline.process_frame(frame, timestamp_sec=timestamp_sec)
            raw_state_vector = list(result["bridge_vector"])
            window_state_vectors.append(raw_state_vector)
            if raw_state_handle is not None:
                raw_state_handle.write(
                    json.dumps(
                        {
                            "timestamp_sec": round(timestamp_sec, 3),
                            "state": raw_state_vector,
                        }
                    )
                    + "\n"
                )

            annotated = None
            if args.show or args.annotated_output:
                annotated = render_annotations(frame, result, pipeline)
                if args.annotated_output:
                    if writer is None:
                        output_fps = args.process_fps if args.process_fps > 0.0 else fps / max(args.frame_stride, 1)
                        writer = open_video_writer(
                            args.annotated_output,
                            annotated.shape,
                            output_fps,
                        )
                    writer.write(annotated)
                if args.show and not maybe_show(annotated, 1):
                    break

            should_emit, next_emit_timestamp = should_emit_state(
                timestamp_sec=timestamp_sec,
                emit_interval_sec=args.emit_interval_sec,
                next_emit_timestamp=next_emit_timestamp,
            )
            if not should_emit:
                continue

            payload = build_state_payload(
                window_state_vectors,
                queue_cap=args.queue_cap,
                wait_cap_sec=wait_cap_sec,
            )
            window_state_vectors = []
            payload_json = json.dumps(payload)
            print(payload_json)
            if output_handle is not None:
                output_handle.write(payload_json + "\n")

            if args.post_url:
                try:
                    status_code, response_body = post_state(
                        url=args.post_url,
                        payload=payload,
                        timeout_sec=args.timeout_sec,
                    )
                    print(
                        json.dumps(
                            {
                                "timestamp_sec": round(timestamp_sec, 3),
                                "status_code": status_code,
                                "response_body": response_body,
                            }
                        ),
                        file=sys.stderr,
                    )
                except error.URLError as exc:
                    print(
                        f"POST failed at {timestamp_sec:.3f}s: {exc}",
                        file=sys.stderr,
                    )

            emitted += 1
            if args.max_emits and emitted >= args.max_emits:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if output_handle is not None:
            output_handle.close()
        if raw_state_handle is not None:
            raw_state_handle.close()
        if args.show:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    pipeline = TrafficPipeline(load_config(args.config))
    process_video(args.source, pipeline, args)


if __name__ == "__main__":
    main()
