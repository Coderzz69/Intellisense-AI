#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.analysis import (
    LANES,
    build_analysis_payload,
    build_sequence_summary,
    build_video_metadata,
    build_video_summary,
    format_timestamp,
    json_bytes,
    parse_bool,
    parse_non_negative_int,
    parse_positive_float,
    record_analytics_event,
    resolve_existing_path,
    resolve_video_paths,
    should_emit_state,
    should_process_frame,
)
from backend.cloud import S3AnalyticsSink
from backend.runtime import (
    BUNDLE_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_EMIT_INTERVAL_SEC,
    DEFAULT_PROCESS_FPS,
    DEFAULT_REALTIME,
    DEFAULT_VIDEO_PATH,
    get_inference_provider,
    load_processing_pipeline,
)
from backend.settings import resolve_sagemaker_endpoint_name


class TrafficDashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TrafficDashboard/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[traffic-dashboard] {self.address_string()} - {fmt % args}")

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._write_json({"status": "ok"})
            return
        if parsed.path == "/api/config":
            self._handle_config()
            return
        if parsed.path == "/api/stream":
            self._handle_stream(parsed.query)
            return
        if parsed.path == "/api/video":
            self._handle_video(parsed.query)
            return
        self._write_json(
            {
                "error": "Not found",
                "path": parsed.path,
            },
            status=HTTPStatus.NOT_FOUND,
        )

    def _handle_config(self) -> None:
        video = None
        if DEFAULT_VIDEO_PATH is not None:
            capture = cv2.VideoCapture(str(DEFAULT_VIDEO_PATH))
            try:
                video = build_video_metadata(capture, DEFAULT_VIDEO_PATH) if capture.isOpened() else None
            finally:
                capture.release()

        self._write_json(
            {
                "bundlePath": str(BUNDLE_DIR),
                "configPath": str(DEFAULT_CONFIG_PATH),
                "defaultVideoPath": None if DEFAULT_VIDEO_PATH is None else str(DEFAULT_VIDEO_PATH),
                "defaults": {
                    "processFps": DEFAULT_PROCESS_FPS,
                    "emitIntervalSec": DEFAULT_EMIT_INTERVAL_SEC,
                    "realtime": DEFAULT_REALTIME,
                },
                "inferenceProvider": get_inference_provider(),
                "sagemakerEndpointName": resolve_sagemaker_endpoint_name(),
                "s3AnalyticsEnabled": bool(os.environ.get("TRAFFIC_S3_BUCKET")),
                "lanes": list(LANES),
                "video": video,
            }
        )

    def _handle_stream(self, raw_query: str) -> None:
        query = parse_qs(raw_query)
        analytics_sink: S3AnalyticsSink | None = None
        try:
            video_paths = resolve_video_paths(query.get("videoPath", []), DEFAULT_VIDEO_PATH)
            config_path = resolve_existing_path(
                query.get("configPath", [None])[0],
                DEFAULT_CONFIG_PATH,
                "Config",
            )
            emit_interval_sec = parse_positive_float(
                query.get("emitIntervalSec", [None])[0],
                DEFAULT_EMIT_INTERVAL_SEC,
                "emitIntervalSec",
            )
            process_fps = parse_positive_float(
                query.get("processFps", [None])[0],
                DEFAULT_PROCESS_FPS,
                "processFps",
            )
            max_emits = parse_non_negative_int(
                query.get("maxEmits", [None])[0],
                0,
                "maxEmits",
            )
            realtime = parse_bool(query.get("realtime", [None])[0], DEFAULT_REALTIME)
        except (FileNotFoundError, ValueError) as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            analytics_sink = S3AnalyticsSink.from_env()
        except Exception as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        try:
            sequence_total = len(video_paths)
            video_summaries: list[dict[str, object]] = []

            for sequence_index, video_path in enumerate(video_paths, start=1):
                capture = None
                try:
                    pipeline = load_processing_pipeline(config_path)
                    capture = cv2.VideoCapture(str(video_path))
                    if not capture.isOpened():
                        raise RuntimeError(f"Could not open source: {video_path}")

                    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                    video_metadata = build_video_metadata(capture, video_path)
                    session_payload = {
                        "type": "session",
                        "sequenceIndex": sequence_index,
                        "sequenceTotal": sequence_total,
                        "video": video_metadata,
                        "videoPath": str(video_path),
                        "configPath": str(config_path),
                        "processFps": process_fps,
                        "emitIntervalSec": emit_interval_sec,
                        "realtime": realtime,
                        "inferenceProvider": get_inference_provider(),
                        "sagemakerEndpointName": resolve_sagemaker_endpoint_name(),
                    }
                    self._write_sse(session_payload)
                    record_analytics_event(
                        analytics_sink,
                        {
                            "eventType": "session",
                            "payload": session_payload,
                        },
                    )

                    frame_index = -1
                    next_process_timestamp: float | None = None
                    next_emit_timestamp: float | None = None
                    latest_result: dict[str, object] | None = None
                    latest_timestamp = 0.0
                    emitted = 0
                    stream_started_at = time.perf_counter()

                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break

                        frame_index += 1
                        timestamp_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        if timestamp_sec <= 0.0:
                            timestamp_sec = frame_index / fps

                        process_now, next_process_timestamp = should_process_frame(
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            process_fps=process_fps,
                            next_process_timestamp=next_process_timestamp,
                        )
                        if not process_now:
                            continue

                        latest_result = pipeline.process_frame(frame, timestamp_sec=timestamp_sec)
                        latest_timestamp = timestamp_sec

                        emit_now, next_emit_timestamp = should_emit_state(
                            timestamp_sec=timestamp_sec,
                            emit_interval_sec=emit_interval_sec,
                            next_emit_timestamp=next_emit_timestamp,
                        )
                        if not emit_now:
                            continue

                        emitted += 1
                        payload = build_analysis_payload(
                            latest_result,
                            emitted_index=emitted,
                            timestamp_sec=timestamp_sec,
                        )
                        payload["sequenceIndex"] = sequence_index
                        payload["sequenceTotal"] = sequence_total
                        payload["videoPath"] = str(video_path)
                        payload["video"] = video_metadata
                        self._write_sse(payload)
                        record_analytics_event(
                            analytics_sink,
                            {
                                "eventType": "analysis",
                                "payload": payload,
                            },
                        )

                        if realtime:
                            target_elapsed = emitted * emit_interval_sec
                            remaining = target_elapsed - (time.perf_counter() - stream_started_at)
                            if remaining > 0:
                                time.sleep(remaining)

                        if max_emits and emitted >= max_emits:
                            break

                    final_frame = (
                        None
                        if latest_result is None
                        else build_analysis_payload(
                            latest_result,
                            emitted_index=emitted,
                            timestamp_sec=latest_timestamp,
                        )
                    )
                    if final_frame is not None:
                        final_frame["sequenceIndex"] = sequence_index
                        final_frame["sequenceTotal"] = sequence_total
                        final_frame["videoPath"] = str(video_path)
                        final_frame["video"] = video_metadata

                    video_complete_payload = {
                        "type": "videoComplete",
                        "sequenceIndex": sequence_index,
                        "sequenceTotal": sequence_total,
                        "videoPath": str(video_path),
                        "video": video_metadata,
                        "emittedCount": emitted,
                        "lastTimestampSec": round(latest_timestamp, 2),
                        "lastTimestampLabel": format_timestamp(latest_timestamp),
                        "finalFrame": final_frame,
                    }
                    self._write_sse(video_complete_payload)
                    record_analytics_event(
                        analytics_sink,
                        {
                            "eventType": "videoComplete",
                            "payload": video_complete_payload,
                        },
                    )

                    video_summaries.append(
                        build_video_summary(
                            sequence_index=sequence_index,
                            sequence_total=sequence_total,
                            video_path=video_path,
                            video_metadata=video_metadata,
                            emitted_count=emitted,
                            last_timestamp_sec=latest_timestamp,
                            final_frame=final_frame,
                        )
                    )
                finally:
                    if capture is not None:
                        capture.release()

            overall_summary = build_sequence_summary(video_summaries)
            complete_payload = {
                "type": "complete",
                "sequenceTotal": sequence_total,
                "emittedCount": int(overall_summary["totalSnapshots"]),
                "videoSummaries": video_summaries,
                "overallSummary": overall_summary,
            }
            self._write_sse(complete_payload)
            record_analytics_event(
                analytics_sink,
                {
                    "eventType": "complete",
                    "payload": complete_payload,
                },
            )
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self._write_sse({"type": "error", "message": str(exc)})
            record_analytics_event(
                analytics_sink,
                {
                    "eventType": "error",
                    "payload": {
                        "message": str(exc),
                    },
                },
            )
        finally:
            if analytics_sink is not None:
                analytics_sink.close()

    def _handle_video(self, raw_query: str) -> None:
        query = parse_qs(raw_query)
        try:
            video_path = resolve_existing_path(query.get("path", [None])[0], DEFAULT_VIDEO_PATH, "Video")
        except FileNotFoundError as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        file_size = video_path.stat().st_size
        content_type = mimetypes.guess_type(video_path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header and range_header.startswith("bytes="):
            start_str, _, end_str = range_header[6:].partition("-")
            if start_str:
                start = int(start_str)
            if end_str:
                end = int(end_str)
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT

        content_length = (end - start) + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with video_path.open("rb") as handle:
            try:
                handle.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return

    def _write_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_sse(self, payload: dict[str, object]) -> None:
        body = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve traffic analysis for the React dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), TrafficDashboardHandler)
    print(f"Traffic dashboard backend listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
