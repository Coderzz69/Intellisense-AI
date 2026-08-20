from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

from backend.settings import (
    LEGACY_SAGEMAKER_ENDPOINT_ENV,
    PRIMARY_SAGEMAKER_ENDPOINT_ENV,
    resolve_sagemaker_endpoint_name,
)
from traffic_pipeline import load_config
from traffic_pipeline.bridge import build_bridge_vector
from traffic_pipeline.models import Detection, SignalPlan
from traffic_pipeline.signal_control import PaperSignalController
from traffic_pipeline.tracker import SimpleTracker

SAGEMAKER_REGION_ENV = "AWS_REGION"
JPEG_QUALITY_ENV = "TRAFFIC_SAGEMAKER_JPEG_QUALITY"
CONNECT_TIMEOUT_ENV = "TRAFFIC_SAGEMAKER_CONNECT_TIMEOUT_SEC"
READ_TIMEOUT_ENV = "TRAFFIC_SAGEMAKER_READ_TIMEOUT_SEC"
MAX_RETRIES_ENV = "TRAFFIC_SAGEMAKER_MAX_RETRIES"


def _matches_keyword(name: str, keywords: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


class SageMakerDetectorClient:
    def __init__(self, endpoint_name: str) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "boto3 is required for SageMaker inference. Install backend cloud dependencies first."
            ) from exc

        self.endpoint_name = endpoint_name
        self.jpeg_quality = int(os.environ.get(JPEG_QUALITY_ENV, "85"))
        self.runtime = boto3.client(
            "sagemaker-runtime",
            region_name=os.environ.get(SAGEMAKER_REGION_ENV),
            config=Config(
                connect_timeout=int(os.environ.get(CONNECT_TIMEOUT_ENV, "5")),
                read_timeout=int(os.environ.get(READ_TIMEOUT_ENV, "70")),
                retries={
                    "max_attempts": int(os.environ.get(MAX_RETRIES_ENV, "3")),
                    "mode": "standard",
                },
            ),
        )

    def detect(self, frame) -> tuple[List[Detection], Dict[str, object]]:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Could not encode frame for SageMaker inference.")

        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/x-image",
            Accept="application/json",
            Body=encoded.tobytes(),
        )
        payload = json.loads(response["Body"].read().decode("utf-8"))

        detections: List[Detection] = []
        for item in payload.get("detections", []):
            class_name = str(item.get("class_name", item.get("className", "")))
            detections.append(
                Detection(
                    bbox=tuple(float(value) for value in item["bbox"]),
                    confidence=float(item.get("confidence", 0.0)),
                    class_id=int(item.get("class_id", item.get("classId", 0))),
                    class_name=class_name,
                )
            )

        metadata = {
            "endpointName": self.endpoint_name,
            "invokedProductionVariant": response.get("InvokedProductionVariant"),
            "modelLatencyMs": payload.get("model_latency_ms", payload.get("modelLatencyMs")),
            "imageShape": payload.get("image_shape", payload.get("imageShape")),
        }
        return detections, metadata


class SageMakerTrafficPipeline:
    def __init__(self, config_path) -> None:
        self.config = load_config(str(config_path))
        endpoint_name = resolve_sagemaker_endpoint_name()
        if not endpoint_name:
            raise ValueError(
                f"{PRIMARY_SAGEMAKER_ENDPOINT_ENV} or {LEGACY_SAGEMAKER_ENDPOINT_ENV} "
                "must be configured when using SageMaker inference."
            )

        self.detector = SageMakerDetectorClient(endpoint_name)
        self.vehicle_keywords = list(self.config.vehicle_keywords)
        self.tracker = SimpleTracker(self.config.tracking)
        self.signal_controller = PaperSignalController(
            self.config.signal,
            self.config.calibration,
        )
        self._signal_plan: Optional[SignalPlan] = None
        self._cycle_started_at = 0.0

    def process_frame(
        self,
        frame,
        timestamp_sec: float,
        temporal_metrics_available: bool = True,
    ) -> Dict[str, object]:
        detections, endpoint_metadata = self.detector.detect(frame)
        filtered_detections = [
            detection
            for detection in detections
            if _matches_keyword(detection.class_name, self.vehicle_keywords)
        ]

        snapshots = self.tracker.update(
            detections=filtered_detections,
            frame_shape=frame.shape,
            timestamp_sec=timestamp_sec,
            lane_polygons=self.config.lane_polygons,
            emergency_keywords=self.config.emergency_keywords,
        )

        if self._should_refresh_plan(timestamp_sec, snapshots):
            self._signal_plan = self.signal_controller.build_signal_plan(snapshots)
            self._cycle_started_at = timestamp_sec

        current_green_lane = None
        if self._signal_plan is not None:
            current_green_lane = self.signal_controller.active_green_lane(
                self._signal_plan,
                max(0.0, timestamp_sec - self._cycle_started_at),
            )

        return {
            "vehicles": [
                snapshot.to_output_dict(
                    temporal_metrics_available=temporal_metrics_available
                )
                for snapshot in snapshots
            ],
            "snapshots": snapshots,
            "temporal_metrics_available": temporal_metrics_available,
            "bridge_vector": build_bridge_vector(
                snapshots=snapshots,
                current_green_lane=current_green_lane,
                lane_order=self.config.signal.lane_order,
            ),
            "current_green_lane": current_green_lane,
            "signal_plan": None if self._signal_plan is None else self._signal_plan.to_dict(),
            "inference": endpoint_metadata,
            "inferenceProvider": "sagemaker",
        }

    def _should_refresh_plan(self, timestamp_sec, snapshots) -> bool:
        emergency_present = any(snapshot.is_emergency for snapshot in snapshots)
        if self._signal_plan is None:
            return True
        if emergency_present:
            return True
        if self._signal_plan.emergency_override_lane is not None and not emergency_present:
            return True
        return (timestamp_sec - self._cycle_started_at) >= self._signal_plan.total_cycle_time
