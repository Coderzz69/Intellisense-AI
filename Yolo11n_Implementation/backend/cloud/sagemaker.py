"""SageMaker-backed detector.

This is deliberately *only* a detector. It plugs into the same
``TrafficPipeline`` / ``AITrafficPipeline`` as the local YOLO detector, so
tracking, lane metrics and the RL controller behave identically no matter where
the boxes came from.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, List, Optional

import cv2

from backend.settings import (
    LEGACY_SAGEMAKER_ENDPOINT_ENV,
    PRIMARY_SAGEMAKER_ENDPOINT_ENV,
    resolve_sagemaker_endpoint_name,
)
from traffic_pipeline.models import Detection

SAGEMAKER_REGION_ENV = "AWS_REGION"
JPEG_QUALITY_ENV = "TRAFFIC_SAGEMAKER_JPEG_QUALITY"
CONNECT_TIMEOUT_ENV = "TRAFFIC_SAGEMAKER_CONNECT_TIMEOUT_SEC"
READ_TIMEOUT_ENV = "TRAFFIC_SAGEMAKER_READ_TIMEOUT_SEC"
MAX_RETRIES_ENV = "TRAFFIC_SAGEMAKER_MAX_RETRIES"


def _matches_keyword(name: str, keywords: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


class SageMakerDetector:
    """Satisfies ``traffic_pipeline.TrafficDetector`` against a SageMaker endpoint."""

    provider = "sagemaker"

    def __init__(self, vehicle_keywords: List[str], endpoint_name: Optional[str] = None) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "boto3 is required for SageMaker inference. "
                "Install it with: pip install -r backend/requirements-cloud.txt"
            ) from exc

        endpoint_name = endpoint_name or resolve_sagemaker_endpoint_name()
        if not endpoint_name:
            raise ValueError(
                f"{PRIMARY_SAGEMAKER_ENDPOINT_ENV} or {LEGACY_SAGEMAKER_ENDPOINT_ENV} "
                "must be set when TRAFFIC_INFERENCE_PROVIDER=sagemaker."
            )

        self.endpoint_name = endpoint_name
        self.vehicle_keywords = list(vehicle_keywords)
        self.jpeg_quality = int(os.environ.get(JPEG_QUALITY_ENV, "85"))
        self.inference_metadata: Optional[Dict[str, object]] = None
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

    def detect(self, frame) -> List[Detection]:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Could not encode frame for SageMaker inference.")

        started_at = time.perf_counter()
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/x-image",
            Accept="application/json",
            Body=encoded.tobytes(),
        )
        payload = json.loads(response["Body"].read().decode("utf-8"))
        round_trip_ms = round((time.perf_counter() - started_at) * 1000.0, 2)

        detections: List[Detection] = []
        for item in payload.get("detections", []):
            class_name = str(item.get("class_name", item.get("className", "")))
            if not _matches_keyword(class_name, self.vehicle_keywords):
                continue
            detections.append(
                Detection(
                    bbox=tuple(float(value) for value in item["bbox"]),
                    confidence=float(item.get("confidence", 0.0)),
                    class_id=int(item.get("class_id", item.get("classId", 0))),
                    class_name=class_name,
                )
            )

        self.inference_metadata = {
            "endpointName": self.endpoint_name,
            "invokedProductionVariant": response.get("InvokedProductionVariant"),
            "modelLatencyMs": payload.get("model_latency_ms", payload.get("modelLatencyMs")),
            "roundTripMs": round_trip_ms,
            "imageShape": payload.get("image_shape", payload.get("imageShape")),
            "rawDetectionCount": len(payload.get("detections", [])),
            "keptDetectionCount": len(detections),
        }
        return detections
