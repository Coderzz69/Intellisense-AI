from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

from .config import DetectorConfig
from .models import Detection


@runtime_checkable
class TrafficDetector(Protocol):
    """What :class:`~traffic_pipeline.pipeline.TrafficPipeline` needs from a detector.

    Implementations may additionally expose ``provider`` (a short backend name)
    and ``inference_metadata`` (details of the most recent call, e.g. endpoint
    latency); the pipeline surfaces both when present.
    """

    def detect(self, frame) -> List[Detection]:
        ...


def _matches_keyword(name: str, keywords: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _is_explicit_local_weights_path(weights_path: str) -> bool:
    candidate = Path(weights_path).expanduser()
    if candidate.is_absolute():
        return True
    return "/" in weights_path or "\\" in weights_path


def _resolve_device(device: str) -> str:
    normalized = str(device).strip().lower() if device is not None else ""

    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if normalized in {"", "auto", "none"}:
        return "0" if cuda_available else "cpu"
    if normalized == "0" and not cuda_available:
        return "cpu"
    if normalized.startswith("cuda") and not cuda_available:
        return "cpu"
    return str(device)


def detection_roi_bounds(
    frame_shape: Tuple[int, int, int],
    config: DetectorConfig,
) -> Tuple[int, int, int, int]:
    frame_height, frame_width = frame_shape[:2]
    left_fraction = min(max(float(config.crop_left_fraction), 0.0), 1.0)
    right_fraction = min(max(float(config.crop_right_fraction), 0.0), 1.0)
    top_fraction = min(max(float(config.crop_top_fraction), 0.0), 1.0)
    bottom_fraction = min(max(float(config.crop_bottom_fraction), 0.0), 1.0)

    x1 = int(round(frame_width * min(left_fraction, right_fraction)))
    x2 = int(round(frame_width * max(left_fraction, right_fraction)))
    y1 = int(round(frame_height * min(top_fraction, bottom_fraction)))
    y2 = int(round(frame_height * max(top_fraction, bottom_fraction)))

    if frame_width > 0:
        x1 = min(max(x1, 0), frame_width - 1)
        x2 = min(max(x2, x1 + 1), frame_width)
    if frame_height > 0:
        y1 = min(max(y1, 0), frame_height - 1)
        y2 = min(max(y2, y1 + 1), frame_height)
    return x1, y1, x2, y2


def offset_bbox_to_frame(
    bbox: Tuple[float, float, float, float],
    roi_bounds: Tuple[int, int, int, int],
) -> Tuple[float, float, float, float]:
    x_offset, y_offset, _, _ = roi_bounds
    x1, y1, x2, y2 = bbox
    return (
        float(x1 + x_offset),
        float(y1 + y_offset),
        float(x2 + x_offset),
        float(y2 + y_offset),
    )


def crop_frame_to_detection_roi(
    frame,
    config: DetectorConfig,
):
    x1, y1, x2, y2 = detection_roi_bounds(frame.shape, config)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


class YoloTrafficDetector:
    provider = "local"

    def __init__(self, config: DetectorConfig, vehicle_keywords: List[str]) -> None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ultralytics is required to instantiate YoloTrafficDetector. "
                "Install project dependencies before running detection."
            ) from exc

        self.config = config
        self.vehicle_keywords = vehicle_keywords
        weights_path = str(config.weights_path)
        if _is_explicit_local_weights_path(weights_path):
            expanded_path = Path(weights_path).expanduser()
            if not expanded_path.exists():
                raise FileNotFoundError(f"YOLO weights not found: {expanded_path}")
        self.device = _resolve_device(config.device)
        self.model = YOLO(weights_path)
        self.inference_metadata: Optional[Dict[str, object]] = {
            "weightsPath": weights_path,
            "device": self.device,
            "imgsz": config.imgsz,
            "confidence": config.confidence,
        }

    def detection_roi(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        return detection_roi_bounds(frame_shape, self.config)

    def _extract_detections(
        self,
        result,
        roi_bounds: Tuple[int, int, int, int],
    ) -> List[Detection]:
        if result.boxes is None or result.boxes.cls is None:
            return []

        names = result.names
        detections: List[Detection] = []
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for bbox, confidence, class_id in zip(boxes, confidences, class_ids):
            class_name = names.get(class_id, str(class_id))
            if not _matches_keyword(class_name, self.vehicle_keywords):
                continue
            detections.append(
                Detection(
                    bbox=offset_bbox_to_frame(
                        tuple(float(value) for value in bbox),
                        roi_bounds,
                    ),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )
        return detections

    def detect(self, frame) -> List[Detection]:
        cropped_frame, roi_bounds = crop_frame_to_detection_roi(frame, self.config)
        results = self.model.predict(
            source=cropped_frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._extract_detections(results[0], roi_bounds)

    def detect_many(self, frames) -> List[List[Detection]]:
        if not frames:
            return []

        cropped_frames = []
        roi_bounds_by_frame = []
        for frame in frames:
            cropped_frame, roi_bounds = crop_frame_to_detection_roi(frame, self.config)
            cropped_frames.append(cropped_frame)
            roi_bounds_by_frame.append(roi_bounds)

        results = self.model.predict(
            source=cropped_frames,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False,
        )
        return [
            self._extract_detections(result, roi_bounds)
            for result, roi_bounds in zip(results, roi_bounds_by_frame)
        ]
