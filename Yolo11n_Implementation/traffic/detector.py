from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from .config import DetectorConfig
from .models import Detection


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


class YoloTrafficDetector:
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

    def _extract_detections(self, result) -> List[Detection]:
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
                    bbox=tuple(float(value) for value in bbox),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )
        return detections

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._extract_detections(results[0])

    def detect_many(self, frames) -> List[List[Detection]]:
        if not frames:
            return []

        results = self.model.predict(
            source=list(frames),
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False,
        )
        return [self._extract_detections(result) for result in results]
