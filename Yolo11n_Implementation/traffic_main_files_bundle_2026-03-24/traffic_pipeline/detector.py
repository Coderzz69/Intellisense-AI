from __future__ import annotations

from typing import Iterable, List

from .config import DetectorConfig
from .models import Detection


def _matches_keyword(name: str, keywords: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


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
        self.model = YOLO(config.weights_path)

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )
        result = results[0]
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
