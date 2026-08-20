from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.25"))
DEFAULT_IOU = float(os.environ.get("YOLO_IOU", "0.45"))
DEFAULT_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
DEFAULT_DEVICE = os.environ.get("YOLO_DEVICE", "cuda")


def _resolve_model_path(model_dir: str) -> Path:
    root = Path(model_dir)
    for candidate in ("best.pt", "model.pt", "weights/best.pt"):
        model_path = root / candidate
        if model_path.exists():
            return model_path
    raise FileNotFoundError(f"Could not find a YOLO weights file under {model_dir}")


def model_fn(model_dir: str):
    model_path = _resolve_model_path(model_dir)
    return YOLO(str(model_path))


def input_fn(request_body, request_content_type: str):
    if request_content_type != "application/x-image":
        raise ValueError("Unsupported content type. Expected application/x-image.")

    frame = cv2.imdecode(np.frombuffer(request_body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the incoming frame.")
    return frame


def predict_fn(frame, model):
    results = model.predict(
        source=frame,
        conf=DEFAULT_CONFIDENCE,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        verbose=False,
    )
    result = results[0]
    detections = []
    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []

    for bbox, confidence, class_id in zip(boxes, confidences, class_ids):
        detections.append(
            {
                "bbox": [float(value) for value in bbox],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": names.get(class_id, str(class_id)),
            }
        )

    return {
        "detections": detections,
        "image_shape": list(frame.shape),
        "model_latency_ms": getattr(result, "speed", {}).get("inference"),
    }


def output_fn(prediction, accept: str):
    if accept not in {"application/json", "*/*"}:
        raise ValueError("Unsupported Accept header. Expected application/json.")
    return json.dumps(prediction), "application/json"
