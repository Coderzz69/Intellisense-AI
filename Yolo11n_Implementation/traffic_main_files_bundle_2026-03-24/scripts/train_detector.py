#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _ai_path  # noqa: F401,E402  (puts the canonical traffic_pipeline on sys.path)


DEFAULT_DATASET = str(
    Path(__file__).resolve().parents[1] / "configs" / "local_dataset.yaml"
)
DEFAULT_WEIGHTS = "/home/anuj/Desktop/image-processing/yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a traffic detector on the local Indian vehicle dataset."
    )
    parser.add_argument("--data", default=DEFAULT_DATASET, help="Dataset YAML.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Base model checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="traffic-paper-model")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--fraction", type=float, default=1.0, help="Use a subset for quick experiments.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the provided checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print the training config without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    weights_path = Path(args.weights)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if args.dry_run:
        print(f"dataset={data_path}")
        print(f"weights={weights_path}")
        print(f"epochs={args.epochs}")
        print(f"imgsz={args.imgsz}")
        print(f"batch={args.batch}")
        print(f"project={args.project}")
        print(f"name={args.name}")
        print(f"fraction={args.fraction}")
        print(f"resume={args.resume}")
        return

    model = YOLO(str(weights_path))
    if args.resume:
        model.train(resume=True)
        return
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        fraction=args.fraction,
    )


if __name__ == "__main__":
    main()
