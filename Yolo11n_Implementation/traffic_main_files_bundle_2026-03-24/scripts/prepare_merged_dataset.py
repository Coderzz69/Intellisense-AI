#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional


CURRENT_CLASSES = [
    "ambulance",
    "auto-rikshaw",
    "bike",
    "bus",
    "car",
    "police vehicle",
    "truck",
]
TARGET_CLASS_IDS = {name: index for index, name in enumerate(CURRENT_CLASSES)}

# UVH-26 majority-vote COCO categories.
UVH_TO_TARGET = {
    "Hatchback": "car",
    "Sedan": "car",
    "SUV": "car",
    "MUV": "car",
    "Bus": "bus",
    "Truck": "truck",
    "Three-wheeler": "auto-rikshaw",
    "Two-wheeler": "bike",
    "LCV": "truck",
    "Mini-bus": "bus",
    "Tempo-traveller": "bus",
    "Bicycle": "bike",
    "Van": "car",
    "Others": None,
}

# IRUVD does not publish an explicit class-id map in the on-repo sample.
# This mapping assumes class 0 is pedestrian and the remaining 13 vehicle
# classes follow the same Indian vehicle taxonomy used in the paper/repo.
IRUVD_ID_TO_NAME = {
    0: "Pedestrian",
    1: "Hatchback",
    2: "Sedan",
    3: "SUV",
    4: "MUV",
    5: "Bus",
    6: "Truck",
    7: "Three-wheeler",
    8: "Two-wheeler",
    9: "LCV",
    10: "Mini-bus",
    11: "Tempo-traveller",
    12: "Bicycle",
    13: "Van",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the local dataset with downloaded Indian traffic sources."
    )
    parser.add_argument(
        "--base-dataset",
        default="/home/anuj/Desktop/image-processing/DataSet",
        help="Existing YOLO dataset root.",
    )
    parser.add_argument(
        "--uvh-root",
        default="external_data/uvh26/UVH-26-Train",
        help="UVH-26 train root containing the COCO JSON and downloaded images.",
    )
    parser.add_argument(
        "--iruvd-root",
        default="external_data/IRUVD",
        help="IRUVD sample root containing images/ and labels/.",
    )
    parser.add_argument(
        "--output-root",
        default="prepared_data/merged_indian_traffic",
        help="Where to write the normalized merged YOLO dataset.",
    )
    parser.add_argument(
        "--skip-uvh",
        action="store_true",
        help="Skip UVH-26 conversion.",
    )
    parser.add_argument(
        "--skip-iruvd",
        action="store_true",
        help="Skip IRUVD conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output dataset root if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned counts without writing files.",
    )
    return parser.parse_args()


def ensure_output_root(root: Path, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {root}. Re-run with --overwrite."
            )
        shutil.rmtree(root)

    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def normalize_coco_bbox(
    bbox: Iterable[float], image_width: int, image_height: int
) -> Optional[tuple[float, float, float, float]]:
    x, y, w, h = (float(value) for value in bbox)
    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + w))
    y2 = max(0.0, min(float(image_height), y + h))
    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    return (
        x_center / float(image_width),
        y_center / float(image_height),
        width / float(image_width),
        height / float(image_height),
    )


def clamp_normalized_box(
    x_center: float, y_center: float, width: float, height: float
) -> Optional[tuple[float, float, float, float]]:
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    if width <= 0.0 or height <= 0.0:
        return None
    return x_center, y_center, width, height


def yolo_line(class_name: str, box: tuple[float, float, float, float]) -> str:
    class_id = TARGET_CLASS_IDS[class_name]
    return (
        f"{class_id} "
        f"{box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
    )


def write_label_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def add_summary_label(
    summary: dict,
    source: str,
    split: str,
    class_name: str,
    count: int = 1,
) -> None:
    summary["sources"][source][split]["labels"][class_name] += count


def init_summary() -> dict:
    return {
        "classes": list(CURRENT_CLASSES),
        "sources": defaultdict(
            lambda: defaultdict(
                lambda: {"images": 0, "background_images": 0, "labels": Counter()}
            )
        ),
    }


def add_base_dataset(base_root: Path, output_root: Path, summary: dict, dry_run: bool) -> None:
    for split in ("train", "valid", "test"):
        image_dir = base_root / split / "images"
        label_dir = base_root / split / "labels"
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing split images directory: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing split labels directory: {label_dir}")

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            output_image_name = f"base__{image_path.name}"
            output_label_name = f"base__{image_path.stem}.txt"
            label_path = label_dir / f"{image_path.stem}.txt"

            summary["sources"]["base"][split]["images"] += 1
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    class_id = int(line.split()[0])
                    add_summary_label(
                        summary,
                        source="base",
                        split=split,
                        class_name=CURRENT_CLASSES[class_id],
                    )
            else:
                summary["sources"]["base"][split]["background_images"] += 1

            if dry_run:
                continue

            symlink_file(image_path, output_root / split / "images" / output_image_name)
            if label_path.exists():
                symlink_file(label_path, output_root / split / "labels" / output_label_name)
            else:
                write_label_file(output_root / split / "labels" / output_label_name, [])


def add_uvh26(uvh_root: Path, output_root: Path, summary: dict, dry_run: bool) -> None:
    annotation_path = uvh_root / "UVH-26-MV-Train.json"
    image_root = uvh_root / "data"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing UVH-26 annotation JSON: {annotation_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Missing UVH-26 image root: {image_root}")

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_info_by_name = {item["file_name"]: item for item in payload["images"]}
    annotations_by_image_id: Dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image_id[int(annotation["image_id"])].append(annotation)

    category_name_by_id = {
        int(category["id"]): str(category["name"])
        for category in payload["categories"]
    }

    for image_path in sorted(path for path in image_root.rglob("*") if path.is_file()):
        image_name = image_path.name
        image_info = image_info_by_name.get(image_name)
        if image_info is None:
            continue

        label_lines: list[str] = []
        for annotation in annotations_by_image_id.get(int(image_info["id"]), []):
            if int(annotation.get("iscrowd", 0)) != 0:
                continue
            category_name = category_name_by_id[int(annotation["category_id"])]
            target_name = UVH_TO_TARGET.get(category_name)
            if target_name is None:
                continue
            normalized_box = normalize_coco_bbox(
                annotation["bbox"],
                image_width=int(image_info["width"]),
                image_height=int(image_info["height"]),
            )
            if normalized_box is None:
                continue
            label_lines.append(yolo_line(target_name, normalized_box))
            add_summary_label(summary, "uvh26", "train", target_name)

        summary["sources"]["uvh26"]["train"]["images"] += 1
        if not label_lines:
            summary["sources"]["uvh26"]["train"]["background_images"] += 1

        if dry_run:
            continue

        image_output_name = f"uvh26__{image_path.parent.name}__{image_name}"
        label_output_name = f"uvh26__{image_path.parent.name}__{image_path.stem}.txt"
        symlink_file(image_path, output_root / "train" / "images" / image_output_name)
        write_label_file(output_root / "train" / "labels" / label_output_name, label_lines)


def add_iruvd(iruvd_root: Path, output_root: Path, summary: dict, dry_run: bool) -> None:
    image_dir = iruvd_root / "images"
    label_dir = iruvd_root / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing IRUVD images directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing IRUVD labels directory: {label_dir}")

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        label_lines: list[str] = []

        if label_path.exists():
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                parts = raw_line.split()
                if len(parts) != 5:
                    continue
                source_id = int(float(parts[0]))
                source_name = IRUVD_ID_TO_NAME.get(source_id)
                target_name = UVH_TO_TARGET.get(source_name) if source_name else None
                if target_name is None:
                    continue
                normalized_box = clamp_normalized_box(
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
                if normalized_box is None:
                    continue
                label_lines.append(yolo_line(target_name, normalized_box))
                add_summary_label(summary, "iruvd", "train", target_name)

        summary["sources"]["iruvd"]["train"]["images"] += 1
        if not label_lines:
            summary["sources"]["iruvd"]["train"]["background_images"] += 1

        if dry_run:
            continue

        image_output_name = f"iruvd__{image_path.name}"
        label_output_name = f"iruvd__{image_path.stem}.txt"
        symlink_file(image_path, output_root / "train" / "images" / image_output_name)
        write_label_file(output_root / "train" / "labels" / label_output_name, label_lines)


def write_dataset_yaml(output_root: Path) -> None:
    yaml_lines = [
        f"path: {output_root.resolve()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        "names:",
    ]
    for index, name in enumerate(CURRENT_CLASSES):
        yaml_lines.append(f"  {index}: {name}")
    (output_root / "dataset.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )


def freeze_summary(summary: dict) -> dict:
    frozen_sources: dict[str, dict[str, dict[str, object]]] = {}
    for source_name, split_payload in summary["sources"].items():
        frozen_sources[source_name] = {}
        for split_name, split_summary in split_payload.items():
            frozen_sources[source_name][split_name] = {
                "images": split_summary["images"],
                "background_images": split_summary["background_images"],
                "labels": dict(sorted(split_summary["labels"].items())),
            }
    return {"classes": summary["classes"], "sources": frozen_sources}


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_dataset)
    uvh_root = Path(args.uvh_root)
    iruvd_root = Path(args.iruvd_root)
    output_root = Path(args.output_root)

    if not base_root.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_root}")
    if not args.skip_uvh and not uvh_root.exists():
        raise FileNotFoundError(f"UVH-26 root not found: {uvh_root}")
    if not args.skip_iruvd and not iruvd_root.exists():
        raise FileNotFoundError(f"IRUVD root not found: {iruvd_root}")

    summary = init_summary()
    if not args.dry_run:
        ensure_output_root(output_root, overwrite=args.overwrite)

    add_base_dataset(base_root, output_root, summary, dry_run=args.dry_run)
    if not args.skip_uvh:
        add_uvh26(uvh_root, output_root, summary, dry_run=args.dry_run)
    if not args.skip_iruvd:
        add_iruvd(iruvd_root, output_root, summary, dry_run=args.dry_run)

    frozen_summary = freeze_summary(summary)
    print(json.dumps(frozen_summary, indent=2))

    if args.dry_run:
        return

    write_dataset_yaml(output_root)
    (output_root / "summary.json").write_text(
        json.dumps(frozen_summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
