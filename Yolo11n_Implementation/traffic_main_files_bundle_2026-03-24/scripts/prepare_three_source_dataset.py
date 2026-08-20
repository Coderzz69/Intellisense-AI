#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional


CURRENT_CLASSES = [
    "ambulance",
    "auto-rikshaw",
    "bike",
    "bus",
    "car",
    "truck",
]
TARGET_CLASS_IDS = {name: index for index, name in enumerate(CURRENT_CLASSES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

AMBULANCE_COCO_TO_TARGET = {
    "ambulance": "ambulance",
}

NORMAL_VEHICLE2_COCO_TO_TARGET = {
    "ambulance": "ambulance",
    "army vehicle": "truck",
    "auto rickshaw": "auto-rikshaw",
    "bicycle": "bike",
    "bus": "bus",
    "car": "car",
    "garbagevan": "truck",
    "human hauler": "auto-rikshaw",
    "minibus": "bus",
    "minivan": "car",
    "motorbike": "bike",
    "pickup": "truck",
    "policecar": None,
    "rickshaw": "auto-rikshaw",
    "scooter": "bike",
    "suv": "car",
    "taxi": "car",
    "three wheelers -cng-": "auto-rikshaw",
    "truck": "truck",
    "van": "car",
}

NORMAL_VEHICLE_XML_TO_TARGET = {
    "auto": "auto-rikshaw",
    "bicycle": "bike",
    "bus": "bus",
    "car": "car",
    "concrete_mixer": "truck",
    "tempo": "truck",
    "tractor": "truck",
    "truck_tanker": "truck",
    "two_wheelers": "bike",
    "vehicle_truck": "truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a fresh YOLO dataset from three local Indian vehicle sources."
    )
    parser.add_argument(
        "--ambulance-root",
        default="/home/anuj/Desktop/image-processing/dataset_ambulance",
        help="Flat COCO dataset root containing ambulance images and _annotations.coco.json.",
    )
    parser.add_argument(
        "--normal-small-root",
        default="/home/anuj/Desktop/image-processing/dataset_normal_vehicle/Indian_vehicle_dataset",
        help="Pascal VOC dataset root containing Indian vehicle images and XML labels.",
    )
    parser.add_argument(
        "--normal-large-root",
        default="/home/anuj/Desktop/image-processing/dataset_normal_vehicle2",
        help="Flat COCO dataset root containing mixed vehicle images and _annotations.coco.json.",
    )
    parser.add_argument(
        "--output-root",
        default="prepared_data/three_source_no_police",
        help="Where to write the merged YOLO dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of each source assigned to train.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of each source assigned to valid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split assignment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output root if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned summary without writing files.",
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
    bbox: Iterable[float],
    image_width: int,
    image_height: int,
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


def normalize_voc_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> Optional[tuple[float, float, float, float]]:
    x1 = max(0.0, min(float(image_width), xmin))
    y1 = max(0.0, min(float(image_height), ymin))
    x2 = max(0.0, min(float(image_width), xmax))
    y2 = max(0.0, min(float(image_height), ymax))
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


def init_summary() -> dict:
    return {
        "classes": list(CURRENT_CLASSES),
        "sources": defaultdict(
            lambda: defaultdict(
                lambda: {
                    "images": 0,
                    "background_images": 0,
                    "labels": Counter(),
                    "ignored_labels": Counter(),
                }
            )
        ),
    }


def add_summary_label(
    summary: dict,
    source: str,
    split: str,
    class_name: str,
    count: int = 1,
) -> None:
    summary["sources"][source][split]["labels"][class_name] += count


def add_ignored_label(
    summary: dict,
    source: str,
    split: str,
    raw_label: str,
    count: int = 1,
) -> None:
    summary["sources"][source][split]["ignored_labels"][raw_label] += count


def split_assignment(
    items: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[int, str]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be < 1")

    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    total = len(ordered)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    assignments: Dict[int, str] = {}
    for index, item in enumerate(ordered):
        if index < train_end:
            split = "train"
        elif index < val_end:
            split = "valid"
        else:
            split = "test"
        assignments[int(item["id"])] = split
    return assignments


def add_coco_dataset(
    source_name: str,
    source_root: Path,
    label_mapping: dict[str, str],
    output_root: Path,
    summary: dict,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    dry_run: bool,
) -> None:
    annotations_path = source_root / "_annotations.coco.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing COCO annotations: {annotations_path}")

    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    category_name_by_id = {
        int(category["id"]): str(category.get("name", "")).strip().lower()
        for category in payload.get("categories", [])
    }
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    image_infos = [
        image_info
        for image_info in payload.get("images", [])
        if (source_root / image_info["file_name"]).exists()
    ]
    assignments = split_assignment(image_infos, train_ratio, val_ratio, seed)

    for image_info in image_infos:
        image_id = int(image_info["id"])
        split = assignments[image_id]
        image_path = source_root / image_info["file_name"]
        lines: list[str] = []

        for annotation in annotations_by_image.get(image_id, []):
            if int(annotation.get("iscrowd", 0)) != 0:
                continue
            raw_category = category_name_by_id.get(int(annotation["category_id"]), "")
            target_name = label_mapping.get(raw_category)
            if target_name is None:
                add_ignored_label(summary, source_name, split, raw_category or "unknown")
                continue
            box = normalize_coco_bbox(
                annotation["bbox"],
                int(image_info["width"]),
                int(image_info["height"]),
            )
            if box is None:
                continue
            lines.append(yolo_line(target_name, box))
            add_summary_label(summary, source_name, split, target_name)

        summary["sources"][source_name][split]["images"] += 1
        if not lines:
            summary["sources"][source_name][split]["background_images"] += 1

        if dry_run:
            continue

        output_image_name = f"{source_name}__{image_path.name}"
        output_label_name = f"{source_name}__{image_path.stem}.txt"
        symlink_file(image_path, output_root / split / "images" / output_image_name)
        write_label_file(output_root / split / "labels" / output_label_name, lines)


def add_pascal_voc_dataset(
    source_name: str,
    source_root: Path,
    label_mapping: dict[str, str],
    output_root: Path,
    summary: dict,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    dry_run: bool,
) -> None:
    xml_files = sorted(source_root.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found under: {source_root}")

    items = [{"id": index, "xml_path": str(xml_path)} for index, xml_path in enumerate(xml_files)]
    assignments = split_assignment(items, train_ratio, val_ratio, seed)

    for item in items:
        item_id = int(item["id"])
        split = assignments[item_id]
        xml_path = Path(item["xml_path"])
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_name = root.findtext("filename", "").strip()
        if not image_name:
            raise ValueError(f"Missing filename in XML: {xml_path}")
        image_path = source_root / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image referenced by XML is missing: {image_path}")

        size_node = root.find("size")
        if size_node is None:
            raise ValueError(f"Missing size in XML: {xml_path}")
        image_width = int(float(size_node.findtext("width", "0")))
        image_height = int(float(size_node.findtext("height", "0")))
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image size in XML: {xml_path}")

        lines: list[str] = []
        for obj in root.findall("object"):
            raw_category = obj.findtext("name", "").strip().lower()
            target_name = label_mapping.get(raw_category)
            if target_name is None:
                add_ignored_label(summary, source_name, split, raw_category or "unknown")
                continue

            box_node = obj.find("bndbox")
            if box_node is None:
                continue
            box = normalize_voc_bbox(
                float(box_node.findtext("xmin", "0")),
                float(box_node.findtext("ymin", "0")),
                float(box_node.findtext("xmax", "0")),
                float(box_node.findtext("ymax", "0")),
                image_width,
                image_height,
            )
            if box is None:
                continue
            lines.append(yolo_line(target_name, box))
            add_summary_label(summary, source_name, split, target_name)

        summary["sources"][source_name][split]["images"] += 1
        if not lines:
            summary["sources"][source_name][split]["background_images"] += 1

        if dry_run:
            continue

        output_image_name = f"{source_name}__{image_path.name}"
        output_label_name = f"{source_name}__{image_path.stem}.txt"
        symlink_file(image_path, output_root / split / "images" / output_image_name)
        write_label_file(output_root / split / "labels" / output_label_name, lines)


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


def materialize_summary(summary: dict) -> dict:
    sources: dict[str, dict[str, dict[str, object]]] = {}
    for source_name, split_map in summary["sources"].items():
        sources[source_name] = {}
        for split_name, values in split_map.items():
            sources[source_name][split_name] = {
                "images": int(values["images"]),
                "background_images": int(values["background_images"]),
                "labels": {
                    class_name: int(count)
                    for class_name, count in sorted(values["labels"].items())
                },
                "ignored_labels": {
                    class_name: int(count)
                    for class_name, count in sorted(values["ignored_labels"].items())
                },
            }
    return {"classes": list(summary["classes"]), "sources": sources}


def main() -> None:
    args = parse_args()
    ambulance_root = Path(args.ambulance_root)
    normal_small_root = Path(args.normal_small_root)
    normal_large_root = Path(args.normal_large_root)
    output_root = Path(args.output_root)

    if not ambulance_root.exists():
        raise FileNotFoundError(f"Ambulance dataset root not found: {ambulance_root}")
    if not normal_small_root.exists():
        raise FileNotFoundError(f"Normal-vehicle VOC dataset root not found: {normal_small_root}")
    if not normal_large_root.exists():
        raise FileNotFoundError(f"Normal-vehicle COCO dataset root not found: {normal_large_root}")

    summary = init_summary()
    if not args.dry_run:
        ensure_output_root(output_root, overwrite=args.overwrite)

    add_coco_dataset(
        source_name="ambulance_coco",
        source_root=ambulance_root,
        label_mapping=AMBULANCE_COCO_TO_TARGET,
        output_root=output_root,
        summary=summary,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    add_pascal_voc_dataset(
        source_name="normal_vehicle_voc",
        source_root=normal_small_root,
        label_mapping=NORMAL_VEHICLE_XML_TO_TARGET,
        output_root=output_root,
        summary=summary,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed + 1,
        dry_run=args.dry_run,
    )
    add_coco_dataset(
        source_name="normal_vehicle_coco",
        source_root=normal_large_root,
        label_mapping=NORMAL_VEHICLE2_COCO_TO_TARGET,
        output_root=output_root,
        summary=summary,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed + 2,
        dry_run=args.dry_run,
    )

    frozen_summary = materialize_summary(summary)
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
