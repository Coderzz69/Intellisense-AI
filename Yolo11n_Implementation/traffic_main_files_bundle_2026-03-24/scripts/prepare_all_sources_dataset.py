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
    "truck",
]

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
        description="Prepare a combined YOLO dataset from the prepared three-source base plus UVH-26 and IRUVD."
    )
    parser.add_argument(
        "--base-prepared-root",
        default="prepared_data/three_source_no_police",
        help="Prepared YOLO dataset root to copy first.",
    )
    parser.add_argument(
        "--uvh-root",
        default="external_data/uvh26/UVH-26-Train",
        help="UVH-26 train root containing the COCO JSON and images.",
    )
    parser.add_argument(
        "--iruvd-root",
        default="external_data/IRUVD",
        help="IRUVD root containing images/ and labels/.",
    )
    parser.add_argument(
        "--output-root",
        default="prepared_data/all_sources_no_police",
        help="Where to write the merged YOLO dataset.",
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
                lambda: {"images": 0, "background_images": 0, "labels": Counter()}
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


def yolo_line(class_name: str, box: tuple[float, float, float, float]) -> str:
    class_id = CURRENT_CLASSES.index(class_name)
    return (
        f"{class_id} "
        f"{box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
    )


def has_broken_symlinks(root: Path) -> bool:
    for path in root.rglob("*"):
        if path.is_symlink() and not path.exists():
            return True
    return False


def copy_existing_dataset(
    source_root: Path,
    output_root: Path,
    summary: dict,
    dry_run: bool,
) -> None:
    for split in ("train", "valid", "test"):
        image_dir = source_root / split / "images"
        label_dir = source_root / split / "labels"
        if not image_dir.exists():
            continue
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing split labels directory: {label_dir}")

        for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
            label_path = label_dir / f"{image_path.stem}.txt"
            summary["sources"]["base_prepared"][split]["images"] += 1

            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    class_id = int(line.split()[0])
                    add_summary_label(
                        summary,
                        source="base_prepared",
                        split=split,
                        class_name=CURRENT_CLASSES[class_id],
                    )
            else:
                summary["sources"]["base_prepared"][split]["background_images"] += 1

            if dry_run:
                continue

            symlink_file(image_path, output_root / split / "images" / image_path.name)
            if label_path.exists():
                symlink_file(label_path, output_root / split / "labels" / label_path.name)
            else:
                write_label_file(output_root / split / "labels" / f"{image_path.stem}.txt", [])


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
        image_info = image_info_by_name.get(image_path.name)
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
            box = normalize_coco_bbox(
                annotation["bbox"],
                image_width=int(image_info["width"]),
                image_height=int(image_info["height"]),
            )
            if box is None:
                continue
            label_lines.append(yolo_line(target_name, box))
            add_summary_label(summary, "uvh26", "train", target_name)

        summary["sources"]["uvh26"]["train"]["images"] += 1
        if not label_lines:
            summary["sources"]["uvh26"]["train"]["background_images"] += 1

        if dry_run:
            continue

        image_output_name = f"uvh26__{image_path.parent.name}__{image_path.name}"
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
                box = clamp_normalized_box(
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
                if box is None:
                    continue
                label_lines.append(yolo_line(target_name, box))
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


def materialize_summary(summary: dict) -> dict:
    sources = {}
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
            }
    return {"classes": list(summary["classes"]), "sources": sources}


def main() -> None:
    args = parse_args()
    base_prepared_root = Path(args.base_prepared_root)
    uvh_root = Path(args.uvh_root)
    iruvd_root = Path(args.iruvd_root)
    output_root = Path(args.output_root)

    if not base_prepared_root.exists():
        raise FileNotFoundError(f"Prepared dataset root not found: {base_prepared_root}")
    if has_broken_symlinks(base_prepared_root):
        raise ValueError(f"Prepared dataset contains broken symlinks: {base_prepared_root}")
    if not uvh_root.exists():
        raise FileNotFoundError(f"UVH-26 root not found: {uvh_root}")
    if not iruvd_root.exists():
        raise FileNotFoundError(f"IRUVD root not found: {iruvd_root}")

    summary = init_summary()
    if not args.dry_run:
        ensure_output_root(output_root, args.overwrite)

    copy_existing_dataset(base_prepared_root, output_root, summary, dry_run=args.dry_run)
    add_uvh26(uvh_root, output_root, summary, dry_run=args.dry_run)
    add_iruvd(iruvd_root, output_root, summary, dry_run=args.dry_run)

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
