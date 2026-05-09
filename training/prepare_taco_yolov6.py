#!/usr/bin/env python3
"""Convert TACO COCO annotations into a YOLOv6 dataset with six bin classes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


DEFAULT_CLASSES = ["plastic", "metal", "paper", "glass", "organic", "generic"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    source_path: Path
    labels: tuple[tuple[int, float, float, float, float], ...]
    copy_index: int = 0


def canonical_label(value: str) -> str:
    """Normalize TACO labels across spaces, underscores, case, and punctuation."""
    return "".join(re.findall(r"[a-z0-9]+", value.lower()))


def load_mapping(mapping_path: Path) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """Load the JSON-compatible YAML mapping and return class IDs by source label."""
    with mapping_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    classes = data.get("classes")
    raw_mapping = data.get("mapping")
    if classes != DEFAULT_CLASSES:
        raise ValueError(
            f"{mapping_path} must define classes in this order: {DEFAULT_CLASSES}"
        )
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"{mapping_path} must contain a mapping object")

    class_to_id = {name: index for index, name in enumerate(classes)}
    normalized: dict[str, int] = {}
    for source_label, target_class in raw_mapping.items():
        if target_class not in class_to_id:
            raise ValueError(
                f"Mapping for {source_label!r} uses unknown class {target_class!r}"
            )
        key = canonical_label(source_label)
        if key in normalized:
            raise ValueError(f"Duplicate normalized mapping key for {source_label!r}")
        normalized[key] = class_to_id[target_class]
    return list(classes), normalized, dict(raw_mapping)


def find_annotations_json(source_root: Path, explicit_path: Path | None = None) -> Path:
    """Find a COCO annotations.json under source_root."""
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Annotations file not found: {explicit_path}")
        return explicit_path

    candidates = sorted(source_root.rglob("annotations.json"))
    valid: list[Path] = []
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            continue
        if {"images", "annotations", "categories"}.issubset(data):
            valid.append(candidate)

    if not valid:
        raise FileNotFoundError(
            f"No COCO annotations.json found under {source_root}. "
            "Pass --annotations if the file has another name."
        )
    if len(valid) > 1:
        options = "\n".join(f"  - {path}" for path in valid)
        raise ValueError(
            "Multiple COCO annotations.json files were found. "
            f"Pass --annotations explicitly.\n{options}"
        )
    return valid[0]


def build_image_index(source_root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_relative_path: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(source_root).as_posix()
        by_relative_path[relative] = path
        by_name[path.name].append(path)
    return by_relative_path, by_name


def resolve_image_path(
    source_root: Path,
    annotations_path: Path,
    file_name: str,
    by_relative_path: dict[str, Path],
    by_name: dict[str, list[Path]],
) -> Path | None:
    normalized_name = file_name.replace("\\", "/")
    direct_candidates = [
        source_root / normalized_name,
        annotations_path.parent / normalized_name,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    indexed = by_relative_path.get(normalized_name)
    if indexed is not None:
        return indexed

    basename_matches = by_name.get(Path(normalized_name).name, [])
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None


def coco_bbox_to_yolo(
    bbox: Iterable[float], image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    x, y, width, height = [float(value) for value in bbox]
    if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
        return None

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    center_x = (x1 + x2) / 2.0 / image_width
    center_y = (y1 + y2) / 2.0 / image_height
    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height
    return center_x, center_y, normalized_width, normalized_height


def load_coco_records(
    source_root: Path,
    annotations_path: Path,
    mapping_by_label: dict[str, int],
    allow_missing_images: bool = False,
) -> tuple[list[ImageRecord], dict[str, object]]:
    with annotations_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    categories = {category["id"]: category["name"] for category in coco["categories"]}
    unmapped = sorted(
        name for name in categories.values() if canonical_label(name) not in mapping_by_label
    )
    if unmapped:
        raise ValueError("Unmapped TACO categories: " + ", ".join(unmapped))

    category_to_class = {
        category_id: mapping_by_label[canonical_label(name)]
        for category_id, name in categories.items()
    }
    annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    by_relative_path, by_name = build_image_index(source_root)
    missing_images: list[str] = []
    skipped_invalid_boxes = 0
    records: list[ImageRecord] = []

    for image in sorted(coco["images"], key=lambda item: (int(item["id"]), item["file_name"])):
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        source_path = resolve_image_path(
            source_root, annotations_path, file_name, by_relative_path, by_name
        )
        if source_path is None:
            missing_images.append(file_name)
            if allow_missing_images:
                continue
            continue

        width = int(image["width"])
        height = int(image["height"])
        labels: list[tuple[int, float, float, float, float]] = []
        for annotation in annotations_by_image.get(image_id, []):
            converted = coco_bbox_to_yolo(annotation["bbox"], width, height)
            if converted is None:
                skipped_invalid_boxes += 1
                continue
            class_id = category_to_class[int(annotation["category_id"])]
            labels.append((class_id, *converted))
        labels.sort()
        records.append(
            ImageRecord(
                image_id=image_id,
                file_name=file_name,
                width=width,
                height=height,
                source_path=source_path,
                labels=tuple(labels),
            )
        )

    if missing_images and not allow_missing_images:
        preview = "\n".join(f"  - {name}" for name in missing_images[:20])
        remaining = "" if len(missing_images) <= 20 else f"\n  ... {len(missing_images) - 20} more"
        raise FileNotFoundError(
            f"{len(missing_images)} COCO image files could not be found:\n"
            f"{preview}{remaining}"
        )

    diagnostics = {
        "missing_images": missing_images,
        "skipped_invalid_boxes": skipped_invalid_boxes,
        "source_images": len(coco["images"]),
        "source_annotations": len(coco["annotations"]),
    }
    return records, diagnostics


def split_records(
    records: list[ImageRecord],
    train_ratio: float,
    seed: int,
    limit_images: int | None = None,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    shuffled = sorted(records, key=lambda record: (record.image_id, record.file_name))
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if limit_images is not None:
        if limit_images <= 0:
            raise ValueError("--limit-images must be positive")
        shuffled = shuffled[:limit_images]

    if len(shuffled) <= 1:
        return shuffled, []

    train_count = int(round(len(shuffled) * train_ratio))
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    return shuffled[:train_count], shuffled[train_count:]


def bins_for_record(record: ImageRecord) -> set[int]:
    return {label[0] for label in record.labels}


def count_bin_images(records: Iterable[ImageRecord], num_classes: int) -> dict[str, int]:
    counts = {str(class_id): 0 for class_id in range(num_classes)}
    for record in records:
        for class_id in bins_for_record(record):
            counts[str(class_id)] += 1
    return counts


def count_annotations(records: Iterable[ImageRecord], num_classes: int) -> dict[str, int]:
    counts = {str(class_id): 0 for class_id in range(num_classes)}
    for record in records:
        for label in record.labels:
            counts[str(label[0])] += 1
    return counts


def oversample_rare_bins(
    records: list[ImageRecord],
    num_classes: int,
    seed: int,
    target_fraction: float = 0.5,
    max_appearances: int = 3,
) -> tuple[list[ImageRecord], dict[str, object]]:
    if not records:
        return records, {
            "enabled": True,
            "target_fraction": target_fraction,
            "max_appearances": max_appearances,
            "target_image_count": 0,
            "before_bin_image_counts": {str(index): 0 for index in range(num_classes)},
            "after_bin_image_counts": {str(index): 0 for index in range(num_classes)},
            "duplicates_added": 0,
            "shortfalls": {},
        }

    before_counts = count_bin_images(records, num_classes)
    numeric_counts = {int(key): value for key, value in before_counts.items()}
    majority = max(numeric_counts.values(), default=0)
    target = int(math.ceil(majority * target_fraction))
    if majority == 0 or target == 0:
        return records, {
            "enabled": True,
            "target_fraction": target_fraction,
            "max_appearances": max_appearances,
            "target_image_count": target,
            "before_bin_image_counts": before_counts,
            "after_bin_image_counts": before_counts,
            "duplicates_added": 0,
            "shortfalls": {},
        }

    candidates_by_bin: dict[int, list[ImageRecord]] = {}
    rng = random.Random(seed)
    for class_id in range(num_classes):
        candidates = [record for record in records if class_id in bins_for_record(record)]
        candidates.sort(key=lambda record: (record.image_id, record.file_name))
        rng.shuffle(candidates)
        candidates_by_bin[class_id] = candidates

    appearances = {record.image_id: 1 for record in records}
    cursors = {class_id: 0 for class_id in range(num_classes)}
    duplicates: list[ImageRecord] = []
    current_counts = dict(numeric_counts)

    while True:
        made_progress = False
        for class_id in range(num_classes):
            if current_counts[class_id] >= target:
                continue
            candidates = candidates_by_bin[class_id]
            for _ in range(len(candidates)):
                candidate = candidates[cursors[class_id] % len(candidates)]
                cursors[class_id] += 1
                if appearances[candidate.image_id] >= max_appearances:
                    continue
                appearances[candidate.image_id] += 1
                duplicate = replace(candidate, copy_index=appearances[candidate.image_id] - 1)
                duplicates.append(duplicate)
                for duplicate_class_id in bins_for_record(duplicate):
                    current_counts[duplicate_class_id] += 1
                made_progress = True
                break
        if not made_progress:
            break
        if all(
            current_counts[class_id] >= target or not candidates_by_bin[class_id]
            for class_id in range(num_classes)
        ):
            break

    shortfalls = {
        str(class_id): {"actual": current_counts[class_id], "target": target}
        for class_id in range(num_classes)
        if current_counts[class_id] < target
    }
    oversampled = records + duplicates
    info = {
        "enabled": True,
        "target_fraction": target_fraction,
        "max_appearances": max_appearances,
        "target_image_count": target,
        "before_bin_image_counts": before_counts,
        "after_bin_image_counts": {str(key): value for key, value in current_counts.items()},
        "duplicates_added": len(duplicates),
        "shortfalls": shortfalls,
    }
    return oversampled, info


def safe_stem(file_name: str) -> str:
    stem = Path(file_name.replace("\\", "/")).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return cleaned or "image"


def output_stem(record: ImageRecord) -> str:
    suffix = "" if record.copy_index == 0 else f"_dup{record.copy_index}"
    return f"{record.image_id:08d}_{safe_stem(record.file_name)}{suffix}"


def place_image(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "hardlink":
        os.link(source, destination)
        return
    if mode == "symlink":
        os.symlink(source, destination)
        return
    raise ValueError(f"Unsupported copy mode: {mode}")


def write_labels(record: ImageRecord, destination: Path) -> None:
    lines = [
        f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        for class_id, center_x, center_y, width, height in record.labels
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_split(
    records: list[ImageRecord],
    output_dir: Path,
    split_name: str,
    copy_mode: str,
) -> dict[str, object]:
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    empty_label_files = 0
    for record in records:
        stem = output_stem(record)
        image_destination = images_dir / f"{stem}{record.source_path.suffix.lower()}"
        label_destination = labels_dir / f"{stem}.txt"
        place_image(record.source_path, image_destination, copy_mode)
        write_labels(record, label_destination)
        if not record.labels:
            empty_label_files += 1

    return {
        "images": len(records),
        "empty_label_files": empty_label_files,
    }


def write_dataset_yaml(output_dir: Path, classes: list[str]) -> None:
    names = ", ".join(json.dumps(name) for name in classes)
    content = "\n".join(
        [
            f"train: {json.dumps(str((output_dir / 'images' / 'train').resolve()))}",
            f"val: {json.dumps(str((output_dir / 'images' / 'val').resolve()))}",
            "is_coco: False",
            f"nc: {len(classes)}",
            f"names: [{names}]",
            "",
        ]
    )
    (output_dir / "dataset.yaml").write_text(content, encoding="utf-8")


def prepare_dataset(args: argparse.Namespace) -> dict[str, object]:
    source_root = Path(args.source_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mapping_path = Path(args.mapping).expanduser().resolve()
    annotations_path = find_annotations_json(
        source_root,
        Path(args.annotations).expanduser().resolve() if args.annotations else None,
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists and is not empty. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes, mapping_by_label, raw_mapping = load_mapping(mapping_path)
    records, diagnostics = load_coco_records(
        source_root,
        annotations_path,
        mapping_by_label,
        allow_missing_images=args.allow_missing_images,
    )
    train_records, val_records = split_records(
        records,
        train_ratio=args.train_ratio,
        seed=args.seed,
        limit_images=args.limit_images,
    )

    if args.no_oversample:
        oversampling_info = {
            "enabled": False,
            "target_fraction": args.rare_target_fraction,
            "max_appearances": args.max_appearances,
            "before_bin_image_counts": count_bin_images(train_records, len(classes)),
            "after_bin_image_counts": count_bin_images(train_records, len(classes)),
            "duplicates_added": 0,
            "shortfalls": {},
        }
        final_train_records = train_records
    else:
        final_train_records, oversampling_info = oversample_rare_bins(
            train_records,
            num_classes=len(classes),
            seed=args.seed,
            target_fraction=args.rare_target_fraction,
            max_appearances=args.max_appearances,
        )

    train_write_info = write_split(
        final_train_records, output_dir, "train", copy_mode=args.copy_mode
    )
    val_write_info = write_split(val_records, output_dir, "val", copy_mode=args.copy_mode)
    write_dataset_yaml(output_dir, classes)

    summary = {
        "classes": classes,
        "class_ids": {name: index for index, name in enumerate(classes)},
        "source_root": str(source_root),
        "annotations_path": str(annotations_path),
        "mapping_path": str(mapping_path),
        "mapped_taco_categories": raw_mapping,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "limit_images": args.limit_images,
        "copy_mode": args.copy_mode,
        "diagnostics": diagnostics,
        "original_records": len(records),
        "split_records": {"train": len(train_records), "val": len(val_records)},
        "written_records": {"train": train_write_info, "val": val_write_info},
        "annotation_counts": {
            "train": count_annotations(final_train_records, len(classes)),
            "val": count_annotations(val_records, len(classes)),
        },
        "bin_image_counts": {
            "train": count_bin_images(final_train_records, len(classes)),
            "val": count_bin_images(val_records, len(classes)),
        },
        "oversampling": oversampling_info,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    default_mapping = Path(__file__).with_name("taco6_map.yaml")
    parser = argparse.ArgumentParser(
        description="Prepare the TACO Trash Dataset for YOLOv6 TACO6 training."
    )
    parser.add_argument("--source-root", required=True, help="Kaggle TACO dataset root.")
    parser.add_argument("--output-dir", required=True, help="Prepared YOLOv6 dataset dir.")
    parser.add_argument(
        "--mapping",
        default=str(default_mapping),
        help="JSON-compatible YAML TACO-to-TACO6 mapping.",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Optional explicit path to COCO annotations.json.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How images are placed in the prepared dataset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--no-oversample", action="store_true")
    parser.add_argument("--rare-target-fraction", type=float, default=0.5)
    parser.add_argument("--max-appearances", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = prepare_dataset(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
