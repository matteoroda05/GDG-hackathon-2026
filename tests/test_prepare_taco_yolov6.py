import argparse
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from training.prepare_taco_yolov6 import (
    DEFAULT_CLASSES,
    ImageRecord,
    canonical_label,
    load_coco_records,
    load_mapping,
    oversample_rare_bins,
    prepare_dataset,
    split_records,
)


TACO_CATEGORIES = [
    "Aluminium foil",
    "Battery",
    "Aluminium blister pack",
    "Carded blister pack",
    "Other plastic bottle",
    "Clear plastic bottle",
    "Glass bottle",
    "Plastic bottle cap",
    "Metal bottle cap",
    "Broken glass",
    "Food Can",
    "Aerosol",
    "Drink can",
    "Toilet tube",
    "Other carton",
    "Egg carton",
    "Drink carton",
    "Corrugated carton",
    "Meal carton",
    "Pizza box",
    "Paper cup",
    "Disposable plastic cup",
    "Foam cup",
    "Glass cup",
    "Other plastic cup",
    "Food waste",
    "Glass jar",
    "Plastic lid",
    "Metal lid",
    "Other plastic",
    "Magazine paper",
    "Tissues",
    "Wrapping paper",
    "Normal paper",
    "Paper bag",
    "Plastified paper bag",
    "Plastic film",
    "Six pack rings",
    "Garbage bag",
    "Other plastic wrapper",
    "Single-use carrier bag",
    "Polypropylene bag",
    "Crisp packet",
    "Spread tub",
    "Tupperware",
    "Disposable food container",
    "Foam food container",
    "Other plastic container",
    "Plastic glooves",
    "Plastic utensils",
    "Pop tab",
    "Rope & strings",
    "Scrap metal",
    "Shoe",
    "Squeezable tube",
    "Plastic straw",
    "Paper straw",
    "Styrofoam piece",
    "Unlabeled litter",
    "Cigarette",
]


def mapping_path() -> Path:
    return Path(__file__).resolve().parents[1] / "training" / "taco6_map.yaml"


def make_fixture(root: Path) -> Path:
    batch = root / "batch_1"
    batch.mkdir()
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        (batch / name).write_bytes(b"fake image bytes")

    coco = {
        "images": [
            {"id": 1, "file_name": "batch_1/a.jpg", "width": 100, "height": 200},
            {"id": 2, "file_name": "batch_1/b.jpg", "width": 50, "height": 50},
            {"id": 3, "file_name": "batch_1/c.jpg", "width": 80, "height": 40},
        ],
        "categories": [
            {"id": 10, "name": "Clear plastic bottle"},
            {"id": 11, "name": "Food waste"},
            {"id": 12, "name": "Battery"},
        ],
        "annotations": [
            {"id": 100, "image_id": 1, "category_id": 10, "bbox": [10, 20, 30, 40]},
            {"id": 101, "image_id": 1, "category_id": 11, "bbox": [-10, -10, 20, 20]},
            {"id": 102, "image_id": 2, "category_id": 12, "bbox": [1, 1, 0, 10]},
        ],
    }
    annotations = root / "annotations.json"
    annotations.write_text(json.dumps(coco), encoding="utf-8")
    return annotations


def make_args(source_root: Path, output_dir: Path, **overrides):
    defaults = {
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "mapping": str(mapping_path()),
        "annotations": None,
        "train_ratio": 0.8,
        "seed": 2026,
        "limit_images": None,
        "copy_mode": "copy",
        "overwrite": False,
        "allow_missing_images": False,
        "no_oversample": False,
        "rare_target_fraction": 0.5,
        "max_appearances": 3,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class Taco6MappingTests(unittest.TestCase):
    def test_mapping_covers_all_taco_categories_once(self):
        classes, normalized_mapping, raw_mapping = load_mapping(mapping_path())

        self.assertEqual(classes, DEFAULT_CLASSES)
        self.assertEqual(len(raw_mapping), 60)
        self.assertEqual(set(raw_mapping.values()), set(DEFAULT_CLASSES))
        for category in TACO_CATEGORIES:
            self.assertIn(canonical_label(category), normalized_mapping)

    def test_class_ids_are_stable(self):
        classes, normalized_mapping, _ = load_mapping(mapping_path())

        self.assertEqual(classes, ["plastic", "metal", "paper", "glass", "organic", "generic"])
        self.assertEqual(normalized_mapping[canonical_label("Clear plastic bottle")], 0)
        self.assertEqual(normalized_mapping[canonical_label("Drink can")], 1)
        self.assertEqual(normalized_mapping[canonical_label("Normal paper")], 2)
        self.assertEqual(normalized_mapping[canonical_label("Glass jar")], 3)
        self.assertEqual(normalized_mapping[canonical_label("Food waste")], 4)
        self.assertEqual(normalized_mapping[canonical_label("Paper cup")], 5)


class CocoConversionTests(unittest.TestCase):
    def test_load_coco_records_normalizes_boxes_and_keeps_empty_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            annotations = make_fixture(root)
            _, mapping_by_label, _ = load_mapping(mapping_path())

            records, diagnostics = load_coco_records(root, annotations, mapping_by_label)

        self.assertEqual(diagnostics["skipped_invalid_boxes"], 1)
        self.assertEqual(len(records), 3)
        first = records[0]
        self.assertEqual(first.image_id, 1)
        self.assertEqual(len(first.labels), 2)
        self.assertEqual(first.labels[0][0], 0)
        self.assertEqual(first.labels[0][1:], (0.25, 0.2, 0.3, 0.2))
        self.assertEqual(first.labels[1][0], 4)
        self.assertEqual(first.labels[1][1:], (0.05, 0.025, 0.1, 0.05))
        self.assertEqual(records[1].labels, ())
        self.assertEqual(records[2].labels, ())

    def test_prepare_dataset_writes_yolov6_files_and_deterministic_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "source"
            root.mkdir()
            make_fixture(root)
            output = Path(tmpdir) / "prepared"
            args = make_args(
                root,
                output,
                no_oversample=True,
                train_ratio=0.67,
            )

            summary = prepare_dataset(args)
            _, mapping_by_label, _ = load_mapping(mapping_path())
            records, _ = load_coco_records(root, root / "annotations.json", mapping_by_label)
            split_a = split_records(records, 0.67, 2026)
            split_b = split_records(records, 0.67, 2026)

            label_files = sorted(output.glob("labels/*/*.txt"))
            self.assertEqual(
                [[record.image_id for record in split] for split in split_a],
                [[record.image_id for record in split] for split in split_b],
            )
            self.assertTrue((output / "dataset.yaml").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertEqual(summary["split_records"], {"train": 2, "val": 1})
            self.assertEqual(len(label_files), 3)
            for label_file in label_files:
                for line in label_file.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    self.assertEqual(len(parts), 5)
                    self.assertIn(int(parts[0]), range(6))
                    for value in parts[1:]:
                        self.assertGreaterEqual(float(value), 0.0)
                        self.assertLessEqual(float(value), 1.0)

    def test_missing_images_raise_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            annotations = make_fixture(root)
            (root / "batch_1" / "b.jpg").unlink()
            _, mapping_by_label, _ = load_mapping(mapping_path())

            with self.assertRaises(FileNotFoundError) as raised:
                load_coco_records(root, annotations, mapping_by_label)

        self.assertIn("batch_1/b.jpg", str(raised.exception))


class OversamplingTests(unittest.TestCase):
    def record(self, image_id, class_id):
        return ImageRecord(
            image_id=image_id,
            file_name=f"{image_id}.jpg",
            width=10,
            height=10,
            source_path=Path(f"{image_id}.jpg"),
            labels=((class_id, 0.5, 0.5, 0.2, 0.2),),
        )

    def test_oversampling_reaches_feasible_rare_targets_and_preserves_cap(self):
        records = [
            self.record(1, 0),
            self.record(2, 0),
            self.record(3, 0),
            self.record(4, 0),
            self.record(5, 1),
            self.record(6, 4),
        ]

        oversampled, info = oversample_rare_bins(records, 6, seed=2026)
        appearances = Counter(record.image_id for record in oversampled)

        self.assertGreater(len(oversampled), len(records))
        self.assertLessEqual(max(appearances.values()), 3)
        self.assertGreaterEqual(info["after_bin_image_counts"]["1"], info["target_image_count"])
        self.assertGreaterEqual(info["after_bin_image_counts"]["4"], info["target_image_count"])

    def test_oversampling_reports_shortfall_when_cap_blocks_target(self):
        records = [self.record(image_id, 0) for image_id in range(1, 11)]
        records.append(self.record(100, 4))

        oversampled, info = oversample_rare_bins(records, 6, seed=2026)
        appearances = Counter(record.image_id for record in oversampled)

        self.assertEqual(appearances[100], 3)
        self.assertEqual(info["target_image_count"], 5)
        self.assertEqual(info["after_bin_image_counts"]["4"], 3)
        self.assertEqual(info["shortfalls"]["4"], {"actual": 3, "target": 5})


if __name__ == "__main__":
    unittest.main()
