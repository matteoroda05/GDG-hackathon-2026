import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_APP = REPO_ROOT / "base-app"
sys.path.insert(0, str(BASE_APP))

from webapp import (  # noqa: E402
    DEFAULT_BIN_SETTINGS,
    TRASH_CLASSES,
    DashboardState,
    build_state_payload,
    compute_fullness_percent,
    load_bin_config,
)


class BinMonitorConfigTests(unittest.TestCase):
    def test_missing_config_uses_defaults_for_all_classes(self):
        missing_path = Path(tempfile.mkdtemp()) / "missing.json"

        config = load_bin_config(missing_path)

        self.assertEqual(set(config), set(TRASH_CLASSES))
        for class_name in TRASH_CLASSES:
            self.assertEqual(
                config[class_name]["max_average_depth_mm"],
                DEFAULT_BIN_SETTINGS["max_average_depth_mm"],
            )
            self.assertEqual(
                config[class_name]["empty_threshold_percent"],
                DEFAULT_BIN_SETTINGS["empty_threshold_percent"],
            )


class BinMonitorSummaryTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            class_name: {
                "max_average_depth_mm": 1000.0,
                "empty_threshold_percent": 80.0,
            }
            for class_name in TRASH_CLASSES
        }

    def test_top_camera_fullness_rises_as_depth_gets_smaller(self):
        self.assertEqual(compute_fullness_percent(1000.0, 1000.0), 0.0)
        self.assertEqual(compute_fullness_percent(200.0, 1000.0), 80.0)
        self.assertEqual(compute_fullness_percent(-50.0, 1000.0), 100.0)

    def test_missing_depth_has_no_fullness(self):
        state = DashboardState(depth_mm=None)

        payload = build_state_payload(state, self.config, "plastic")

        self.assertIsNone(payload["fullness_percent"])
        self.assertFalse(payload["is_full_enough_to_empty"])

    def test_unknown_expected_class_falls_back_to_generic(self):
        state = DashboardState(depth_mm=500.0)

        payload = build_state_payload(state, self.config, "unknown")

        self.assertEqual(payload["expected_class"], "generic")
        self.assertEqual(payload["fullness_percent"], 50.0)

    def test_any_mismatch_is_reported_as_wrong_object(self):
        state = DashboardState(
            detections=[
                {"label": "plastic", "confidence": 0.92},
                {"label": "metal", "confidence": 0.81},
            ],
            depth_mm=200.0,
        )

        payload = build_state_payload(state, self.config, "plastic")

        self.assertTrue(payload["has_wrong_object"])
        self.assertEqual(payload["wrong_objects"], [{"label": "metal", "confidence": 0.81}])
        self.assertTrue(payload["is_full_enough_to_empty"])


if __name__ == "__main__":
    unittest.main()
