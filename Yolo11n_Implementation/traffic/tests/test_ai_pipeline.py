"""Exercises the wired-up AI chain: detections -> tracker -> DQN decision.

These bypass YOLO by feeding synthetic ``Detection`` objects straight into
``AITrafficPipeline.process_detections``, so the tracker, the bridge vector and
the trained checkpoint are all covered without needing a video asset.
"""

import unittest
from pathlib import Path

from traffic_control import AITrafficPipeline
from traffic_pipeline.models import Detection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime.yaml"
FRAME_SHAPE = (480, 640, 3)


class _StubDetector:
    """Stands in for YOLO so the tests never touch the .pt weights."""

    provider = "stub"
    inference_metadata = {"stub": True}

    def detect(self, frame):
        return []


def _box(cx, cy, class_name, size=40):
    half = size / 2.0
    return Detection(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        confidence=0.9,
        class_id=0,
        class_name=class_name,
    )


def _lane_boxes(lane, count, class_name="car"):
    """Boxes whose bottom-centre anchor lands in the requested compass lane."""
    cx, cy = {"N": (320, 80), "E": (560, 240), "S": (320, 400), "W": (80, 240)}[lane]
    return [_box(cx + index * 5, cy, class_name) for index in range(count)]


def _build_pipeline(**kwargs):
    return AITrafficPipeline(CONFIG_PATH, detector=_StubDetector(), **kwargs)


class AIPipelineTest(unittest.TestCase):
    def test_produces_a_decision_on_the_first_frame(self):
        ai = _build_pipeline()
        result = ai.process_detections([], FRAME_SHAPE, timestamp_sec=0.0)

        decision = result["rl_decision"]
        self.assertIsNotNone(decision)
        self.assertIn(decision["actionName"], {"NS_GREEN", "EW_GREEN"})
        self.assertEqual(len(decision["qValues"]), 2)
        self.assertEqual(len(decision["state"]), 16)
        self.assertEqual(
            set(decision["greenLanes"]) | set(decision["redLanes"]),
            {"N", "E", "S", "W"},
        )

    def test_tracker_assigns_detections_to_compass_lanes(self):
        ai = _build_pipeline()
        detections = _lane_boxes("N", 2) + _lane_boxes("E", 3)
        result = ai.process_detections(detections, FRAME_SHAPE, timestamp_sec=0.0)

        lanes = sorted(vehicle["lane"] for vehicle in result["vehicles"])
        self.assertEqual(lanes, ["E", "E", "E", "N", "N"])

    def test_phase_is_held_for_minimum_green_then_reconsidered(self):
        ai = _build_pipeline(min_green_sec=5.0)
        ai.process_detections([], FRAME_SHAPE, timestamp_sec=0.0)

        held = ai.process_detections([], FRAME_SHAPE, timestamp_sec=2.0)["rl_decision"]
        self.assertEqual(held["decidedAtSec"], 0.0)
        self.assertEqual(held["phaseHeldForSec"], 2.0)

        refreshed = ai.process_detections([], FRAME_SHAPE, timestamp_sec=6.0)["rl_decision"]
        self.assertEqual(refreshed["decidedAtSec"], 6.0)

    def test_ambulance_on_the_red_pair_overrides_the_minimum_green_hold(self):
        ai = _build_pipeline(min_green_sec=60.0)
        # Settle the controller onto the N/S phase.
        ai.controller.current_action = 0
        ai.process_detections([], FRAME_SHAPE, timestamp_sec=0.0)

        # An ambulance appears on the East approach, which is currently red.
        # It must be halting for the bridge vector to register it, so hold it
        # in place across a few frames.
        ambulance = _lane_boxes("E", 1, class_name="ambulance")
        for timestamp in (0.5, 1.0, 1.5, 2.0, 2.5):
            result = ai.process_detections(ambulance, FRAME_SHAPE, timestamp_sec=timestamp)

        decision = result["rl_decision"]
        self.assertEqual(decision["decisionSource"], "emergency_override")
        self.assertIn("E", decision["greenLanes"])

    def test_reports_the_detector_backend_it_is_running(self):
        ai = _build_pipeline()
        self.assertEqual(ai.provider, "stub")
        self.assertEqual(ai.describe()["inferenceProvider"], "stub")


if __name__ == "__main__":
    unittest.main()
