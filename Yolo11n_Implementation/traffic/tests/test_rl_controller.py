import unittest

from traffic_control.controller import lane_signals_from_action, phase_one_hot_from_action


class RLControllerHelpersTest(unittest.TestCase):
    def test_lane_signals_for_ns_action(self):
        self.assertEqual(
            lane_signals_from_action(0),
            {"N": "green", "E": "red", "S": "green", "W": "red"},
        )

    def test_lane_signals_for_ew_action(self):
        self.assertEqual(
            lane_signals_from_action(1),
            {"N": "red", "E": "green", "S": "red", "W": "green"},
        )

    def test_phase_one_hot_uses_first_two_indices(self):
        self.assertEqual(phase_one_hot_from_action(0), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(phase_one_hot_from_action(1), [0.0, 1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
