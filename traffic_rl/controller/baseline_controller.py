"""
controller/baseline_controller.py

A simple fixed-timer baseline controller that applies action '0' to all traffic lights.
By passing '0', the environment allows SUMO to handle its static pre-timed logic.
"""

class FixedTimerController:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def select_action(self, state, verbose: bool = False) -> int:
        # Hold phase, letting SUMO auto-cycle via default timetable
        return 0
