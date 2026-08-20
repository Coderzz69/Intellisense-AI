"""
controller/hybrid_controller.py

A rule-based wrapper that evaluates the universal MARL environment state.
- Priority 1: Emergency Vehicle detected on a red lane -> Switch phase to serve it.
- Fallback : Ask the underlying RL Agent (DQN) for the optimal action.
"""

class HybridController:
    def __init__(self, agent, queue_threshold_ratio: float = 0.85):
        """
        agent: The trained DQN agent.
        queue_threshold_ratio is maintained for API signature compatibility.
        """
        self.agent = agent

    def select_action(self, state, verbose: bool = False) -> int:
        """
        State array structure (5-dim):
        [0] phase_progress
        [1] is_yellow
        [2] norm_queue
        [3] norm_max_wait
        [4] has_emergency (on a red lane)
        """
        has_emergency_on_red = state[4]

        # 1. Emergency Override
        if has_emergency_on_red > 0.5:
            if verbose:
                print("\033[91m[HYBRID] Emergency blocked on red! Overriding AI to switch phases.\033[0m")
            return 1 # Force switch phase to eventually turn green

        # 2. Fallback to Deep Q-Network
        return self.agent.select_action(state)
