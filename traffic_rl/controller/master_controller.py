"""
controller/master_controller.py

Hierarchical Master Supervisor Agent for the Multi-Agent Traffic Control System.

Priority Architecture (highest to lowest):
  P1 [EMERGENCY CORRIDOR]  - Detected ambulance/emergency → forces a green wave
                             across ALL corridors leading towards network edge.
  P2 [CONGESTION RELIEF]   - Critically congested junction → sends hold signals to
                             upstream neighbours to prevent gridlock propagation.
  P3 [LOCAL RL AGENT]      - Falls through to each junction's trained DQN policy.

The MasterController wraps around the per-junction HybridController and augments
it with a global view of the network.  It returns the same {tls_id: int} action
dict that TrafficEnv.step() expects.

State vector (per TLS, as returned by env):
  [0] phase_progress   (float, 0-1)
  [1] is_yellow        (0 or 1)
  [2] norm_queue       (float, 0-1)
  [3] norm_max_wait    (float, 0-1)
  [4] has_emergency    (0 or 1)  ← emergency BLOCKED at a red lane
"""

from __future__ import annotations
from controller.hybrid_controller import HybridController


# Thresholds
EMERGENCY_FLAG_IDX  = 4   # index in the 5-dim state vector
QUEUE_IDX           = 2   # index for normalised queue
WAIT_IDX            = 3   # index for normalised max wait

CRITICAL_QUEUE      = 0.80  # 80 % of max_queue → critical
HOLD_QUEUE_THRESH   = 0.60  # tell upstream to hold once their queue > 60 %


class MasterController:
    """
    Hierarchical master supervisor for all junction agents.

    Parameters
    ----------
    agent            : trained DQNAgent; shared across all junctions
    tls_adjacency    : dict[str, list[str]]  mapping tls_id → list of neighbouring
                       tls_ids.  Build this once from TrafficEnv.tls_ids and the
                       network topology.
    queue_threshold  : float — normalised queue above which a junction is
                       considered "critically congested" for global relief.
    verbose          : print override reasons to stdout.
    """

    def __init__(
        self,
        agent,
        tls_adjacency: dict[str, list[str]] | None = None,
        queue_threshold: float = CRITICAL_QUEUE,
        verbose: bool = False,
    ):
        self.agent           = agent
        self.local           = HybridController(agent=agent)  # per-junction fallback
        self.tls_adjacency   = tls_adjacency or {}
        self.queue_threshold = queue_threshold
        self.verbose         = verbose

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def select_actions(self, states: dict[str, "np.ndarray"]) -> dict[str, int]:
        """
        Given the full network state, return a dict of {tls_id: action}.

        Action codes
        ------------
          0 = hold current phase
          1 = switch to next phase
        """
        actions: dict[str, int] = {}

        # ── P1: Emergency scan ────────────────────────────────────────
        emergency_tls = [
            tls for tls, s in states.items() if s[EMERGENCY_FLAG_IDX] > 0.5
        ]
        if emergency_tls:
            actions = self._emergency_corridor(states, emergency_tls)
            return actions   # highest priority — no need to continue

        # ── P2: Congestion relief ─────────────────────────────────────
        critical_tls = [
            tls for tls, s in states.items() if s[QUEUE_IDX] >= self.queue_threshold
        ]
        congestion_overrides: dict[str, int] = {}
        for crit_tls in critical_tls:
            congestion_overrides.update(self._congestion_relief(states, crit_tls))

        # ── P3: Local RL agent for every remaining junction ───────────
        for tls, state in states.items():
            if tls not in congestion_overrides:
                actions[tls] = self.local.select_action(state, verbose=self.verbose)

        # Merge congestion overrides (they win over local for affected junctions)
        actions.update(congestion_overrides)
        return actions

    # ------------------------------------------------------------------
    # P1 – Emergency Corridor
    # ------------------------------------------------------------------

    def _emergency_corridor(
        self, states: dict, emergency_tls: list[str]
    ) -> dict[str, int]:
        """
        Forces a green wave corridor for every junction that has an emergency.

        Strategy:
        • The junction WITH the blocked emergency vehicle → force SWITCH so it
          cycles towards a green phase for that vehicle (action = 1).
        • All immediate NEIGHBOURS of that junction → hold their current phase
          (action = 0) so approaching traffic does not pile into the corridor.
        • All remaining junctions → fall through to local RL policy.
        """
        if self.verbose:
            print(
                f"\033[91m[MASTER-P1] EMERGENCY CORRIDOR ACTIVE "
                f"at junction(s): {emergency_tls}\033[0m"
            )

        actions: dict[str, int] = {}
        affected_neighbours: set[str] = set()

        for emg_tls in emergency_tls:
            # Force switch at the junction blocking the ambulance
            actions[emg_tls] = 1
            # Freeze neighbouring intersections to keep the corridor clear
            for nb in self.tls_adjacency.get(emg_tls, []):
                if nb not in emergency_tls:
                    actions[nb] = 0
                    affected_neighbours.add(nb)

        # Local RL for everything else
        for tls, state in states.items():
            if tls not in actions:
                actions[tls] = self.local.select_action(state, verbose=False)

        return actions

    # ------------------------------------------------------------------
    # P2 – Congestion Relief
    # ------------------------------------------------------------------

    def _congestion_relief(
        self, states: dict, crit_tls: str
    ) -> dict[str, int]:
        """
        When junction `crit_tls` is critically congested:
        • Tell upstream neighbours whose queues are building (> HOLD threshold)
          to HOLD their phase so they momentarily throttle outflow into crit_tls.
        • crit_tls itself gets a SWITCH so it can clear its queues.
        """
        overrides: dict[str, int] = {}
        overrides[crit_tls] = 1   # crit junction: switch → try to clear

        if self.verbose:
            print(
                f"\033[93m[MASTER-P2] Congestion relief: "
                f"switching {crit_tls} and throttling upstream\033[0m"
            )

        for nb in self.tls_adjacency.get(crit_tls, []):
            nb_state = states.get(nb)
            if nb_state is not None and nb_state[QUEUE_IDX] > HOLD_QUEUE_THRESH:
                overrides[nb] = 0   # tell upstream to slow down

        return overrides

    # ------------------------------------------------------------------
    # Convenience: build adjacency from a flat list of TLS IDs
    # ------------------------------------------------------------------

    @staticmethod
    def build_grid_adjacency(
        tls_ids: list[str], grid_cols: int = 3
    ) -> dict[str, list[str]]:
        """
        Build a simple rectangular-grid adjacency map.

        tls_ids  : list of TLS IDs in row-major order (top-left → bottom-right).
        grid_cols: number of columns in the grid (default 3, i.e. 4-col grid has 3
                   internal nodes per row if the outer ring lacks TLS).

        Returns a dict mapping each TLS ID to its direct (N/S/E/W) neighbours.
        """
        n = len(tls_ids)
        adj: dict[str, list[str]] = {tls: [] for tls in tls_ids}

        for i, tls in enumerate(tls_ids):
            row, col = divmod(i, grid_cols)
            candidates = []
            if col > 0:                         candidates.append(tls_ids[i - 1])        # W
            if col < grid_cols - 1:             candidates.append(tls_ids[i + 1])        # E
            if row > 0:                         candidates.append(tls_ids[i - grid_cols])# N
            if i + grid_cols < n:               candidates.append(tls_ids[i + grid_cols])# S
            adj[tls] = candidates

        return adj
