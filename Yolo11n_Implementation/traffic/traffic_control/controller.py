from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch

from traffic_pipeline.bridge import build_bridge_vector
from traffic_pipeline.models import LANE_ORDER, TrackSnapshot

from .config import RLConfig
from .dqn import QNetwork

NS_LANES = ("N", "S")
EW_LANES = ("E", "W")


@dataclass(frozen=True)
class RLDecision:
    state: List[float]
    q_values: List[float]
    action: int
    action_name: str
    lane_signals: Dict[str, str]
    green_lanes: List[str]
    red_lanes: List[str]
    decision_source: str
    current_phase_action: int


def lane_signals_from_action(action: int) -> Dict[str, str]:
    if action == 0:
        return {"N": "green", "E": "red", "S": "green", "W": "red"}
    if action == 1:
        return {"N": "red", "E": "green", "S": "red", "W": "green"}
    raise ValueError(f"Unsupported action: {action}")


def phase_one_hot_from_action(action: int) -> List[float]:
    phase = [0.0, 0.0, 0.0, 0.0]
    if 0 <= action < len(phase):
        phase[action] = 1.0
    return phase


def _resolve_device(device: str) -> torch.device:
    normalized = str(device).strip().lower()
    if normalized in {"", "auto", "none"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "0" and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized.startswith("cuda") and torch.cuda.is_available():
        return torch.device(normalized)
    return torch.device("cpu")


class RLTrafficController:
    def __init__(self, config: RLConfig):
        self.config = config
        self.device = _resolve_device(config.device)
        self.current_action = int(config.initial_action)
        self.model = QNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_layers=config.hidden_layers,
        ).to(self.device)
        checkpoint = torch.load(config.checkpoint_path, map_location=self.device)
        policy_state = checkpoint.get("policy_net", checkpoint)
        self.model.load_state_dict(policy_state)
        self.model.eval()

    def raw_bridge_from_snapshots(
        self,
        snapshots: Iterable[TrackSnapshot],
    ) -> List[float]:
        raw = build_bridge_vector(
            snapshots=snapshots,
            current_green_lane=None,
            lane_order=LANE_ORDER,
        )
        # Keep the bridge-vector shape stable but let the controller own the phase segment.
        return raw[:8] + [0.0, 0.0, 0.0, 0.0] + raw[12:16]

    def aggregate_state(self, raw_vectors: List[List[float]]) -> List[float]:
        if not raw_vectors:
            queue = [0.0] * 4
            wait = [0.0] * 4
            emergency = [0.0] * 4
        else:
            count = float(len(raw_vectors))
            queue = [
                round(min(sum(vector[index] for vector in raw_vectors) / count / self.config.max_queue, 1.0), 4)
                for index in range(4)
            ]
            wait = [
                round(min(sum(vector[index + 4] for vector in raw_vectors) / count / self.config.max_wait, 1.0), 4)
                for index in range(4)
            ]
            emergency = [
                round(max(vector[index + 12] for vector in raw_vectors), 4)
                for index in range(4)
            ]
        phase = phase_one_hot_from_action(self.current_action)
        return queue + wait + phase + emergency

    def decide(self, raw_vectors: List[List[float]]) -> RLDecision:
        state = self.aggregate_state(raw_vectors)
        q_values = self._predict_q_values(state)
        override_action = self._priority_override(state)
        action = override_action if override_action is not None else int(max(range(len(q_values)), key=q_values.__getitem__))
        decision_source = "emergency_override" if override_action is not None else "rl_model"
        lane_signals = lane_signals_from_action(action)
        green_lanes = [lane for lane, signal in lane_signals.items() if signal == "green"]
        red_lanes = [lane for lane, signal in lane_signals.items() if signal == "red"]
        phase_action = self.current_action
        self.current_action = action
        return RLDecision(
            state=state,
            q_values=q_values,
            action=action,
            action_name="NS_GREEN" if action == 0 else "EW_GREEN",
            lane_signals=lane_signals,
            green_lanes=green_lanes,
            red_lanes=red_lanes,
            decision_source=decision_source,
            current_phase_action=phase_action,
        )

    def _predict_q_values(self, state: List[float]) -> List[float]:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).cpu().tolist()
        return [round(float(value), 6) for value in q_values]

    def _priority_override(self, state: List[float]) -> int | None:
        emergency = state[12:16]
        queue = state[0:4]
        wait = state[4:8]

        ns_emergency = emergency[0] > 0.5 or emergency[2] > 0.5
        ew_emergency = emergency[1] > 0.5 or emergency[3] > 0.5

        if not ns_emergency and not ew_emergency:
            return None

        if self.current_action == 0 and ew_emergency and not ns_emergency:
            return 1
        if self.current_action == 1 and ns_emergency and not ew_emergency:
            return 0

        if ns_emergency and ew_emergency:
            ns_score = queue[0] + queue[2] + wait[0] + wait[2]
            ew_score = queue[1] + queue[3] + wait[1] + wait[3]
            return 0 if ns_score >= ew_score else 1

        if ns_emergency:
            return 0
        return 1
