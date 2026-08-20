from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class RLConfig:
    checkpoint_path: str = "weights/rl_latest.pt"
    state_dim: int = 16
    action_dim: int = 2
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    device: str = "auto"
    max_queue: float = 50.0
    max_wait: float = 200.0
    initial_action: int = 0


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(raw_path: str, base_dir: Path) -> str:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def load_rl_config(path: Optional[str] = None) -> RLConfig:
    if path is None:
        return RLConfig()

    config_path = Path(path).expanduser().resolve()
    payload = _read_yaml(config_path)
    rl_config = RLConfig(**payload.get("rl", {}))
    rl_config.checkpoint_path = _resolve_path(rl_config.checkpoint_path, config_path.parent.parent)
    return rl_config
