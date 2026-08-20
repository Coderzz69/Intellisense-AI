"""
model/dqn.py

Q-Network: 2-layer fully-connected MLP.
Architecture (from model.md):
  Linear(state_dim, 128) → ReLU
  Linear(128, 64)        → ReLU
  Linear(64, action_dim)
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Fully-connected Q-value estimator."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: list = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64]

        layers = []
        in_dim = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
