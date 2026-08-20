"""
Minimal Q-network definition used for inference in the integrated project.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Fully connected Q-value estimator."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list[int] | None = None):
        super().__init__()
        hidden_layers = hidden_layers or [128, 64]

        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
