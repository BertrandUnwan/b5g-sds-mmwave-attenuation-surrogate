"""Neural network architecture used by the packaged surrogate.

Important:
- This architecture MUST match the one used during training, otherwise loading
  saved checkpoints will fail or produce incorrect results.
- Keep layer sizes / order stable for backward compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ShallowMLP"]


class ShallowMLP(nn.Module):
    """A small feed-forward MLP used for the 28/38 GHz attenuation surrogates."""

    def __init__(self, input_dim: int):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        # NOTE: Do not change this stack without retraining + re-exporting weights.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (N, input_dim)

        Returns:
            Tensor of shape (N,) in the model's output space.
        """
        return self.net(x).squeeze(-1)