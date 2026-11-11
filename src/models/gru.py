"""Sensor-local temporal GRU for 12-step METR-LA forecasting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GRUConfig:
    """Architecture configuration saved with every checkpoint."""

    history_features: int = 3
    hidden_size: int = 64
    sensor_count: int = 207
    sensor_embedding_dim: int = 8
    output_length: int = 12
    future_time_features: int = 2

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class SensorLocalGRU(nn.Module):
    """Shared GRU whose individual forecasts contain one sensor's history only."""

    def __init__(self, config: GRUConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.GRU(
            input_size=config.history_features,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.sensor_embedding = nn.Embedding(
            config.sensor_count, config.sensor_embedding_dim
        )
        head_input = (
            config.hidden_size
            + config.sensor_embedding_dim
            + config.future_time_features
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(head_input, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        sensor_indices: torch.Tensor,
        future_time: torch.Tensor,
    ) -> torch.Tensor:
        """Return standardized predictions with shape ``(batch, 12)``."""
        expected_history = (history.shape[0], 12, self.config.history_features)
        expected_future = (
            history.shape[0], self.config.output_length, self.config.future_time_features
        )
        if tuple(history.shape) != expected_history:
            raise ValueError(f"history must have shape {expected_history}")
        if tuple(future_time.shape) != expected_future:
            raise ValueError(f"future_time must have shape {expected_future}")
        if tuple(sensor_indices.shape) != (history.shape[0],):
            raise ValueError("sensor_indices must have shape (batch,)")

        _, final_state = self.encoder(history)
        encoded = final_state[-1]
        identity = self.sensor_embedding(sensor_indices)
        static = torch.cat((encoded, identity), dim=-1)
        repeated = static[:, None, :].expand(-1, self.config.output_length, -1)
        return self.forecast_head(torch.cat((repeated, future_time), dim=-1)).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        """Return the exact number of trainable parameters in this instance."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

