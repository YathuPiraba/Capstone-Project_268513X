"""DCRNN-style diffusion-convolutional GRU with a direct forecast head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn


GraphMode = Literal["graph", "identity"]


@dataclass(frozen=True)
class DCGRUConfig:
    """Immutable architecture configuration stored with Task 07 checkpoints."""

    history_features: int = 3
    hidden_size: int = 64
    sensor_count: int = 207
    sensor_embedding_dim: int = 8
    output_length: int = 12
    future_time_features: int = 2
    diffusion_order: int = 2
    graph_mode: GraphMode = "graph"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def random_walk_matrix(adjacency: np.ndarray) -> np.ndarray:
    """Return ``D^-1 A`` with explicit all-zero rows for zero-degree nodes."""
    values = np.asarray(adjacency, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Adjacency must be a square matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Adjacency weights must be finite and nonnegative")
    degree = values.sum(axis=1)
    inverse = np.zeros_like(degree)
    np.divide(1.0, degree, out=inverse, where=degree != 0)
    return (inverse[:, None] * values).astype(np.float32)


def directed_random_walk_supports(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return forward and reverse row-normalized directed supports."""
    values = np.asarray(adjacency)
    return random_walk_matrix(values), random_walk_matrix(values.T)


class DiffusionGraphConv(nn.Module):
    """Learn from identity plus powers of each supplied diffusion support."""

    def __init__(self, input_size: int, output_size: int, support_count: int, order: int) -> None:
        super().__init__()
        if input_size <= 0 or output_size <= 0 or support_count <= 0 or order <= 0:
            raise ValueError("Diffusion dimensions, support count, and order must be positive")
        self.input_size = input_size
        self.output_size = output_size
        self.support_count = support_count
        self.order = order
        basis_count = 1 + support_count * order
        self.linear = nn.Linear(input_size * basis_count, output_size)

    def forward(self, values: torch.Tensor, supports: Sequence[torch.Tensor]) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_size:
            raise ValueError("Diffusion input must have shape (batch, nodes, input_size)")
        if len(supports) != self.support_count:
            raise ValueError(f"Expected exactly {self.support_count} diffusion supports")
        basis = [values]
        for support in supports:
            if tuple(support.shape) != (values.shape[1], values.shape[1]):
                raise ValueError("Support shape must match the node dimension")
            diffused = values
            for _ in range(self.order):
                diffused = torch.einsum("nm,bmf->bnf", support, diffused)
                basis.append(diffused)
        return self.linear(torch.cat(basis, dim=-1))


class DCGRUCell(nn.Module):
    """GRU cell whose gates and candidate use directed graph diffusion."""

    def __init__(self, input_size: int, hidden_size: int, order: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined = input_size + hidden_size
        self.gates = DiffusionGraphConv(combined, 2 * hidden_size, 2, order)
        self.candidate = DiffusionGraphConv(combined, hidden_size, 2, order)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        supports: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        expected_state = (inputs.shape[0], inputs.shape[1], self.hidden_size)
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("Cell inputs have an invalid shape")
        if tuple(state.shape) != expected_state:
            raise ValueError(f"State must have shape {expected_state}")
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, state), dim=-1), supports)
        ).chunk(2, dim=-1)
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * state), dim=-1), supports)
        )
        return update * state + (1.0 - update) * candidate


class DirectHeadDCGRU(nn.Module):
    """Encode all sensors jointly and forecast all 12 steps without a decoder."""

    architecture_id = "dcrnn_style_direct_head_dcgru_v1"

    def __init__(
        self,
        config: DCGRUConfig,
        forward_support: np.ndarray | torch.Tensor,
        reverse_support: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        if config.graph_mode not in ("graph", "identity"):
            raise ValueError("graph_mode must be 'graph' or 'identity'")
        self.config = config
        forward = torch.as_tensor(forward_support, dtype=torch.float32)
        reverse = torch.as_tensor(reverse_support, dtype=torch.float32)
        expected = (config.sensor_count, config.sensor_count)
        if tuple(forward.shape) != expected or tuple(reverse.shape) != expected:
            raise ValueError(f"Supports must both have shape {expected}")
        if not torch.isfinite(forward).all() or not torch.isfinite(reverse).all():
            raise ValueError("Supports must be finite")
        self.register_buffer("forward_support", forward.clone())
        self.register_buffer("reverse_support", reverse.clone())
        self.register_buffer("identity_support", torch.eye(config.sensor_count))
        self.encoder_cell = DCGRUCell(
            config.history_features, config.hidden_size, config.diffusion_order
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

    def active_supports(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.graph_mode == "identity":
            # These terms are intentionally redundant. Equal nominal parameter
            # counts do not imply equal effective capacity in this control mode.
            return self.identity_support, self.identity_support
        return self.forward_support, self.reverse_support

    def forward(self, history: torch.Tensor, future_time: torch.Tensor) -> torch.Tensor:
        expected_history = (
            history.shape[0], 12, self.config.sensor_count, self.config.history_features
        )
        expected_future = (
            history.shape[0], self.config.output_length, self.config.future_time_features
        )
        if tuple(history.shape) != expected_history:
            raise ValueError(f"history must have shape {expected_history}")
        if tuple(future_time.shape) != expected_future:
            raise ValueError(f"future_time must have shape {expected_future}")
        state = history.new_zeros(
            history.shape[0], self.config.sensor_count, self.config.hidden_size
        )
        supports = self.active_supports()
        for step in range(history.shape[1]):
            state = self.encoder_cell(history[:, step], state, supports)

        sensor_indices = torch.arange(self.config.sensor_count, device=history.device)
        identity = self.sensor_embedding(sensor_indices)
        identity = identity[None, None].expand(
            history.shape[0], self.config.output_length, -1, -1
        )
        encoded = state[:, None].expand(-1, self.config.output_length, -1, -1)
        time = future_time[:, :, None].expand(-1, -1, self.config.sensor_count, -1)
        return self.forecast_head(torch.cat((encoded, identity, time), dim=-1)).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
