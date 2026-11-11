"""Data, loss, validation, and recoverable training utilities for Task 06."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import PerSensorStandardizer
from src.data.windows import ForecastWindows
from src.models.gru import GRUConfig, SensorLocalGRU


DEFAULT_SEED = 268513
REPORTED_STEPS = (2, 5, 11)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration that must match before a checkpoint can be resumed."""

    candidate_name: str
    model: GRUConfig
    learning_rate: float = 1e-3
    batch_size: int = 4096
    max_epochs: int = 40
    patience: int = 6
    gradient_clip_norm: float = 1.0
    seed: int = DEFAULT_SEED

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model"] = self.model.as_dict()
        return result


class SensorLocalWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy mapping from flat indices to independent (window, sensor) examples."""

    def __init__(
        self,
        windows: ForecastWindows,
        standardizer: PerSensorStandardizer,
        future_time: np.ndarray,
        sensor_ids: Sequence[str],
    ) -> None:
        self.windows = windows
        self.standardizer = standardizer
        self.future_time = np.asarray(future_time, dtype=np.float32)
        self.sensor_ids = tuple(str(value) for value in sensor_ids)
        self.sensor_count = windows.X_speed.shape[2]
        if windows.X_speed.shape[1] != 12 or windows.y_speed.shape[1] != 12:
            raise ValueError("Task 06 requires 12 input and 12 output steps")
        if self.future_time.shape != (len(windows.X_speed), 12, 2):
            raise ValueError("future_time must have shape (windows, 12, 2)")
        if len(self.sensor_ids) != self.sensor_count or len(set(self.sensor_ids)) != self.sensor_count:
            raise ValueError("Canonical sensor IDs must be unique and match the windows")
        if standardizer.mean.size != self.sensor_count:
            raise ValueError("Standardizer does not match the sensor dimension")

    def __len__(self) -> int:
        return len(self.windows.X_speed) * self.sensor_count

    def coordinates(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return divmod(index, self.sensor_count)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window_index, sensor_index = self.coordinates(index)
        raw_history = np.asarray(
            self.windows.X_speed[window_index, :, sensor_index], dtype=np.float32
        )
        finite = np.isfinite(raw_history)
        standardized = (
            (raw_history - self.standardizer.mean[sensor_index])
            / self.standardizer.scale[sensor_index]
        ).astype(np.float32)
        # A neutral numeric placeholder is required by GRU kernels. The finite
        # channel preserves the observation semantics; verified METR-LA has no
        # non-finite inputs.
        standardized = np.where(finite, standardized, 0.0).astype(np.float32)
        history = np.column_stack(
            (standardized, finite & (raw_history == 0), finite)
        ).astype(np.float32)
        targets = np.asarray(
            self.windows.y_speed[window_index, :, sensor_index], dtype=np.float32
        )
        target_mask = np.isfinite(targets) & (targets != 0)
        return {
            "history": torch.from_numpy(history),
            "sensor_index": torch.tensor(sensor_index, dtype=torch.long),
            "future_time": torch.from_numpy(self.future_time[window_index]),
            "targets_mph": torch.from_numpy(targets),
            "target_mask": torch.from_numpy(target_mask),
            "mean": torch.tensor(self.standardizer.mean[sensor_index], dtype=torch.float32),
            "scale": torch.tensor(self.standardizer.scale[sensor_index], dtype=torch.float32),
            "window_index": torch.tensor(window_index, dtype=torch.long),
        }


def predictions_to_mph(
    standardized: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Apply the selected sensor's training-only inverse transform."""
    return standardized * scale[:, None] + mean[:, None]


def masked_mae_loss(
    predictions_mph: torch.Tensor,
    targets_mph: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor, int]:
    """Return differentiable MAE, absolute-error sum, and eligible count."""
    if predictions_mph.shape != targets_mph.shape or mask.shape != targets_mph.shape:
        raise ValueError("Predictions, targets, and mask must have identical shapes")
    eligible = mask.bool()
    count = int(eligible.sum().item())
    if count == 0:
        zero = predictions_mph.sum().detach() * 0
        return None, zero, 0
    error_sum = torch.abs(predictions_mph[eligible] - targets_mph[eligible]).sum()
    return error_sum / count, error_sum.detach(), count


def validation_horizon_mae(
    absolute_error_sums: Sequence[float], eligible_counts: Sequence[int]
) -> tuple[list[float], float]:
    """Return horizon MAEs and their equal-weight arithmetic mean."""
    if len(absolute_error_sums) != 3 or len(eligible_counts) != 3:
        raise ValueError("Exactly three reported horizons are required")
    if any(count <= 0 for count in eligible_counts):
        raise ValueError("Every reported horizon needs eligible validation targets")
    values = [float(total / count) for total, count in zip(absolute_error_sums, eligible_counts)]
    return values, float(np.mean(values))


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def capture_rng_state(data_generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "data_generator": data_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], data_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # A CUDA map_location migrates every tensor in the checkpoint, including
    # RNG byte tensors that these APIs require on CPU.
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    data_generator.set_state(state["data_generator"].detach().cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.detach().cpu() for value in state["torch_cuda"]])


def save_checkpoint(
    path: Path,
    *,
    model: SensorLocalGRU,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    epoch: int,
    best_score: float,
    patience_used: int,
    data_generator: torch.Generator,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "training_config": training_config.as_dict(),
            "parameter_count": model.parameter_count,
            "epoch": epoch,
            "best_score": best_score,
            "patience_used": patience_used,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state(data_generator),
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: SensorLocalGRU,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    data_generator: torch.Generator,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported GRU checkpoint format")
    if checkpoint.get("training_config") != training_config.as_dict():
        raise ValueError("Checkpoint configuration does not match the requested run")
    if checkpoint.get("parameter_count") != model.parameter_count:
        raise ValueError("Checkpoint parameter count does not match the model")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    restore_rng_state(checkpoint["rng_state"], data_generator)
    return checkpoint


def load_checkpoint_for_inference(
    path: Path,
    *,
    model: SensorLocalGRU,
    training_config: TrainingConfig,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load validated model weights without mutating optimizer or RNG state."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported GRU checkpoint format")
    if checkpoint.get("training_config") != training_config.as_dict():
        raise ValueError("Checkpoint configuration does not match the requested run")
    if checkpoint.get("parameter_count") != model.parameter_count:
        raise ValueError("Checkpoint parameter count does not match the model")
    if "model_state" not in checkpoint:
        raise ValueError("Checkpoint is missing model state")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint

