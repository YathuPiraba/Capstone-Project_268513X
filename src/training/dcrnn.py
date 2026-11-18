"""Task 07 graph-window data, training, checkpoint, and reproducibility utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import PerSensorStandardizer
from src.data.windows import ForecastWindows
from src.models.dcrnn import DCGRUConfig, DirectHeadDCGRU


DEFAULT_SEED = 268513
REPORTED_STEPS = (2, 5, 11)
HORIZON_MINUTES = (15, 30, 60)


@dataclass(frozen=True)
class DCGRUTrainingConfig:
    candidate_name: str
    model: DCGRUConfig
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    max_epochs: int = 30
    patience: int = 5
    gradient_clip_norm: float = 1.0
    seed: int = DEFAULT_SEED
    deterministic_algorithms: bool = False

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model"] = self.model.as_dict()
        return result


class GraphWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """One sample contains a complete 207-node forecasting window."""

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
            raise ValueError("Task 07 requires 12 input and 12 output steps")
        if self.future_time.shape != (len(windows.X_speed), 12, 2):
            raise ValueError("future_time must have shape (windows, 12, 2)")
        if len(self.sensor_ids) != self.sensor_count or len(set(self.sensor_ids)) != self.sensor_count:
            raise ValueError("Canonical sensor IDs must be unique and match the windows")
        if standardizer.mean.size != self.sensor_count:
            raise ValueError("Standardizer does not match the sensor dimension")

    def __len__(self) -> int:
        return len(self.windows.X_speed)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw = np.asarray(self.windows.X_speed[index], dtype=np.float32)
        finite = np.isfinite(raw)
        standardized = ((raw - self.standardizer.mean) / self.standardizer.scale).astype(np.float32)
        standardized = np.where(finite, standardized, 0.0).astype(np.float32)
        history = np.stack((standardized, finite & (raw == 0), finite), axis=-1).astype(np.float32)
        targets = np.asarray(self.windows.y_speed[index], dtype=np.float32)
        return {
            "history": torch.from_numpy(history),
            "future_time": torch.from_numpy(self.future_time[index]),
            "targets_mph": torch.from_numpy(targets),
            "target_mask": torch.from_numpy(np.isfinite(targets) & (targets != 0)),
            "window_index": torch.tensor(index, dtype=torch.long),
        }


def predictions_to_mph(
    standardized: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return standardized * scale[None, None, :] + mean[None, None, :]


def masked_mae_loss(
    predictions_mph: torch.Tensor, targets_mph: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor, int]:
    if predictions_mph.shape != targets_mph.shape or mask.shape != targets_mph.shape:
        raise ValueError("Predictions, targets, and mask must have identical shapes")
    eligible = mask.bool()
    count = int(eligible.sum().item())
    if count == 0:
        return None, predictions_mph.sum().detach() * 0, 0
    error_sum = torch.abs(predictions_mph[eligible] - targets_mph[eligible]).sum()
    return error_sum / count, error_sum.detach(), count


def validation_horizon_mae(
    error_sums: Sequence[float], eligible_counts: Sequence[int]
) -> tuple[list[float], float]:
    if len(error_sums) != 3 or len(eligible_counts) != 3 or any(v <= 0 for v in eligible_counts):
        raise ValueError("All three reported horizons require eligible targets")
    values = [float(total / count) for total, count in zip(error_sums, eligible_counts)]
    return values, float(np.mean(values))


def configure_reproducibility(seed: int, deterministic_algorithms: bool) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic_algorithms
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "data_generator": generator.get_state().cpu(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return result


def restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    generator.set_state(state["data_generator"].detach().cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.detach().cpu() for value in state["torch_cuda"]])


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    *,
    model: DirectHeadDCGRU,
    optimizer: torch.optim.Optimizer,
    training_config: DCGRUTrainingConfig,
    immutable_metadata: dict[str, Any],
    epoch: int,
    global_step: int,
    best_score: float,
    best_epoch: int,
    patience_used: int,
    data_generator: torch.Generator,
    history: list[dict[str, Any]],
    amp_scaler_state: dict[str, Any] | None = None,
) -> None:
    atomic_torch_save(
        {
            "format_version": 1,
            "architecture_id": model.architecture_id,
            "training_config": training_config.as_dict(),
            "immutable_metadata": immutable_metadata,
            "parameter_count": model.parameter_count,
            "epoch": epoch,
            "global_step": global_step,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "patience_used": patience_used,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "amp_scaler_state": amp_scaler_state,
            "rng_state": capture_rng_state(data_generator),
            "history": history,
        },
        path,
    )


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    model: DirectHeadDCGRU,
    config: DCGRUTrainingConfig,
    immutable_metadata: dict[str, Any],
) -> None:
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported DCGRU checkpoint format")
    if checkpoint.get("architecture_id") != model.architecture_id:
        raise ValueError("Checkpoint architecture is incompatible")
    if checkpoint.get("training_config") != config.as_dict():
        raise ValueError("Checkpoint configuration does not match the requested run")
    if checkpoint.get("immutable_metadata") != immutable_metadata:
        raise ValueError("Checkpoint data or graph identity is incompatible")
    if checkpoint.get("parameter_count") != model.parameter_count:
        raise ValueError("Checkpoint parameter count does not match the model")


def load_checkpoint_for_resume(
    path: Path,
    *,
    model: DirectHeadDCGRU,
    optimizer: torch.optim.Optimizer,
    training_config: DCGRUTrainingConfig,
    immutable_metadata: dict[str, Any],
    data_generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    # CPU-first deserialization prevents CUDA-mapped CPU RNG byte tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint(checkpoint, model, training_config, immutable_metadata)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    _optimizer_to(optimizer, device)
    restore_rng_state(checkpoint["rng_state"], data_generator)
    return checkpoint


def load_checkpoint_for_inference(
    path: Path,
    *,
    model: DirectHeadDCGRU,
    training_config: DCGRUTrainingConfig,
    immutable_metadata: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint(checkpoint, model, training_config, immutable_metadata)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    return checkpoint
