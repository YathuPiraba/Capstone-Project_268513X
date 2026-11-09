"""Mask-aware regression metrics in original traffic-speed units."""

from __future__ import annotations

import numpy as np


def masked_mae(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Return mean absolute error over explicitly eligible observations."""
    target, prediction, eligible = _validated_inputs(targets, predictions, mask)
    return float(np.mean(np.abs(prediction[eligible] - target[eligible])))


def masked_rmse(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Return root mean squared error over explicitly eligible observations."""
    target, prediction, eligible = _validated_inputs(targets, predictions, mask)
    errors = prediction[eligible] - target[eligible]
    return float(np.sqrt(np.mean(np.square(errors))))


def _validated_inputs(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    eligible = np.asarray(mask)
    if target.shape != prediction.shape or target.shape != eligible.shape:
        raise ValueError("Targets, predictions, and mask must have identical shapes")
    if eligible.dtype.kind != "b":
        raise TypeError("Evaluation mask must be boolean")
    if not np.any(eligible):
        raise ValueError("Evaluation mask contains no eligible observations")
    if not np.all(np.isfinite(target[eligible])):
        raise ValueError("Masked targets must be finite")
    if not np.all(np.isfinite(prediction[eligible])):
        raise ValueError("Masked predictions must be finite")
    return target, prediction, eligible
