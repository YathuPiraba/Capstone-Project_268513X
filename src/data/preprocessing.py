"""Leakage-safe splitting, scaling, masks, and time features for METR-LA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# All learned models receive the recorded speed value (standardized with the
# training-only scaler) plus explicit masks. Task 03 performs no imputation.
INPUT_ZERO_POLICY = "preserve_recorded_zeros_with_masks"


@dataclass(frozen=True)
class SplitRange:
    """A chronological half-open row range, ``[start, stop)``."""

    name: str
    start: int
    stop: int
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp

    @property
    def count(self) -> int:
        return self.stop - self.start


def chronological_split_ranges(
    timestamps: pd.DatetimeIndex,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
) -> dict[str, SplitRange]:
    """Return ordered train/validation/test ranges without shuffling.

    Boundaries use the floor of cumulative proportions. The test split receives
    all remaining rows, so no observation is dropped.
    """
    index = pd.DatetimeIndex(timestamps)
    if len(index) < 3:
        raise ValueError("At least three timestamps are required")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and strictly increasing")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test split")

    train_stop = int(len(index) * train_fraction)
    validation_stop = int(len(index) * (train_fraction + validation_fraction))
    boundaries = {
        "train": (0, train_stop),
        "validation": (train_stop, validation_stop),
        "test": (validation_stop, len(index)),
    }
    if any(stop <= start for start, stop in boundaries.values()):
        raise ValueError("Every chronological split must contain at least one row")

    return {
        name: SplitRange(name, start, stop, index[start], index[stop - 1])
        for name, (start, stop) in boundaries.items()
    }


@dataclass(frozen=True)
class PerSensorStandardizer:
    """Per-sensor standardization parameters fitted on training rows only."""

    mean: np.ndarray
    scale: np.ndarray
    valid_count: np.ndarray
    exclude_zeros: bool

    @classmethod
    def fit(
        cls,
        training_speeds: np.ndarray,
        *,
        exclude_zeros: bool = True,
    ) -> "PerSensorStandardizer":
        """Fit per-sensor statistics, excluding non-finite values and optionally zeros."""
        values = _as_speed_matrix(training_speeds)
        valid = np.isfinite(values)
        if exclude_zeros:
            valid &= values != 0
        valid_count = valid.sum(axis=0)
        if np.any(valid_count == 0):
            sensors = np.flatnonzero(valid_count == 0).tolist()
            raise ValueError(f"No valid training observations for sensor indices {sensors}")

        masked = np.where(valid, values, np.nan)
        mean = np.nanmean(masked, axis=0)
        scale = np.nanstd(masked, axis=0)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("Normalization statistics must be finite")
        # Constant sensors remain invertible and map to zero after centering.
        scale = np.where(scale == 0, 1.0, scale)
        return cls(mean=mean, scale=scale, valid_count=valid_count, exclude_zeros=exclude_zeros)

    def transform(self, speeds: np.ndarray) -> np.ndarray:
        """Apply stored training parameters without refitting."""
        values = _as_speed_array(speeds, self.mean.size)
        return (values - self.mean) / self.scale

    def inverse_transform(self, standardized: np.ndarray) -> np.ndarray:
        """Convert standardized values back to original speed units (mph)."""
        values = _as_speed_array(standardized, self.mean.size)
        return values * self.scale + self.mean


def measurement_masks(speeds: np.ndarray) -> dict[str, np.ndarray]:
    """Return explicit finite, zero, and nonzero-valid masks without altering data."""
    values = np.asarray(speeds)
    finite = np.isfinite(values)
    zero = finite & (values == 0)
    return {"finite": finite, "zero": zero, "nonzero_valid": finite & ~zero}


def prepare_speed_inputs(
    speeds: np.ndarray,
    standardizer: PerSensorStandardizer,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Standardize recorded inputs and return their masks without imputation.

    Exact zeros remain recorded zero-speed observations before standardization.
    The returned ``zero`` mask lets every learned model identify those values;
    no past or future traffic value is substituted for them.
    """
    values = _as_speed_array(speeds, standardizer.mean.size)
    return standardizer.transform(values), measurement_masks(values)


def target_evaluation_mask(
    targets: np.ndarray,
    *,
    exclude_zeros: bool = True,
) -> np.ndarray:
    """Build the common target mask used by every later forecasting model."""
    masks = measurement_masks(targets)
    return masks["nonzero_valid"] if exclude_zeros else masks["finite"]


def all_sensor_zero_mask(speeds: np.ndarray) -> np.ndarray:
    """Flag timestamps where every sensor has an observed exact-zero reading."""
    values = _as_speed_matrix(speeds)
    return np.all(np.isfinite(values) & (values == 0), axis=1)


def cyclical_time_of_day(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Return known-at-prediction-time sine/cosine features with shape ``(time, 2)``."""
    index = pd.DatetimeIndex(timestamps)
    if index.hasnans:
        raise ValueError("Time-of-day features require valid timestamps")
    seconds = (
        index.hour * 3600
        + index.minute * 60
        + index.second
        + index.microsecond / 1_000_000
    )
    angle = 2 * np.pi * np.asarray(seconds, dtype=np.float64) / (24 * 3600)
    return np.column_stack((np.sin(angle), np.cos(angle)))


def _as_speed_matrix(speeds: np.ndarray) -> np.ndarray:
    values = np.asarray(speeds, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional (time, sensor) speed array")
    if values.shape[1] == 0:
        raise ValueError("Speed array must contain at least one sensor")
    return values


def _as_speed_array(speeds: np.ndarray, sensor_count: int) -> np.ndarray:
    values = np.asarray(speeds, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != sensor_count:
        raise ValueError(f"Expected final dimension to contain {sensor_count} sensors")
    return values
