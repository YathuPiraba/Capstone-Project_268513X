"""Sensor-local feature construction for conventional XGBoost forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from src.data.preprocessing import PerSensorStandardizer
from src.data.windows import ForecastWindows


HORIZON_STEPS = {15: 2, 30: 5, 60: 11}
HISTORY_LENGTH = 12
FEATURE_NAMES = tuple(
    [f"speed_lag_{lag}" for lag in range(HISTORY_LENGTH, 0, -1)]
    + [f"zero_lag_{lag}" for lag in range(HISTORY_LENGTH, 0, -1)]
    + [f"finite_lag_{lag}" for lag in range(HISTORY_LENGTH, 0, -1)]
    + ["target_time_sin", "target_time_cos", "sensor_id"]
)
FEATURE_TYPES = tuple(["q"] * (len(FEATURE_NAMES) - 1) + ["c"])


@dataclass(frozen=True)
class SensorCategoryMap:
    """Stable categorical codes derived from canonical sensor ordering."""

    sensor_ids: tuple[str, ...]

    @classmethod
    def from_ids(cls, sensor_ids: Sequence[object]) -> "SensorCategoryMap":
        canonical = tuple(str(sensor_id) for sensor_id in sensor_ids)
        if not canonical:
            raise ValueError("At least one sensor ID is required")
        if len(set(canonical)) != len(canonical):
            raise ValueError("Sensor IDs must be unique")
        return cls(canonical)

    @property
    def codes(self) -> np.ndarray:
        return np.arange(len(self.sensor_ids), dtype=np.int32)

    def assert_same_ids(self, sensor_ids: Sequence[object]) -> None:
        observed = tuple(str(sensor_id) for sensor_id in sensor_ids)
        if observed != self.sensor_ids:
            raise ValueError("Sensor IDs or ordering differ from the canonical category map")


@dataclass(frozen=True)
class PooledHorizonBatch:
    """One memory-bounded block of sensor-local supervised examples."""

    features: np.ndarray
    targets_mph: np.ndarray
    sample_indices: np.ndarray
    sensor_indices: np.ndarray
    candidate_count: int
    included_count: int
    excluded_count: int


def validate_horizon_minutes(horizon_minutes: int) -> int:
    """Return the zero-based target step for one required forecast horizon."""
    try:
        return HORIZON_STEPS[int(horizon_minutes)]
    except (KeyError, ValueError) as exc:
        raise ValueError("Horizon must be one of 15, 30, or 60 minutes") from exc


def build_pooled_horizon_batch(
    windows: ForecastWindows,
    standardizer: PerSensorStandardizer,
    future_time_features: np.ndarray,
    categories: SensorCategoryMap,
    horizon_minutes: int,
    *,
    sample_start: int = 0,
    sample_stop: int | None = None,
    filter_primary_targets: bool = True,
) -> PooledHorizonBatch:
    """Build sensor-local rows without exposing neighbouring sensor histories.

    Rows are ordered forecast-origin first and sensor second. Exact-zero inputs
    remain recorded values before training-only standardization and receive an
    explicit indicator. Non-finite input values remain ``NaN`` for XGBoost and
    receive a finite indicator. The final column is a stable categorical sensor
    code; callers must pass ``FEATURE_TYPES`` to XGBoost so it is not ordinal.
    """
    horizon_step = validate_horizon_minutes(horizon_minutes)
    raw_inputs = np.asarray(windows.X_speed)
    targets = np.asarray(windows.y_speed)
    time_features = np.asarray(future_time_features)
    if raw_inputs.ndim != 3 or raw_inputs.shape[1] != HISTORY_LENGTH:
        raise ValueError("Expected input windows with shape (samples, 12, sensors)")
    if targets.ndim != 3 or targets.shape[:1] != raw_inputs.shape[:1]:
        raise ValueError("Input and target windows must contain the same samples")
    if targets.shape[1] <= horizon_step or targets.shape[2] != raw_inputs.shape[2]:
        raise ValueError("Target windows do not contain the requested horizon and sensors")
    if time_features.shape != (raw_inputs.shape[0], targets.shape[1], 2):
        raise ValueError("Future time features must have shape (samples, horizons, 2)")
    if len(categories.sensor_ids) != raw_inputs.shape[2]:
        raise ValueError("Sensor categories do not match the window sensor dimension")
    if standardizer.mean.size != raw_inputs.shape[2]:
        raise ValueError("Standardizer does not match the window sensor dimension")

    stop = raw_inputs.shape[0] if sample_stop is None else sample_stop
    if not (0 <= sample_start < stop <= raw_inputs.shape[0]):
        raise ValueError("Sample range is empty or outside the forecasting windows")

    selected_inputs = np.asarray(raw_inputs[sample_start:stop], dtype=np.float64)
    sample_count, _, sensor_count = selected_inputs.shape
    standardized = standardizer.transform(selected_inputs)
    speed_features = standardized.transpose(0, 2, 1).reshape(-1, HISTORY_LENGTH)
    zero_features = (
        (np.isfinite(selected_inputs) & (selected_inputs == 0))
        .transpose(0, 2, 1)
        .reshape(-1, HISTORY_LENGTH)
    )
    finite_features = (
        np.isfinite(selected_inputs).transpose(0, 2, 1).reshape(-1, HISTORY_LENGTH)
    )
    target_time = np.repeat(time_features[sample_start:stop, horizon_step], sensor_count, axis=0)
    sensor_indices = np.tile(categories.codes, sample_count)
    features = np.column_stack(
        (
            speed_features.astype(np.float32, copy=False),
            zero_features.astype(np.float32, copy=False),
            finite_features.astype(np.float32, copy=False),
            target_time.astype(np.float32, copy=False),
            sensor_indices.astype(np.float32, copy=False),
        )
    ).astype(np.float32, copy=False)
    horizon_targets = np.asarray(
        targets[sample_start:stop, horizon_step, :], dtype=np.float32
    ).reshape(-1)
    sample_indices = np.repeat(
        np.arange(sample_start, stop, dtype=np.int64), sensor_count
    )
    eligible = np.isfinite(horizon_targets) & (horizon_targets != 0)
    candidate_count = int(horizon_targets.size)
    included_count = int(eligible.sum()) if filter_primary_targets else candidate_count
    excluded_count = candidate_count - included_count
    if filter_primary_targets:
        if not np.any(eligible):
            raise ValueError("No finite nonzero targets are available in this batch")
        features = features[eligible]
        horizon_targets = horizon_targets[eligible]
        sample_indices = sample_indices[eligible]
        sensor_indices = sensor_indices[eligible]

    return PooledHorizonBatch(
        features=features,
        targets_mph=horizon_targets,
        sample_indices=sample_indices,
        sensor_indices=sensor_indices,
        candidate_count=candidate_count,
        included_count=included_count,
        excluded_count=excluded_count,
    )


def iter_pooled_horizon_batches(
    windows: ForecastWindows,
    standardizer: PerSensorStandardizer,
    future_time_features: np.ndarray,
    categories: SensorCategoryMap,
    horizon_minutes: int,
    *,
    batch_samples: int,
    sample_limit: int | None = None,
    filter_primary_targets: bool = True,
) -> Iterator[PooledHorizonBatch]:
    """Yield chronological, memory-bounded pooled feature blocks."""
    if batch_samples <= 0:
        raise ValueError("batch_samples must be positive")
    total_samples = windows.X_speed.shape[0]
    if sample_limit is not None:
        if sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        total_samples = min(total_samples, sample_limit)
    for start in range(0, total_samples, batch_samples):
        stop = min(start + batch_samples, total_samples)
        try:
            yield build_pooled_horizon_batch(
                windows,
                standardizer,
                future_time_features,
                categories,
                horizon_minutes,
                sample_start=start,
                sample_stop=stop,
                filter_primary_targets=filter_primary_targets,
            )
        except ValueError as exc:
            if "No finite nonzero targets" not in str(exc):
                raise

