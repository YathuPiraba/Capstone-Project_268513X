"""Forecast-window construction with explicit timestamp and split alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.preprocessing import SplitRange


@dataclass(frozen=True)
class ForecastWindows:
    """Views/copies and metadata for one split's forecasting samples."""

    X_speed: np.ndarray
    y_speed: np.ndarray
    input_timestamps: np.ndarray
    target_timestamps: np.ndarray
    forecast_origin_timestamps: np.ndarray
    input_start_rows: np.ndarray
    target_start_rows: np.ndarray


def build_forecast_windows(
    speeds: np.ndarray,
    timestamps: pd.DatetimeIndex,
    split: SplitRange,
    *,
    input_length: int = 12,
    output_length: int = 12,
) -> ForecastWindows:
    """Create windows whose targets lie completely inside ``split``.

    Earlier rows may provide input context for validation and test forecasts.
    The forecast origin is the final input timestamp; target step 1 is the next
    five-minute observation.
    """
    values = np.asarray(speeds)
    index = pd.DatetimeIndex(timestamps)
    if values.ndim != 2 or len(values) != len(index):
        raise ValueError("Speeds must have shape (timestamps, sensors)")
    if input_length <= 0 or output_length <= 0:
        raise ValueError("Input and output lengths must be positive")
    if not (0 <= split.start < split.stop <= len(index)):
        raise ValueError("Split row range is outside the supplied data")

    first_target = max(split.start, input_length)
    final_target = split.stop - output_length
    if final_target < first_target:
        raise ValueError(f"Split {split.name!r} is too short for the requested windows")

    target_starts = np.arange(first_target, final_target + 1, dtype=np.int64)
    input_starts = target_starts - input_length
    timestamp_values = index.to_numpy()
    # Sliding-window views avoid copying the large speed matrix. Target starts
    # are consecutive, so ordinary slices preserve those views.
    all_input_windows = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(values, input_length, axis=0), -1, 1
    )
    all_target_windows = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(values, output_length, axis=0), -1, 1
    )
    all_input_times = np.lib.stride_tricks.sliding_window_view(timestamp_values, input_length)
    all_target_times = np.lib.stride_tricks.sliding_window_view(timestamp_values, output_length)
    input_slice = slice(input_starts[0], input_starts[-1] + 1)
    target_slice = slice(target_starts[0], target_starts[-1] + 1)
    return ForecastWindows(
        X_speed=all_input_windows[input_slice],
        y_speed=all_target_windows[target_slice],
        input_timestamps=all_input_times[input_slice],
        target_timestamps=all_target_times[target_slice],
        forecast_origin_timestamps=timestamp_values[target_starts - 1],
        input_start_rows=input_starts,
        target_start_rows=target_starts,
    )


def window_time_features(
    features: np.ndarray,
    windows: ForecastWindows,
) -> tuple[np.ndarray, np.ndarray]:
    """Align historical and known-future time features to existing windows."""
    values = np.asarray(features)
    if values.ndim != 2:
        raise ValueError("Time features must have shape (timestamps, features)")
    input_length = windows.X_speed.shape[1]
    output_length = windows.y_speed.shape[1]
    input_rows = windows.input_start_rows[:, None] + np.arange(input_length)
    target_rows = windows.target_start_rows[:, None] + np.arange(output_length)
    if target_rows.max(initial=-1) >= len(values):
        raise ValueError("Time features do not cover all forecasting windows")
    return values[input_rows], values[target_rows]
