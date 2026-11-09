import numpy as np
import pandas as pd

from src.data.preprocessing import chronological_split_ranges, cyclical_time_of_day
from src.data.windows import build_forecast_windows, window_time_features


def test_window_values_shapes_and_horizons_are_exact() -> None:
    timestamps = pd.date_range("2026-01-01", periods=40, freq="5min")
    speeds = np.arange(40 * 2).reshape(40, 2)
    train = chronological_split_ranges(timestamps)["train"]
    windows = build_forecast_windows(speeds, timestamps, train, input_length=12, output_length=12)

    assert windows.X_speed.shape == (5, 12, 2)
    assert windows.y_speed.shape == (5, 12, 2)
    np.testing.assert_array_equal(windows.X_speed[0], speeds[0:12])
    np.testing.assert_array_equal(windows.y_speed[0], speeds[12:24])
    assert windows.forecast_origin_timestamps[0] == timestamps[11]
    assert windows.target_timestamps[0, 2] - windows.forecast_origin_timestamps[0] == np.timedelta64(15, "m")
    assert windows.target_timestamps[0, 5] - windows.forecast_origin_timestamps[0] == np.timedelta64(30, "m")
    assert windows.target_timestamps[0, 11] - windows.forecast_origin_timestamps[0] == np.timedelta64(60, "m")


def test_validation_and_test_targets_never_cross_split_boundaries() -> None:
    timestamps = pd.date_range("2026-01-01", periods=200, freq="5min")
    speeds = np.arange(200)[:, None]
    splits = chronological_split_ranges(timestamps)
    for name in ("validation", "test"):
        split = splits[name]
        windows = build_forecast_windows(speeds, timestamps, split, input_length=12, output_length=12)
        assert windows.target_start_rows.min() >= split.start
        assert (windows.target_start_rows + 12).max() <= split.stop
        assert windows.input_start_rows[0] == split.start - 12
        assert windows.target_timestamps.min() == timestamps[split.start]


def test_training_targets_do_not_enter_later_splits() -> None:
    timestamps = pd.date_range("2026-01-01", periods=100, freq="5min")
    speeds = np.arange(100)[:, None]
    train = chronological_split_ranges(timestamps)["train"]
    windows = build_forecast_windows(speeds, timestamps, train)
    assert (windows.target_start_rows + 12).max() == train.stop
    assert windows.target_timestamps.max() == timestamps[train.stop - 1]


def test_time_features_follow_the_same_window_rows() -> None:
    timestamps = pd.date_range("2026-01-01", periods=200, freq="5min")
    speeds = np.arange(200)[:, None]
    features = cyclical_time_of_day(timestamps)
    validation = chronological_split_ranges(timestamps)["validation"]
    windows = build_forecast_windows(speeds, timestamps, validation)
    input_features, future_features = window_time_features(features, windows)
    assert input_features.shape == (validation.count - 12 + 1, 12, 2)
    assert future_features.shape == (validation.count - 12 + 1, 12, 2)
    np.testing.assert_allclose(input_features[0], features[validation.start - 12:validation.start])
    np.testing.assert_allclose(future_features[0], features[validation.start:validation.start + 12])
