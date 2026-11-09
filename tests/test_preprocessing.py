import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    INPUT_ZERO_POLICY,
    PerSensorStandardizer,
    all_sensor_zero_mask,
    chronological_split_ranges,
    cyclical_time_of_day,
    measurement_masks,
    prepare_speed_inputs,
    target_evaluation_mask,
)


def test_chronological_split_ranges_are_contiguous_and_complete() -> None:
    timestamps = pd.date_range("2026-01-01", periods=100, freq="5min")
    splits = chronological_split_ranges(timestamps)
    assert (splits["train"].start, splits["train"].stop) == (0, 70)
    assert (splits["validation"].start, splits["validation"].stop) == (70, 80)
    assert (splits["test"].start, splits["test"].stop) == (80, 100)
    assert splits["validation"].start_timestamp == timestamps[70]
    assert splits["test"].end_timestamp == timestamps[-1]


def test_standardizer_uses_only_nonzero_training_values_and_inverts() -> None:
    train = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    scaler = PerSensorStandardizer.fit(train, exclude_zeros=True)
    np.testing.assert_allclose(scaler.mean, [3.0, 20.0])
    np.testing.assert_array_equal(scaler.valid_count, [2, 3])
    transformed = scaler.transform(train)
    np.testing.assert_allclose(scaler.inverse_transform(transformed), train)


def test_validation_and_test_changes_cannot_change_fitted_statistics() -> None:
    train = np.array([[1.0, 10.0], [3.0, 30.0]])
    held_out_a = np.array([[5.0, 50.0]])
    held_out_b = np.array([[5000.0, -999.0]])
    scaler_a = PerSensorStandardizer.fit(train)
    scaler_b = PerSensorStandardizer.fit(train)
    np.testing.assert_array_equal(scaler_a.mean, scaler_b.mean)
    np.testing.assert_array_equal(scaler_a.scale, scaler_b.scale)
    assert not np.array_equal(scaler_a.transform(held_out_a), scaler_b.transform(held_out_b))


def test_zero_variance_is_safe_and_empty_valid_sensor_fails() -> None:
    scaler = PerSensorStandardizer.fit(np.array([[5.0], [5.0]]))
    np.testing.assert_array_equal(scaler.scale, [1.0])
    np.testing.assert_array_equal(scaler.transform(np.array([[5.0]])), [[0.0]])
    with pytest.raises(ValueError, match="No valid training observations"):
        PerSensorStandardizer.fit(np.zeros((3, 1)))


def test_masks_preserve_zero_and_nonfinite_meanings() -> None:
    values = np.array([[0.0, 1.0, np.nan], [0.0, 0.0, 0.0]])
    masks = measurement_masks(values)
    np.testing.assert_array_equal(masks["zero"], [[True, False, False], [True, True, True]])
    np.testing.assert_array_equal(target_evaluation_mask(values), masks["nonzero_valid"])
    np.testing.assert_array_equal(target_evaluation_mask(values, exclude_zeros=False), masks["finite"])
    np.testing.assert_array_equal(all_sensor_zero_mask(values), [False, True])


def test_all_zero_input_window_is_preserved_and_masked_without_imputation() -> None:
    scaler = PerSensorStandardizer.fit(np.array([[10.0, 20.0], [12.0, 24.0]]))
    recorded_window = np.zeros((1, 12, 2))
    standardized, masks = prepare_speed_inputs(recorded_window, scaler)

    assert INPUT_ZERO_POLICY == "preserve_recorded_zeros_with_masks"
    np.testing.assert_array_equal(masks["zero"], np.ones_like(recorded_window, dtype=bool))
    np.testing.assert_array_equal(masks["nonzero_valid"], np.zeros_like(recorded_window, dtype=bool))
    np.testing.assert_allclose(scaler.inverse_transform(standardized), recorded_window, atol=1e-12)


def test_time_of_day_features_have_expected_cycle() -> None:
    timestamps = pd.DatetimeIndex([
        "2026-01-01 00:00", "2026-01-01 06:00", "2026-01-01 12:00", "2026-01-01 18:00"
    ])
    features = cyclical_time_of_day(timestamps)
    assert features.shape == (4, 2)
    np.testing.assert_allclose(features, [[0, 1], [1, 0], [0, -1], [-1, 0]], atol=1e-12)
