import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import target_evaluation_mask
from src.evaluation.metrics import masked_mae, masked_rmse
from src.models.baselines import HistoricalAverage, persistence_forecast


def test_persistence_repeats_last_observation_for_every_horizon() -> None:
    inputs = np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    predicted = persistence_forecast(inputs, output_length=3)
    np.testing.assert_array_equal(predicted, [[[3.0, 30.0]] * 3])


def test_historical_average_uses_training_slot_of_week_and_preserves_zeros() -> None:
    timestamps = pd.DatetimeIndex([
        "2026-01-05 00:00", "2026-01-05 00:05",
        "2026-01-12 00:00", "2026-01-12 00:05",
    ])
    speeds = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0], [6.0, 40.0]])
    model = HistoricalAverage.fit(speeds, timestamps)
    targets = np.array([[np.datetime64("2026-01-19T00:00"), np.datetime64("2026-01-19T00:05")]])
    np.testing.assert_allclose(model.predict(targets), [[[2.0, 20.0], [4.0, 30.0]]])


def test_historical_average_falls_back_to_training_sensor_mean_for_unseen_slot() -> None:
    timestamps = pd.date_range("2026-01-05", periods=2, freq="5min")
    model = HistoricalAverage.fit(np.array([[0.0, 10.0], [2.0, 30.0]]), timestamps)
    unseen = np.array([[np.datetime64("2026-01-05T00:10")]])
    np.testing.assert_allclose(model.predict(unseen), [[[1.0, 20.0]]])


def test_masked_metrics_exclude_zero_targets_only_when_policy_requests_it() -> None:
    targets = np.array([0.0, 2.0, 4.0])
    predictions = np.array([10.0, 4.0, 1.0])
    nonzero_mask = target_evaluation_mask(targets, exclude_zeros=True)
    finite_mask = target_evaluation_mask(targets, exclude_zeros=False)
    assert masked_mae(targets, predictions, nonzero_mask) == pytest.approx(2.5)
    assert masked_rmse(targets, predictions, nonzero_mask) == pytest.approx(np.sqrt(6.5))
    assert masked_mae(targets, predictions, finite_mask) == pytest.approx(5.0)
    assert masked_rmse(targets, predictions, finite_mask) == pytest.approx(np.sqrt(113 / 3))


def test_masked_metrics_reject_empty_mask_shape_mismatch_and_nonfinite_prediction() -> None:
    with pytest.raises(ValueError, match="no eligible"):
        masked_mae(np.array([1.0]), np.array([1.0]), np.array([False]))
    with pytest.raises(ValueError, match="identical shapes"):
        masked_rmse(np.array([1.0]), np.array([[1.0]]), np.array([True]))
    with pytest.raises(ValueError, match="predictions must be finite"):
        masked_mae(np.array([1.0]), np.array([np.nan]), np.array([True]))
    with pytest.raises(TypeError, match="boolean"):
        masked_mae(np.array([1.0]), np.array([1.0]), np.array([1]))
