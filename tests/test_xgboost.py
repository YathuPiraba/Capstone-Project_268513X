from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.data.preprocessing import (
    PerSensorStandardizer,
    chronological_split_ranges,
    cyclical_time_of_day,
)
from src.data.windows import build_forecast_windows, window_time_features
from src.models.xgboost import (
    FEATURE_NAMES,
    FEATURE_TYPES,
    HORIZON_STEPS,
    SensorCategoryMap,
    build_pooled_horizon_batch,
    iter_pooled_horizon_batches,
    validate_horizon_minutes,
)
from src.training.xgboost import (
    CandidateConfig,
    load_booster,
    make_quantile_dmatrix,
    predict_batches,
    save_booster,
    select_shared_candidate,
    train_with_early_stopping,
)
from scripts.run_xgboost import assert_task04_alignment


def _prepared_windows(periods: int = 100, sensors: int = 3):
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="5min")
    speeds = np.arange(1, periods * sensors + 1, dtype=np.float64).reshape(periods, sensors)
    splits = chronological_split_ranges(timestamps)
    standardizer = PerSensorStandardizer.fit(speeds[:splits["train"].stop])
    windows = build_forecast_windows(speeds, timestamps, splits["train"])
    _, future_time = window_time_features(cyclical_time_of_day(timestamps), windows)
    categories = SensorCategoryMap.from_ids([f"sensor-{index}" for index in range(sensors)])
    return timestamps, speeds, splits, standardizer, windows, future_time, categories


def test_horizon_mapping_is_exact_and_rejects_other_steps() -> None:
    assert HORIZON_STEPS == {15: 2, 30: 5, 60: 11}
    assert [validate_horizon_minutes(value) + 1 for value in (15, 30, 60)] == [3, 6, 12]
    with pytest.raises(ValueError, match="15, 30, or 60"):
        validate_horizon_minutes(45)


def test_pooled_features_are_sensor_local_and_aligned() -> None:
    _, speeds, _, standardizer, windows, future_time, categories = _prepared_windows()
    batch = build_pooled_horizon_batch(
        windows,
        standardizer,
        future_time,
        categories,
        15,
        sample_start=0,
        sample_stop=2,
    )
    assert batch.features.shape == (2 * 3, 39)
    assert len(FEATURE_NAMES) == len(FEATURE_TYPES) == 39
    assert FEATURE_TYPES[-1] == "c"
    np.testing.assert_array_equal(batch.sample_indices, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(batch.sensor_indices, [0, 1, 2, 0, 1, 2])
    expected_sensor_zero = standardizer.transform(windows.X_speed[:1])[0, :, 0]
    expected_sensor_one = standardizer.transform(windows.X_speed[:1])[0, :, 1]
    np.testing.assert_allclose(batch.features[0, :12], expected_sensor_zero)
    np.testing.assert_allclose(batch.features[1, :12], expected_sensor_one)
    np.testing.assert_allclose(batch.features[0, 36:38], future_time[0, 2])
    assert batch.targets_mph[0] == speeds[14, 0]


def test_recorded_zeros_and_nonfinite_values_keep_explicit_indicators() -> None:
    _, _, _, standardizer, windows, future_time, categories = _prepared_windows()
    raw = np.array(windows.X_speed, copy=True)
    raw[0, 0, 0] = 0.0
    raw[0, 1, 0] = np.nan
    modified = type(windows)(
        X_speed=raw,
        y_speed=windows.y_speed,
        input_timestamps=windows.input_timestamps,
        target_timestamps=windows.target_timestamps,
        forecast_origin_timestamps=windows.forecast_origin_timestamps,
        input_start_rows=windows.input_start_rows,
        target_start_rows=windows.target_start_rows,
    )
    batch = build_pooled_horizon_batch(
        modified, standardizer, future_time, categories, 15, sample_stop=1
    )
    assert batch.features[0, 12] == 1.0
    assert batch.features[0, 13] == 0.0
    assert batch.features[0, 24] == 1.0
    assert batch.features[0, 25] == 0.0
    assert np.isnan(batch.features[0, 1])


def test_primary_target_filter_removes_rows_instead_of_relabeling() -> None:
    _, _, _, standardizer, windows, future_time, categories = _prepared_windows()
    targets = np.array(windows.y_speed, copy=True)
    targets[0, 2, 0] = 0.0
    targets[0, 2, 1] = np.nan
    modified = type(windows)(
        X_speed=windows.X_speed,
        y_speed=targets,
        input_timestamps=windows.input_timestamps,
        target_timestamps=windows.target_timestamps,
        forecast_origin_timestamps=windows.forecast_origin_timestamps,
        input_start_rows=windows.input_start_rows,
        target_start_rows=windows.target_start_rows,
    )
    batch = build_pooled_horizon_batch(
        modified, standardizer, future_time, categories, 15, sample_stop=1
    )
    assert (batch.candidate_count, batch.included_count, batch.excluded_count) == (3, 1, 2)
    np.testing.assert_array_equal(batch.targets_mph, [targets[0, 2, 2]])
    assert np.all(np.isfinite(batch.targets_mph))
    assert np.all(batch.targets_mph != 0)

    targets[:, 2, :] = 0
    all_invalid = type(windows)(
        X_speed=windows.X_speed,
        y_speed=targets,
        input_timestamps=windows.input_timestamps,
        target_timestamps=windows.target_timestamps,
        forecast_origin_timestamps=windows.forecast_origin_timestamps,
        input_start_rows=windows.input_start_rows,
        target_start_rows=windows.target_start_rows,
    )
    with pytest.raises(ValueError, match="No finite nonzero targets"):
        build_pooled_horizon_batch(
            all_invalid, standardizer, future_time, categories, 15, sample_stop=1
        )


def test_category_mapping_is_stable_and_rejects_reordering() -> None:
    categories = SensorCategoryMap.from_ids(["A", "B", "C"])
    np.testing.assert_array_equal(categories.codes, [0, 1, 2])
    categories.assert_same_ids(["A", "B", "C"])
    with pytest.raises(ValueError, match="ordering"):
        categories.assert_same_ids(["B", "A", "C"])
    with pytest.raises(ValueError, match="unique"):
        SensorCategoryMap.from_ids(["A", "A"])


def test_windows_at_split_boundary_do_not_leak_targets() -> None:
    timestamps, speeds, splits, standardizer, _, _, categories = _prepared_windows(periods=200)
    validation = build_forecast_windows(speeds, timestamps, splits["validation"])
    _, future = window_time_features(cyclical_time_of_day(timestamps), validation)
    batch = build_pooled_horizon_batch(
        validation, standardizer, future, categories, 60, sample_stop=1
    )
    assert validation.input_start_rows[0] == splits["validation"].start - 12
    assert validation.target_start_rows[0] == splits["validation"].start
    assert batch.targets_mph[0] == speeds[splits["validation"].start + 11, 0]


def test_shared_selection_uses_mean_mae_then_name() -> None:
    trials = [
        {"candidate": name, "validation_mae_mph": score}
        for name, scores in (("b", [1.0, 2.0, 3.0]), ("a", [2.0, 2.0, 2.0]))
        for score in scores
    ]
    assert select_shared_candidate(trials) == "a"
    with pytest.raises(ValueError, match="exactly three"):
        select_shared_candidate(trials[:-1])


def test_small_training_is_deterministic_and_json_reload_matches(tmp_path: Path) -> None:
    _, _, _, standardizer, windows, future_time, categories = _prepared_windows()

    def factory():
        return iter_pooled_horizon_batches(
            windows,
            standardizer,
            future_time,
            categories,
            15,
            batch_samples=8,
            sample_limit=24,
        )

    train = make_quantile_dmatrix(factory, nthread=1)
    candidate = CandidateConfig("tiny", 2, 0.1, 1.0, 1.0, 1.0)
    model_a, metadata_a = train_with_early_stopping(
        train, train, candidate, nthread=1, num_boost_round=8, early_stopping_rounds=3
    )
    model_b, metadata_b = train_with_early_stopping(
        train, train, candidate, nthread=1, num_boost_round=8, early_stopping_rounds=3
    )
    prediction_a, sample_a, sensor_a = predict_batches(
        model_a, factory, iteration_count=metadata_a["boosting_rounds"]
    )
    prediction_b, _, _ = predict_batches(
        model_b, factory, iteration_count=metadata_b["boosting_rounds"]
    )
    np.testing.assert_array_equal(prediction_a, prediction_b)

    path = tmp_path / "model.json"
    save_booster(model_a, path)
    reloaded = load_booster(path)
    prediction_reloaded, sample_reloaded, sensor_reloaded = predict_batches(
        reloaded, factory, iteration_count=metadata_a["boosting_rounds"]
    )
    np.testing.assert_array_equal(prediction_a, prediction_reloaded)
    np.testing.assert_array_equal(sample_a, sample_reloaded)
    np.testing.assert_array_equal(sensor_a, sensor_reloaded)
    config = json_config = reloaded.save_config()
    assert "categorical" in config or json_config


def test_categorical_column_is_not_declared_numeric() -> None:
    matrix = xgb.DMatrix(
        np.array([[0.0] * 38 + [0.0], [1.0] * 38 + [1.0]], dtype=np.float32),
        feature_names=list(FEATURE_NAMES),
        feature_types=list(FEATURE_TYPES),
        enable_categorical=True,
    )
    assert matrix.feature_types[-1] == "c"


def test_task04_reference_alignment_is_exact(tmp_path: Path) -> None:
    timestamps, speeds, splits, _, _, _, _ = _prepared_windows(periods=200)
    frame = pd.DataFrame(speeds, index=timestamps, columns=["sensor-0", "sensor-1", "sensor-2"])
    windows = build_forecast_windows(speeds, timestamps, splits["test"])
    steps = np.asarray(list(HORIZON_STEPS.values()))
    targets = windows.y_speed[:, steps, :]
    primary = np.isfinite(targets) & (targets != 0)
    sensitivity = np.isfinite(targets)
    artifact = tmp_path / "task04.npz"
    np.savez_compressed(
        artifact,
        sensor_ids=np.asarray(frame.columns, dtype=str),
        horizon_minutes=np.asarray([15, 30, 60]),
        targets_mph=targets,
        target_timestamps=windows.target_timestamps[:, steps],
        primary_nonzero_mask=primary,
        finite_including_zero_mask=sensitivity,
    )
    assert_task04_alignment(
        artifact, frame, windows, targets, primary, sensitivity
    )
    with np.load(artifact) as reference:
        altered_targets = np.array(reference["targets_mph"], copy=True)
    altered_targets[0, 0, 0] += 1
    with pytest.raises(AssertionError):
        assert_task04_alignment(
            artifact, frame, windows, altered_targets, primary, sensitivity
        )
