from pathlib import Path
import inspect
import random

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.run_gru as run_gru
from scripts.run_gru import assert_reference_alignment
from src.data.preprocessing import PerSensorStandardizer, chronological_split_ranges, cyclical_time_of_day
from src.data.windows import build_forecast_windows, window_time_features
from src.models.gru import GRUConfig, SensorLocalGRU
from src.training.gru import (
    REPORTED_STEPS,
    SensorLocalWindowDataset,
    TrainingConfig,
    load_checkpoint,
    load_checkpoint_for_inference,
    masked_mae_loss,
    predictions_to_mph,
    save_checkpoint,
    seed_everything,
    validation_horizon_mae,
)


def _dataset(periods=100, sensors=3):
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="5min")
    speeds = np.arange(1, periods * sensors + 1, dtype=np.float64).reshape(periods, sensors)
    split = chronological_split_ranges(timestamps)["train"]
    scaler = PerSensorStandardizer.fit(speeds[:split.stop])
    windows = build_forecast_windows(speeds, timestamps, split)
    _, future = window_time_features(cyclical_time_of_day(timestamps), windows)
    return speeds, timestamps, split, scaler, windows, SensorLocalWindowDataset(windows, scaler, future, [f"s{i}" for i in range(sensors)])


def test_dataset_is_sensor_local_and_shapes_are_12_to_12():
    _, _, _, _, _, dataset = _dataset()
    item = dataset[1]
    assert item["history"].shape == (12, 3)
    assert item["future_time"].shape == (12, 2)
    assert item["targets_mph"].shape == item["target_mask"].shape == (12,)
    assert dataset.coordinates(4) == (1, 1)


def test_changing_other_sensor_does_not_change_selected_sensor_prediction():
    speeds, timestamps, split, scaler, windows, dataset = _dataset()
    model = SensorLocalGRU(GRUConfig(hidden_size=8, sensor_count=3, sensor_embedding_dim=4))
    selected = dataset[0]
    with torch.no_grad():
        before = model(selected["history"][None], selected["sensor_index"][None], selected["future_time"][None])
    altered = speeds.copy()
    altered[:, 1] *= -100
    altered_windows = build_forecast_windows(altered, timestamps, split)
    _, future = window_time_features(cyclical_time_of_day(timestamps), altered_windows)
    altered_dataset = SensorLocalWindowDataset(altered_windows, scaler, future, ["s0", "s1", "s2"])
    changed = altered_dataset[0]
    with torch.no_grad():
        after = model(changed["history"][None], changed["sensor_index"][None], changed["future_time"][None])
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_model_parameter_counts_are_actual_and_shapes_validated():
    assert SensorLocalGRU(GRUConfig(hidden_size=64)).parameter_count == 19769
    assert SensorLocalGRU(GRUConfig(hidden_size=128)).parameter_count == 70649
    model = SensorLocalGRU(GRUConfig(hidden_size=8, sensor_count=3))
    output = model(torch.zeros(2, 12, 3), torch.tensor([0, 2]), torch.zeros(2, 12, 2))
    assert output.shape == (2, 12)
    with pytest.raises(ValueError, match="history"):
        model(torch.zeros(2, 11, 3), torch.tensor([0, 2]), torch.zeros(2, 12, 2))


def test_horizon_indices_and_model_api_exclude_graph_inputs():
    assert REPORTED_STEPS == (2, 5, 11)
    assert tuple(step + 1 for step in REPORTED_STEPS) == (3, 6, 12)
    parameters = tuple(inspect.signature(SensorLocalGRU.forward).parameters)
    assert parameters == ("self", "history", "sensor_indices", "future_time")


def test_seeded_cpu_initialization_and_forward_are_deterministic():
    config = GRUConfig(hidden_size=8, sensor_count=3)
    seed_everything(268513)
    first = SensorLocalGRU(config)
    seed_everything(268513)
    second = SensorLocalGRU(config)
    history = torch.arange(72, dtype=torch.float32).reshape(2, 12, 3)
    sensors = torch.tensor([0, 2])
    future = torch.zeros(2, 12, 2)
    torch.testing.assert_close(first(history, sensors, future), second(history, sensors, future), rtol=0, atol=0)


def test_masks_loss_inverse_scaling_and_horizon_aggregation():
    standardized = torch.tensor([[0.0, 1.0]])
    mph = predictions_to_mph(standardized, torch.tensor([10.0]), torch.tensor([2.0]))
    torch.testing.assert_close(mph, torch.tensor([[10.0, 12.0]]))
    loss, error_sum, count = masked_mae_loss(mph, torch.tensor([[8.0, 0.0]]), torch.tensor([[True, False]]))
    assert loss.item() == error_sum.item() == 2.0 and count == 1
    empty, _, empty_count = masked_mae_loss(mph, mph, torch.zeros_like(mph, dtype=torch.bool))
    assert empty is None and empty_count == 0
    values, mean = validation_horizon_mae([10, 12, 14], [5, 4, 2])
    assert values == [2.0, 3.0, 7.0] and mean == 4.0


def test_checkpoint_restores_model_optimizer_rng_and_rejects_config(tmp_path: Path):
    config = TrainingConfig("tiny", GRUConfig(hidden_size=8, sensor_count=3), batch_size=4, max_epochs=2)
    generator = seed_everything(config.seed)
    model = SensorLocalGRU(config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, training_config=config, epoch=1, best_score=2.5, patience_used=1, data_generator=generator, history=[{"epoch": 1}])
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    expected_shuffle = torch.randperm(10, generator=generator)
    restored_generator = torch.Generator()
    restored = load_checkpoint(path, model=model, optimizer=optimizer, training_config=config, data_generator=restored_generator)
    assert restored["epoch"] == 1 and restored["parameter_count"] == model.parameter_count
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(1), expected_torch, rtol=0, atol=0)
    torch.testing.assert_close(torch.randperm(10, generator=restored_generator), expected_shuffle, rtol=0, atol=0)
    wrong = TrainingConfig("other", config.model, batch_size=4, max_epochs=2)
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, model=model, optimizer=optimizer, training_config=wrong, data_generator=restored_generator)

    seed_everything(99)
    expected = (random.random(), np.random.random(), torch.rand(1))
    seed_everything(99)
    load_checkpoint_for_inference(path, model=model, training_config=config)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2], rtol=0, atol=0)


def test_reference_alignment_accepts_exact_task_contract(tmp_path: Path):
    speeds, timestamps, split, _, windows, _ = _dataset(periods=200)
    frame = pd.DataFrame(speeds, index=timestamps, columns=["s0", "s1", "s2"])
    targets = windows.y_speed[:, [2, 5, 11], :]
    path = tmp_path / "reference.npz"
    np.savez_compressed(path, sensor_ids=np.asarray(frame.columns, dtype=str), horizon_minutes=np.asarray([15, 30, 60]), targets_mph=targets, target_timestamps=windows.target_timestamps[:, [2, 5, 11]], primary_nonzero_mask=np.isfinite(targets) & (targets != 0), finite_including_zero_mask=np.isfinite(targets))
    assert_reference_alignment(path, frame, windows)


def test_probe_only_is_mutually_exclusive(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_gru.py", "--probe-only", "--smoke"])
    with pytest.raises(SystemExit):
        run_gru.parse_args()


def test_probe_only_benchmarks_both_candidates_without_scientific_artifacts():
    _, _, _, _, _, dataset = _dataset(periods=60, sensors=3)
    report = run_gru.run_probe_only(
        {"train": dataset, "validation": dataset},
        torch.device("cpu"),
        workers=0,
        batch_size=4,
        probe_batches=1,
        warmup_batches=0,
    )
    assert report["mode"] == "probe-only"
    assert report["scientific_artifacts_written"] is False
    assert report["test_dataset_constructed_or_evaluated"] is False
    assert [row["candidate"] for row in report["candidates"]] == ["gru_h64", "gru_h128"]
    assert [row["parameter_count"] for row in report["candidates"]] == [19769, 70649]
    for row in report["candidates"]:
        assert row["steady_state_training_examples_per_second"] > 0
        assert row["estimated_full_training_epoch_seconds"] > 0
        assert row["estimated_full_validation_seconds"] > 0
        assert row["peak_allocated_gpu_bytes"] == 0


def test_probe_only_main_skips_test_alignment_and_artifact_paths(monkeypatch, tmp_path: Path, capsys):
    data_path = tmp_path / "METR-LA.csv"
    data_path.write_bytes(b"probe")
    sentinel = object()
    fake_dataset = [sentinel]
    monkeypatch.setattr(
        run_gru,
        "prepare_datasets",
        lambda split_names: (data_path, sentinel, sentinel, sentinel, {"train": fake_dataset, "validation": fake_dataset}, {"train": sentinel, "validation": sentinel}),
    )
    monkeypatch.setattr(run_gru, "assert_reference_alignment", lambda *args, **kwargs: pytest.fail("test alignment must not run"))
    monkeypatch.setattr(run_gru, "finalize_full_run", lambda *args, **kwargs: pytest.fail("full finalization must not run"))
    monkeypatch.setattr(
        run_gru,
        "run_probe_only",
        lambda *args, **kwargs: {"mode": "probe-only", "scientific_artifacts_written": False},
    )
    monkeypatch.setattr("sys.argv", ["run_gru.py", "--probe-only", "--device", "cpu", "--throughput-probe-batches", "1"])
    run_gru.main()
    output = capsys.readouterr().out
    assert '"mode": "probe-only"' in output


def test_incomplete_candidate_evidence_is_rejected():
    configs = list(run_gru.approved_configs(4))
    with pytest.raises(ValueError, match="exactly two"):
        run_gru.validate_completed_candidates([], configs)


def test_finalize_only_main_never_trains(monkeypatch, tmp_path: Path, capsys):
    data_path = tmp_path / "METR-LA.csv"
    data_path.write_bytes(b"synthetic")
    fake = [object()]
    splits = {name: type("Split", (), {"start": index, "stop": index + 1})() for index, name in enumerate(("train", "validation", "test"))}
    monkeypatch.setattr(run_gru, "prepare_datasets", lambda names: (data_path, object(), splits, object(), {name: fake for name in names}, {name: object() for name in names}))
    monkeypatch.setattr(run_gru, "assert_reference_alignment", lambda *args: None)
    monkeypatch.setattr(run_gru, "load_finalization_evidence", lambda *args: [{"candidate": "saved"}])
    monkeypatch.setattr(run_gru, "fit_candidate", lambda *args, **kwargs: pytest.fail("training must not run"))
    monkeypatch.setattr(run_gru, "finalize_full_run", lambda *args, **kwargs: {"candidate": "saved"})
    monkeypatch.setattr("sys.argv", ["run_gru.py", "--finalize-only", "--device", "cpu", "--batch-size", "4"])
    run_gru.main()
    assert '"mode": "finalize-only"' in capsys.readouterr().out


def test_synthetic_finalization_writes_expected_outputs(monkeypatch, tmp_path: Path):
    speeds, timestamps, _, scaler, windows, dataset = _dataset(periods=40, sensors=207)
    frame = pd.DataFrame(speeds, index=timestamps, columns=[f"s{i}" for i in range(207)])
    splits = chronological_split_ranges(timestamps)
    data_path = tmp_path / "data" / "raw" / "metr-la" / "METR-LA.csv"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(run_gru, "ROOT", tmp_path)
    configs = list(run_gru.approved_configs(4))
    trials = []
    for position, config in enumerate(configs, start=1):
        model = SensorLocalGRU(config.model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        generator = seed_everything(config.seed)
        checkpoint = tmp_path / "artifacts" / "task06_gru" / "checkpoints" / f"{config.candidate_name}_best.pt"
        history = [{"epoch": 0, "selection_score_mph": float(position)}]
        save_checkpoint(checkpoint, model=model, optimizer=optimizer, training_config=config, epoch=0, best_score=float(position), patience_used=0, data_generator=generator, history=history)
        trials.append({"candidate": config.candidate_name, "best_score_mph": float(position), "parameter_count": model.parameter_count, "best_checkpoint": checkpoint.relative_to(tmp_path).as_posix(), "history": history})
    selected = run_gru.finalize_full_run(trials, configs, {"test": dataset}, {"test": windows}, frame, torch.device("cpu"), 0, 4, data_path, splits)
    assert selected["candidate"] == "gru_h64"
    assert (tmp_path / "artifacts" / "task06_gru" / "test_predictions_horizons.npz").is_file()
    assert (tmp_path / "results" / "task06_gru" / "metrics.json").is_file()

