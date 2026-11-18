from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.run_dcrnn as run_dcrnn
from src.data.preprocessing import PerSensorStandardizer, chronological_split_ranges, cyclical_time_of_day
from src.data.windows import build_forecast_windows, window_time_features
from src.models.dcrnn import (
    DCGRUConfig,
    DCGRUCell,
    DiffusionGraphConv,
    DirectHeadDCGRU,
    directed_random_walk_supports,
    random_walk_matrix,
)
from src.training.dcrnn import (
    DCGRUTrainingConfig,
    GraphWindowDataset,
    configure_reproducibility,
    load_checkpoint_for_inference,
    load_checkpoint_for_resume,
    masked_mae_loss,
    predictions_to_mph,
    save_checkpoint,
    validation_horizon_mae,
)


def _supports(nodes=3):
    adjacency = np.eye(nodes, dtype=np.float32)
    if nodes > 1:
        adjacency[0, 1] = 1.0
    return directed_random_walk_supports(adjacency)


def _model(hidden=8, mode="graph", nodes=3):
    forward, reverse = _supports(nodes)
    config = DCGRUConfig(hidden_size=hidden, sensor_count=nodes, sensor_embedding_dim=4, graph_mode=mode)
    return DirectHeadDCGRU(config, forward, reverse)


def _dataset(periods=80, sensors=3):
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="5min")
    speeds = np.arange(1, periods * sensors + 1, dtype=np.float64).reshape(periods, sensors)
    splits = chronological_split_ranges(timestamps)
    scaler = PerSensorStandardizer.fit(speeds[:splits["train"].stop])
    windows = build_forecast_windows(speeds, timestamps, splits["train"])
    _, future = window_time_features(cyclical_time_of_day(timestamps), windows)
    return scaler, windows, GraphWindowDataset(windows, scaler, future, [f"s{i}" for i in range(sensors)])


def test_random_walk_supports_preserve_direction_and_handle_zero_degree():
    adjacency = np.array([[1.0, 3.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    forward, reverse = directed_random_walk_supports(adjacency)
    np.testing.assert_allclose(forward[0], [0.25, 0.75, 0.0])
    np.testing.assert_allclose(forward[2], 0.0)
    np.testing.assert_allclose(reverse, random_walk_matrix(adjacency.T))
    assert not np.array_equal(forward, reverse)


def test_diffusion_convolution_and_cell_shapes():
    forward, reverse = _supports()
    supports = [torch.from_numpy(forward), torch.from_numpy(reverse)]
    convolution = DiffusionGraphConv(5, 7, support_count=2, order=2)
    assert convolution(torch.zeros(2, 3, 5), supports).shape == (2, 3, 7)
    cell = DCGRUCell(3, 8, order=2)
    assert cell(torch.zeros(2, 3, 3), torch.zeros(2, 3, 8), supports).shape == (2, 3, 8)


def test_model_contract_embedding_placement_and_parameter_count():
    forward, reverse = directed_random_walk_supports(np.eye(207, dtype=np.float32))
    model = DirectHeadDCGRU(DCGRUConfig(), forward, reverse)
    assert model.parameter_count == 71033
    assert model.sensor_embedding.embedding_dim == 8
    assert model.encoder_cell.input_size == 3
    output = model(torch.zeros(2, 12, 207, 3), torch.zeros(2, 12, 2))
    assert output.shape == (2, 12, 207)


def test_cpu_initialization_and_inference_are_deterministic():
    configure_reproducibility(268513, False)
    first = _model()
    configure_reproducibility(268513, False)
    second = _model()
    history = torch.arange(2 * 12 * 3 * 3, dtype=torch.float32).reshape(2, 12, 3, 3) / 100
    future = torch.zeros(2, 12, 2)
    torch.testing.assert_close(first(history, future), second(history, future), rtol=0, atol=0)


def test_graph_mode_uses_neighbours_and_identity_mode_does_not():
    configure_reproducibility(7, False)
    graph = _model(mode="graph")
    identity = _model(mode="identity")
    identity.load_state_dict(graph.state_dict())
    base = torch.zeros(1, 12, 3, 3)
    changed = base.clone()
    changed[:, :, 1, 0] = 4.0
    future = torch.zeros(1, 12, 2)
    graph_difference = (graph(base, future)[:, :, 0] - graph(changed, future)[:, :, 0]).abs().max()
    identity_difference = (identity(base, future)[:, :, 0] - identity(changed, future)[:, :, 0]).abs().max()
    assert graph_difference > 1e-7
    assert identity_difference == 0


def test_disconnected_node_does_not_influence_other_component():
    adjacency = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
    forward, reverse = directed_random_walk_supports(adjacency)
    model = DirectHeadDCGRU(DCGRUConfig(hidden_size=8, sensor_count=3, sensor_embedding_dim=4), forward, reverse)
    base = torch.zeros(1, 12, 3, 3)
    changed = base.clone(); changed[:, :, 2, 0] = 10
    future = torch.zeros(1, 12, 2)
    torch.testing.assert_close(model(base, future)[:, :, :2], model(changed, future)[:, :, :2], rtol=0, atol=0)


def test_changing_graph_changes_predictions_with_fixed_parameters():
    configure_reproducibility(12, False)
    forward, reverse = _supports()
    first = DirectHeadDCGRU(DCGRUConfig(hidden_size=8, sensor_count=3, sensor_embedding_dim=4), forward, reverse)
    identity = np.eye(3, dtype=np.float32)
    second = DirectHeadDCGRU(first.config, identity, identity)
    second.load_state_dict(first.state_dict())
    # Restore distinct supports after state_dict copied registered buffers.
    second.forward_support.copy_(torch.eye(3)); second.reverse_support.copy_(torch.eye(3))
    history = torch.randn(1, 12, 3, 3); future = torch.zeros(1, 12, 2)
    assert not torch.allclose(first(history, future), second(history, future))


def test_graph_window_dataset_and_loss_match_task06_policy():
    scaler, _, dataset = _dataset()
    item = dataset[0]
    assert item["history"].shape == (12, 3, 3)
    assert item["targets_mph"].shape == item["target_mask"].shape == (12, 3)
    standardized = torch.zeros(1, 12, 3)
    mph = predictions_to_mph(standardized, torch.tensor(scaler.mean, dtype=torch.float32), torch.tensor(scaler.scale, dtype=torch.float32))
    assert mph.shape == standardized.shape
    loss, error_sum, count = masked_mae_loss(mph, item["targets_mph"][None], item["target_mask"][None])
    assert loss is not None and loss.item() == pytest.approx(error_sum.item() / count)
    empty, _, empty_count = masked_mae_loss(mph, mph, torch.zeros_like(mph, dtype=torch.bool))
    assert empty is None and empty_count == 0
    values, mean = validation_horizon_mae([10, 12, 14], [5, 4, 2])
    assert values == [2, 3, 7] and mean == 4


def test_training_configuration_matches_selected_task06_optimizer_contract():
    config = DCGRUTrainingConfig("dcgru_h64_k2", DCGRUConfig())
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.0
    assert config.gradient_clip_norm == 1.0
    assert config.max_epochs == 30 and config.patience == 5


def test_checkpoint_round_trip_rng_isolation_and_identity_rejection(tmp_path: Path):
    config = DCGRUTrainingConfig("tiny", DCGRUConfig(hidden_size=8, sensor_count=3, sensor_embedding_dim=4), batch_size=2)
    metadata = {"data_sha256": "data", "adjacency_sha256": "graph", "graph_mode": "graph"}
    generator = configure_reproducibility(config.seed, False)
    model = _model(); optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Populate Adam state.
    model(torch.zeros(1, 12, 3, 3), torch.zeros(1, 12, 2)).sum().backward(); optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, training_config=config, immutable_metadata=metadata, epoch=0, global_step=1, best_score=2.0, best_epoch=0, patience_used=0, data_generator=generator, history=[{"epoch": 0}])
    assert path.is_file() and not path.with_name(path.name + ".tmp").exists()
    expected_python = random.random(); expected_numpy = np.random.random(); expected_torch = torch.rand(1); expected_shuffle = torch.randperm(6, generator=generator)
    restored_generator = torch.Generator()
    restored = load_checkpoint_for_resume(path, model=model, optimizer=optimizer, training_config=config, immutable_metadata=metadata, data_generator=restored_generator, device=torch.device("cpu"))
    assert restored["global_step"] == 1
    assert random.random() == expected_python and np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(1), expected_torch, rtol=0, atol=0)
    torch.testing.assert_close(torch.randperm(6, generator=restored_generator), expected_shuffle, rtol=0, atol=0)

    configure_reproducibility(99, False)
    expected = (random.random(), np.random.random(), torch.rand(1))
    configure_reproducibility(99, False)
    load_checkpoint_for_inference(path, model=model, training_config=config, immutable_metadata=metadata, device=torch.device("cpu"))
    assert random.random() == expected[0] and np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint_for_inference(path, model=model, training_config=config, immutable_metadata={**metadata, "adjacency_sha256": "wrong"}, device=torch.device("cpu"))


def test_corrupt_checkpoint_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.pt"; path.write_bytes(b"not a checkpoint")
    config = DCGRUTrainingConfig("tiny", DCGRUConfig(hidden_size=8, sensor_count=3, sensor_embedding_dim=4))
    with pytest.raises(Exception):
        load_checkpoint_for_inference(path, model=_model(), training_config=config, immutable_metadata={}, device=torch.device("cpu"))


def test_runtime_projection_requires_owner_inputs_for_feasibility():
    unknown = run_dcrnn.runtime_projection(4.0, None, None, 0.25)
    assert unknown["status"] == "unknown"
    assert run_dcrnn.runtime_projection(4.0, 8.0, 8.0, 0.25)["status"] == "feasible"
    assert run_dcrnn.runtime_projection(8.0, 8.0, 20.0, 0.25)["status"] == "infeasible"


def test_finalize_only_is_isolated_from_training_preparation(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_dcrnn.py", "--finalize-only", "--device", "cpu"])
    monkeypatch.setattr(run_dcrnn, "prepare_data", lambda *args, **kwargs: pytest.fail("generic train/validation preparation must not run"))
    monkeypatch.setattr(run_dcrnn, "finalize_completed_run", lambda args, device: {"mode": "finalize-only", "trained": False})
    run_dcrnn.main()
    assert '"trained": false' in capsys.readouterr().out.lower()


def test_probe_only_requests_train_and_validation_without_test(monkeypatch, tmp_path: Path, capsys):
    data_path = tmp_path / "METR-LA.csv"; data_path.write_bytes(b"probe")
    splits = {name: type("Split", (), {"start": index, "stop": index + 1})() for index, name in enumerate(("train", "validation", "test"))}
    scaler = type("Scaler", (), {"mean": np.zeros(207), "scale": np.ones(207)})()
    observed = []
    def fake_prepare(names):
        observed.append(names)
        datasets = {name: [object()] for name in names}
        return data_path, object(), splits, scaler, np.eye(207, dtype=np.float32), np.eye(207, dtype=np.float32), {"adjacency_sha256": "a", "sensor_order_sha256": "s", "normalization": "n", "supports": ["f", "r"], "diffusion_order": 2}, datasets, {}
    monkeypatch.setattr(run_dcrnn, "prepare_data", fake_prepare)
    monkeypatch.setattr(run_dcrnn, "benchmark", lambda *args, **kwargs: {"measured": True})
    monkeypatch.setattr("sys.argv", ["run_dcrnn.py", "--probe-only", "--device", "cpu", "--batch-size", "1"])
    run_dcrnn.main()
    assert observed == [("train", "validation")]
    assert '"test_loaded": false' in capsys.readouterr().out.lower()


def test_real_graph_integrity_and_manifest_contract():
    frame = run_dcrnn.load_speed_data(run_dcrnn.DATA_DIR / "METR-LA.csv")
    _, forward, reverse, manifest = run_dcrnn.inspect_graph(frame)
    assert manifest["adjacency_sha256"] == run_dcrnn.EXPECTED_ADJACENCY_SHA256
    assert manifest["nonzero_entries"] == 1722
    np.testing.assert_allclose(forward.sum(1), 1.0, atol=1e-6)
    np.testing.assert_allclose(reverse.sum(1), 1.0, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-specific RNG restoration requires CUDA")
def test_cuda_rng_state_restoration_when_available():
    generator = configure_reproducibility(123, False)
    state = torch.cuda.get_rng_state_all()
    assert all(value.device.type == "cpu" for value in state)
    # The full resume path is covered on CPU; this guards the API contract that
    # prevented Task 06 finalization.
    torch.cuda.set_rng_state_all([value.cpu() for value in state])
    assert generator.get_state().device.type == "cpu"
