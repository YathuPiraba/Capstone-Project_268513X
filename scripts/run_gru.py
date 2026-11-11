"""Run guarded Task 06 GRU smoke checks or owner-operated full training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.inspection import load_speed_data  # noqa: E402
from src.data.preprocessing import PerSensorStandardizer, chronological_split_ranges, cyclical_time_of_day  # noqa: E402
from src.data.windows import build_forecast_windows, window_time_features  # noqa: E402
from src.evaluation.metrics import masked_mae, masked_rmse  # noqa: E402
from src.models.gru import GRUConfig, SensorLocalGRU  # noqa: E402
from src.training.gru import (  # noqa: E402
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


HORIZON_MINUTES = (15, 30, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 06 sensor-local GRU")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--probe-only", action="store_true")
    mode.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--throughput-probe-batches", type=int, default=0)
    parser.add_argument("--probe-warmup-batches", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def canonical_sensor_ids(path: Path, observed) -> tuple[str, ...]:
    sensor_ids = tuple(value.strip() for value in path.read_text(encoding="utf-8").strip().split(","))
    if sensor_ids != tuple(str(value) for value in observed):
        raise ValueError("Canonical sensor IDs or ordering do not match METR-LA columns")
    return sensor_ids


def approved_configs(batch_size: int) -> tuple[TrainingConfig, TrainingConfig]:
    return (
        TrainingConfig("gru_h64", GRUConfig(hidden_size=64), batch_size=batch_size),
        TrainingConfig("gru_h128", GRUConfig(hidden_size=128), batch_size=batch_size),
    )


def prepare_datasets(split_names=("train", "validation", "test")):
    data_path = ROOT / "data" / "raw" / "metr-la" / "METR-LA.csv"
    frame = load_speed_data(data_path)
    speeds = frame.to_numpy(dtype=np.float64, copy=False)
    splits = chronological_split_ranges(frame.index)
    standardizer = PerSensorStandardizer.fit(speeds[:splits["train"].stop], exclude_zeros=True)
    ids = canonical_sensor_ids(ROOT / "data" / "raw" / "metr-la" / "graph_sensor_ids.txt", frame.columns)
    time_features = cyclical_time_of_day(frame.index)
    datasets = {}
    windows_by_split = {}
    for name in split_names:
        split = splits[name]
        windows = build_forecast_windows(speeds, frame.index, split)
        _, future = window_time_features(time_features, windows)
        datasets[name] = SensorLocalWindowDataset(windows, standardizer, future, ids)
        windows_by_split[name] = windows
    return data_path, frame, splits, standardizer, datasets, windows_by_split


def make_loader(dataset, *, batch_size: int, shuffle: bool, generator, workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def move_batch(batch, device: torch.device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_epoch(model, loader, optimizer, device, clip_norm: float) -> tuple[float, int]:
    model.train()
    error_sum = 0.0
    eligible_count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        standardized = model(batch["history"], batch["sensor_index"], batch["future_time"])
        predictions = predictions_to_mph(standardized, batch["mean"], batch["scale"])
        loss, detached_sum, count = masked_mae_loss(
            predictions, batch["targets_mph"], batch["target_mask"]
        )
        if loss is None:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        error_sum += float(detached_sum.cpu())
        eligible_count += count
    if eligible_count == 0:
        raise ValueError("Training epoch contained no eligible targets")
    return error_sum / eligible_count, eligible_count


@torch.no_grad()
def validate(model, loader, device) -> tuple[list[float], float]:
    model.eval()
    sums = np.zeros(3, dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        standardized = model(batch["history"], batch["sensor_index"], batch["future_time"])
        predictions = predictions_to_mph(standardized, batch["mean"], batch["scale"])
        for position, step in enumerate(REPORTED_STEPS):
            mask = batch["target_mask"][:, step]
            sums[position] += torch.abs(
                predictions[:, step][mask] - batch["targets_mph"][:, step][mask]
            ).sum().item()
            counts[position] += int(mask.sum().item())
    return validation_horizon_mae(sums.tolist(), counts.tolist())


def throughput_probe(model, loader, device, batches: int) -> dict[str, float]:
    if batches <= 0:
        return {}
    model.eval()
    examples = 0
    started = time.perf_counter()
    with torch.no_grad():
        for number, raw_batch in enumerate(loader, start=1):
            batch = move_batch(raw_batch, device)
            model(batch["history"], batch["sensor_index"], batch["future_time"])
            examples += len(batch["sensor_index"])
            if number >= batches:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"batches": number, "examples": examples, "seconds": elapsed, "examples_per_second": examples / elapsed}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def benchmark_candidate(
    config: TrainingConfig,
    train_dataset,
    validation_dataset,
    device: torch.device,
    workers: int,
    *,
    probe_batches: int,
    warmup_batches: int,
) -> dict[str, object]:
    """Benchmark unchanged training/validation work without saving artifacts."""
    if probe_batches <= 0 or warmup_batches < 0:
        raise ValueError("Probe batches must be positive and warm-up batches cannot be negative")
    generator = seed_everything(config.seed)
    model = SensorLocalGRU(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        generator=generator, workers=workers,
    )
    validation_loader = make_loader(
        validation_dataset, batch_size=config.batch_size, shuffle=False,
        generator=None, workers=workers,
    )

    def training_step(raw_batch) -> int:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        standardized = model(batch["history"], batch["sensor_index"], batch["future_time"])
        predictions = predictions_to_mph(standardized, batch["mean"], batch["scale"])
        loss, _, count = masked_mae_loss(predictions, batch["targets_mph"], batch["target_mask"])
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
        return len(batch["sensor_index"])

    train_iterator = iter(train_loader)
    model.train()
    for _ in range(warmup_batches):
        raw_batch, train_iterator = _next_batch(train_iterator, train_loader)
        training_step(raw_batch)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_examples = 0
    train_started = time.perf_counter()
    for _ in range(probe_batches):
        raw_batch, train_iterator = _next_batch(train_iterator, train_loader)
        train_examples += training_step(raw_batch)
    _synchronize(device)
    train_seconds = time.perf_counter() - train_started
    train_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    validation_iterator = iter(validation_loader)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_batches):
            raw_batch, validation_iterator = _next_batch(validation_iterator, validation_loader)
            batch = move_batch(raw_batch, device)
            standardized = model(batch["history"], batch["sensor_index"], batch["future_time"])
            predictions = predictions_to_mph(standardized, batch["mean"], batch["scale"])
            for step in REPORTED_STEPS:
                mask = batch["target_mask"][:, step]
                torch.abs(predictions[:, step][mask] - batch["targets_mph"][:, step][mask]).sum()
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        validation_examples = 0
        validation_started = time.perf_counter()
        for _ in range(probe_batches):
            raw_batch, validation_iterator = _next_batch(validation_iterator, validation_loader)
            batch = move_batch(raw_batch, device)
            standardized = model(batch["history"], batch["sensor_index"], batch["future_time"])
            predictions = predictions_to_mph(standardized, batch["mean"], batch["scale"])
            for step in REPORTED_STEPS:
                mask = batch["target_mask"][:, step]
                torch.abs(predictions[:, step][mask] - batch["targets_mph"][:, step][mask]).sum()
            validation_examples += len(batch["sensor_index"])
        _synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
    validation_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    train_batch_seconds = train_seconds / probe_batches
    validation_batch_seconds = validation_seconds / probe_batches
    return {
        "candidate": config.candidate_name,
        "parameter_count": model.parameter_count,
        "batch_size": config.batch_size,
        "warmup_batches_excluded": warmup_batches,
        "measured_training_batches": probe_batches,
        "training_examples": train_examples,
        "steady_state_training_seconds": train_seconds,
        "steady_state_training_examples_per_second": train_examples / train_seconds,
        "estimated_full_training_epoch_seconds": train_batch_seconds * math.ceil(len(train_dataset) / config.batch_size),
        "estimated_full_validation_seconds": validation_batch_seconds * math.ceil(len(validation_dataset) / config.batch_size),
        "validation_measurement_examples": validation_examples,
        "peak_allocated_gpu_bytes": max(train_peak, validation_peak),
        "estimates_note": "Epoch and validation durations are linear estimates from steady-state measured batches, not completed full passes.",
    }


def run_probe_only(datasets, device, workers: int, batch_size: int, probe_batches: int, warmup_batches: int) -> dict[str, object]:
    configs = approved_configs(batch_size)
    results = [
        benchmark_candidate(
            config, datasets["train"], datasets["validation"], device, workers,
            probe_batches=probe_batches, warmup_batches=warmup_batches,
        )
        for config in configs
    ]
    return {
        "task": "Task 06 - GRU throughput probe only",
        "mode": "probe-only",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "scientific_artifacts_written": False,
        "test_dataset_constructed_or_evaluated": False,
        "candidates": results,
    }


def fit_candidate(config, train_dataset, validation_dataset, device, workers, output_dir, resume, probe_batches):
    generator = seed_everything(config.seed)
    model = SensorLocalGRU(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator, workers=workers)
    validation_loader = make_loader(validation_dataset, batch_size=config.batch_size, shuffle=False, generator=None, workers=workers)
    probe = throughput_probe(model, train_loader, device, probe_batches)
    # The optional probe must not change initialization or shuffle order.
    generator = seed_everything(config.seed)
    model = SensorLocalGRU(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator, workers=workers)
    latest_path = output_dir / f"{config.candidate_name}_latest.pt"
    best_path = output_dir / f"{config.candidate_name}_best.pt"
    start_epoch, best_score, patience_used, history = 0, float("inf"), 0, []
    if resume and latest_path.exists():
        restored = load_checkpoint(latest_path, model=model, optimizer=optimizer, training_config=config, data_generator=generator, map_location=device)
        start_epoch = int(restored["epoch"]) + 1
        best_score = float(restored["best_score"])
        patience_used = int(restored["patience_used"])
        history = list(restored["history"])
    for epoch in range(start_epoch, config.max_epochs):
        if patience_used >= config.patience:
            break
        train_mae, train_count = train_epoch(model, train_loader, optimizer, device, config.gradient_clip_norm)
        horizon_mae, selection_score = validate(model, validation_loader, device)
        row = {"epoch": epoch, "train_mae_mph": train_mae, "train_eligible_count": train_count, "validation_horizon_mae_mph": dict(zip(HORIZON_MINUTES, horizon_mae)), "selection_score_mph": selection_score}
        history.append(row)
        improved = selection_score < best_score
        if improved:
            best_score, patience_used = selection_score, 0
            save_checkpoint(best_path, model=model, optimizer=optimizer, training_config=config, epoch=epoch, best_score=best_score, patience_used=patience_used, data_generator=generator, history=history)
        else:
            patience_used += 1
        save_checkpoint(latest_path, model=model, optimizer=optimizer, training_config=config, epoch=epoch, best_score=best_score, patience_used=patience_used, data_generator=generator, history=history)
        print(json.dumps(row), flush=True)
        if patience_used >= config.patience:
            break
    return {"candidate": config.candidate_name, "best_score_mph": best_score, "parameter_count": model.parameter_count, "best_checkpoint": best_path.relative_to(ROOT).as_posix(), "latest_checkpoint": latest_path.relative_to(ROOT).as_posix(), "throughput_probe": probe, "history": history}


def assert_reference_alignment(reference_path: Path, frame, test_windows) -> None:
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    targets = test_windows.y_speed[:, REPORTED_STEPS, :]
    with np.load(reference_path) as reference:
        np.testing.assert_array_equal(reference["sensor_ids"], np.asarray(frame.columns, dtype=str))
        np.testing.assert_array_equal(reference["horizon_minutes"], HORIZON_MINUTES)
        np.testing.assert_array_equal(reference["targets_mph"], targets)
        np.testing.assert_array_equal(reference["target_timestamps"], test_windows.target_timestamps[:, REPORTED_STEPS])
        np.testing.assert_array_equal(reference["primary_nonzero_mask"], np.isfinite(targets) & (targets != 0))
        np.testing.assert_array_equal(reference["finite_including_zero_mask"], np.isfinite(targets))


@torch.no_grad()
def predict_locked_test(model, dataset, device, batch_size: int, workers: int) -> np.ndarray:
    """Return window-major predictions at the three reported horizons."""
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, generator=None, workers=workers)
    model.eval()
    chunks = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        standardized = model(batch["history"], batch["sensor_index"], batch["future_time"])
        mph = predictions_to_mph(standardized, batch["mean"], batch["scale"])
        chunks.append(mph[:, REPORTED_STEPS].cpu().numpy())
    flat = np.concatenate(chunks, axis=0)
    return flat.reshape(len(dataset.windows.X_speed), dataset.sensor_count, 3).transpose(0, 2, 1)


def validate_completed_candidates(trials, configs) -> list[dict[str, object]]:
    """Fail closed unless both approved best checkpoints and scores are complete."""
    expected = {config.candidate_name: config for config in configs}
    if not isinstance(trials, list) or len(trials) != len(expected):
        raise ValueError("Finalization requires exactly two completed candidate trials")
    observed = {str(trial.get("candidate")): trial for trial in trials}
    if set(observed) != set(expected):
        raise ValueError("Candidate evidence must contain exactly gru_h64 and gru_h128")
    validated = []
    for name in sorted(expected):
        trial = observed[name]
        score = float(trial.get("best_score_mph", float("nan")))
        history = trial.get("history")
        if not np.isfinite(score) or not isinstance(history, list) or not history:
            raise ValueError(f"Candidate {name} has incomplete validation evidence")
        history_scores = [float(row["selection_score_mph"]) for row in history]
        if not all(np.isfinite(history_scores)) or not np.isclose(score, min(history_scores), rtol=0, atol=1e-12):
            raise ValueError(f"Candidate {name} best score disagrees with validation history")
        checkpoint_path = ROOT / str(trial.get("best_checkpoint", ""))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Required best checkpoint is missing: {checkpoint_path}")
        model = SensorLocalGRU(expected[name].model)
        checkpoint = load_checkpoint_for_inference(
            checkpoint_path, model=model, training_config=expected[name], map_location="cpu"
        )
        if int(trial.get("parameter_count", -1)) != model.parameter_count:
            raise ValueError(f"Candidate {name} parameter-count evidence is inconsistent")
        if not np.isclose(float(checkpoint.get("best_score", float("nan"))), score, rtol=0, atol=1e-12):
            raise ValueError(f"Candidate {name} checkpoint score is inconsistent")
        best_row = min(history, key=lambda row: (float(row["selection_score_mph"]), int(row["epoch"])))
        if int(checkpoint.get("epoch", -1)) != int(best_row["epoch"]):
            raise ValueError(f"Candidate {name} checkpoint epoch is inconsistent")
        validated.append(trial)
    return validated


def load_finalization_evidence(path: Path, data_path: Path, splits, datasets) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation selection evidence is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("mode") != "full":
        raise ValueError("Finalization requires completed full-training evidence")
    if document.get("data_sha256") != _sha256(data_path):
        raise ValueError("Validation evidence dataset hash does not match current METR-LA")
    expected_rows = {name: [split.start, split.stop] for name, split in splits.items()}
    if document.get("split_rows") != expected_rows:
        raise ValueError("Validation evidence split boundaries are inconsistent")
    expected_examples = {name: len(dataset) for name, dataset in datasets.items()}
    if document.get("dataset_examples") != expected_examples:
        raise ValueError("Validation evidence dataset counts are inconsistent")
    return document.get("trials")


def finalize_full_run(trials, configs, datasets, windows, frame, device, workers, batch_size, data_path, splits):
    trials = validate_completed_candidates(trials, configs)
    selected = min(trials, key=lambda trial: (float(trial["best_score_mph"]), str(trial["candidate"])))
    config = next(value for value in configs if value.candidate_name == selected["candidate"])
    model = SensorLocalGRU(config.model).to(device)
    checkpoint = load_checkpoint_for_inference(
        ROOT / selected["best_checkpoint"], model=model,
        training_config=config, map_location="cpu",
    )
    predictions = predict_locked_test(model, datasets["test"], device, batch_size, workers)
    targets = windows["test"].y_speed[:, REPORTED_STEPS, :]
    primary = np.isfinite(targets) & (targets != 0)
    sensitivity = np.isfinite(targets)
    artifact_dir = ROOT / "artifacts" / "task06_gru"
    result_dir = ROOT / "results" / "task06_gru"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = artifact_dir / "test_predictions_horizons.npz"
    np.savez_compressed(
        prediction_path,
        targets_mph=targets,
        gru_mph=predictions,
        target_timestamps=windows["test"].target_timestamps[:, REPORTED_STEPS],
        forecast_origin_timestamps=windows["test"].forecast_origin_timestamps,
        sensor_ids=np.asarray(frame.columns, dtype=str),
        horizon_minutes=np.asarray(HORIZON_MINUTES),
        primary_nonzero_mask=primary,
        finite_including_zero_mask=sensitivity,
    )
    rows = []
    for policy, mask in (("finite_nonzero_targets", primary), ("all_finite_targets_including_zeros", sensitivity)):
        for position, minutes in enumerate(HORIZON_MINUTES):
            rows.append({
                "model": "gru", "horizon": f"+{minutes}",
                "forecast_step_1_based": REPORTED_STEPS[position] + 1,
                "mask_policy": policy,
                "eligible_count": int(mask[:, position].sum()),
                "excluded_count": int(mask[:, position].size - mask[:, position].sum()),
                "mae_mph": masked_mae(targets[:, position], predictions[:, position], mask[:, position]),
                "rmse_mph": masked_rmse(targets[:, position], predictions[:, position], mask[:, position]),
            })
    selected_document = {
        "candidate": selected["candidate"], "training_config": config.as_dict(),
        "parameter_count": model.parameter_count, "best_epoch": checkpoint["epoch"],
        "best_validation_score_mph": checkpoint["best_score"],
        "checkpoint": selected["best_checkpoint"],
        "checkpoint_sha256": _sha256(ROOT / selected["best_checkpoint"]),
    }
    (artifact_dir / "selected_config.json").write_text(json.dumps(selected_document, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "task": "Task 06 - Sensor-local GRU", "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path.relative_to(ROOT).as_posix(), "data_sha256": _sha256(data_path),
        "split_policy": "chronological_70_10_20",
        "splits": {name: {"start_row": split.start, "stop_row_exclusive": split.stop} for name, split in splits.items()},
        "window_policy": "12 sensor-local input steps to 12 jointly trained target steps",
        "feature_policy": "local standardized speed, zero/finite indicators, target time-of-day, learned sensor identity; no graph or neighbouring speeds",
        "selection_policy": "lowest equal-weight mean validation MAE at +15/+30/+60, then candidate name",
        "selected_config": "artifacts/task06_gru/selected_config.json",
        "prediction_artifact": prediction_path.relative_to(ROOT).as_posix(),
        "prediction_sha256": _sha256(prediction_path), "task04_alignment_verified": True,
        "task05_alignment_verified": True, "metrics": rows,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return selected_document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.num_workers < 0
        or args.throughput_probe_batches < 0
        or args.probe_warmup_batches < 0
    ):
        raise ValueError("Batch size must be positive; worker/probe counts cannot be negative")
    if args.probe_only and args.throughput_probe_batches <= 0:
        raise ValueError("--probe-only requires --throughput-probe-batches greater than zero")
    if args.finalize_only and args.resume:
        raise ValueError("--finalize-only does not accept --resume")
    device = resolve_device(args.device)
    split_names = ("train", "validation") if args.probe_only else ("train", "validation", "test")
    data_path, frame, splits, _, datasets, windows = prepare_datasets(split_names)
    if args.probe_only:
        report = run_probe_only(
            datasets, device, args.num_workers, args.batch_size,
            args.throughput_probe_batches, args.probe_warmup_batches,
        )
        report["data_sha256"] = _sha256(data_path)
        report["dataset_examples"] = {name: len(dataset) for name, dataset in datasets.items()}
        print(json.dumps(report, indent=2))
        return
    assert_reference_alignment(ROOT / "artifacts" / "task04_baselines" / "test_predictions_horizons.npz", frame, windows["test"])
    assert_reference_alignment(ROOT / "artifacts" / "task05_xgboost" / "test_predictions_horizons.npz", frame, windows["test"])
    if args.finalize_only:
        configs = list(approved_configs(args.batch_size))
        evidence_path = ROOT / "results" / "task06_gru" / "validation_trials.json"
        trials = load_finalization_evidence(evidence_path, data_path, splits, datasets)
        selected = finalize_full_run(
            trials, configs, datasets, windows, frame, device, args.num_workers,
            args.batch_size, data_path, splits,
        )
        print(json.dumps({"mode": "finalize-only", "selected": selected}, indent=2))
        return
    artifact_dir = ROOT / (".tmp/task06_gru_smoke" if args.smoke else "artifacts/task06_gru/checkpoints")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        train_data = Subset(datasets["train"], range(min(512, len(datasets["train"]))))
        validation_data = Subset(datasets["validation"], range(min(256, len(datasets["validation"]))))
        configs = [TrainingConfig("gru_smoke", GRUConfig(hidden_size=16), batch_size=min(args.batch_size, 64), max_epochs=2, patience=2)]
    else:
        train_data, validation_data = datasets["train"], datasets["validation"]
        configs = list(approved_configs(args.batch_size))
    trials = [fit_candidate(config, train_data, validation_data, device, args.num_workers, artifact_dir, args.resume, args.throughput_probe_batches) for config in configs]
    summary = {
        "task": "Task 06 - Sensor-local GRU",
        "mode": "smoke" if args.smoke else "full",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": _sha256(data_path),
        "split_rows": {name: [split.start, split.stop] for name, split in splits.items()},
        "dataset_examples": {name: len(dataset) for name, dataset in datasets.items()},
        "device": str(device),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "trials": trials,
        "note": "Smoke metrics are engineering checks only." if args.smoke else "Full results require locked test inference after validation selection.",
    }
    summary_path = artifact_dir / "smoke_summary.json" if args.smoke else ROOT / "results/task06_gru/validation_trials.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.full:
        summary["selected"] = finalize_full_run(
            trials, configs, datasets, windows, frame, device, args.num_workers,
            args.batch_size, data_path, splits,
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

