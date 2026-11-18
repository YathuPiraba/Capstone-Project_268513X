"""Task 07 local smoke, GPU-probe preparation, training, and finalization entry point.

Full training, GPU probing, and locked-test finalization are implemented for later
owner-authorized use. This module does not execute any mode on import.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.inspection import (
    load_graph_bundle,
    load_speed_data,
    read_sensor_ids,
    validate_graph_alignment,
)
from src.data.preprocessing import PerSensorStandardizer, chronological_split_ranges, cyclical_time_of_day
from src.data.windows import build_forecast_windows, window_time_features
from src.evaluation.metrics import masked_mae, masked_rmse
from src.models.dcrnn import DCGRUConfig, DirectHeadDCGRU, directed_random_walk_supports
from src.training.dcrnn import (
    DCGRUTrainingConfig,
    GraphWindowDataset,
    HORIZON_MINUTES,
    REPORTED_STEPS,
    configure_reproducibility,
    load_checkpoint_for_inference,
    load_checkpoint_for_resume,
    masked_mae_loss,
    predictions_to_mph,
    save_checkpoint,
    validation_horizon_mae,
)


DATA_DIR = ROOT / "data" / "raw" / "metr-la"
EXPECTED_ADJACENCY_SHA256 = "A35687C6E15AA228DC45027B0ED2A0EA0F4EC78F573DEB992C595092D12F61B3"
EXPECTED_GRAPH_STATS = {
    "node_count": 207,
    "nonzero_entries": 1722,
    "off_diagonal_nonzero_entries": 1515,
    "diagonal_nonzero_entries": 207,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 07 direct-head DCGRU")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--integrity-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--probe-only", action="store_true")
    modes.add_argument("--full", action="store_true")
    modes.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, choices=(32, 64), default=64)
    parser.add_argument("--graph-mode", choices=("graph", "identity"), default="graph")
    parser.add_argument("--seed", type=int, default=268513)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--probe-train-batches", type=int, default=50)
    parser.add_argument("--probe-validation-batches", type=int, default=20)
    parser.add_argument("--session-limit-hours", type=float)
    parser.add_argument("--remaining-gpu-quota-hours", type=float)
    parser.add_argument("--safety-margin", type=float, default=0.30)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sensor_order_sha256(sensor_ids: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in sensor_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def inspect_graph(frame: pd.DataFrame) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    graph_path = DATA_DIR / "adj_mx_METR-LA.pkl"
    actual_hash = sha256(graph_path)
    if actual_hash != EXPECTED_ADJACENCY_SHA256:
        raise ValueError("METR-LA adjacency SHA-256 does not match the approved graph")
    graph = load_graph_bundle(graph_path)
    reference_ids = read_sensor_ids(DATA_DIR / "graph_sensor_ids.txt")
    alignment = validate_graph_alignment(frame.columns, graph, reference_ids)
    required_alignment = (
        "csv_graph_same_set", "csv_graph_same_order", "reference_graph_same_order",
        "mapping_matches_graph_order", "adjacency_shape_matches_ids",
    )
    if not all(alignment[name] for name in required_alignment):
        raise ValueError(f"METR-LA graph alignment failed: {alignment}")
    adjacency = np.asarray(graph.adjacency, dtype=np.float32)
    if adjacency.shape != (207, 207) or not np.isfinite(adjacency).all():
        raise ValueError("Adjacency must be finite with shape (207, 207)")
    if np.any(adjacency < 0) or float(adjacency.max()) > 1.0:
        raise ValueError("Adjacency weights must lie in [0, 1]")
    diagonal = np.diag(adjacency)
    nonzero = adjacency != 0
    off_diagonal = nonzero & ~np.eye(207, dtype=bool)
    observed = {
        "node_count": 207,
        "nonzero_entries": int(nonzero.sum()),
        "off_diagonal_nonzero_entries": int(off_diagonal.sum()),
        "diagonal_nonzero_entries": int(np.count_nonzero(diagonal)),
    }
    if observed != EXPECTED_GRAPH_STATS:
        raise ValueError(f"Adjacency statistics differ from the approved graph: {observed}")
    if not np.all(diagonal == 1.0):
        raise ValueError("Approved adjacency must retain exactly one recorded self-loop per node")
    if np.array_equal(adjacency, adjacency.T):
        raise ValueError("Approved METR-LA adjacency is expected to be directed/asymmetric")
    forward, reverse = directed_random_walk_supports(adjacency)
    row_sums = adjacency.sum(axis=1)
    column_sums = adjacency.sum(axis=0)
    if np.any(row_sums == 0) or np.any(column_sums == 0):
        raise ValueError("Approved graph unexpectedly contains a zero-degree node")
    np.testing.assert_allclose(forward.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(reverse.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    manifest = {
        "format_version": 1,
        "source": "data/raw/metr-la/adj_mx_METR-LA.pkl",
        "adjacency_sha256": actual_hash,
        "sensor_order_sha256": sensor_order_sha256(graph.sensor_ids),
        "sensor_count": 207,
        "shape": [207, 207],
        "dtype": str(graph.adjacency.dtype),
        "weight_min": float(adjacency.min()),
        "weight_max": float(adjacency.max()),
        "nonzero_entries": observed["nonzero_entries"],
        "off_diagonal_nonzero_entries": observed["off_diagonal_nonzero_entries"],
        "density": float(nonzero.mean()),
        "self_loops": "source adjacency has diagonal weight 1.0 for all 207 nodes; no loops added",
        "asymmetric_entry_count": int(np.count_nonzero(adjacency != adjacency.T)),
        "zero_out_degree_nodes": [],
        "zero_in_degree_nodes": [],
        "normalization": "row random walk",
        "supports": ["D_out^-1 A", "D_in^-1 A^T"],
        "diffusion_order": 2,
        "basis": "order-zero identity plus orders 1 and 2 for both directed supports",
        "forward_row_sum_range": [float(forward.sum(1).min()), float(forward.sum(1).max())],
        "reverse_row_sum_range": [float(reverse.sum(1).min()), float(reverse.sum(1).max())],
        "alignment_verified": True,
    }
    return graph, forward, reverse, manifest


def verify_locked_artifact_contract(frame: pd.DataFrame, test_windows) -> dict[str, str]:
    """Read alignment fields only; never read prediction arrays or metric documents."""
    result: dict[str, str] = {}
    expected_targets = test_windows.y_speed[:, REPORTED_STEPS, :]
    expected_times = test_windows.target_timestamps[:, REPORTED_STEPS]
    expected_primary = np.isfinite(expected_targets) & (expected_targets != 0)
    expected_finite = np.isfinite(expected_targets)
    for task in ("task04_baselines", "task05_xgboost", "task06_gru"):
        path = ROOT / "artifacts" / task / "test_predictions_horizons.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Locked reference artifact is missing: {path}")
        with np.load(path) as reference:
            np.testing.assert_array_equal(reference["sensor_ids"], np.asarray(frame.columns, dtype=str))
            np.testing.assert_array_equal(reference["horizon_minutes"], HORIZON_MINUTES)
            np.testing.assert_array_equal(reference["targets_mph"], expected_targets)
            np.testing.assert_array_equal(reference["target_timestamps"], expected_times)
            np.testing.assert_array_equal(reference["primary_nonzero_mask"], expected_primary)
            np.testing.assert_array_equal(reference["finite_including_zero_mask"], expected_finite)
        result[task] = sha256(path)
    return result


def prepare_data(split_names: tuple[str, ...]):
    data_path = DATA_DIR / "METR-LA.csv"
    frame = load_speed_data(data_path)
    if frame.shape != (34272, 207):
        raise ValueError("METR-LA dimensions do not match the locked Task 07 contract")
    graph, forward, reverse, manifest = inspect_graph(frame)
    splits = chronological_split_ranges(frame.index)
    expected_rows = {"train": (0, 23990), "validation": (23990, 27417), "test": (27417, 34272)}
    if {name: (value.start, value.stop) for name, value in splits.items()} != expected_rows:
        raise ValueError("Chronological split boundaries changed")
    scaler = PerSensorStandardizer.fit(frame.iloc[:splits["train"].stop].to_numpy(), exclude_zeros=True)
    time_features = cyclical_time_of_day(frame.index)
    windows = {}
    datasets = {}
    for name in split_names:
        split_windows = build_forecast_windows(frame.to_numpy(), frame.index, splits[name])
        _, future = window_time_features(time_features, split_windows)
        windows[name] = split_windows
        datasets[name] = GraphWindowDataset(split_windows, scaler, future, graph.sensor_ids)
    return data_path, frame, splits, scaler, forward, reverse, manifest, datasets, windows


def training_config(args: argparse.Namespace) -> DCGRUTrainingConfig:
    return DCGRUTrainingConfig(
        candidate_name=f"dcgru_h{args.hidden_size}_k2",
        model=DCGRUConfig(hidden_size=args.hidden_size, graph_mode=args.graph_mode),
        batch_size=args.batch_size,
        seed=args.seed,
    )


def immutable_metadata(data_path: Path, splits, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_sha256": sha256(data_path),
        "adjacency_sha256": manifest["adjacency_sha256"],
        "sensor_order_sha256": manifest["sensor_order_sha256"],
        "graph_normalization": manifest["normalization"],
        "support_definitions": manifest["supports"],
        "diffusion_order": manifest["diffusion_order"],
        "split_rows": {name: [split.start, split.stop] for name, split in splits.items()},
        "feature_policy": "standardized recorded speed, zero/finite indicators, future time-of-day, sensor identity",
        "mask_policy": "training and primary validation exclude finite exact-zero targets",
        "window_policy": "12 history steps to 12 direct forecast steps",
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested if requested != "auto" else "cpu")


def make_loader(dataset, *, batch_size: int, shuffle: bool, generator, workers: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=workers)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_batches(model, loader, optimizer, device, mean, scale, clip_norm, max_batches=None):
    model.train()
    total_error, total_count, completed = 0.0, 0, 0
    for raw in loader:
        batch = move_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        standardized = model(batch["history"], batch["future_time"])
        mph = predictions_to_mph(standardized, mean, scale)
        loss, error_sum, count = masked_mae_loss(mph, batch["targets_mph"], batch["target_mask"])
        if loss is not None:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradients encountered")
            optimizer.step()
            total_error += float(error_sum.item())
            total_count += count
        completed += 1
        if max_batches is not None and completed >= max_batches:
            break
    return total_error / total_count if total_count else float("nan"), total_count, completed


@torch.no_grad()
def validate(model, loader, device, mean, scale):
    model.eval()
    sums = [0.0, 0.0, 0.0]
    counts = [0, 0, 0]
    for raw in loader:
        batch = move_batch(raw, device)
        mph = predictions_to_mph(model(batch["history"], batch["future_time"]), mean, scale)
        for position, step in enumerate(REPORTED_STEPS):
            eligible = batch["target_mask"][:, step]
            sums[position] += float(torch.abs(mph[:, step][eligible] - batch["targets_mph"][:, step][eligible]).sum().item())
            counts[position] += int(eligible.sum().item())
    return validation_horizon_mae(sums, counts)


def fit(config, datasets, forward, reverse, scaler, device, workers, output_dir, metadata, resume):
    generator = configure_reproducibility(config.seed, config.deterministic_algorithms)
    model = DirectHeadDCGRU(config.model, forward, reverse).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = make_loader(datasets["train"], batch_size=config.batch_size, shuffle=True, generator=generator, workers=workers)
    validation_loader = make_loader(datasets["validation"], batch_size=config.batch_size, shuffle=False, generator=None, workers=workers)
    latest = output_dir / f"{config.candidate_name}_latest.pt"
    best = output_dir / f"{config.candidate_name}_best.pt"
    start_epoch, global_step, best_score, best_epoch, patience_used, history = 0, 0, float("inf"), -1, 0, []
    if resume and latest.is_file():
        restored = load_checkpoint_for_resume(
            latest, model=model, optimizer=optimizer, training_config=config,
            immutable_metadata=metadata, data_generator=generator, device=device,
        )
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        best_score = float(restored["best_score"])
        best_epoch = int(restored["best_epoch"])
        patience_used = int(restored["patience_used"])
        history = list(restored["history"])
    mean = torch.as_tensor(scaler.mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(scaler.scale, dtype=torch.float32, device=device)
    for epoch in range(start_epoch, config.max_epochs):
        if patience_used >= config.patience:
            break
        train_mae, count, batches = train_batches(model, train_loader, optimizer, device, mean, scale, config.gradient_clip_norm)
        global_step += batches
        horizon_mae, score = validate(model, validation_loader, device, mean, scale)
        row = {"epoch": epoch, "train_mae_mph": train_mae, "train_eligible_count": count, "validation_horizon_mae_mph": dict(zip(HORIZON_MINUTES, horizon_mae)), "selection_score_mph": score}
        history.append(row)
        improved = score < best_score
        if improved:
            best_score, best_epoch, patience_used = score, epoch, 0
        else:
            patience_used += 1
        common = dict(model=model, optimizer=optimizer, training_config=config, immutable_metadata=metadata, epoch=epoch, global_step=global_step, best_score=best_score, best_epoch=best_epoch, patience_used=patience_used, data_generator=generator, history=history)
        if improved:
            save_checkpoint(best, **common)
        save_checkpoint(latest, **common)
        print(json.dumps(row), flush=True)
    return {"candidate": config.candidate_name, "parameter_count": model.parameter_count, "best_score_mph": best_score, "best_epoch": best_epoch, "best_checkpoint": best.relative_to(ROOT).as_posix(), "latest_checkpoint": latest.relative_to(ROOT).as_posix(), "history": history}


def runtime_projection(measured_hours: float, session_limit: float | None, quota: float | None, safety_margin: float) -> dict[str, Any]:
    if measured_hours < 0 or not 0 <= safety_margin < 1:
        raise ValueError("Runtime must be nonnegative and safety margin must be in [0, 1)")
    required = measured_hours / (1.0 - safety_margin)
    session_ok = None if session_limit is None else required <= session_limit
    quota_ok = None if quota is None else required <= quota
    if session_ok is False or quota_ok is False:
        status = "infeasible"
    elif session_ok is True and quota_ok is True:
        status = "feasible"
    else:
        status = "unknown"
    return {"projected_hours": measured_hours, "safety_adjusted_required_hours": required, "session_limit_hours": session_limit, "remaining_gpu_quota_hours": quota, "safety_margin": safety_margin, "status": status}


def benchmark(model, train_loader, validation_loader, optimizer, device, mean, scale, args, config, metadata, generator):
    def synchronize():
        if device.type == "cuda": torch.cuda.synchronize(device)
    iterator = iter(train_loader)
    for _ in range(args.warmup_batches):
        try: batch = next(iterator)
        except StopIteration: iterator, batch = iter(train_loader), next(iter(train_loader))
        train_batches(model, [batch], optimizer, device, mean, scale, 1.0, max_batches=1)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    _, _, measured = train_batches(model, train_loader, optimizer, device, mean, scale, 1.0, max_batches=args.probe_train_batches)
    synchronize()
    train_seconds = time.perf_counter() - started
    model.eval(); validation_examples = 0
    started = time.perf_counter()
    with torch.no_grad():
        for index, raw in enumerate(validation_loader):
            batch = move_batch(raw, device)
            predictions_to_mph(model(batch["history"], batch["future_time"]), mean, scale)
            validation_examples += len(batch["history"])
            if index + 1 >= args.probe_validation_batches: break
    synchronize(); validation_seconds = time.perf_counter() - started
    train_batch_seconds = train_seconds / measured
    validation_batches = min(args.probe_validation_batches, math.ceil(len(validation_loader.dataset) / args.batch_size))
    validation_batch_seconds = validation_seconds / validation_batches
    epoch_seconds = train_batch_seconds * math.ceil(len(train_loader.dataset) / args.batch_size)
    full_validation_seconds = validation_batch_seconds * math.ceil(len(validation_loader.dataset) / args.batch_size)
    checkpoint_path = ROOT / ".tmp" / "task07_probe" / "probe_checkpoint.pt"
    save_started = time.perf_counter()
    save_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer, training_config=config,
        immutable_metadata=metadata, epoch=0, global_step=measured,
        best_score=1.0, best_epoch=0, patience_used=0,
        data_generator=generator, history=[{"epoch": 0, "probe_only": True}],
    )
    synchronize(); checkpoint_save_seconds = time.perf_counter() - save_started
    restored_model = DirectHeadDCGRU(
        config.model, model.forward_support.detach().cpu(), model.reverse_support.detach().cpu()
    )
    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    restored_generator = torch.Generator()
    load_started = time.perf_counter()
    restored_checkpoint = load_checkpoint_for_resume(
        checkpoint_path, model=restored_model, optimizer=restored_optimizer,
        training_config=config, immutable_metadata=metadata,
        data_generator=restored_generator, device=device,
    )
    synchronize(); checkpoint_load_seconds = time.perf_counter() - load_started
    cpu_rng_restored = torch.equal(
        torch.get_rng_state().cpu(), restored_checkpoint["rng_state"]["torch_cpu"].cpu()
    )
    loader_rng_restored = torch.equal(
        restored_generator.get_state().cpu(),
        restored_checkpoint["rng_state"]["data_generator"].cpu(),
    )
    if device.type == "cuda":
        saved_cuda = restored_checkpoint["rng_state"].get("torch_cuda", [])
        current_cuda = torch.cuda.get_rng_state_all()
        cuda_rng_restored = len(saved_cuda) == len(current_cuda) and all(
            torch.equal(current.cpu(), saved.cpu())
            for current, saved in zip(current_cuda, saved_cuda)
        )
    else:
        cuda_rng_restored = None
    if not cpu_rng_restored or not loader_rng_restored or cuda_rng_restored is False:
        raise RuntimeError("Checkpoint RNG restoration verification failed")
    checkpoint_path.unlink(missing_ok=True)
    maximum_hours = 30 * (
        epoch_seconds + full_validation_seconds + checkpoint_save_seconds
    ) / 3600
    return {
        "measured_training_batches": measured,
        "training_batches_per_second": measured / train_seconds,
        "training_batch_seconds": train_batch_seconds,
        "windows_per_second": measured * args.batch_size / train_seconds,
        "validation_windows_per_second": validation_examples / validation_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "estimated_epoch_seconds": epoch_seconds,
        "estimated_validation_seconds": full_validation_seconds,
        "estimated_30_epoch_hours": maximum_hours,
        "estimated_30_epoch_includes": "training, validation, and one checkpoint save per epoch; locked-test finalization is separately unmeasured",
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "checkpoint_load_resume_seconds": checkpoint_load_seconds,
        "checkpoint_cpu_rng_restored": cpu_rng_restored,
        "checkpoint_dataloader_rng_restored": loader_rng_restored,
        "checkpoint_cuda_rng_restored": cuda_rng_restored,
        "finite_loss_and_gradients": True,
        "runtime_projection": runtime_projection(maximum_hours, args.session_limit_hours, args.remaining_gpu_quota_hours, args.safety_margin),
        "identity_control_projection_note": "Measure identity-only mode separately; do not assume graph-enabled throughput is identical.",
    }


def load_completed_validation_evidence(path: Path, config: DCGRUTrainingConfig, metadata: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Completed Task 07 validation evidence is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("mode") != "full-training-validation-only":
        raise ValueError("Finalization requires completed validation-only training evidence")
    if document.get("metadata") != metadata:
        raise ValueError("Validation evidence data or graph identity is incompatible")
    trial = document.get("trial")
    if not isinstance(trial, dict) or trial.get("candidate") != config.candidate_name:
        raise ValueError("Validation evidence does not match the requested candidate")
    history = trial.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Validation evidence has no completed history")
    scores = [float(row["selection_score_mph"]) for row in history]
    if not all(np.isfinite(scores)):
        raise ValueError("Validation history contains a non-finite score")
    best_row = min(history, key=lambda row: (float(row["selection_score_mph"]), int(row["epoch"])))
    if int(trial.get("best_epoch", -1)) != int(best_row["epoch"]):
        raise ValueError("Recorded best epoch disagrees with validation history")
    if not np.isclose(float(trial.get("best_score_mph", np.nan)), float(best_row["selection_score_mph"]), rtol=0, atol=1e-12):
        raise ValueError("Recorded best score disagrees with validation history")
    checkpoint = ROOT / str(trial.get("best_checkpoint", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected best checkpoint is missing: {checkpoint}")
    return trial


@torch.no_grad()
def predict_locked_test(model, dataset, device, scaler, batch_size: int, workers: int) -> np.ndarray:
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, generator=None, workers=workers)
    mean = torch.as_tensor(scaler.mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(scaler.scale, dtype=torch.float32, device=device)
    model.eval(); chunks = []
    for raw in loader:
        batch = move_batch(raw, device)
        mph = predictions_to_mph(model(batch["history"], batch["future_time"]), mean, scale)
        chunks.append(mph[:, REPORTED_STEPS].cpu().numpy())
    return np.concatenate(chunks, axis=0)


def finalize_completed_run(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    # Test windows are constructed only inside this explicitly selected mode.
    data_path, frame, splits, scaler, forward, reverse, manifest, datasets, windows = prepare_data(("test",))
    references = verify_locked_artifact_contract(frame, windows["test"])
    config = training_config(args)
    metadata = immutable_metadata(data_path, splits, manifest)
    evidence_path = ROOT / "results" / "task07_dcrnn" / "validation_trial.json"
    trial = load_completed_validation_evidence(evidence_path, config, metadata)
    model = DirectHeadDCGRU(config.model, forward, reverse)
    checkpoint_path = ROOT / str(trial["best_checkpoint"])
    checkpoint = load_checkpoint_for_inference(
        checkpoint_path, model=model, training_config=config,
        immutable_metadata=metadata, device=device,
    )
    if int(checkpoint["best_epoch"]) != int(trial["best_epoch"]):
        raise ValueError("Best checkpoint epoch disagrees with validation evidence")
    if not np.isclose(float(checkpoint["best_score"]), float(trial["best_score_mph"]), rtol=0, atol=1e-12):
        raise ValueError("Best checkpoint score disagrees with validation evidence")
    predictions = predict_locked_test(model, datasets["test"], device, scaler, args.batch_size, args.num_workers)
    targets = windows["test"].y_speed[:, REPORTED_STEPS, :]
    primary = np.isfinite(targets) & (targets != 0)
    sensitivity = np.isfinite(targets)
    if predictions.shape != (6844, 3, 207):
        raise ValueError(f"Locked Task 07 prediction shape is invalid: {predictions.shape}")
    artifact_dir = ROOT / "artifacts" / "task07_dcrnn"
    result_dir = ROOT / "results" / "task07_dcrnn"
    artifact_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = artifact_dir / "test_predictions_horizons.npz"
    np.savez_compressed(
        prediction_path,
        targets_mph=targets,
        dcrnn_mph=predictions,
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
                "model": "direct_head_dcgru", "horizon": f"+{minutes}",
                "forecast_step_1_based": REPORTED_STEPS[position] + 1,
                "mask_policy": policy,
                "eligible_count": int(mask[:, position].sum()),
                "excluded_count": int(mask[:, position].size - mask[:, position].sum()),
                "mae_mph": masked_mae(targets[:, position], predictions[:, position], mask[:, position]),
                "rmse_mph": masked_rmse(targets[:, position], predictions[:, position], mask[:, position]),
            })
    metrics = {
        "task": "Task 07 - DCRNN-style direct-head DCGRU",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config.as_dict(), "metadata": metadata,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "prediction_artifact": prediction_path.relative_to(ROOT).as_posix(),
        "prediction_sha256": sha256(prediction_path),
        "locked_reference_hashes": references,
        "metrics": rows,
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def integrity_only() -> dict[str, Any]:
    data_path, frame, splits, _, _, _, manifest, _, windows = prepare_data(("test",))
    references = verify_locked_artifact_contract(frame, windows["test"])
    output = ROOT / "results" / "task07_dcrnn" / "graph_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"mode": "integrity-only", "data_sha256": sha256(data_path), "split_rows": {k: [v.start, v.stop] for k, v in splits.items()}, "graph_manifest": output.relative_to(ROOT).as_posix(), "locked_artifact_hashes": references, "test_inputs_passed_to_task07_model": False, "test_predictions_or_metrics_calculated": False}


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers cannot be negative")
    if args.warmup_batches < 0 or args.probe_train_batches <= 0 or args.probe_validation_batches <= 0:
        raise ValueError("Probe counts require nonnegative warm-up and positive measured batches")
    if not 0 <= args.safety_margin < 1:
        raise ValueError("Safety margin must be in [0, 1)")
    if args.finalize_only and args.resume:
        raise ValueError("Finalize-only cannot resume training")
    if args.graph_mode == "identity" and (args.full or args.finalize_only):
        raise ValueError("Identity-only full experiments require separate owner approval")
    if args.integrity_only:
        print(json.dumps(integrity_only(), indent=2)); return
    device = resolve_device(args.device)
    if args.finalize_only:
        # This branch is deliberately the only model path allowed to construct test inputs.
        print(json.dumps(finalize_completed_run(args, device), indent=2)); return
    config = training_config(args)
    split_names = ("train",) if args.smoke else ("train", "validation")
    data_path, frame, splits, scaler, forward, reverse, manifest, datasets, windows = prepare_data(split_names)
    metadata = immutable_metadata(data_path, splits, manifest)
    model = DirectHeadDCGRU(config.model, forward, reverse).to(device)
    if args.smoke:
        # At most two optimizer steps, using training windows only.
        generator = configure_reproducibility(config.seed, config.deterministic_algorithms)
        subset = Subset(datasets["train"], range(min(4, len(datasets["train"]))))
        loader = make_loader(subset, batch_size=2, shuffle=False, generator=generator, workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        mean = torch.as_tensor(scaler.mean, dtype=torch.float32, device=device)
        scale = torch.as_tensor(scaler.scale, dtype=torch.float32, device=device)
        loss, count, steps = train_batches(model, loader, optimizer, device, mean, scale, config.gradient_clip_norm, max_batches=2)
        print(json.dumps({"mode": "smoke", "optimizer_steps_at_most": steps, "train_mae_mph": loss, "eligible_count": count, "parameter_count": model.parameter_count, "splits_loaded": ["train"], "test_loaded": False}, indent=2)); return
    probe_generator = configure_reproducibility(config.seed, config.deterministic_algorithms)
    train_loader = make_loader(datasets["train"], batch_size=config.batch_size, shuffle=True, generator=probe_generator, workers=args.num_workers)
    validation_loader = make_loader(datasets["validation"], batch_size=config.batch_size, shuffle=False, generator=None, workers=args.num_workers)
    if args.probe_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        mean = torch.as_tensor(scaler.mean, dtype=torch.float32, device=device)
        scale = torch.as_tensor(scaler.scale, dtype=torch.float32, device=device)
        report = benchmark(model, train_loader, validation_loader, optimizer, device, mean, scale, args, config, metadata, probe_generator)
        report.update({"mode": "probe-only", "device": str(device), "parameter_count": model.parameter_count, "test_loaded": False, "scientific_artifacts_written": False})
        print(json.dumps(report, indent=2)); return
    if args.full:
        output_dir = ROOT / "artifacts" / "task07_dcrnn" / "checkpoints"
        trial = fit(config, datasets, forward, reverse, scaler, device, args.num_workers, output_dir, metadata, args.resume)
        result = {"task": "Task 07", "mode": "full-training-validation-only", "created_utc": datetime.now(timezone.utc).isoformat(), "trial": trial, "metadata": metadata, "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}, "locked_test_accessed": False}
        result_path = ROOT / "results" / "task07_dcrnn" / "validation_trial.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2)); return


if __name__ == "__main__":
    main()
