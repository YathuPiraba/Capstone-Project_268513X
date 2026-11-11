"""Run the Task 05 XGBoost smoke check or owner-approved full experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.inspection import load_speed_data  # noqa: E402
from src.data.preprocessing import (  # noqa: E402
    INPUT_ZERO_POLICY,
    PerSensorStandardizer,
    chronological_split_ranges,
    cyclical_time_of_day,
    target_evaluation_mask,
)
from src.data.windows import build_forecast_windows, window_time_features  # noqa: E402
from src.evaluation.metrics import masked_mae, masked_rmse  # noqa: E402
from src.models.xgboost import (  # noqa: E402
    FEATURE_NAMES,
    HORIZON_STEPS,
    SensorCategoryMap,
    iter_pooled_horizon_batches,
)
from src.training.xgboost import (  # noqa: E402
    DEFAULT_CANDIDATES,
    DEFAULT_SEED,
    load_booster,
    make_quantile_dmatrix,
    predict_batches,
    save_booster,
    select_shared_candidate,
    train_with_early_stopping,
)


PRIMARY_MASK_POLICY = "finite_nonzero_targets"
SENSITIVITY_MASK_POLICY = "all_finite_targets_including_zeros"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 05 XGBoost experiment; full training requires explicit --full."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run a tiny deterministic check")
    mode.add_argument("--full", action="store_true", help="Run full validation selection and test")
    parser.add_argument("--nthread", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--batch-samples", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nthread <= 0 or args.batch_samples <= 0:
        raise ValueError("nthread and batch-samples must be positive")

    mode_name = "SMOKE" if args.smoke else "FULL"
    overall_started = time.perf_counter()
    print(
        f"Task 05 XGBoost {mode_name} run started. Progress is printed live; "
        "keep this terminal open. Press Ctrl+C for a controlled interruption.",
        flush=True,
    )
    print("Phase 1/4: loading data and preparing leakage-safe windows...", flush=True)
    data_path = ROOT / "data" / "raw" / "metr-la" / "METR-LA.csv"
    frame = load_speed_data(data_path)
    speeds = frame.to_numpy(dtype=np.float64, copy=False)
    splits = chronological_split_ranges(frame.index)
    train_range = splits["train"]
    standardizer = PerSensorStandardizer.fit(
        speeds[train_range.start:train_range.stop], exclude_zeros=True
    )
    sensor_id_path = ROOT / "data" / "raw" / "metr-la" / "graph_sensor_ids.txt"
    canonical_sensor_ids = tuple(
        sensor_id.strip()
        for sensor_id in sensor_id_path.read_text(encoding="utf-8").strip().split(",")
    )
    categories = SensorCategoryMap.from_ids(canonical_sensor_ids)
    categories.assert_same_ids(frame.columns)
    if len(categories.sensor_ids) != 207:
        raise ValueError("Canonical METR-LA sensor map must contain exactly 207 sensors")
    time_features = cyclical_time_of_day(frame.index)
    train_windows = build_forecast_windows(speeds, frame.index, splits["train"])
    validation_windows = build_forecast_windows(speeds, frame.index, splits["validation"])
    _, train_future_time = window_time_features(time_features, train_windows)
    _, validation_future_time = window_time_features(time_features, validation_windows)
    print(
        f"Phase 1/4 complete in {_duration(time.perf_counter() - overall_started)}: "
        f"{train_windows.X_speed.shape[0]:,} train and "
        f"{validation_windows.X_speed.shape[0]:,} validation windows.",
        flush=True,
    )

    if args.smoke:
        summary = run_selection(
            train_windows,
            validation_windows,
            train_future_time,
            validation_future_time,
            standardizer,
            categories,
            nthread=args.nthread,
            batch_samples=min(args.batch_samples, 64),
            sample_limit=128,
            candidates=DEFAULT_CANDIDATES[:2],
            num_boost_round=25,
            early_stopping_rounds=5,
            output_dir=ROOT / ".tmp" / "task05_xgboost_smoke",
            retain_selected_models=False,
        )
        summary["mode"] = "smoke"
        summary["input_finite_indicator"] = {
            "nonfinite_observation_count": int((~np.isfinite(speeds)).sum()),
            "constant_on_verified_metr_la": bool(np.isfinite(speeds).all()),
            "purpose": "Retained for preprocessing-contract consistency and future datasets with missing inputs.",
        }
        summary["sensor_id_sha256"] = _sha256(sensor_id_path)
        summary["note"] = (
            "Tiny train/validation subsets only; metrics are engineering checks, "
            "not experimental results."
        )
        print(json.dumps(summary, indent=2))
        return

    artifact_dir = ROOT / "artifacts" / "task05_xgboost"
    result_dir = ROOT / "results" / "task05_xgboost"
    print("Phase 2/4: validation-only selection (9 fits total)...", flush=True)
    selection = run_selection(
        train_windows,
        validation_windows,
        train_future_time,
        validation_future_time,
        standardizer,
        categories,
        nthread=args.nthread,
        batch_samples=args.batch_samples,
        sample_limit=None,
        candidates=DEFAULT_CANDIDATES,
        num_boost_round=1000,
        early_stopping_rounds=50,
        output_dir=artifact_dir,
        retain_selected_models=True,
    )
    selection["input_finite_indicator"] = {
        "nonfinite_observation_count": int((~np.isfinite(speeds)).sum()),
        "constant_on_verified_metr_la": bool(np.isfinite(speeds).all()),
        "purpose": "Retained for preprocessing-contract consistency and future datasets with missing inputs.",
    }
    selection["sensor_id_sha256"] = _sha256(sensor_id_path)
    (artifact_dir / "selected_config.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "validation_trials.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Phase 2/4 complete after {_duration(time.perf_counter() - overall_started)}. "
        f"Selected {selection['selected_candidate']}.",
        flush=True,
    )
    print("Phase 3/4: verifying Task 04 alignment and running locked test inference...", flush=True)
    evaluate_locked_test(
        frame,
        speeds,
        splits,
        standardizer,
        categories,
        time_features,
        selection,
        artifact_dir,
        result_dir,
        data_path,
        nthread=args.nthread,
        batch_samples=args.batch_samples,
    )
    print(
        f"Phase 4/4 complete: artifacts and results saved. Total elapsed "
        f"{_duration(time.perf_counter() - overall_started)}.",
        flush=True,
    )


def run_selection(
    train_windows,
    validation_windows,
    train_future_time: np.ndarray,
    validation_future_time: np.ndarray,
    standardizer: PerSensorStandardizer,
    categories: SensorCategoryMap,
    *,
    nthread: int,
    batch_samples: int,
    sample_limit: int | None,
    candidates,
    num_boost_round: int,
    early_stopping_rounds: int,
    output_dir: Path,
    retain_selected_models: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    memory_start = _peak_working_set_bytes()
    trial_dir = output_dir / "selection_trials"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, object]] = []
    counts: dict[str, dict[str, int]] = {}
    completed_fit_durations: list[float] = []
    total_fits = len(HORIZON_STEPS) * len(candidates)
    fit_number = 0

    for horizon_position, horizon_minutes in enumerate(HORIZON_STEPS, start=1):
        train_factory = _batch_factory(
            train_windows, standardizer, train_future_time, categories, horizon_minutes,
            batch_samples, sample_limit, True
        )
        validation_factory = _batch_factory(
            validation_windows, standardizer, validation_future_time, categories,
            horizon_minutes, batch_samples, sample_limit, True
        )
        counts[str(horizon_minutes)] = {
            "train_included": sum(batch.included_count for batch in train_factory()),
            "train_excluded": sum(batch.excluded_count for batch in train_factory()),
            "validation_included": sum(batch.included_count for batch in validation_factory()),
            "validation_excluded": sum(batch.excluded_count for batch in validation_factory()),
        }
        train_sample_count = min(
            train_windows.X_speed.shape[0],
            sample_limit or train_windows.X_speed.shape[0],
        )
        validation_sample_count = min(
            validation_windows.X_speed.shape[0],
            sample_limit or validation_windows.X_speed.shape[0],
        )
        print(
            f"  Horizon {horizon_position}/3 (+{horizon_minutes} min): building "
            "training and validation matrices...",
            flush=True,
        )
        matrix_started = time.perf_counter()
        dtrain = make_quantile_dmatrix(
            train_factory,
            nthread=nthread,
            progress_label=f"+{horizon_minutes} train matrix",
            expected_batches=math.ceil(train_sample_count / batch_samples),
        )
        dvalidation = make_quantile_dmatrix(
            validation_factory,
            nthread=nthread,
            reference=dtrain,
            progress_label=f"+{horizon_minutes} validation matrix",
            expected_batches=math.ceil(validation_sample_count / batch_samples),
        )
        print(
            f"  Horizon +{horizon_minutes} matrices ready in "
            f"{_duration(time.perf_counter() - matrix_started)}.",
            flush=True,
        )
        for candidate in candidates:
            fit_number += 1
            fit_started = time.perf_counter()
            remaining_fit_eta = (
                np.mean(completed_fit_durations) * (total_fits - fit_number + 1)
                if completed_fit_durations
                else None
            )
            eta_text = (
                f"; rolling remaining-fit ETA {_duration(float(remaining_fit_eta))}"
                if remaining_fit_eta is not None
                else ""
            )
            print(
                f"  Validation fit {fit_number}/{total_fits} "
                f"({(fit_number - 1) / total_fits:.1%} complete): "
                f"+{horizon_minutes} min, {candidate.name}{eta_text}",
                flush=True,
            )
            booster, trial = train_with_early_stopping(
                dtrain,
                dvalidation,
                candidate,
                nthread=nthread,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                progress_label=f"fit {fit_number}/{total_fits} +{horizon_minutes} {candidate.name}",
                progress_interval=5 if sample_limit is not None else 25,
            )
            completed_fit_durations.append(time.perf_counter() - fit_started)
            trial["horizon_minutes"] = horizon_minutes
            trial.pop("history", None)
            trials.append(trial)
            save_booster(
                booster,
                trial_dir / f"{candidate.name}_horizon_{horizon_minutes}.json",
            )
            average_fit = float(np.mean(completed_fit_durations))
            print(
                f"  Validation fit {fit_number}/{total_fits} complete "
                f"({fit_number / total_fits:.1%}); fit elapsed "
                f"{_duration(completed_fit_durations[-1])}; estimated remaining fits "
                f"{_duration(average_fit * (total_fits - fit_number))}.",
                flush=True,
            )

    selected_name = select_shared_candidate(trials)
    selected_trials = {
        str(trial["horizon_minutes"]): trial
        for trial in trials
        if trial["candidate"] == selected_name
    }
    selected_models: dict[str, str] = {}
    if retain_selected_models:
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for horizon_minutes in HORIZON_STEPS:
            source = trial_dir / f"{selected_name}_horizon_{horizon_minutes}.json"
            destination = model_dir / f"horizon_{horizon_minutes}.json"
            shutil.copy2(source, destination)
            selected_models[str(horizon_minutes)] = destination.relative_to(ROOT).as_posix()
    shutil.rmtree(trial_dir)

    elapsed = time.perf_counter() - started
    summary: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": DEFAULT_SEED,
        "nthread": nthread,
        "batch_samples": batch_samples,
        "sample_limit_per_split": sample_limit,
        "feature_count": len(FEATURE_NAMES),
        "input_zero_policy": INPUT_ZERO_POLICY,
        "supervised_target_policy": PRIMARY_MASK_POLICY,
        "candidates": [candidate.__dict__ for candidate in candidates],
        "tie_breaking": "lowest mean three-horizon validation MAE, then candidate name",
        "trials": trials,
        "selected_candidate": selected_name,
        "selected_horizon_trials": selected_trials,
        "selected_models": selected_models,
        "target_counts": counts,
        "resource_bounds": {
            "max_float32_feature_batch_bytes": batch_samples * len(categories.sensor_ids) * len(FEATURE_NAMES) * 4,
            "fully_materialized_train_features_bytes_avoided": train_windows.X_speed.shape[0] * len(categories.sensor_ids) * len(FEATURE_NAMES) * 4,
            "note": "XGBoost native quantile storage is additional and runtime-dependent.",
        },
        "elapsed_seconds": elapsed,
        "peak_working_set_bytes": _peak_working_set_bytes(),
        "peak_working_set_start_bytes": memory_start,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
    }
    return summary


def evaluate_locked_test(
    frame,
    speeds,
    splits,
    standardizer,
    categories,
    time_features,
    selection,
    artifact_dir,
    result_dir,
    data_path,
    *,
    nthread: int,
    batch_samples: int,
) -> None:
    test_windows = build_forecast_windows(speeds, frame.index, splits["test"])
    _, test_future_time = window_time_features(time_features, test_windows)
    selected_steps = np.asarray(list(HORIZON_STEPS.values()), dtype=np.int64)
    targets = np.asarray(test_windows.y_speed[:, selected_steps, :], dtype=np.float64)
    primary_mask = target_evaluation_mask(targets, exclude_zeros=True)
    sensitivity_mask = target_evaluation_mask(targets, exclude_zeros=False)

    baseline_path = ROOT / "artifacts" / "task04_baselines" / "test_predictions_horizons.npz"
    assert_task04_alignment(
        baseline_path, frame, test_windows, targets, primary_mask, sensitivity_mask
    )

    predictions = np.empty_like(targets, dtype=np.float32)
    for horizon_position, horizon_minutes in enumerate(HORIZON_STEPS):
        print(
            f"  Test inference horizon {horizon_position + 1}/3 (+{horizon_minutes} min)...",
            flush=True,
        )
        factory = _batch_factory(
            test_windows, standardizer, test_future_time, categories, horizon_minutes,
            batch_samples, None, False
        )
        booster = load_booster(artifact_dir / "models" / f"horizon_{horizon_minutes}.json")
        rounds = int(
            selection["selected_horizon_trials"][str(horizon_minutes)]["boosting_rounds"]
        )
        values, sample_indices, sensor_indices = predict_batches(
            booster, factory, iteration_count=rounds
        )
        expected_samples = np.repeat(np.arange(targets.shape[0]), targets.shape[2])
        expected_sensors = np.tile(np.arange(targets.shape[2]), targets.shape[0])
        np.testing.assert_array_equal(sample_indices, expected_samples)
        np.testing.assert_array_equal(sensor_indices, expected_sensors)
        predictions[:, horizon_position, :] = values.reshape(targets.shape[0], targets.shape[2])
        print(
            f"  Test inference horizon {horizon_position + 1}/3 complete.",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    for policy, mask in (
        (PRIMARY_MASK_POLICY, primary_mask),
        (SENSITIVITY_MASK_POLICY, sensitivity_mask),
    ):
        for position, horizon_minutes in enumerate(HORIZON_STEPS):
            horizon_mask = mask[:, position, :]
            horizon_targets = targets[:, position, :]
            horizon_predictions = predictions[:, position, :]
            rows.append({
                "model": "xgboost",
                "horizon": f"+{horizon_minutes}",
                "forecast_step_1_based": HORIZON_STEPS[horizon_minutes] + 1,
                "mask_policy": policy,
                "eligible_count": int(horizon_mask.sum()),
                "excluded_count": int(horizon_mask.size - horizon_mask.sum()),
                "mae_mph": masked_mae(horizon_targets, horizon_predictions, horizon_mask),
                "rmse_mph": masked_rmse(horizon_targets, horizon_predictions, horizon_mask),
            })

    prediction_path = artifact_dir / "test_predictions_horizons.npz"
    np.savez_compressed(
        prediction_path,
        targets_mph=targets,
        xgboost_mph=predictions,
        target_timestamps=test_windows.target_timestamps[:, selected_steps],
        forecast_origin_timestamps=test_windows.forecast_origin_timestamps,
        sensor_ids=np.asarray(frame.columns, dtype=str),
        horizon_minutes=np.asarray(list(HORIZON_STEPS), dtype=np.int64),
        primary_nonzero_mask=primary_mask,
        finite_including_zero_mask=sensitivity_mask,
    )
    metrics = {
        "task": "Task 05 - Conventional ML with XGBoost",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path.relative_to(ROOT).as_posix(),
        "data_sha256": _sha256(data_path),
        "split_policy": "chronological_70_10_20",
        "splits": {
            name: {
                "start_row": split.start,
                "stop_row_exclusive": split.stop,
                "row_count": split.count,
                "start_timestamp": split.start_timestamp.isoformat(),
                "end_timestamp": split.end_timestamp.isoformat(),
            }
            for name, split in splits.items()
        },
        "window_policy": "12 input steps to direct target steps 3, 6, and 12",
        "feature_policy": "sensor-local history, input masks, target time-of-day, categorical sensor identity; no graph or neighbouring speeds",
        "selection_artifact": "results/task05_xgboost/validation_trials.json",
        "prediction_artifact": prediction_path.relative_to(ROOT).as_posix(),
        "prediction_sha256": _sha256(prediction_path),
        "baseline_alignment_verified": True,
        "metrics": rows,
        "versions": selection["versions"],
        "seed": DEFAULT_SEED,
        "nthread": nthread,
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def _batch_factory(
    windows,
    standardizer,
    future_time,
    categories,
    horizon_minutes,
    batch_samples,
    sample_limit,
    filter_primary_targets,
):
    return lambda: iter_pooled_horizon_batches(
        windows,
        standardizer,
        future_time,
        categories,
        horizon_minutes,
        batch_samples=batch_samples,
        sample_limit=sample_limit,
        filter_primary_targets=filter_primary_targets,
    )


def assert_task04_alignment(
    baseline_path: Path,
    frame,
    test_windows,
    targets,
    primary_mask,
    sensitivity_mask,
) -> None:
    if not baseline_path.exists():
        raise FileNotFoundError(
            "Task 04 prediction artifact is required before locked test inference"
        )
    with np.load(baseline_path) as baseline:
        np.testing.assert_array_equal(baseline["sensor_ids"], np.asarray(frame.columns, dtype=str))
        np.testing.assert_array_equal(baseline["horizon_minutes"], [15, 30, 60])
        np.testing.assert_array_equal(baseline["targets_mph"], targets)
        np.testing.assert_array_equal(
            baseline["target_timestamps"],
            test_windows.target_timestamps[:, list(HORIZON_STEPS.values())],
        )
        np.testing.assert_array_equal(baseline["primary_nonzero_mask"], primary_mask)
        np.testing.assert_array_equal(
            baseline["finite_including_zero_mask"], sensitivity_mask
        )


def _peak_working_set_bytes() -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    ):
        return None
    return int(counters.PeakWorkingSetSize)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nTask 05 run interrupted by the user. No test metrics are valid unless "
            "results/task05_xgboost/metrics.json was written by a completed run. "
            "Rerunning the same command safely replaces Task 05 temporary selection files.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130)
