"""Locked-test comparison utilities for UrbanFlow AI Task 08.

The module reads existing prediction artifacts only. It never trains models,
runs inference, changes evaluation masks, or rewrites source artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256 as _hash_factory
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .metrics import masked_mae, masked_rmse


HORIZONS = np.asarray([15, 30, 60], dtype=np.int64)
EXPECTED_SHAPE = (6844, 3, 207)
PRIMARY_POLICY = "finite_nonzero_targets"
SENSITIVITY_POLICY = "all_finite_targets_including_zeros"
POLICY_MASK_KEYS = {
    PRIMARY_POLICY: "primary_nonzero_mask",
    SENSITIVITY_POLICY: "finite_including_zero_mask",
}
MODEL_LABELS = {
    "persistence": "Persistence",
    "historical_average": "Historical Average",
    "xgboost": "XGBoost",
    "gru": "GRU-128",
    "dcrnn": "DCGRU-64",
}
PREDICTION_KEYS = {
    "persistence": "persistence_mph",
    "historical_average": "historical_average_mph",
    "xgboost": "xgboost_mph",
    "gru": "gru_mph",
    "dcrnn": "dcrnn_mph",
}
COMPARATORS = ("gru", "xgboost", "persistence", "historical_average")
TIME_OF_DAY_GROUPS = (
    ("overnight_00_06", 0, 6),
    ("morning_06_12", 6, 12),
    ("afternoon_12_18", 12, 18),
    ("evening_18_24", 18, 24),
)


@dataclass(frozen=True)
class ArtifactSpec:
    task: str
    artifact_path: Path
    manifest_path: Path
    models: tuple[str, ...]


@dataclass
class ComparisonData:
    targets: np.ndarray
    predictions: dict[str, np.ndarray]
    target_timestamps: np.ndarray
    forecast_origin_timestamps: np.ndarray
    sensor_ids: np.ndarray
    horizons: np.ndarray
    masks: dict[str, np.ndarray]
    artifact_hashes: dict[str, str]
    data_sha256: str
    split_rows: dict[str, tuple[int, int]]


def artifact_specs(root: Path) -> tuple[ArtifactSpec, ...]:
    """Return the four immutable Task 04-07 prediction sources."""
    return (
        ArtifactSpec(
            "task04_baselines",
            root / "artifacts/task04_baselines/test_predictions_horizons.npz",
            root / "results/task04_baselines/metrics.json",
            ("persistence", "historical_average"),
        ),
        ArtifactSpec(
            "task05_xgboost",
            root / "artifacts/task05_xgboost/test_predictions_horizons.npz",
            root / "results/task05_xgboost/metrics.json",
            ("xgboost",),
        ),
        ArtifactSpec(
            "task06_gru",
            root / "artifacts/task06_gru/test_predictions_horizons.npz",
            root / "results/task06_gru/metrics.json",
            ("gru",),
        ),
        ArtifactSpec(
            "task07_dcrnn",
            root / "artifacts/task07_dcrnn/test_predictions_horizons.npz",
            root / "results/task07_dcrnn/metrics.json",
            ("dcrnn",),
        ),
    )


def file_sha256(path: Path) -> str:
    digest = _hash_factory()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def source_hashes(root: Path) -> dict[str, str]:
    return {spec.task: file_sha256(spec.artifact_path) for spec in artifact_specs(root)}


def load_and_validate_artifacts(
    root: Path,
    *,
    expected_shape: tuple[int, int, int] = EXPECTED_SHAPE,
) -> ComparisonData:
    """Load frozen predictions after strict manifest and array alignment checks."""
    specs = artifact_specs(root)
    for spec in specs:
        if not spec.artifact_path.is_file():
            raise FileNotFoundError(f"Missing prediction artifact: {spec.artifact_path}")
        if not spec.manifest_path.is_file():
            raise FileNotFoundError(f"Missing metrics manifest: {spec.manifest_path}")

    reference: dict[str, np.ndarray] | None = None
    predictions: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    common_data_hash: str | None = None
    common_splits: dict[str, tuple[int, int]] | None = None

    shared_keys = (
        "targets_mph",
        "forecast_origin_timestamps",
        "target_timestamps",
        "sensor_ids",
        "horizon_minutes",
        "primary_nonzero_mask",
        "finite_including_zero_mask",
    )

    for spec in specs:
        manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
        actual_hash = file_sha256(spec.artifact_path)
        recorded_hash = str(manifest.get("prediction_sha256", "")).upper()
        if actual_hash != recorded_hash:
            raise ValueError(
                f"Prediction SHA-256 mismatch for {spec.task}: "
                f"expected {recorded_hash}, found {actual_hash}"
            )
        hashes[spec.task] = actual_hash

        data_hash = _manifest_data_hash(manifest)
        splits = _manifest_split_rows(manifest)
        if common_data_hash is None:
            common_data_hash = data_hash
            common_splits = splits
        elif data_hash != common_data_hash or splits != common_splits:
            raise ValueError(f"Dataset or split provenance mismatch in {spec.task}")
        _validate_manifest_policies(manifest, spec.task)

        with np.load(spec.artifact_path, allow_pickle=False) as archive:
            missing = set(shared_keys) - set(archive.files)
            missing.update(PREDICTION_KEYS[m] for m in spec.models if PREDICTION_KEYS[m] not in archive.files)
            if missing:
                raise ValueError(f"Missing arrays in {spec.task}: {sorted(missing)}")
            current = {key: np.array(archive[key], copy=True) for key in shared_keys}
            if current["targets_mph"].shape != expected_shape:
                raise ValueError(
                    f"Unexpected prediction shape in {spec.task}: "
                    f"{current['targets_mph'].shape}, expected {expected_shape}"
                )
            if current["target_timestamps"].shape != expected_shape[:2]:
                raise ValueError(f"Unexpected target timestamp shape in {spec.task}")
            if current["forecast_origin_timestamps"].shape != (expected_shape[0],):
                raise ValueError(f"Unexpected forecast-origin shape in {spec.task}")
            if current["sensor_ids"].shape != (expected_shape[2],):
                raise ValueError(f"Unexpected sensor-ID shape in {spec.task}")
            np.testing.assert_array_equal(
                current["horizon_minutes"], HORIZONS,
                err_msg=f"Incorrect horizon array in {spec.task}",
            )
            for mask_key in POLICY_MASK_KEYS.values():
                if current[mask_key].dtype.kind != "b" or current[mask_key].shape != expected_shape:
                    raise ValueError(f"Invalid evaluation mask {mask_key} in {spec.task}")

            if reference is None:
                reference = current
            else:
                for key in shared_keys:
                    np.testing.assert_array_equal(
                        reference[key], current[key],
                        err_msg=f"Locked-test alignment mismatch for {key} in {spec.task}",
                    )

            for model in spec.models:
                values = np.asarray(archive[PREDICTION_KEYS[model]])
                if values.shape != expected_shape:
                    raise ValueError(f"Unexpected {model} prediction shape: {values.shape}")
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Nonfinite predictions found for {model}")
                predictions[model] = np.array(values, copy=True)

    assert reference is not None and common_data_hash is not None and common_splits is not None
    if set(predictions) != set(MODEL_LABELS):
        raise ValueError(f"Model set mismatch: {sorted(predictions)}")
    return ComparisonData(
        targets=reference["targets_mph"],
        predictions=predictions,
        target_timestamps=reference["target_timestamps"],
        forecast_origin_timestamps=reference["forecast_origin_timestamps"],
        sensor_ids=reference["sensor_ids"],
        horizons=reference["horizon_minutes"],
        masks={policy: reference[key] for policy, key in POLICY_MASK_KEYS.items()},
        artifact_hashes=hashes,
        data_sha256=common_data_hash,
        split_rows=common_splits,
    )


def overall_metric_rows(data: ComparisonData) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODEL_LABELS:
        for policy, mask in data.masks.items():
            for j, horizon in enumerate(data.horizons):
                eligible = mask[:, j, :]
                rows.append(_metric_row(
                    model, int(horizon), policy, data.targets[:, j, :],
                    data.predictions[model][:, j, :], eligible,
                ))
    return rows


def reconcile_published_metrics(
    root: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    atol: float = 1e-9,
) -> None:
    """Require every published Task 04-07 metric to match recomputation."""
    computed = {
        (str(r["model"]), int(r["horizon_minutes"]), str(r["mask_policy"])): r
        for r in rows
    }
    manifest_model_names = {
        "persistence": "persistence",
        "historical_average": "historical_average",
        "xgboost": "xgboost",
        "gru": "gru",
        "direct_head_dcgru": "dcrnn",
    }
    for spec in artifact_specs(root):
        manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
        for published in manifest["metrics"]:
            model = manifest_model_names[str(published["model"])]
            horizon = int(str(published["horizon"]).lstrip("+"))
            key = (model, horizon, str(published["mask_policy"]))
            actual = computed[key]
            for metric in ("mae_mph", "rmse_mph"):
                if not np.isclose(float(actual[metric]), float(published[metric]), rtol=0.0, atol=atol):
                    raise ValueError(f"Published {metric} does not reconcile for {key}")
            for count in ("eligible_count", "excluded_count"):
                if int(actual[count]) != int(published[count]):
                    raise ValueError(f"Published {count} does not reconcile for {key}")


def equal_horizon_mean_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["model", "model_label", "mask_policy"], sort=False)
    result: list[dict[str, object]] = []
    for keys, group in grouped:
        if set(group["horizon_minutes"]) != set(HORIZONS.tolist()):
            raise ValueError(f"Cannot calculate equal-horizon mean for incomplete group {keys}")
        result.append({
            "model": keys[0],
            "model_label": keys[1],
            "mask_policy": keys[2],
            "equal_horizon_mean_mae_mph": float(group["mae_mph"].mean()),
        })
    return result


def pairwise_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (str(r["model"]), int(r["horizon_minutes"]), str(r["mask_policy"])): r
        for r in rows
    }
    output: list[dict[str, object]] = []
    for policy in POLICY_MASK_KEYS:
        horizon_improvements: dict[str, list[float]] = {name: [] for name in COMPARATORS}
        horizon_differences: dict[str, list[float]] = {name: [] for name in COMPARATORS}
        for horizon in HORIZONS:
            dcgru = float(lookup[("dcrnn", int(horizon), policy)]["mae_mph"])
            for comparator in COMPARATORS:
                baseline = float(lookup[(comparator, int(horizon), policy)]["mae_mph"])
                improvement = percentage_improvement(baseline, dcgru)
                difference = dcgru - baseline
                horizon_improvements[comparator].append(improvement)
                horizon_differences[comparator].append(difference)
                output.append({
                    "mask_policy": policy,
                    "horizon_minutes": int(horizon),
                    "summary": "per_horizon",
                    "comparator": comparator,
                    "comparator_label": MODEL_LABELS[comparator],
                    "dcgru_mae_mph": dcgru,
                    "comparator_mae_mph": baseline,
                    "dcgru_minus_comparator_mae_mph": difference,
                    "improvement_percent": improvement,
                })
        means = equal_horizon_mean_rows(rows)
        mean_lookup = {(r["model"], r["mask_policy"]): r for r in means}
        dcgru_mean = float(mean_lookup[("dcrnn", policy)]["equal_horizon_mean_mae_mph"])
        for comparator in COMPARATORS:
            baseline_mean = float(mean_lookup[(comparator, policy)]["equal_horizon_mean_mae_mph"])
            output.append({
                "mask_policy": policy,
                "horizon_minutes": "mean",
                "summary": "equal_horizon_mean",
                "comparator": comparator,
                "comparator_label": MODEL_LABELS[comparator],
                "dcgru_mae_mph": dcgru_mean,
                "comparator_mae_mph": baseline_mean,
                "dcgru_minus_comparator_mae_mph": dcgru_mean - baseline_mean,
                "improvement_percent": percentage_improvement(baseline_mean, dcgru_mean),
            })
    return output


def percentage_improvement(comparator_mae: float, dcgru_mae: float) -> float:
    if not np.isfinite(comparator_mae) or comparator_mae == 0:
        return float("nan")
    return float(100.0 * (comparator_mae - dcgru_mae) / comparator_mae)


def sensor_metric_rows(data: ComparisonData) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    pairwise: list[dict[str, object]] = []
    mask = data.masks[PRIMARY_POLICY]
    for j, horizon in enumerate(data.horizons):
        for sensor_index, sensor_id in enumerate(data.sensor_ids):
            eligible = mask[:, j, sensor_index]
            count = int(eligible.sum())
            metric_by_model: dict[str, dict[str, float]] = {}
            for model in MODEL_LABELS:
                if count:
                    target = data.targets[:, j, sensor_index]
                    pred = data.predictions[model][:, j, sensor_index]
                    mae = masked_mae(target, pred, eligible)
                    rmse = masked_rmse(target, pred, eligible)
                else:
                    mae = rmse = float("nan")
                metric_by_model[model] = {"mae_mph": mae, "rmse_mph": rmse}
                rows.append({
                    "sensor_index": sensor_index,
                    "sensor_id": str(sensor_id),
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "horizon_minutes": int(horizon),
                    "mask_policy": PRIMARY_POLICY,
                    "eligible_count": count,
                    "mae_mph": mae,
                    "rmse_mph": rmse,
                })
            gru_mae = metric_by_model["gru"]["mae_mph"]
            dcgru_mae = metric_by_model["dcrnn"]["mae_mph"]
            pairwise.append({
                "sensor_index": sensor_index,
                "sensor_id": str(sensor_id),
                "horizon_minutes": int(horizon),
                "eligible_count": count,
                "dcgru_minus_gru_mae_mph": dcgru_mae - gru_mae,
                "improvement_percent": percentage_improvement(gru_mae, dcgru_mae),
                "dcgru_improved": bool(dcgru_mae < gru_mae),
            })
    pair_lookup = {
        (int(row["sensor_index"]), int(row["horizon_minutes"])): row
        for row in pairwise
    }
    for row in rows:
        paired = pair_lookup[(int(row["sensor_index"]), int(row["horizon_minutes"]))]
        row.update({
            "dcgru_minus_gru_mae_mph": paired["dcgru_minus_gru_mae_mph"],
            "dcgru_vs_gru_improvement_percent": paired["improvement_percent"],
            "dcgru_improved_vs_gru": paired["dcgru_improved"],
        })
    return rows, pairwise


def sensor_pairwise_summary(pairwise_rows_: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(pairwise_rows_)
    summaries: list[dict[str, object]] = []
    for horizon, group in frame.groupby("horizon_minutes", sort=True):
        valid = group[np.isfinite(group["dcgru_minus_gru_mae_mph"])]
        best = valid.loc[valid["dcgru_minus_gru_mae_mph"].idxmin()]
        worst = valid.loc[valid["dcgru_minus_gru_mae_mph"].idxmax()]
        improved = int(valid["dcgru_improved"].sum())
        summaries.append({
            "horizon_minutes": int(horizon),
            "valid_sensor_count": int(len(valid)),
            "improved_sensor_count": improved,
            "improved_sensor_percent": float(100.0 * improved / len(valid)),
            "median_dcgru_minus_gru_mae_mph": float(valid["dcgru_minus_gru_mae_mph"].median()),
            "median_improvement_percent": float(valid["improvement_percent"].median()),
            "best_sensor_id": str(best["sensor_id"]),
            "best_dcgru_minus_gru_mae_mph": float(best["dcgru_minus_gru_mae_mph"]),
            "worst_sensor_id": str(worst["sensor_id"]),
            "worst_dcgru_minus_gru_mae_mph": float(worst["dcgru_minus_gru_mae_mph"]),
        })
    return summaries


def observed_speed_strata_rows(data: ComparisonData) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    bands = (
        ("low_below_20", lambda x: x < 20.0),
        ("medium_20_to_below_40", lambda x: (x >= 20.0) & (x < 40.0)),
        ("high_40_and_above", lambda x: x >= 40.0),
    )
    for policy, policy_mask in data.masks.items():
        for j, horizon in enumerate(data.horizons):
            eligible_total = int(policy_mask[:, j, :].sum())
            coverage = np.zeros(policy_mask[:, j, :].shape, dtype=np.int8)
            for band, predicate in bands:
                band_mask = policy_mask[:, j, :] & predicate(data.targets[:, j, :])
                coverage += band_mask.astype(np.int8)
                count = int(band_mask.sum())
                for model in MODEL_LABELS:
                    mae, rmse = _masked_metrics_or_nan(
                        data.targets[:, j, :], data.predictions[model][:, j, :], band_mask
                    )
                    rows.append({
                        "stratum_label": "Observed-Speed Evaluation Strata",
                        "speed_band": band,
                        "mask_policy": policy,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "horizon_minutes": int(horizon),
                        "eligible_count": count,
                        "eligible_percent": float(100.0 * count / eligible_total),
                        "mae_mph": mae,
                        "rmse_mph": rmse,
                    })
            if not np.array_equal(coverage[policy_mask[:, j, :]], np.ones(eligible_total, dtype=np.int8)):
                raise ValueError(f"Speed bands do not partition eligible cases at +{horizon}")
    return rows


def temporal_metric_rows(data: ComparisonData) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for j, horizon in enumerate(data.horizons):
        timestamps = pd.DatetimeIndex(data.target_timestamps[:, j])
        groupings: list[tuple[str, np.ndarray]] = []
        hours = timestamps.hour.to_numpy()
        for name, start, stop in TIME_OF_DAY_GROUPS:
            groupings.append((f"time_of_day:{name}", (hours >= start) & (hours < stop)))
        groupings.extend((
            ("day_type:weekday", timestamps.day_of_week.to_numpy() < 5),
            ("day_type:weekend", timestamps.day_of_week.to_numpy() >= 5),
        ))
        dates = timestamps.normalize()
        for date in dates.unique().sort_values():
            groupings.append((f"target_date:{date.date().isoformat()}", np.asarray(dates == date)))

        for policy, mask in data.masks.items():
            for group_key, window_selector in groupings:
                group_type, group = group_key.split(":", 1)
                group_mask = mask[:, j, :] & window_selector[:, None]
                count = int(group_mask.sum())
                for model in MODEL_LABELS:
                    mae, rmse = _masked_metrics_or_nan(
                        data.targets[:, j, :], data.predictions[model][:, j, :], group_mask
                    )
                    rows.append({
                        "group_type": group_type,
                        "group": group,
                        "mask_policy": policy,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "horizon_minutes": int(horizon),
                        "eligible_count": count,
                        "mae_mph": mae,
                        "rmse_mph": rmse,
                    })
    return rows


def daily_absolute_error_summaries(
    data: ComparisonData,
    models: Iterable[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Return date labels, model error sums, and counts for paired bootstrapping."""
    date_matrix = data.target_timestamps.astype("datetime64[D]")
    dates = np.unique(date_matrix)
    counts = np.zeros((len(dates), len(HORIZONS)), dtype=np.int64)
    sums = {model: np.zeros_like(counts, dtype=np.float64) for model in models}
    mask = data.masks[PRIMARY_POLICY]
    for d, date in enumerate(dates):
        for j in range(len(HORIZONS)):
            eligible = mask[:, j, :] & (date_matrix[:, j] == date)[:, None]
            counts[d, j] = int(eligible.sum())
            for model in sums:
                errors = np.abs(data.predictions[model][:, j, :] - data.targets[:, j, :])
                sums[model][d, j] = float(errors[eligible].sum())
    if np.any(counts == 0):
        raise ValueError("At least one target-date/horizon cluster has no eligible cases")
    return dates, sums, counts


def paired_day_cluster_bootstrap(
    daily_sums: Mapping[str, np.ndarray],
    daily_counts: np.ndarray,
    *,
    comparator: str,
    repetitions: int = 10_000,
    seed: int = 268_513,
) -> dict[str, object]:
    """Bootstrap paired pooled-MAE differences using whole target dates."""
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive")
    dcgru = np.asarray(daily_sums["dcrnn"], dtype=np.float64)
    other = np.asarray(daily_sums[comparator], dtype=np.float64)
    counts = np.asarray(daily_counts, dtype=np.int64)
    if dcgru.shape != other.shape or dcgru.shape != counts.shape or dcgru.ndim != 2:
        raise ValueError("Daily sums and counts must have matching date-by-horizon shapes")
    if np.any(counts <= 0):
        raise ValueError("Daily eligible counts must be positive")

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, counts.shape[0], size=(repetitions, counts.shape[0]))
    sampled_counts = counts[sampled].sum(axis=1)
    dcgru_mae = dcgru[sampled].sum(axis=1) / sampled_counts
    comparator_mae = other[sampled].sum(axis=1) / sampled_counts
    differences = dcgru_mae - comparator_mae
    point = dcgru.sum(axis=0) / counts.sum(axis=0) - other.sum(axis=0) / counts.sum(axis=0)
    mean_differences = differences.mean(axis=1)
    rows = []
    for j, horizon in enumerate(HORIZONS):
        low, high = np.percentile(differences[:, j], [2.5, 97.5])
        rows.append({
            "horizon_minutes": int(horizon),
            "dcgru_minus_comparator_mae_mph": float(point[j]),
            "interval_95_low_mph": float(low),
            "interval_95_high_mph": float(high),
        })
    low, high = np.percentile(mean_differences, [2.5, 97.5])
    return {
        "comparator": comparator,
        "comparator_label": MODEL_LABELS[comparator],
        "rows": rows,
        "equal_horizon_mean": {
            "dcgru_minus_comparator_mae_mph": float(point.mean()),
            "interval_95_low_mph": float(low),
            "interval_95_high_mph": float(high),
        },
    }


def best_primary_models(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    frame = frame[frame["mask_policy"] == PRIMARY_POLICY]
    output: list[dict[str, object]] = []
    for horizon, group in frame.groupby("horizon_minutes", sort=True):
        for metric in ("mae_mph", "rmse_mph"):
            best = group.loc[group[metric].idxmin()]
            output.append({
                "horizon_minutes": int(horizon),
                "metric": metric,
                "model": str(best["model"]),
                "model_label": str(best["model_label"]),
                "value_mph": float(best[metric]),
            })
    return output


def _metric_row(
    model: str,
    horizon: int,
    policy: str,
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    count = int(mask.sum())
    return {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "horizon_minutes": horizon,
        "mask_policy": policy,
        "eligible_count": count,
        "excluded_count": int(mask.size - count),
        "mae_mph": masked_mae(targets, predictions, mask),
        "rmse_mph": masked_rmse(targets, predictions, mask),
    }


def _masked_metrics_or_nan(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Return transparent NaNs for an empty exploratory subgroup."""
    if not np.any(mask):
        return float("nan"), float("nan")
    return masked_mae(targets, predictions, mask), masked_rmse(targets, predictions, mask)


def _manifest_data_hash(manifest: Mapping[str, object]) -> str:
    if "data_sha256" in manifest:
        return str(manifest["data_sha256"]).upper()
    metadata = manifest.get("metadata")
    if isinstance(metadata, Mapping) and "data_sha256" in metadata:
        return str(metadata["data_sha256"]).upper()
    raise ValueError("Manifest is missing dataset SHA-256")


def _manifest_split_rows(manifest: Mapping[str, object]) -> dict[str, tuple[int, int]]:
    if isinstance(manifest.get("splits"), Mapping):
        result = {}
        for name, value in manifest["splits"].items():
            if not isinstance(value, Mapping):
                raise ValueError("Invalid split manifest")
            result[str(name)] = (int(value["start_row"]), int(value["stop_row_exclusive"]))
        return result
    metadata = manifest.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("split_rows"), Mapping):
        return {
            str(name): (int(bounds[0]), int(bounds[1]))
            for name, bounds in metadata["split_rows"].items()
        }
    raise ValueError("Manifest is missing split rows")


def _validate_manifest_policies(manifest: Mapping[str, object], task: str) -> None:
    rows = manifest.get("metrics")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Manifest metrics missing in {task}")
    policies = {str(row.get("mask_policy")) for row in rows if isinstance(row, Mapping)}
    if policies != set(POLICY_MASK_KEYS):
        raise ValueError(f"Evaluation-policy mismatch in {task}: {sorted(policies)}")

