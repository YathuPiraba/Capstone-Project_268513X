"""Run reproducible Task 04 baselines on the locked chronological test split."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.inspection import load_speed_data  # noqa: E402
from src.data.preprocessing import chronological_split_ranges, target_evaluation_mask  # noqa: E402
from src.data.windows import build_forecast_windows  # noqa: E402
from src.evaluation.metrics import masked_mae, masked_rmse  # noqa: E402
from src.models.baselines import HistoricalAverage, persistence_forecast  # noqa: E402


HORIZONS = {"+15": 2, "+30": 5, "+60": 11}
PRIMARY_MASK_POLICY = "finite_nonzero_targets"
SENSITIVITY_MASK_POLICY = "all_finite_targets_including_zeros"


def main() -> None:
    data_path = ROOT / "data" / "raw" / "metr-la" / "METR-LA.csv"
    artifact_dir = ROOT / "artifacts" / "task04_baselines"
    result_dir = ROOT / "results" / "task04_baselines"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    frame = load_speed_data(data_path)
    speeds = frame.to_numpy(dtype=np.float64, copy=False)
    splits = chronological_split_ranges(frame.index)
    train = splits["train"]
    test_windows = build_forecast_windows(speeds, frame.index, splits["test"])

    historical_average = HistoricalAverage.fit(
        speeds[train.start:train.stop], frame.index[train.start:train.stop]
    )
    predictions = {
        "persistence": persistence_forecast(test_windows.X_speed),
        "historical_average": historical_average.predict(test_windows.target_timestamps),
    }
    selected_steps = np.asarray(list(HORIZONS.values()), dtype=np.int64)
    targets = test_windows.y_speed[:, selected_steps, :]
    selected_predictions = {
        name: values[:, selected_steps, :] for name, values in predictions.items()
    }
    masks = {
        PRIMARY_MASK_POLICY: target_evaluation_mask(targets, exclude_zeros=True),
        SENSITIVITY_MASK_POLICY: target_evaluation_mask(targets, exclude_zeros=False),
    }

    rows: list[dict[str, object]] = []
    for model_name, model_predictions in selected_predictions.items():
        for policy, mask in masks.items():
            for horizon_position, horizon_label in enumerate(HORIZONS):
                horizon_mask = mask[:, horizon_position, :]
                horizon_targets = targets[:, horizon_position, :]
                horizon_predictions = model_predictions[:, horizon_position, :]
                rows.append({
                    "model": model_name,
                    "horizon": horizon_label,
                    "forecast_step_1_based": HORIZONS[horizon_label] + 1,
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
        persistence_mph=selected_predictions["persistence"],
        historical_average_mph=selected_predictions["historical_average"],
        target_timestamps=test_windows.target_timestamps[:, selected_steps],
        forecast_origin_timestamps=test_windows.forecast_origin_timestamps,
        sensor_ids=np.asarray(frame.columns, dtype=str),
        horizon_minutes=np.asarray([15, 30, 60], dtype=np.int64),
        primary_nonzero_mask=masks[PRIMARY_MASK_POLICY],
        finite_including_zero_mask=masks[SENSITIVITY_MASK_POLICY],
    )

    split_manifest = {
        name: {
            "start_row": split.start,
            "stop_row_exclusive": split.stop,
            "row_count": split.count,
            "start_timestamp": split.start_timestamp.isoformat(),
            "end_timestamp": split.end_timestamp.isoformat(),
        }
        for name, split in splits.items()
    }
    metrics_document = {
        "task": "Task 04 - Simple baselines",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path.relative_to(ROOT)).replace("\\", "/"),
        "data_sha256": _sha256(data_path),
        "split_policy": "chronological_70_10_20",
        "splits": split_manifest,
        "window_policy": "12 input steps to 12 target steps; test inputs may use earlier observed context; all targets remain in test",
        "models": {
            "persistence": "repeat final observed input speed for all horizons",
            "historical_average": "training-only per-sensor 5-minute slot-of-week finite-value mean; training-only per-sensor fallback",
        },
        "zero_target_policy": {
            "primary": PRIMARY_MASK_POLICY,
            "sensitivity": SENSITIVITY_MASK_POLICY,
            "note": "Recorded zeros are preserved in targets and predictions. Primary metrics exclude exact-zero targets because their meaning is unresolved; sensitivity metrics include every finite target, including zero.",
        },
        "metrics": rows,
        "prediction_artifact": str(prediction_path.relative_to(ROOT)).replace("\\", "/"),
        "prediction_sha256": _sha256(prediction_path),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "random_seed": None,
        "random_seed_note": "Both baselines are deterministic and use no random operations.",
    }
    metrics_path = result_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_document, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {prediction_path.relative_to(ROOT)} ({prediction_path.stat().st_size} bytes)")
    print(f"Saved {metrics_path.relative_to(ROOT)}")
    for row in rows:
        if row["mask_policy"] == PRIMARY_MASK_POLICY:
            print(
                f"{row['model']} {row['horizon']}: MAE={row['mae_mph']:.6f} mph, "
                f"RMSE={row['rmse_mph']:.6f} mph, n={row['eligible_count']}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    main()
