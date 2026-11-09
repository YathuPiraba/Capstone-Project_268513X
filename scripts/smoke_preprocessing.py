"""Small real-data verification for the Task 03 preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.inspection import (  # noqa: E402
    load_graph_bundle,
    load_speed_data,
    read_sensor_ids,
    validate_graph_alignment,
)
from src.data.preprocessing import (  # noqa: E402
    PerSensorStandardizer,
    all_sensor_zero_mask,
    chronological_split_ranges,
    cyclical_time_of_day,
    target_evaluation_mask,
)
from src.data.windows import build_forecast_windows, window_time_features  # noqa: E402


def main() -> None:
    data_dir = ROOT / "data" / "raw" / "metr-la"
    frame = load_speed_data(data_dir / "METR-LA.csv")
    speeds = frame.to_numpy(dtype=np.float64, copy=False)
    splits = chronological_split_ranges(frame.index)

    graph = load_graph_bundle(data_dir / "adj_mx_METR-LA.pkl")
    reference_ids = read_sensor_ids(data_dir / "graph_sensor_ids.txt")
    alignment = validate_graph_alignment(frame.columns, graph, reference_ids)
    required_alignment = (
        "csv_graph_same_order",
        "reference_graph_same_order",
        "mapping_matches_graph_order",
        "adjacency_shape_matches_ids",
    )
    assert all(alignment[key] for key in required_alignment)

    scaler = PerSensorStandardizer.fit(speeds[splits["train"].start:splits["train"].stop])
    sample = speeds[:24]
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(sample)), sample, atol=1e-12
    )
    time_features = cyclical_time_of_day(frame.index)

    print(f"data_shape={speeds.shape}; adjacency_shape={graph.adjacency.shape}")
    print(
        "normalization=per-sensor, train-only, nonzero finite observations; "
        f"mean_range=({scaler.mean.min():.6f}, {scaler.mean.max():.6f}); "
        f"scale_range=({scaler.scale.min():.6f}, {scaler.scale.max():.6f})"
    )
    print(f"time_features_shape={time_features.shape}")

    horizon_indices = {"+15": 2, "+30": 5, "+60": 11}
    for name, split in splits.items():
        windows = build_forecast_windows(speeds, frame.index, split)
        input_time, future_time = window_time_features(time_features, windows)
        evaluation_mask = target_evaluation_mask(windows.y_speed)
        horizon_valid = {
            label: float(100 * evaluation_mask[:, step].mean())
            for label, step in horizon_indices.items()
        }
        synchronized_zeros = int(
            all_sensor_zero_mask(speeds[split.start:split.stop]).sum()
        )
        print(
            f"{name}: rows=[{split.start}, {split.stop}) count={split.count}; "
            f"timestamps={split.start_timestamp}..{split.end_timestamp}; "
            f"X={windows.X_speed.shape}; y={windows.y_speed.shape}; "
            f"input_time={input_time.shape}; future_time={future_time.shape}; "
            f"valid_all={100 * evaluation_mask.mean():.6f}%; "
            f"valid_horizons={horizon_valid}; all_sensor_zero_rows={synchronized_zeros}"
        )
        assert windows.target_start_rows.min() >= split.start
        assert (windows.target_start_rows + windows.y_speed.shape[1]).max() <= split.stop

    print("Task 03 real-data smoke test passed.")


if __name__ == "__main__":
    main()
