"""Reusable, non-mutating inspection helpers for the trusted METR-LA assets."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_TIME_STEPS = 34_272
EXPECTED_SENSORS = 207
EXPECTED_FREQUENCY = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class TimestampReport:
    row_count: int
    start: pd.Timestamp
    end: pd.Timestamp
    duplicate_count: int
    non_increasing_count: int
    irregular_interval_count: int
    interval_counts: pd.Series


@dataclass(frozen=True)
class GraphBundle:
    sensor_ids: list[str]
    sensor_id_to_index: dict[str, int]
    adjacency: np.ndarray


def load_speed_data(csv_path: str | Path) -> pd.DataFrame:
    """Load METR-LA speeds with a parsed, named timestamp index."""
    frame = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
    frame.index.name = "timestamp"
    frame.columns = frame.columns.astype(str)
    return frame


def inspect_timestamps(index: pd.Index) -> TimestampReport:
    """Summarize timestamp ordering, duplication, and 5-minute regularity."""
    timestamps = pd.DatetimeIndex(index)
    if timestamps.empty:
        raise ValueError("Timestamp index is empty")
    deltas = timestamps.to_series().diff().dropna()
    interval_counts = deltas.value_counts().sort_index()
    return TimestampReport(
        row_count=len(timestamps),
        start=timestamps[0],
        end=timestamps[-1],
        duplicate_count=int(timestamps.duplicated().sum()),
        non_increasing_count=int((deltas <= pd.Timedelta(0)).sum()),
        irregular_interval_count=int((deltas != EXPECTED_FREQUENCY).sum()),
        interval_counts=interval_counts,
    )


def load_graph_bundle(pickle_path: str | Path) -> GraphBundle:
    """Load the trusted DCRNN-format graph pickle.

    Pickle loading can execute code. Call this only for a verified, trusted file.
    """
    with Path(pickle_path).open("rb") as stream:
        payload = pickle.load(stream, encoding="latin1")  # noqa: S301
    if not isinstance(payload, (list, tuple)) or len(payload) != 3:
        raise ValueError("Expected [sensor_ids, sensor_id_to_index, adjacency]")
    sensor_ids, sensor_id_to_index, adjacency = payload
    return GraphBundle(
        sensor_ids=[str(sensor_id) for sensor_id in sensor_ids],
        sensor_id_to_index={str(key): int(value) for key, value in sensor_id_to_index.items()},
        adjacency=np.asarray(adjacency),
    )


def read_sensor_ids(path: str | Path) -> list[str]:
    """Read the comma-separated DCRNN sensor-ID reference."""
    return [value.strip() for value in Path(path).read_text().strip().split(",") if value.strip()]


def validate_graph_alignment(
    csv_sensor_ids: list[str] | pd.Index,
    graph: GraphBundle,
    reference_sensor_ids: list[str],
) -> dict[str, object]:
    """Return explicit set, order, mapping, and shape alignment checks."""
    csv_ids = [str(value) for value in csv_sensor_ids]
    graph_ids = graph.sensor_ids
    expected_mapping = {sensor_id: index for index, sensor_id in enumerate(graph_ids)}
    shape_expected = (len(graph_ids), len(graph_ids))
    return {
        "csv_count": len(csv_ids),
        "graph_count": len(graph_ids),
        "reference_count": len(reference_sensor_ids),
        "csv_unique": len(set(csv_ids)) == len(csv_ids),
        "graph_unique": len(set(graph_ids)) == len(graph_ids),
        "reference_unique": len(set(reference_sensor_ids)) == len(reference_sensor_ids),
        "csv_graph_same_set": set(csv_ids) == set(graph_ids),
        "csv_graph_same_order": csv_ids == graph_ids,
        "reference_graph_same_order": reference_sensor_ids == graph_ids,
        "mapping_matches_graph_order": graph.sensor_id_to_index == expected_mapping,
        "adjacency_shape": tuple(graph.adjacency.shape),
        "adjacency_shape_matches_ids": tuple(graph.adjacency.shape) == shape_expected,
        "csv_only_ids": sorted(set(csv_ids) - set(graph_ids)),
        "graph_only_ids": sorted(set(graph_ids) - set(csv_ids)),
    }


def longest_true_run(mask: pd.Series) -> int:
    """Return the longest consecutive run of True values in a boolean series."""
    values = np.asarray(mask, dtype=bool)
    if not values.any():
        return 0
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    boundaries = np.flatnonzero(np.diff(padded))
    return int(np.max(boundaries[1::2] - boundaries[::2]))
