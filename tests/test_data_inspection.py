from pathlib import Path

from src.data.inspection import (
    EXPECTED_FREQUENCY,
    EXPECTED_SENSORS,
    EXPECTED_TIME_STEPS,
    inspect_timestamps,
    load_graph_bundle,
    load_speed_data,
    read_sensor_ids,
    validate_graph_alignment,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "metr-la"


def test_metr_la_expected_dimensions() -> None:
    speeds = load_speed_data(DATA_DIR / "METR-LA.csv")
    assert speeds.shape == (EXPECTED_TIME_STEPS, EXPECTED_SENSORS)
    assert speeds.columns.is_unique


def test_metr_la_timestamps_are_regular_and_unique() -> None:
    speeds = load_speed_data(DATA_DIR / "METR-LA.csv")
    report = inspect_timestamps(speeds.index)
    assert report.duplicate_count == 0
    assert report.non_increasing_count == 0
    assert report.irregular_interval_count == 0
    assert report.interval_counts.to_dict() == {EXPECTED_FREQUENCY: EXPECTED_TIME_STEPS - 1}


def test_graph_sensor_ids_align_with_csv_and_reference() -> None:
    speeds = load_speed_data(DATA_DIR / "METR-LA.csv")
    graph = load_graph_bundle(DATA_DIR / "adj_mx_METR-LA.pkl")
    reference_ids = read_sensor_ids(DATA_DIR / "graph_sensor_ids.txt")
    alignment = validate_graph_alignment(speeds.columns, graph, reference_ids)
    assert alignment["csv_graph_same_set"]
    assert alignment["csv_graph_same_order"]
    assert alignment["reference_graph_same_order"]
    assert alignment["mapping_matches_graph_order"]
    assert alignment["adjacency_shape_matches_ids"]
