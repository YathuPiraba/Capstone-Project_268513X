import ast
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

import src.evaluation.comparison as comparison
from src.evaluation.comparison import (
    ArtifactSpec,
    ComparisonData,
    PRIMARY_POLICY,
    SENSITIVITY_POLICY,
    daily_absolute_error_summaries,
    load_and_validate_artifacts,
    observed_speed_strata_rows,
    overall_metric_rows,
    paired_day_cluster_bootstrap,
    percentage_improvement,
    reconcile_published_metrics,
    sensor_metric_rows,
    source_hashes,
    temporal_metric_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _fixture_artifacts(tmp_path: Path, monkeypatch) -> tuple[tuple[ArtifactSpec, ...], dict[str, dict[str, np.ndarray]]]:
    shape = (2, 3, 2)
    targets = np.arange(1, 13, dtype=np.float64).reshape(shape)
    origins = np.array(["2012-06-04T00:00", "2012-06-04T00:05"], dtype="datetime64[m]")
    times = origins[:, None] + np.array([15, 30, 60], dtype="timedelta64[m]")[None, :]
    shared = {
        "targets_mph": targets,
        "forecast_origin_timestamps": origins,
        "target_timestamps": times,
        "sensor_ids": np.array(["a", "b"]),
        "horizon_minutes": np.array([15, 30, 60]),
        "primary_nonzero_mask": np.ones(shape, dtype=bool),
        "finite_including_zero_mask": np.ones(shape, dtype=bool),
    }
    definitions = (
        ("task04", ("persistence", "historical_average")),
        ("task05", ("xgboost",)),
        ("task06", ("gru",)),
        ("task07", ("dcrnn",)),
    )
    specs = []
    arrays_by_task = {}
    for task, models in definitions:
        directory = tmp_path / task
        directory.mkdir()
        artifact = directory / "predictions.npz"
        arrays = {key: np.array(value, copy=True) for key, value in shared.items()}
        for offset, model in enumerate(models, start=1):
            arrays[comparison.PREDICTION_KEYS[model]] = targets + offset
        np.savez_compressed(artifact, **arrays)
        manifest = directory / "metrics.json"
        manifest.write_text(json.dumps({
            "data_sha256": "DATA",
            "splits": {
                "train": {"start_row": 0, "stop_row_exclusive": 10},
                "validation": {"start_row": 10, "stop_row_exclusive": 12},
                "test": {"start_row": 12, "stop_row_exclusive": 16},
            },
            "prediction_sha256": _sha(artifact),
            "metrics": [
                {"mask_policy": PRIMARY_POLICY},
                {"mask_policy": SENSITIVITY_POLICY},
            ],
        }), encoding="utf-8")
        specs.append(ArtifactSpec(task, artifact, manifest, models))
        arrays_by_task[task] = arrays
    monkeypatch.setattr(comparison, "artifact_specs", lambda root: tuple(specs))
    monkeypatch.setattr(comparison, "EXPECTED_SHAPE", shape)
    return tuple(specs), arrays_by_task


def _rewrite(spec: ArtifactSpec, arrays: dict[str, np.ndarray], *, update_hash: bool = True) -> None:
    np.savez_compressed(spec.artifact_path, **arrays)
    manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
    if update_hash:
        manifest["prediction_sha256"] = _sha(spec.artifact_path)
    spec.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    "key,mutation",
    [
        ("targets_mph", lambda a: a.__setitem__((0, 0, 0), 999.0)),
        ("target_timestamps", lambda a: a.__setitem__((0, 0), np.datetime64("2012-07-01"))),
        ("forecast_origin_timestamps", lambda a: a.__setitem__(0, np.datetime64("2012-07-01"))),
        ("sensor_ids", lambda a: a.__setitem__(slice(None), a[::-1])),
        ("horizon_minutes", lambda a: a.__setitem__(0, 10)),
        ("primary_nonzero_mask", lambda a: a.__setitem__((0, 0, 0), False)),
        ("finite_including_zero_mask", lambda a: a.__setitem__((0, 0, 0), False)),
    ],
)
def test_integrity_gate_rejects_shared_array_mismatches(tmp_path, monkeypatch, key, mutation) -> None:
    specs, arrays = _fixture_artifacts(tmp_path, monkeypatch)
    changed = arrays["task05"]
    mutation(changed[key])
    _rewrite(specs[1], changed)
    with pytest.raises((AssertionError, ValueError)):
        load_and_validate_artifacts(tmp_path, expected_shape=(2, 3, 2))


def test_integrity_gate_rejects_incorrect_hash(tmp_path, monkeypatch) -> None:
    specs, arrays = _fixture_artifacts(tmp_path, monkeypatch)
    arrays["task05"]["xgboost_mph"][0, 0, 0] += 1
    _rewrite(specs[1], arrays["task05"], update_hash=False)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_and_validate_artifacts(tmp_path, expected_shape=(2, 3, 2))


def test_integrity_gate_rejects_nonfinite_predictions(tmp_path, monkeypatch) -> None:
    specs, arrays = _fixture_artifacts(tmp_path, monkeypatch)
    arrays["task06"]["gru_mph"][0, 0, 0] = np.nan
    _rewrite(specs[2], arrays["task06"])
    with pytest.raises(ValueError, match="Nonfinite"):
        load_and_validate_artifacts(tmp_path, expected_shape=(2, 3, 2))


def test_real_artifacts_reproduce_all_published_metrics() -> None:
    data = load_and_validate_artifacts(ROOT)
    rows = overall_metric_rows(data)
    reconcile_published_metrics(ROOT, rows, atol=1e-9)


def test_percentage_improvement_sign_and_zero_denominator() -> None:
    assert percentage_improvement(10.0, 8.0) == pytest.approx(20.0)
    assert percentage_improvement(10.0, 12.0) == pytest.approx(-20.0)
    assert np.isnan(percentage_improvement(0.0, 1.0))


def _small_data() -> ComparisonData:
    targets = np.array([
        [[0.0, 20.0], [40.0, 10.0], [50.0, 30.0]],
        [[19.0, 39.0], [20.0, 40.0], [60.0, 5.0]],
    ])
    predictions = {
        model: targets + offset
        for model, offset in zip(comparison.MODEL_LABELS, [5.0, 4.0, 3.0, 2.0, 1.0])
    }
    finite = np.ones_like(targets, dtype=bool)
    primary = finite & (targets != 0)
    origins = np.array(["2012-06-04T23:30", "2012-06-05T00:00"], dtype="datetime64[m]")
    timestamps = origins[:, None] + np.array([15, 30, 60], dtype="timedelta64[m]")[None, :]
    return ComparisonData(
        targets=targets,
        predictions=predictions,
        target_timestamps=timestamps,
        forecast_origin_timestamps=origins,
        sensor_ids=np.array(["a", "b"]),
        horizons=np.array([15, 30, 60]),
        masks={PRIMARY_POLICY: primary, SENSITIVITY_POLICY: finite},
        artifact_hashes={},
        data_sha256="x",
        split_rows={},
    )


def test_sensor_aggregation_uses_primary_eligibility() -> None:
    rows, pairs = sensor_metric_rows(_small_data())
    first = next(r for r in rows if r["sensor_id"] == "a" and r["horizon_minutes"] == 15 and r["model"] == "dcrnn")
    assert first["eligible_count"] == 1
    assert first["mae_mph"] == pytest.approx(1.0)
    assert all(p["dcgru_improved"] for p in pairs)


def test_speed_bands_use_exact_boundaries_and_separate_zero_policy() -> None:
    rows = pd.DataFrame(observed_speed_strata_rows(_small_data()))
    primary = rows[(rows.mask_policy == PRIMARY_POLICY) & (rows.model == "dcrnn")]
    sensitivity = rows[(rows.mask_policy == SENSITIVITY_POLICY) & (rows.model == "dcrnn")]
    primary_counts = primary.groupby("speed_band").eligible_count.sum().to_dict()
    sensitivity_counts = sensitivity.groupby("speed_band").eligible_count.sum().to_dict()
    assert primary_counts["low_below_20"] == 3
    assert primary_counts["medium_20_to_below_40"] == 4
    assert primary_counts["high_40_and_above"] == 4
    assert sensitivity_counts["low_below_20"] == 4
    assert set(primary.speed_band) == {
        "low_below_20", "medium_20_to_below_40", "high_40_and_above"
    }


def test_temporal_groups_use_target_timestamps_and_retain_partial_dates() -> None:
    rows = pd.DataFrame(temporal_metric_rows(_small_data()))
    dates = rows[rows.group_type == "target_date"]
    assert set(dates.group) == {"2012-06-04", "2012-06-05"}
    assert not dates.empty


def test_weighted_paired_bootstrap_uses_pooled_sums_and_is_reproducible() -> None:
    counts = np.array([[1, 1, 1], [9, 9, 9]])
    sums = {
        "dcrnn": np.array([[0.0, 0.0, 0.0], [18.0, 18.0, 18.0]]),
        "gru": np.array([[10.0, 10.0, 10.0], [9.0, 9.0, 9.0]]),
    }
    first = paired_day_cluster_bootstrap(sums, counts, comparator="gru", repetitions=500, seed=7)
    second = paired_day_cluster_bootstrap(sums, counts, comparator="gru", repetitions=500, seed=7)
    assert first == second
    # Pooled point difference: (18/10) - (19/10) = -0.1, not mean daily difference.
    assert first["rows"][0]["dcgru_minus_comparator_mae_mph"] == pytest.approx(-0.1)


def test_bootstrap_uses_identical_sampled_dates_for_both_models() -> None:
    counts = np.array([[3, 4, 5], [7, 8, 9], [11, 12, 13]])
    same = np.array([[2.0, 3.0, 4.0], [5.0, 7.0, 11.0], [13.0, 17.0, 19.0]])
    result = paired_day_cluster_bootstrap(
        {"dcrnn": same, "gru": same.copy()}, counts,
        comparator="gru", repetitions=250, seed=19,
    )
    for row in result["rows"]:
        assert row["dcgru_minus_comparator_mae_mph"] == 0.0
        assert row["interval_95_low_mph"] == 0.0
        assert row["interval_95_high_mph"] == 0.0


def test_daily_summaries_use_target_dates() -> None:
    data = _small_data()
    data.target_timestamps = np.array([
        ["2012-06-04T00:15", "2012-06-04T00:30", "2012-06-04T01:00"],
        ["2012-06-05T00:15", "2012-06-05T00:30", "2012-06-05T01:00"],
    ], dtype="datetime64[m]")
    dates, sums, counts = daily_absolute_error_summaries(data, ("dcrnn", "gru"))
    assert dates.tolist() == [np.datetime64("2012-06-04"), np.datetime64("2012-06-05")]
    assert sums["dcrnn"].shape == counts.shape == (2, 3)


def test_model_loading_order_does_not_change_overall_results() -> None:
    data = _small_data()
    original = overall_metric_rows(data)
    data.predictions = dict(reversed(list(data.predictions.items())))
    assert original == overall_metric_rows(data)


def test_real_source_artifacts_remain_unchanged_during_read_only_analysis() -> None:
    before = source_hashes(ROOT)
    data = load_and_validate_artifacts(ROOT)
    overall_metric_rows(data)
    after = source_hashes(ROOT)
    assert before == after


def test_generated_output_schemas() -> None:
    result_dir = ROOT / "results/task08_comparison"
    overall = json.loads((result_dir / "overall_metrics.json").read_text(encoding="utf-8"))
    bootstrap = json.loads((result_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
    assert overall["shape"] == [6844, 3, 207]
    assert len(overall["metrics"]) == 30
    assert bootstrap["repetitions"] == 10_000
    assert bootstrap["cluster_count"] == 24
    for filename in (
        "pairwise_differences.csv", "sensor_metrics.csv",
        "observed_speed_strata.csv", "temporal_metrics.csv",
    ):
        assert not pd.read_csv(result_dir / filename).empty


def test_notebook_is_valid_complete_and_contains_no_training_calls() -> None:
    path = ROOT / "notebooks/08_comparative_analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    headings = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    for required in (
        "Research question", "Artifact integrity", "Observed-Speed Evaluation Strata",
        "Paired bootstrap", "Exact-zero sensitivity", "Scientific limitations",
    ):
        assert required in headings
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    execution_counts = [cell.execution_count for cell in code_cells]
    is_pristine = all(count is None for count in execution_counts)
    is_complete_owner_execution = execution_counts == list(range(1, len(code_cells) + 1))
    assert is_pristine or is_complete_owner_execution
    for cell in code_cells:
        if cell.cell_type == "code":
            assert not any(output.output_type == "error" for output in cell.outputs)
            if is_pristine:
                assert cell.outputs == []
            ast.parse(cell.source)
            lowered = cell.source.lower()
            assert "run_dcrnn" not in lowered
            assert "run_gru" not in lowered
            assert "run_xgboost" not in lowered
            assert "fit(" not in lowered
