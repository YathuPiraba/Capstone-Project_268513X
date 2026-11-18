"""Generate reproducible Task 08 comparisons from frozen prediction artifacts."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.comparison import (  # noqa: E402
    COMPARATORS,
    HORIZONS,
    MODEL_LABELS,
    PRIMARY_POLICY,
    SENSITIVITY_POLICY,
    best_primary_models,
    daily_absolute_error_summaries,
    equal_horizon_mean_rows,
    load_and_validate_artifacts,
    observed_speed_strata_rows,
    overall_metric_rows,
    paired_day_cluster_bootstrap,
    pairwise_rows,
    reconcile_published_metrics,
    sensor_metric_rows,
    sensor_pairwise_summary,
    source_hashes,
    temporal_metric_rows,
)


RESULT_DIR = ROOT / "results/task08_comparison"
FIGURE_DIR = RESULT_DIR / "figures"
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 268_513


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate frozen artifacts and published metrics without writing outputs",
    )
    args = parser.parse_args()

    before = source_hashes(ROOT)
    data = load_and_validate_artifacts(ROOT)
    overall = overall_metric_rows(data)
    reconcile_published_metrics(ROOT, overall)
    if args.verify_only:
        print("Task 08 integrity gate and published-metric reconciliation passed.")
        print(json.dumps(before, indent=2))
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    means = equal_horizon_mean_rows(overall)
    pairwise = pairwise_rows(overall)
    sensors, sensor_pairs = sensor_metric_rows(data)
    sensor_summary = sensor_pairwise_summary(sensor_pairs)
    strata = observed_speed_strata_rows(data)
    temporal = temporal_metric_rows(data)

    dates, daily_sums, daily_counts = daily_absolute_error_summaries(
        data, ("dcrnn", "gru", "xgboost")
    )
    bootstrap_comparisons = [
        paired_day_cluster_bootstrap(
            daily_sums,
            daily_counts,
            comparator=comparator,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
        )
        for comparator in ("gru", "xgboost")
    ]

    timestamp_facts = _timestamp_facts(data)
    exact_zero_counts = [
        int((data.masks[SENSITIVITY_POLICY][:, j, :] & ~data.masks[PRIMARY_POLICY][:, j, :]).sum())
        for j in range(len(HORIZONS))
    ]
    overall_document = {
        "task": "Task 08 - Comparative analysis and model evaluation",
        "generation": "deterministic artifact-only Task 08 pipeline",
        "research_question": (
            "To what extent does the graph-enabled DCGRU improve traffic speed forecasting "
            "compared with conventional and temporal-only approaches, and how does performance "
            "vary across forecasting horizons, sensors, and traffic conditions?"
        ),
        "primary_finding_contract": {
            "metric": "per-horizon MAE in mph",
            "mask_policy": PRIMARY_POLICY,
            "central_comparison": "DCGRU-64 versus GRU-128",
            "horizons_minutes": HORIZONS.tolist(),
        },
        "secondary_summary": "equal-weight mean of +15/+30/+60 MAE",
        "sensitivity_policy": SENSITIVITY_POLICY,
        "models": MODEL_LABELS,
        "shape": list(data.targets.shape),
        "data_sha256": data.data_sha256,
        "split_rows": {name: list(bounds) for name, bounds in data.split_rows.items()},
        "source_artifact_sha256": data.artifact_hashes,
        "source_artifacts_immutable": True,
        "timestamp_convention": "timezone-naive NumPy datetime64; no timezone inferred",
        "timestamp_facts": timestamp_facts,
        "exact_zero_counts_by_horizon": dict(zip(map(str, HORIZONS.tolist()), exact_zero_counts)),
        "metrics": overall,
        "equal_horizon_mean_mae": means,
        "best_primary_models": best_primary_models(overall),
        "sensor_dcgru_vs_gru_summary": sensor_summary,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    bootstrap_document = {
        "analysis_role": "exploratory uncertainty/stability analysis; not a formal significance test",
        "mask_policy": PRIMARY_POLICY,
        "cluster_unit": "target calendar date, separately derived from each horizon target timestamp",
        "calendar_dates": [str(date) for date in dates],
        "cluster_count": int(len(dates)),
        "partial_days_retained": True,
        "aggregation": "sum(sampled daily absolute-error sums) / sum(sampled daily eligible counts)",
        "paired_sampling": "same sampled target-date sequence for both models and all horizons",
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "numpy_bit_generator": "PCG64",
        "interval": "empirical percentile interval, 2.5th to 97.5th percentile",
        "source_artifact_sha256": data.artifact_hashes,
        "comparisons": bootstrap_comparisons,
        "limitations": [
            "Only 24 target-date clusters are available.",
            "Forecast windows overlap and adjacent errors are temporally dependent.",
            "Input history can be shared across midnight and cluster boundaries.",
            "The locked test period is limited to 4-27 June 2012.",
            "The bootstrap was proposed after the primary locked-test results were observed.",
            "Intervals are exploratory and do not establish causality or formal statistical significance.",
        ],
    }

    _write_json(RESULT_DIR / "overall_metrics.json", overall_document)
    _write_json(RESULT_DIR / "bootstrap_summary.json", bootstrap_document)
    pd.DataFrame(pairwise).to_csv(RESULT_DIR / "pairwise_differences.csv", index=False)
    pd.DataFrame(sensors).to_csv(RESULT_DIR / "sensor_metrics.csv", index=False)
    pd.DataFrame(strata).to_csv(RESULT_DIR / "observed_speed_strata.csv", index=False)
    pd.DataFrame(temporal).to_csv(RESULT_DIR / "temporal_metrics.csv", index=False)
    _generate_figures(overall, means, pairwise, sensor_pairs, strata, temporal)

    after = source_hashes(ROOT)
    if before != after:
        raise RuntimeError("A frozen prediction artifact changed during Task 08 generation")
    print("Task 08 comparison generation completed.")
    print(f"Validated {len(MODEL_LABELS)} models on shape {data.targets.shape}.")
    print(f"Wrote results under {RESULT_DIR.relative_to(ROOT)}.")
    print("Frozen prediction artifact hashes remained unchanged.")


def _timestamp_facts(data) -> dict[str, object]:
    target_dates = data.target_timestamps.astype("datetime64[D]")
    facts = {
        "earliest_target": str(data.target_timestamps.min()),
        "latest_target": str(data.target_timestamps.max()),
        "distinct_target_calendar_dates": int(len(np.unique(target_dates))),
        "by_horizon": [],
    }
    for j, horizon in enumerate(HORIZONS):
        dates = target_dates[:, j]
        unique_dates, counts = np.unique(dates, return_counts=True)
        facts["by_horizon"].append({
            "horizon_minutes": int(horizon),
            "earliest_target": str(data.target_timestamps[:, j].min()),
            "latest_target": str(data.target_timestamps[:, j].max()),
            "complete_day_count": int((counts == 288).sum()),
            "partial_days": [
                {"date": str(date), "timestamp_count": int(count)}
                for date, count in zip(unique_dates, counts)
                if count != 288
            ],
        })
    return facts


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _generate_figures(overall, means, pairwise, sensor_pairs, strata, temporal) -> None:
    overall_frame = pd.DataFrame(overall)
    primary = overall_frame[overall_frame["mask_policy"] == PRIMARY_POLICY]
    colors = {
        "Persistence": "#7f8c8d",
        "Historical Average": "#95a5a6",
        "XGBoost": "#f39c12",
        "GRU-128": "#3498db",
        "DCGRU-64": "#8e44ad",
    }
    for metric, filename, ylabel in (
        ("mae_mph", "primary_mae_by_horizon.png", "MAE (mph)"),
        ("rmse_mph", "primary_rmse_by_horizon.png", "RMSE (mph)"),
    ):
        pivot = primary.pivot(index="horizon_minutes", columns="model_label", values=metric)
        ax = pivot[list(colors)].plot(
            kind="bar", figsize=(9, 5), color=[colors[name] for name in colors], width=0.82
        )
        ax.set_xlabel("Forecast horizon (minutes)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Locked-test primary {ylabel} by horizon")
        ax.legend(
            title="Model",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
        )
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / filename, dpi=160)
        plt.close()

    mean_frame = pd.DataFrame(means)
    mean_frame = mean_frame[mean_frame["mask_policy"] == PRIMARY_POLICY].sort_values(
        "equal_horizon_mean_mae_mph"
    )
    plt.figure(figsize=(8, 4.5))
    plt.barh(
        mean_frame["model_label"], mean_frame["equal_horizon_mean_mae_mph"],
        color=[colors[label] for label in mean_frame["model_label"]],
    )
    plt.xlabel("Equal-horizon mean MAE (mph)")
    plt.title("Secondary equal-weight summary across +15/+30/+60 minutes")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "primary_equal_horizon_mean_mae.png", dpi=160)
    plt.close()

    pair = pd.DataFrame(pairwise)
    pair = pair[(pair["mask_policy"] == PRIMARY_POLICY) & (pair["summary"] == "per_horizon")]
    pivot = pair.pivot(index="horizon_minutes", columns="comparator_label", values="improvement_percent")
    ax = pivot.plot(kind="bar", figsize=(9, 5), width=0.82)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("DCGRU MAE improvement (%)")
    ax.set_title("DCGRU improvement relative to comparison models")
    ax.legend(title="Comparator", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "dcgru_pairwise_improvement.png", dpi=160)
    plt.close()

    sensor_frame = pd.DataFrame(sensor_pairs)
    plt.figure(figsize=(9, 5))
    for horizon in HORIZONS:
        values = sensor_frame[sensor_frame["horizon_minutes"] == horizon][
            "dcgru_minus_gru_mae_mph"
        ].sort_values().to_numpy()
        plt.plot(np.arange(1, len(values) + 1), values, label=f"+{horizon} min")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Sensors sorted independently by MAE difference")
    plt.ylabel("DCGRU minus GRU sensor MAE (mph)")
    plt.title("Sensor-level paired differences (negative favors DCGRU)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sensor_dcgru_minus_gru.png", dpi=160)
    plt.close()

    strata_frame = pd.DataFrame(strata)
    strata_frame = strata_frame[
        (strata_frame["mask_policy"] == PRIMARY_POLICY)
        & (strata_frame["model"].isin(["xgboost", "gru", "dcrnn"]))
    ]
    labels = {"low_below_20": "<20", "medium_20_to_below_40": "20-<40", "high_40_and_above": ">=40"}
    for horizon in HORIZONS:
        subset = strata_frame[strata_frame["horizon_minutes"] == horizon].copy()
        subset["speed_band"] = subset["speed_band"].map(labels)
        pivot = subset.pivot(index="speed_band", columns="model_label", values="mae_mph").reindex(labels.values())
        ax = pivot.plot(kind="bar", figsize=(8, 4.5), color=[colors[c] for c in pivot.columns])
        ax.set_xlabel("Observed target-speed band (mph; retrospective)")
        ax.set_ylabel("MAE (mph)")
        ax.set_title(f"Observed-Speed Evaluation Strata at +{horizon} minutes")
        ax.legend(title="Model", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"observed_speed_strata_h{horizon}.png", dpi=160)
        plt.close()

    temporal_frame = pd.DataFrame(temporal)
    daily = temporal_frame[
        (temporal_frame["mask_policy"] == PRIMARY_POLICY)
        & (temporal_frame["group_type"] == "target_date")
        & (temporal_frame["horizon_minutes"] == 60)
        & (temporal_frame["model"].isin(["gru", "dcrnn"]))
    ]
    pivot = daily.pivot(index="group", columns="model_label", values="mae_mph")
    ax = pivot.plot(figsize=(10, 4.5), marker="o", color=[colors[c] for c in pivot.columns])
    ax.set_xlabel("Target calendar date")
    ax.set_ylabel("MAE (mph)")
    ax.set_title("Exploratory daily MAE at +60 minutes")
    ax.legend(title="Model")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "daily_gru_dcgru_h60.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
