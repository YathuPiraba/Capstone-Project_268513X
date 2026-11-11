"""Deterministic XGBoost training and inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
import xgboost as xgb

from src.models.xgboost import FEATURE_NAMES, FEATURE_TYPES, PooledHorizonBatch


DEFAULT_SEED = 268513


class TrainingProgress(xgb.callback.TrainingCallback):
    """Print bounded round progress without treating the maximum as an ETA promise."""

    def __init__(self, label: str, total_rounds: int, every_rounds: int = 25) -> None:
        self.label = label
        self.total_rounds = total_rounds
        self.every_rounds = max(1, every_rounds)
        self.started = 0.0

    def before_training(self, model: xgb.Booster) -> xgb.Booster:
        self.started = time.perf_counter()
        return model

    def after_iteration(
        self,
        model: xgb.Booster,
        epoch: int,
        evals_log: dict[str, dict[str, list[float]]],
    ) -> bool:
        completed = epoch + 1
        if completed != 1 and completed % self.every_rounds and completed != self.total_rounds:
            return False
        elapsed = time.perf_counter() - self.started
        seconds_per_round = elapsed / completed
        upper_eta = seconds_per_round * (self.total_rounds - completed)
        validation_mae = evals_log.get("validation", {}).get("mae", [float("nan")])[-1]
        print(
            f"    {self.label}: round {completed}/{self.total_rounds} "
            f"({completed / self.total_rounds:.1%} of configured maximum), "
            f"validation MAE={validation_mae:.6f} mph, elapsed={_duration(elapsed)}, "
            f"upper-bound round ETA={_duration(upper_eta)}; early stopping may finish sooner",
            flush=True,
        )
        return False


@dataclass(frozen=True)
class CandidateConfig:
    """High-impact XGBoost parameters varied during validation selection."""

    name: str
    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float
    colsample_bytree: float

    def as_params(self, *, nthread: int, seed: int = DEFAULT_SEED) -> dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": seed,
            "nthread": nthread,
            "max_bin": 256,
        }


DEFAULT_CANDIDATES = (
    CandidateConfig("depth6_eta005", 6, 0.05, 1.0, 0.9, 0.9),
    CandidateConfig("depth8_eta005", 8, 0.05, 1.0, 0.9, 0.9),
    CandidateConfig("depth6_eta010_regularized", 6, 0.10, 5.0, 0.8, 0.8),
)


class BatchDataIter(xgb.DataIter):
    """Restartable bridge from chronological feature batches to XGBoost."""

    def __init__(
        self,
        factory: Callable[[], Sequence[PooledHorizonBatch] | Any],
        *,
        progress_label: str | None = None,
        expected_batches: int | None = None,
    ) -> None:
        super().__init__(cache_prefix=None, release_data=True)
        self._factory = factory
        self._iterator: Any = None
        self._progress_label = progress_label
        self._expected_batches = expected_batches
        self._pass_number = 0
        self._batch_number = 0

    def reset(self) -> None:
        self._iterator = iter(self._factory())
        self._pass_number += 1
        self._batch_number = 0

    def next(self, input_data: Callable[..., None]) -> bool:
        if self._iterator is None:
            self.reset()
        try:
            batch = next(self._iterator)
        except StopIteration:
            return False
        self._batch_number += 1
        if self._progress_label and (
            self._batch_number == 1
            or self._batch_number % 10 == 0
            or self._batch_number == self._expected_batches
        ):
            denominator = self._expected_batches or "?"
            print(
                f"    {self._progress_label}: quantization pass {self._pass_number}, "
                f"batch {self._batch_number}/{denominator}",
                flush=True,
            )
        input_data(
            data=batch.features,
            label=batch.targets_mph,
            feature_names=list(FEATURE_NAMES),
            feature_types=list(FEATURE_TYPES),
        )
        return True


def make_quantile_dmatrix(
    batch_factory: Callable[[], Any],
    *,
    nthread: int,
    reference: xgb.DMatrix | None = None,
    progress_label: str | None = None,
    expected_batches: int | None = None,
) -> xgb.QuantileDMatrix:
    """Create a quantized matrix from batches with categorical metadata."""
    return xgb.QuantileDMatrix(
        BatchDataIter(
            batch_factory,
            progress_label=progress_label,
            expected_batches=expected_batches,
        ),
        max_bin=256,
        nthread=nthread,
        ref=reference,
        enable_categorical=True,
    )


def train_with_early_stopping(
    train_matrix: xgb.DMatrix,
    validation_matrix: xgb.DMatrix,
    candidate: CandidateConfig,
    *,
    nthread: int,
    seed: int = DEFAULT_SEED,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    progress_label: str | None = None,
    progress_interval: int = 25,
) -> tuple[xgb.Booster, dict[str, Any]]:
    """Train one horizon model using primary validation MAE for stopping."""
    history: dict[str, dict[str, list[float]]] = {}
    callbacks = (
        [TrainingProgress(progress_label, num_boost_round, progress_interval)]
        if progress_label
        else None
    )
    booster = xgb.train(
        candidate.as_params(nthread=nthread, seed=seed),
        train_matrix,
        num_boost_round=num_boost_round,
        evals=[(validation_matrix, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=history,
        verbose_eval=False,
        callbacks=callbacks,
    )
    best_iteration = int(booster.best_iteration)
    best_score = float(booster.best_score)
    return booster, {
        "candidate": candidate.name,
        "best_iteration_zero_based": best_iteration,
        "boosting_rounds": best_iteration + 1,
        "validation_mae_mph": best_score,
        "history": history,
    }


def select_shared_candidate(trials: Sequence[dict[str, Any]]) -> str:
    """Select lowest mean three-horizon MAE, breaking ties by candidate name."""
    grouped: dict[str, list[float]] = {}
    for trial in trials:
        grouped.setdefault(str(trial["candidate"]), []).append(
            float(trial["validation_mae_mph"])
        )
    if not grouped or any(len(scores) != 3 for scores in grouped.values()):
        raise ValueError("Every candidate must have exactly three horizon validation scores")
    return min(
        grouped,
        key=lambda name: (float(np.mean(grouped[name])), name),
    )


def predict_batches(
    booster: xgb.Booster,
    batch_factory: Callable[[], Any],
    *,
    iteration_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict chronological batches and return values with row coordinates."""
    predictions: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    sensor_indices: list[np.ndarray] = []
    iteration_range = (0, iteration_count) if iteration_count is not None else (0, 0)
    for batch in batch_factory():
        matrix = xgb.DMatrix(
            batch.features,
            feature_names=list(FEATURE_NAMES),
            feature_types=list(FEATURE_TYPES),
            enable_categorical=True,
        )
        predictions.append(booster.predict(matrix, iteration_range=iteration_range))
        sample_indices.append(batch.sample_indices)
        sensor_indices.append(batch.sensor_indices)
    if not predictions:
        raise ValueError("Prediction batch factory produced no rows")
    return (
        np.concatenate(predictions),
        np.concatenate(sample_indices),
        np.concatenate(sensor_indices),
    )


def save_booster(booster: xgb.Booster, path: Path) -> None:
    """Save a categorical XGBoost model in JSON format."""
    if path.suffix.lower() != ".json":
        raise ValueError("Categorical XGBoost models must be saved as JSON")
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(path)


def load_booster(path: Path) -> xgb.Booster:
    """Load a JSON model for reproducible inference."""
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
