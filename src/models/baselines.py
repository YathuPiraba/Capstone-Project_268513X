"""Leakage-safe persistence and historical-average forecasting baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SLOTS_PER_DAY = 24 * 12
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY


def persistence_forecast(input_speeds: np.ndarray, output_length: int = 12) -> np.ndarray:
    """Repeat each sample's final observed speed across every forecast step."""
    values = np.asarray(input_speeds, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("Input speeds must have shape (samples, history, sensors)")
    if output_length <= 0:
        raise ValueError("Output length must be positive")
    return np.repeat(values[:, -1:, :], output_length, axis=1)


@dataclass(frozen=True)
class HistoricalAverage:
    """Per-sensor, slot-of-week averages fitted only from training rows.

    Finite exact zeros are preserved as recorded measurements during fitting.
    A per-sensor training mean is used only if a requested weekly slot has no
    finite training observation for that sensor.
    """

    slot_means: np.ndarray
    sensor_fallback: np.ndarray
    slot_counts: np.ndarray

    @classmethod
    def fit(
        cls,
        training_speeds: np.ndarray,
        training_timestamps: pd.DatetimeIndex,
    ) -> "HistoricalAverage":
        values = np.asarray(training_speeds, dtype=np.float64)
        index = pd.DatetimeIndex(training_timestamps)
        if values.ndim != 2 or values.shape[0] != len(index) or values.shape[1] == 0:
            raise ValueError("Training speeds must have shape (timestamps, sensors)")
        if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("Training timestamps must be valid, unique, and ordered")

        finite = np.isfinite(values)
        sensor_counts = finite.sum(axis=0)
        if np.any(sensor_counts == 0):
            raise ValueError("Every sensor needs at least one finite training observation")
        sensor_fallback = np.sum(np.where(finite, values, 0.0), axis=0) / sensor_counts

        slots = _slot_of_week(index)
        sums = np.zeros((SLOTS_PER_WEEK, values.shape[1]), dtype=np.float64)
        counts = np.zeros((SLOTS_PER_WEEK, values.shape[1]), dtype=np.int64)
        np.add.at(sums, slots, np.where(finite, values, 0.0))
        np.add.at(counts, slots, finite.astype(np.int64))
        means = np.divide(
            sums,
            counts,
            out=np.broadcast_to(sensor_fallback, sums.shape).copy(),
            where=counts > 0,
        )
        return cls(slot_means=means, sensor_fallback=sensor_fallback, slot_counts=counts)

    def predict(self, target_timestamps: np.ndarray) -> np.ndarray:
        """Forecast speeds for target timestamps with shape (samples, horizons)."""
        timestamps = np.asarray(target_timestamps)
        if timestamps.ndim != 2:
            raise ValueError("Target timestamps must have shape (samples, horizons)")
        flat_index = pd.DatetimeIndex(timestamps.reshape(-1))
        if flat_index.hasnans:
            raise ValueError("Target timestamps must be valid")
        predicted = self.slot_means[_slot_of_week(flat_index)]
        return predicted.reshape(*timestamps.shape, self.slot_means.shape[1])


def _slot_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    minute_slot = index.hour * 12 + index.minute // 5
    return np.asarray(index.day_of_week * SLOTS_PER_DAY + minute_slot, dtype=np.int64)
