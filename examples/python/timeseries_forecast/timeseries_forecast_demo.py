#!/usr/bin/env python3
"""Forecast a synthetic daily time series from sliding historical windows."""

from __future__ import annotations

import argparse

import ml_core as ml
import numpy as np


LOOKBACK_DAYS = 28
MINIMUM_DAYS = 300
TEST_FRACTION = 0.2


def simulate_series(days: int, seed: int) -> np.ndarray:
    """Create a noisy daily series with weekly and annual cycles."""
    rng = np.random.default_rng(seed)
    values = np.empty(days, dtype=np.float64)
    values[0] = 100.0
    for day in range(1, days):
        seasonal = (
            12.0 * np.sin(2.0 * np.pi * day / 7.0)
            + 24.0 * np.sin(2.0 * np.pi * day / 365.0)
            + 0.015 * day
        )
        values[day] = (
            seasonal
            + 0.55 * (values[day - 1] - seasonal)
            + rng.normal(0.0, 4.0)
        )
    return values


def make_windows(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create one-step-ahead examples from contiguous past observations."""
    sample_count = len(series) - LOOKBACK_DAYS
    if sample_count <= 0:
        raise ValueError(f"series must contain more than {LOOKBACK_DAYS} values")
    features = np.empty((sample_count, LOOKBACK_DAYS), dtype=np.float64, order="F")
    targets = np.empty(sample_count, dtype=np.float64)
    for row in range(sample_count):
        features[row] = series[row : row + LOOKBACK_DAYS]
        targets[row] = series[row + LOOKBACK_DAYS]
    return features, targets


def standardize(
    train_features: np.ndarray,
    other_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize each feature using training-period statistics only."""
    means = train_features.mean(axis=0)
    scales = train_features.std(axis=0)
    scales[scales == 0.0] = 1.0
    return (
        np.asfortranarray((train_features - means) / scales),
        np.asfortranarray((other_features - means) / scales),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=5 * 365, help="number of synthetic daily values")
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducible simulation")
    parser.add_argument("--epochs", type=int, default=120, help="training epochs")
    args = parser.parse_args()
    if args.days < MINIMUM_DAYS:
        parser.error(f"--days must be at least {MINIMUM_DAYS}")
    if args.epochs < 1:
        parser.error("--epochs must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    series = simulate_series(args.days, args.seed)
    features, targets = make_windows(series)

    split_index = int(len(targets) * (1.0 - TEST_FRACTION))
    if split_index <= 0 or split_index >= len(targets):
        raise ValueError("not enough observations to create chronological train and test sets")

    train_features, test_features = standardize(
        features[:split_index],
        features[split_index:],
    )
    target_mean = float(targets[:split_index].mean())
    target_scale = float(targets[:split_index].std())
    if target_scale == 0.0:
        target_scale = 1.0
    train_targets_scaled = (targets[:split_index] - target_mean) / target_scale
    test_targets_scaled = (targets[split_index:] - target_mean) / target_scale
    persistence_baseline = features[split_index:, -1]
    seasonal_baseline = features[split_index:, -7]

    model = ml.NeuralNetwork(
        [LOOKBACK_DAYS, 32, 1],
        "relu",
        "linear",
        ml.OptimizerType.ADAM,
        0.001,
    )
    model.set_loss_function("mse")
    model.set_epochs(args.epochs)
    model.set_batch_size(32)
    model.set_verbose(False)
    model.fit(train_features, train_targets_scaled)

    predictions_scaled = np.asarray(model.predict(test_features), dtype=np.float64).reshape(-1)
    predictions = predictions_scaled * target_scale + target_mean
    actual = test_targets_scaled * target_scale + target_mean

    model_mae = float(np.mean(np.abs(actual - predictions)))
    model_rmse = float(np.sqrt(np.mean((actual - predictions) ** 2)))
    baseline_mae = float(np.mean(np.abs(actual - persistence_baseline)))
    seasonal_mae = float(np.mean(np.abs(actual - seasonal_baseline)))
    mean_actual = float(actual.mean())
    total_squared_error = float(np.sum((actual - predictions) ** 2))
    total_variation = float(np.sum((actual - mean_actual) ** 2))
    r_squared = 1.0 - total_squared_error / total_variation if total_variation else 0.0

    first_test_target_index = LOOKBACK_DAYS + split_index
    print("Synthetic one-step-ahead time-series forecast")
    print(f"Series length: {args.days} daily observations")
    print(
        f"Chronological split: {split_index} training windows, "
        f"{len(actual)} test windows"
    )
    print(
        f"Input: previous {LOOKBACK_DAYS} days | "
        f"Test period begins at day {first_test_target_index}"
    )
    print(
        f"MLP forecast: MAE={model_mae:.2f}, RMSE={model_rmse:.2f}, "
        f"R²={r_squared:.3f}"
    )
    print(f"Persistence baseline (tomorrow = today): MAE={baseline_mae:.2f}")
    print(f"Weekly seasonal baseline (tomorrow = last week's value): MAE={seasonal_mae:.2f}")
    print("\nSample forecasts (actual vs predicted, test period):")
    for observed, predicted in zip(actual[:7], predictions[:7]):
        print(f"  actual {observed:7.2f} | forecast {predicted:7.2f}")
    print("\nSynthetic data only; these scores do not estimate real-world performance.")


if __name__ == "__main__":
    main()
