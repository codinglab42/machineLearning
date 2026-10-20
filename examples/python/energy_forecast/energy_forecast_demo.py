#!/usr/bin/env python3
"""Forecast next-day electricity demand from synthetic daily observations."""

from __future__ import annotations

import argparse

import ml_core as ml
import numpy as np


DAYS_PER_YEAR = 365.0
MINIMUM_DAYS = 400
TEST_FRACTION = 0.2
HISTORY_DAYS = 7


def simulate_demand(days: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return next-day features, demand targets, and a persistence baseline."""
    rng = np.random.default_rng(seed)
    day_index = np.arange(days, dtype=np.float64)
    annual_phase = 2.0 * np.pi * day_index / DAYS_PER_YEAR
    weekday_phase = 2.0 * np.pi * day_index / 7.0

    actual_temperature = (
        12.0
        + 11.0 * np.sin(annual_phase - 0.8)
        + rng.normal(0.0, 3.0, days)
    )
    forecast_temperature = actual_temperature + rng.normal(0.0, 1.5, days)
    demand = np.empty(days, dtype=np.float64)
    demand[0] = 220.0

    for day in range(1, days):
        weekend_effect = -18.0 if day_index[day] % 7 in (5.0, 6.0) else 0.0
        heating_demand = 2.2 * max(0.0, 16.0 - actual_temperature[day])
        cooling_demand = 1.7 * max(0.0, actual_temperature[day] - 23.0)
        seasonal_demand = 12.0 * np.sin(annual_phase[day] + 0.3)
        weekly_demand = 5.0 * np.sin(weekday_phase[day] - 0.5)
        baseline = 220.0 + weekend_effect + heating_demand + cooling_demand
        demand[day] = (
            0.62 * demand[day - 1]
            + 0.38 * baseline
            + seasonal_demand
            + weekly_demand
            + rng.normal(0.0, 5.0)
        )

    target_days = np.arange(HISTORY_DAYS, days)
    features = np.column_stack(
        (
            demand[target_days - 1],
            demand[target_days - 7],
            np.asarray(
                [demand[day - HISTORY_DAYS : day].mean() for day in target_days],
                dtype=np.float64,
            ),
            forecast_temperature[target_days],
            np.sin(2.0 * np.pi * target_days / 7.0),
            np.cos(2.0 * np.pi * target_days / 7.0),
            np.sin(2.0 * np.pi * target_days / DAYS_PER_YEAR),
            np.cos(2.0 * np.pi * target_days / DAYS_PER_YEAR),
        )
    )
    targets = demand[target_days]
    persistence_baseline = demand[target_days - 1]
    return np.asfortranarray(features), targets, persistence_baseline


def scale_features(
    train_features: np.ndarray,
    other_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize from training data only to avoid test-set leakage."""
    means = train_features.mean(axis=0)
    scales = train_features.std(axis=0)
    scales[scales == 0.0] = 1.0
    return (
        np.asfortranarray((train_features - means) / scales),
        np.asfortranarray((other_features - means) / scales),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=4 * 365, help="number of synthetic days")
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducible simulation")
    args = parser.parse_args()
    if args.days < MINIMUM_DAYS:
        parser.error(f"--days must be at least {MINIMUM_DAYS}")
    return args


def main() -> None:
    args = parse_args()
    features, targets, persistence_baseline = simulate_demand(args.days, args.seed)
    split_index = int(len(targets) * (1.0 - TEST_FRACTION))
    if split_index <= 0 or split_index >= len(targets):
        raise ValueError("not enough observations to create chronological train and test sets")

    train_features, test_features = scale_features(
        features[:split_index],
        features[split_index:],
    )
    train_targets = targets[:split_index]
    test_targets = targets[split_index:]
    naive_predictions = persistence_baseline[split_index:]

    model = ml.LinearRegression(0.01, 2000, 0.01)
    model.fit(train_features, train_targets)
    predictions = np.asarray(model.predict(test_features), dtype=np.float64).reshape(-1)

    model_mae = float(np.mean(np.abs(test_targets - predictions)))
    model_rmse = float(np.sqrt(np.mean((test_targets - predictions) ** 2)))
    baseline_mae = float(np.mean(np.abs(test_targets - naive_predictions)))
    baseline_rmse = float(np.sqrt(np.mean((test_targets - naive_predictions) ** 2)))
    r_squared = float(model.score(test_features, test_targets))

    print("Synthetic next-day electricity demand forecast")
    print(f"Observations: {args.days} daily values")
    print(
        f"Chronological split: {split_index} training samples, "
        f"{len(test_targets)} test samples"
    )
    print("Features: recent demand, 7-day average, calendar cycles, forecast temperature")
    print(f"Test mean demand: {float(test_targets.mean()):.1f} synthetic units/day")
    print(f"Library model: MAE={model_mae:.2f}, RMSE={model_rmse:.2f}, R²={r_squared:.3f}")
    print(
        "Persistence baseline (tomorrow = today): "
        f"MAE={baseline_mae:.2f}, RMSE={baseline_rmse:.2f}"
    )
    print("\nSample forecasts (actual vs predicted, test period):")
    for actual, predicted in zip(test_targets[:7], predictions[:7]):
        print(f"  actual {actual:7.2f} | forecast {predicted:7.2f}")
    print("\nSynthetic data only; these scores do not estimate real-world performance.")


if __name__ == "__main__":
    main()
