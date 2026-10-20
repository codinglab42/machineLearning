#!/usr/bin/env python3
"""Predict simulated machine failures from daily sensor readings."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import ml_core as ml
import numpy as np


MACHINE_COUNT = 20
DAYS_PER_MACHINE = 365
PREDICTION_HORIZON_DAYS = 7
TEST_START_DAY = 270


def simulate_sensor_data(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    """Generate reproducible sensor data, labels, and failure days."""
    rng = np.random.default_rng(seed)
    features: list[list[float]] = []
    labels: list[int] = []
    machine_ids: list[int] = []
    days: list[int] = []
    failures_by_machine: list[list[int]] = []

    for machine_id in range(MACHINE_COUNT):
        degradation = float(rng.uniform(0.0, 0.25))
        daily_wear = float(rng.uniform(0.022, 0.038))
        temperature_offset = float(rng.normal(0.0, 1.5))
        vibration_offset = float(rng.normal(0.0, 0.25))
        current_offset = float(rng.normal(0.0, 0.4))
        sensor_history: list[tuple[float, float, float]] = []
        failure_days: list[int] = []

        for day in range(DAYS_PER_MACHINE):
            degradation += max(0.0, float(rng.normal(daily_wear, daily_wear * 0.2)))
            if degradation >= 1.0:
                failure_days.append(day)
                degradation = float(rng.uniform(0.05, 0.15))

            temperature = (
                55.0
                + temperature_offset
                + 20.0 * degradation
                + float(rng.normal(0.0, 2.0))
            )
            vibration = (
                1.5
                + vibration_offset
                + 5.0 * degradation
                + float(rng.normal(0.0, 0.5))
            )
            current = (
                8.0
                + current_offset
                + 1.8 * degradation
                + float(rng.normal(0.0, 0.4))
            )
            previous = sensor_history[-1] if sensor_history else (temperature, vibration, current)
            sensor_history.append((temperature, vibration, current))

            features.append(
                [
                    temperature,
                    vibration,
                    current,
                    temperature - previous[0],
                    vibration - previous[1],
                    current - previous[2],
                ]
            )
            machine_ids.append(machine_id)
            days.append(day)

        failures_by_machine.append(failure_days)

    for machine_id in range(MACHINE_COUNT):
        machine_failures = failures_by_machine[machine_id]
        for day in range(DAYS_PER_MACHINE):
            labels.append(
                int(
                    any(
                        day < failure_day <= day + PREDICTION_HORIZON_DAYS
                        for failure_day in machine_failures
                    )
                )
            )

    return (
        np.asarray(features, dtype=np.float64, order="F"),
        np.asarray(labels, dtype=np.float64),
        np.asarray(machine_ids, dtype=np.int64),
        np.asarray(days, dtype=np.int64),
        failures_by_machine,
    )


def standardize(
    train_features: np.ndarray,
    other_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale features using training statistics only."""
    means = train_features.mean(axis=0)
    scales = train_features.std(axis=0)
    scales[scales == 0] = 1.0
    return (
        np.asfortranarray((train_features - means) / scales),
        np.asfortranarray((other_features - means) / scales),
    )


def summarize_events(
    predicted: np.ndarray,
    test_machine_ids: np.ndarray,
    test_days: np.ndarray,
    failures_by_machine: Sequence[Sequence[int]],
) -> tuple[int, int, float]:
    """Count test failures detected in advance and compute warning lead time."""
    detected = 0
    eligible = 0
    lead_days: list[int] = []

    for machine_id, machine_failures in enumerate(failures_by_machine):
        machine_mask = test_machine_ids == machine_id
        machine_test_days = test_days[machine_mask]
        machine_predictions = predicted[machine_mask]
        for failure_day in machine_failures:
            if not TEST_START_DAY + PREDICTION_HORIZON_DAYS <= failure_day < DAYS_PER_MACHINE:
                continue
            eligible += 1
            warning_mask = (
                (machine_test_days < failure_day)
                & (machine_test_days >= failure_day - PREDICTION_HORIZON_DAYS)
                & (machine_predictions == 1)
            )
            warning_days = machine_test_days[warning_mask]
            if warning_days.size:
                detected += 1
                lead_days.append(int(failure_day - warning_days.min()))

    mean_lead_hours = 24.0 * float(np.mean(lead_days)) if lead_days else 0.0
    return detected, eligible, mean_lead_hours


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducible simulation")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="probability threshold for raising an alert (default: 0.5)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    features, labels, machine_ids, days, failures_by_machine = simulate_sensor_data(args.seed)

    # Leave a horizon-sized gap so training labels cannot depend on test-period failures.
    train_mask = days < TEST_START_DAY - PREDICTION_HORIZON_DAYS
    test_mask = days >= TEST_START_DAY
    train_features, test_features = standardize(features[train_mask], features[test_mask])
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]
    test_machine_ids = machine_ids[test_mask]
    test_days = days[test_mask]

    model = ml.LogisticRegression(0.05, 1500, 0.01, 1e-4, False)
    model.fit(train_features, train_labels)

    probabilities = np.asarray(model.predict(test_features), dtype=np.float64)
    predicted = (probabilities >= args.threshold).astype(np.int64)
    confusion = np.asarray(model.confusion_matrix(test_features, test_labels, args.threshold))
    true_negative, false_positive = int(confusion[0, 0]), int(confusion[0, 1])
    false_negative, true_positive = int(confusion[1, 0]), int(confusion[1, 1])
    precision, recall, f1 = map(
        float,
        model.precision_recall_f1(test_features, test_labels, args.threshold),
    )
    accuracy = float(np.mean(predicted == test_labels))
    prevalence = float(np.mean(test_labels))
    false_alarms_per_100_machine_days = (
        false_positive / test_labels.size * 100.0
    )
    detected, eligible, mean_lead_hours = summarize_events(
        predicted,
        test_machine_ids,
        test_days,
        failures_by_machine,
    )

    print("Synthetic predictive-maintenance demo")
    print(
        f"Machines: {MACHINE_COUNT} | Days per machine: {DAYS_PER_MACHINE} | "
        f"Prediction horizon: {PREDICTION_HORIZON_DAYS} days"
    )
    print(
        f"Train: {int(train_mask.sum())} daily samples (days 0-{TEST_START_DAY - PREDICTION_HORIZON_DAYS - 1}) | "
        f"Test: {int(test_mask.sum())} daily samples (days {TEST_START_DAY}-{DAYS_PER_MACHINE - 1})"
    )
    print(f"Test failure prevalence: {prevalence:.1%} (always-negative baseline accuracy: {1.0 - prevalence:.1%})")
    print(f"Alert threshold: {args.threshold:.2f}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.1%} | Recall: {recall:.1%} | F1: {f1:.1%}")
    print(
        "Confusion matrix (actual rows, predicted columns): "
        f"TN={true_negative}, FP={false_positive}, FN={false_negative}, TP={true_positive}"
    )
    print(f"False alerts: {false_alarms_per_100_machine_days:.2f} per 100 machine-days")
    if eligible:
        print(f"Failures detected in advance: {detected}/{eligible} ({detected / eligible:.1%})")
        if detected:
            print(f"Mean warning lead time for detected failures: {mean_lead_hours:.1f} hours")
    else:
        print("No eligible test-period failures to evaluate.")
    print("\nThese are simulated data; the scores do not predict real-world performance.")


if __name__ == "__main__":
    main()
