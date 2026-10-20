#!/usr/bin/env python3
"""Classify small synthetic grayscale images with an ml_core neural network."""

from __future__ import annotations

import argparse

import ml_core as ml
import numpy as np


IMAGE_SIDE = 10
MINIMUM_SAMPLES = 100
TEST_FRACTION = 0.2


def generate_images(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy images of vertical or horizontal bars."""
    if samples % 2:
        raise ValueError("samples must be even to keep the two classes balanced")

    rng = np.random.default_rng(seed)
    images = np.empty((samples, IMAGE_SIDE * IMAGE_SIDE), dtype=np.float64, order="F")
    labels = np.empty(samples, dtype=np.float64)

    for sample in range(samples):
        label = sample % 2
        image = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float64)
        position = int(rng.integers(2, IMAGE_SIDE - 3))
        thickness = int(rng.integers(1, 3))
        if label == 0:
            image[:, position : position + thickness] = 1.0
        else:
            image[position : position + thickness, :] = 1.0

        image += rng.normal(0.0, 0.08, image.shape)
        np.clip(image, 0.0, 1.0, out=image)
        images[sample] = image.reshape(-1)
        labels[sample] = label

    return images, labels


def stratified_split(
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices while keeping both classes in train and test."""
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in (0.0, 1.0):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        test_count = max(1, int(round(indices.size * TEST_FRACTION)))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return np.asarray(train_indices), np.asarray(test_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=400,
        help="number of generated images (must be even)",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed for image generation and split")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    args = parser.parse_args()
    if args.samples < MINIMUM_SAMPLES:
        parser.error(f"--samples must be at least {MINIMUM_SAMPLES}")
    if args.samples % 2:
        parser.error("--samples must be even")
    if args.epochs < 1:
        parser.error("--epochs must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    images, labels = generate_images(args.samples, args.seed)
    train_indices, test_indices = stratified_split(labels, args.seed)

    model = ml.NeuralNetwork(
        [IMAGE_SIDE * IMAGE_SIDE, 32, 1],
        "relu",
        "sigmoid",
        ml.OptimizerType.ADAM,
        0.001,
    )
    model.set_loss_function("binary_crossentropy")
    model.set_epochs(args.epochs)
    model.set_batch_size(32)
    model.set_verbose(False)
    model.fit(images[train_indices], labels[train_indices])

    probabilities = np.asarray(model.predict_proba(images[test_indices])).reshape(-1)
    predicted = (probabilities >= 0.5).astype(np.int64)
    actual = labels[test_indices].astype(np.int64)
    true_negative = int(np.sum((actual == 0) & (predicted == 0)))
    false_positive = int(np.sum((actual == 0) & (predicted == 1)))
    false_negative = int(np.sum((actual == 1) & (predicted == 0)))
    true_positive = int(np.sum((actual == 1) & (predicted == 1)))
    accuracy = float(np.mean(predicted == actual))
    baseline_accuracy = float(max(np.mean(actual == 0), np.mean(actual == 1)))

    print("Synthetic grayscale image classification")
    print(f"Image size: {IMAGE_SIDE}x{IMAGE_SIDE} pixels | Classes: vertical bar / horizontal bar")
    print(
        f"Images: {args.samples} | Training: {len(train_indices)} | "
        f"Test: {len(test_indices)}"
    )
    print(f"Model: multilayer perceptron ({IMAGE_SIDE * IMAGE_SIDE}-32-1), sigmoid output")
    print(f"Test accuracy: {accuracy:.1%} | Majority-class baseline: {baseline_accuracy:.1%}")
    print(
        "Confusion matrix (actual rows, predicted columns): "
        f"TN={true_negative}, FP={false_positive}, "
        f"FN={false_negative}, TP={true_positive}"
    )
    print("\nSynthetic data only; these scores do not estimate real-world performance.")


if __name__ == "__main__":
    main()
