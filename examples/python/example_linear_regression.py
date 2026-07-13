#!/usr/bin/env python3
"""
Linear Regression Example
=========================
Demonstrates linear regression using the ML library.
"""

import machine_learning_modules as ml
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=" * 40)
    print("  LINEAR REGRESSION EXAMPLE")
    print("=" * 40)
    print()
    
    # 1. Generate synthetic data
    print("1. Generating synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 5 + 0.1 * np.random.randn(n_samples)
    
    print(f"   Generated {n_samples} samples with 2 features")
    print("   True relationship: y = 2*x1 + 3*x2 + 5 + noise\n")
    
    # 2. Split data
    print("2. Splitting data (80% train, 20% test)...")
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}\n")
    
    # 3. Train model
    print("3. Training model...")
    model = ml.LinearRegression(0.01, 1000)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    print("\n4. Evaluation on test set:")
    r2 = model.score(X_test, y_test)
    print(f"   R² Score: {r2:.4f}")
    
    # 5. Print coefficients
    print("\n5. Learned coefficients:")
    coef = model.coefficients
    intercept = model.intercept
    print(f"   Intercept: {intercept:.4f}")
    for i, c in enumerate(coef):
        print(f"   Coefficient x{i+1}: {c:.4f}")
    print("   True: intercept = 5.00, coef1 = 2.00, coef2 = 3.00\n")
    
    # 6. Make predictions
    print("6. Making predictions:")
    X_new = np.array([[1.5, 2.5]])
    y_pred = model.predict(X_new)
    print(f"   Input: x1=1.5, x2=2.5")
    print(f"   Prediction: {y_pred[0]:.4f}")
    print(f"   True value: {2*1.5 + 3*2.5 + 5:.4f}")
    
    print("\n✅ Linear Regression example completed!")

if __name__ == "__main__":
    main()