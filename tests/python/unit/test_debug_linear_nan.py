import pytest
import numpy as np
import machine_learning_module as ml


def test_debug_linear_nan():
    """Debug del problema NaN in LinearRegression"""

    
    # Dati molto semplici e piccoli
    X = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3]
    ], dtype=np.float64)
    
    y = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    
    print("Testing with small scaled data...")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"y range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Prova con learning rate molto piccolo
    model = ml.LinearRegression(learning_rate=0.001, max_iter=1000)
    model.fit(X, y)
    
    print(f"\nResults:")
    print(f"  coefficients: {model.coefficients}")
    print(f"  intercept: {model.intercept}")
    
    # Prova con scaling manuale
    print("\n" + "="*50)
    print("Testing with manual scaling...")
    
    # Scala i dati
    X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y_scaled = (y - y.mean()) / (y.std() + 1e-8)
    
    model2 = ml.LinearRegression(learning_rate=0.1, max_iter=1000)
    model2.fit(X_scaled, y_scaled)
    
    print(f"Scaled results:")
    print(f"  coefficients: {model2.coefficients}")
    print(f"  intercept: {model2.intercept}")