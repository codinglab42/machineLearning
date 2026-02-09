import pytest
import numpy as np
import machine_learning_module as ml

def test_debug_scaling_issue():
    """Debug per capire cosa succede"""
    X = np.array([[1], [2], [3]], dtype=np.float64)
    y = np.array([6, 7, 8], dtype=np.float64)
    
    print("Original data:")
    print(f"X = {X}")
    print(f"y = {y}")
    
    # Scala X
    scaler = ml.create_scaler("standard")
    X_scaled = scaler.fit_transform(X)
    
    print("\nAfter StandardScaler on X:")
    print(f"X_scaled = {X_scaled}")
    print(f"Mean of X_scaled: {np.mean(X_scaled, axis=0)}")
    print(f"Std of X_scaled: {np.std(X_scaled, axis=0)}")
    
    # Fit modello
    model = ml.LinearRegression(0.1, 1000)
    model.fit(X_scaled, y)
    
    print(f"\nModel results:")
    print(f"Intercept: {model.intercept}")
    print(f"Coefficient: {model.coefficients}")
    
    # Calcola cosa dovrebbe essere
    # Quando X_scaled = 0 (X = mean_X = 2.0), y = 7.0
    print(f"\nExpected: When X_scaled = 0, y should be {7.0}")
    print(f"Actual prediction at X_scaled = 0: {model.intercept}")
    
    # Verifica la predizione
    X_test_scaled = scaler.transform(np.array([[2.0]], dtype=np.float64))
    print(f"\nX = 2.0 -> X_scaled = {X_test_scaled[0][0]}")
    pred = model.predict(X_test_scaled)[0]
    print(f"Prediction for X = 2.0: {pred}")
    print(f"Expected: 7.0")