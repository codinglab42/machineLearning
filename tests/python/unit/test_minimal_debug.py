def test_minimal_debug():
    """Test minimale per debug"""
    import machine_learning_module as ml
    import numpy as np
    
    # CASO MINIMO: 1 feature
    X = np.array([[1], [2], [3]], dtype=np.float64)
    y = np.array([2, 4, 6], dtype=np.float64)  # y = 2x
    
    print(f"Minimal test: X shape {X.shape}, y = 2x")
    
    lr = ml.LinearRegression(solver=ml.LinearSolver.NORMAL_EQUATION)
    lr.fit(X, y)
    
    print(f"  intercept: {lr.intercept} (expected: 0)")
    print(f"  coefficients: {lr.coefficients} (expected: [2])")
    print(f"  len(coefficients): {len(lr.coefficients)} (expected: 1)")
    
    # Prediction test
    X_test = np.array([[4]], dtype=np.float64)
    pred = lr.predict(X_test)[0]
    print(f"  predict(4): {pred} (expected: 8)")