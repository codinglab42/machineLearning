def test_final_verification():
    """Verifica finale dopo tutte le correzioni"""
    import machine_learning_module as ml
    import numpy as np
    
    print("=== FINAL VERIFICATION ===\n")
    
    # 1. LinearRegression coefficient count
    print("1. LinearRegression coefficient count")
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    y = np.array([6, 15, 24], dtype=np.float64)  # y = x1 + x2 + x3
    
    lr = ml.LinearRegression(solver=ml.LinearSolver.NORMAL_EQUATION)
    lr.fit(X, y)
    
    print(f"   X has {X.shape[1]} features")
    print(f"   coefficients has {len(lr.coefficients)} elements")
    print(f"   ✓ OK" if len(lr.coefficients) == X.shape[1] else f"   ✗ WRONG: expected {X.shape[1]}, got {len(lr.coefficients)}")
    
    # 2. Check prediction consistency
    print("\n2. Prediction consistency check")
    X_test = np.array([[2, 3, 4]], dtype=np.float64)
    pred = lr.predict(X_test)[0]
    
    # Calcolo manuale
    manual_pred = lr.intercept + sum(c*x for c, x in zip(lr.coefficients, X_test[0]))
    
    print(f"   Model prediction: {pred:.6f}")
    print(f"   Manual prediction: {manual_pred:.6f}")
    print(f"   ✓ Match" if np.isclose(pred, manual_pred, rtol=1e-5) else f"   ✗ Mismatch")
    
    # 3. LogisticRegression feature check
    print("\n3. LogisticRegression feature check")
    X_log = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0]
    ], dtype=np.float64)
    y_log = np.array([0, 1, 0], dtype=np.float64)
    
    logr = ml.LogisticRegression()
    try:
        logr.fit(X_log, y_log)
        print(f"   Fit successful")
        
        X_test_log = np.array([[1.5, 2.5, 3.5]], dtype=np.float64)
        pred_log = logr.predict(X_test_log)
        print(f"   Prediction: {pred_log[0]:.6f}")
        print(f"   ✓ OK")
    except Exception as e:
        print(f"   ✗ Error: {e}")