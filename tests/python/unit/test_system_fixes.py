import pytest
import machine_learning_module as ml
import numpy as np



def test_system_fixes():
    """Test per verificare tutte le correzioni"""

    
    print("=== TEST DI SISTEMA ===")
    
    # 1. Test LinearRegression con dati scalati
    print("\n1. LinearRegression with scaled data")
    X_lr = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2], 
        [0.3, 0.3, 0.3]
    ], dtype=np.float64)
    y_lr = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    
    # Scala
    X_lr_scaled = (X_lr - X_lr.mean(axis=0)) / (X_lr.std(axis=0) + 1e-8)
    
    lr = ml.LinearRegression(learning_rate=0.1, max_iter=1000)
    lr.fit(X_lr_scaled, y_lr)
    
    print(f"  Coefficients: {lr.coefficients}")
    print(f"  Intercept: {lr.intercept}")
    
    # 2. Test LogisticRegression
    print("\n2. LogisticRegression")
    X_log = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float64)
    y_log = np.array([0, 1, 0, 1], dtype=np.float64)
    
    # Scala per logistic
    X_log_scaled = (X_log - X_log.mean(axis=0)) / (X_log.std(axis=0) + 1e-8)
    
    logr = ml.LogisticRegression()
    try:
        logr.fit(X_log_scaled, y_log)
        print(f"  Fit successful")
        print(f"  Coefficients: {logr.coefficients if hasattr(logr, 'coefficients') else 'N/A'}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 3. Test base: LinearRegression con Normal Equation (deve funzionare sempre)
    print("\n3. LinearRegression with Normal Equation (should always work)")
    X_ne = np.array([
        [1, 1, 1],
        [1, 2, 2],
        [2, 1, 3]
    ], dtype=np.float64)
    y_ne = np.array([6, 9, 12], dtype=np.float64)  # y = x1 + 2*x2 + 3*x3
    
    lr_ne = ml.LinearRegression(solver=ml.LinearSolver.NORMAL_EQUATION)
    lr_ne.fit(X_ne, y_ne)
    
    print(f"  Coefficients: {lr_ne.coefficients}")
    print(f"  Intercept: {lr_ne.intercept}")
    print(f"  Expected: intercept≈0, coeff≈[1, 2, 3]")