def test_after_bug_fix():
    """Test dopo aver corretto il bug del doppio intercept"""
    import machine_learning_module as ml
    import numpy as np
    
    print("=== TEST DOPO CORREZIONE BUG ===")
    
    # Test 1: 1 feature
    print("\n1. Test con 1 feature")
    X1 = np.array([[1], [2], [3]], dtype=np.float64)
    y1 = np.array([2, 4, 6], dtype=np.float64)  # y = 2x
    
    lr1 = ml.LinearRegression(solver=ml.LinearSolver.NORMAL_EQUATION)
    lr1.fit(X1, y1)
    
    print(f"   intercept: {lr1.intercept:.6f} (expected: 0)")
    print(f"   coefficient: {lr1.coefficients[0]:.6f} (expected: 2)")
    print(f"   predict(4): {lr1.predict(np.array([[4]]))[0]:.6f} (expected: 8)")
    
    # Test 2: 3 feature
    print("\n2. Test con 3 feature")
    X3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    y3 = np.array([1, 2, 3], dtype=np.float64)  # y = x1 + 2*x2 + 3*x3
    
    lr3 = ml.LinearRegression(solver=ml.LinearSolver.NORMAL_EQUATION)
    lr3.fit(X3, y3)
    
    print(f"   intercept: {lr3.intercept:.6f} (expected: 0)")
    print(f"   coefficients: {[f'{c:.6f}' for c in lr3.coefficients]}")
    print(f"   expected: [1, 2, 3]")
    
    # Test 3: Gradient Descent
    print("\n3. Test Gradient Descent (con dati scalati)")
    X_gd = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
    y_gd = np.array([0.2, 0.4, 0.6], dtype=np.float64)  # y = 2x
    
    lr_gd = ml.LinearRegression(learning_rate=0.1, max_iter=1000)
    lr_gd.fit(X_gd, y_gd)
    
    print(f"   intercept: {lr_gd.intercept:.6f}")
    print(f"   coefficient: {lr_gd.coefficients[0]:.6f}")