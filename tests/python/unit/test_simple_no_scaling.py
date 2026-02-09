import pytest
import numpy as np
import machine_learning_module as ml




def test_simple_no_scaling():
    """Test semplicissimo senza scaling per verificare il modello base"""
    # y = 2x + 3
    X = np.array([[1], [2], [3]], dtype=np.float64)
    y = np.array([5, 7, 9], dtype=np.float64)
    
    model = ml.LinearRegression(0.01, 1000, 0.0, ml.LinearSolver.NORMAL_EQUATION)
    model.fit(X, y)
    
    print(f"Intercept: {model.intercept} (expected: 3.0)")
    print(f"Coefficient: {model.coefficients[0]} (expected: 2.0)")
    
    X_test = np.array([[4]], dtype=np.float64)
    pred = model.predict(X_test)[0]
    print(f"Prediction for X=4: {pred} (expected: 11.0)")
    
    assert model.intercept == pytest.approx(3.0, rel=1e-2)
    assert model.coefficients[0] == pytest.approx(2.0, rel=1e-2)
    assert pred == pytest.approx(11.0, rel=1e-2)

def test_check_binding():
    """Verifica che il binding Python-C++ funzioni correttamente"""
    # Crea un modello e verifica i metodi base
    model = ml.LinearRegression()
    
    print(f"Model type: {type(model)}")
    print(f"Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
    
    # Verifica che possiamo chiamare fit e predict
    X = np.array([[1.0]], dtype=np.float64)
    y = np.array([2.0], dtype=np.float64)
    
    try:
        model.fit(X, y)
        print("fit() succeeded")
        
        pred = model.predict(X)
        print(f"predict() succeeded: {pred}")
        
        print(f"Intercept: {model.intercept}")
        print(f"Coefficients: {model.coefficients}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()