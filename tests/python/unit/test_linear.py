import pytest
import numpy as np
import machine_learning_module as ml

def test_linear_regression_convergence():
    # Generiamo dati: y = 3 * x1 + 5 * x2 + 10
    np.random.seed(42)
    X = np.random.rand(100, 2)
    weights_true = np.array([[3.0], [5.0]])
    bias_true = 10.0
    y = X @ weights_true + bias_true + np.random.normal(0, 0.01, (100, 1))

    # Inizializzazione modello
    # Assicurati che i parametri (lr, iterations) coincidano con il tuo costruttore C++
    model = ml.LinearRegression(learning_rate=0.1, max_iter=1000)
    
    model.fit(X, y)
    
    # Test su un nuovo punto
    X_new = np.array([[1.0, 1.0]])
    expected = 3.0 * 1.0 + 5.0 * 1.0 + 10.0 # 18.0
    prediction = model.predict(X_new)
    
    assert prediction[0] == pytest.approx(expected, abs=0.1)

def test_linear_errors_on_mismatched_dimensions():
    # Test per verificare che le tue eccezioni C++ (include/exceptions) vengano sollevate
    model = ml.LinearRegression(0.01, 10)
    X = np.ones((5, 2))
    y = np.ones((10, 1)) # Dimensioni incompatibili (5 vs 10)
    
    with pytest.raises(RuntimeError): # Pybind11 traduce le eccezioni C++ in RuntimeError di default
        model.fit(X, y)