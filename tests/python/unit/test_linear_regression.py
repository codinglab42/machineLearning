import pytest
import numpy as np
import machine_learning_module as ml

def test_linear_regression_intercept():
    """Verifica il calcolo dell'intercetta con scaling ESPLICITO"""
    X = np.array([[1], [2], [3]], dtype=np.float64)
    y = np.array([6, 7, 8], dtype=np.float64)
    
    # Creiamo lo scaler tramite la tua nuova factory
    scaler = ml.create_scaler("standard")
    X_scaled = scaler.fit_transform(X)
    
    # Training (usiamo un learning rate più alto dato che i dati sono piccoli e scalati)
    model = ml.LinearRegression(0.1, 1000)
    model.fit(X_scaled, y)
    
    # Ricorda: con i dati scalati, l'intercetta è la media di y (7.0)
    # Se vuoi l'intercetta originale (5.0), devi usare dati NON scalati 
    # ma con il solver NORMAL_EQUATION
    assert model.intercept == pytest.approx(7.0, rel=1e-2)

def test_linear_regression_normal_equation():
    """Verifica l'intercetta originale usando le equazioni normali senza scaling"""
    X = np.array([[1], [2], [3]], dtype=np.float64)
    y = np.array([6, 7, 8], dtype=np.float64)
    
    # Con NORMAL_EQUATION non abbiamo bisogno di scalare per convergere
    model = ml.LinearRegression(0.01, 100, 0.0, ml.LinearSolver.NORMAL_EQUATION)
    model.fit(X, y)
    
    # Qui l'intercetta DEVE essere 5.0
    assert model.intercept == pytest.approx(5.0, rel=1e-2)