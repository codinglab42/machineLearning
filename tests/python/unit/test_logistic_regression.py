import pytest
import numpy as np
import machine_learning_module as ml

def test_logistic_regression_binary_classification():
    """Verifica se il modello separa correttamente due gruppi di punti"""
    # Classe 0: centri intorno a (1,1) - Classe 1: centri intorno a (5,5)
    X = np.array([[0.5, 0.5], [1.0, 1.5], [4.5, 5.0], [5.5, 5.5]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    
    # Scaling (MinMax in questo caso)
    scaler = ml.create_scaler("minmax")
    X_scaled = scaler.fit_transform(X)
    
    model = ml.LogisticRegression(0.5, 2000)
    model.fit(X_scaled, y)
    
    # Testiamo i punti originali (scalati)
    probs = model.predict(X_scaled)
    
    assert probs[0] < 0.5 # Dovrebbe essere classe 0
    assert probs[3] > 0.5 # Dovrebbe essere classe 1
    
    # Verifica accuratezza
    acc = model.score(X_scaled, y)
    assert acc == 1.0

def test_logistic_serialization():
    """Verifica che il modello mantenga i pesi dopo il salvataggio"""
    X = np.array([[1], [2], [3], [4]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    
    model = ml.LogisticRegression()
    model.fit(X, y)
    
    model.save("logistic_tmp.bin")
    
    new_model = ml.LogisticRegression()
    new_model.load("logistic_tmp.bin")
    
    assert new_model.intercept == pytest.approx(model.intercept)
    assert np.allclose(new_model.coefficients, model.coefficients)