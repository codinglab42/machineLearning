import pytest
import numpy as np
import machine_learning_module as ml

def test_logistic_regression_training():
    # 1. Generazione dati: due gruppi distinti
    np.random.seed(42)
    # Cluster 0: centrato in (1, 1)
    X0 = np.random.randn(50, 2) + 1
    y0 = np.zeros((50, 1))
    # Cluster 1: centrato in (4, 4)
    X1 = np.random.randn(50, 2) + 4
    y1 = np.ones((50, 1))
    
    X = np.vstack((X0, X1))
    y = np.vstack((y0, y1))

    # 2. Inizializzazione (usando i nomi scoperti dal debug precedente)
    # Usiamo un learning rate più alto per la logistica
    model = ml.LogisticRegression(learning_rate=0.5, max_iter=1000)
    
    # 3. Training
    model.fit(X, y)
    
    # 4. Verifica predizioni
    # Un punto vicino a (1,1) deve essere classe 0 (prob < 0.5)
    pred_low = model.predict(np.array([[0.5, 0.5]]))
    # Un punto vicino a (4,4) deve essere classe 1 (prob > 0.5)
    pred_high = model.predict(np.array([[4.5, 4.5]]))
    
    print(f"\n[LOGISTIC DEBUG] Pred basso: {pred_low}, Pred alto: {pred_high}")
    
    assert pred_low[0] < 0.5
    assert pred_high[0] > 0.5

def test_logistic_decision_boundary():
    # Test specifico: un punto esattamente in mezzo dovrebbe essere incerto
    # Se il confine è tra (1,1) e (4,4), il punto (2.5, 2.5) è vicino al confine
    model = ml.LogisticRegression(learning_rate=0.1, max_iter=1000)
    
    X = np.array([[1.0, 1.0], [4.0, 4.0]])
    y = np.array([[0.0], [1.0]])
    model.fit(X, y)
    
    # Il punto medio dovrebbe avere una probabilità vicina a 0.5
    mid_pred = model.predict(np.array([[2.5, 2.5]]))
    assert 0.3 <= mid_pred[0] <= 0.7