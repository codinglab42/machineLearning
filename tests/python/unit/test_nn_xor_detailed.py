import numpy as np
import pytest
import machine_learning_module as ml

def test_nn_xor_detailed():
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.float64)
    
    # Test con diverse configurazioni (senza set_learning_rate)
    configs = [
        {"layers": [2, 8, 1], "epochs": 5000},      # Default LR
        {"layers": [2, 16, 1], "epochs": 5000},     # Più largo
        {"layers": [2, 32, 16, 1], "epochs": 5000}, # Più profondo
    ]
    
    for i, cfg in enumerate(configs):
        print(f"\n--- Config {i+1}: layers={cfg['layers']} ---")
        nn = ml.NeuralNetwork(cfg["layers"], "relu", "sigmoid")
        nn.set_epochs(cfg["epochs"])
        nn.set_batch_size(4)
        nn.set_validation_split(0.0)
        nn.set_verbose(True)
        
        nn.fit(X, y)
        
        pred = nn.predict(X)
        print(f"Predictions: {pred.flatten()}")
        print(f"Loss finale: {nn.loss_history[-1]:.6f}")
        
        # Verifica che le predizioni siano nette
        ambiguous = sum(1 for p in pred.flatten() if 0.3 <= p <= 0.7)
        print(f"Predizioni ambigue (0.3-0.7): {ambiguous}/4")