import numpy as np
import pytest
import machine_learning_module as ml

def test_nn_fit_xor():
    """Test classico: la rete deve imparare la funzione XOR."""
    
    # 1. DATI DI INPUT (Sempre MatrixXd 2D)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    X = np.ascontiguousarray(X)
    
    # 2. TARGET (Proviamo la versione 1D "piatta" che Eigen::VectorXd preferisce)
    y = np.array([0, 1, 1, 0], dtype=np.float64)
    y = np.ascontiguousarray(y)
    
    print(f"\n[PYTHON DEBUG] X shape: {X.shape}, y shape: {y.shape}")
    
    # Crea la rete: 2 input, 8 nascosti, 1 output
    nn = ml.NeuralNetwork([2, 8, 1], "relu", "sigmoid")
    
    nn.set_validation_split(0.0)
    nn.set_epochs(5000)
    # nn.set_learning_rate(0.1)
    nn.set_batch_size(4)
    nn.set_verbose(True)

    # TEST A: Verifica se accetta il vettore piatto (4,)
    try:
        print("[PYTHON] Tentativo fit con y 1D (4,)...")
        nn.fit(X, y)
        print("✅ Fit 1D completato!")
    except RuntimeError as e:
        print(f"❌ Fit 1D fallito: {e}")
        
    # TEST B: Verifica se accetta il vettore colonna (4, 1)
    y_col = y.reshape(-1, 1)
    print(f"\n[PYTHON DEBUG] Provando con y_col shape: {y_col.shape}")
    try:
        print("[PYTHON] Tentativo fit con y 2D (4, 1)...")
        nn.fit(X, y_col)
        print("✅ Fit 2D completato!")
    except RuntimeError as e:
        # Qui intercettiamo l'errore che ti sta tormentando
        print(f"🔥 Errore intercettato in y 2D: {e}")
        # Se l'errore contiene ancora (4, 2), la colpa è del binding di X
        if "(4, 2)" in str(e):
            print("⚠️ Il sistema sta ancora scambiando le colonne di X per le righe di y!")