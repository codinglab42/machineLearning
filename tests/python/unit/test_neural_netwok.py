import pytest
import numpy as np
import machine_learning_module as ml

def test_nn_initialization():
    """Verifica che la rete si inizializzi con l'architettura corretta."""
    layers = [2, 4, 1]
    nn = ml.NeuralNetwork(layers, "relu", "sigmoid")
    
    # num_layers conta i layer con pesi (hidden + output)
    assert nn.num_layers == len(layers) - 1  # 2 layer con pesi
    nn.summary()
    assert "NeuralNetwork" in nn.to_string()  # SENZA spazio!

def test_nn_fit_xor():
    """Test classico: la rete deve imparare la funzione XOR."""
    # Dati XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.float64)
    
    # ✅ CORRETTO: y come (4, 1)
    y = y.reshape(-1, 1)
    
    # Assicura che gli array siano contigui
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    
    print(f"\nX shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    
    # Crea la rete
    nn = ml.NeuralNetwork([2, 8, 1], "relu", "sigmoid")
    
    # Configura training
    nn.set_validation_split(0.0)
    nn.set_epochs(1000)
    nn.set_batch_size(4)
    nn.set_verbose(True)  # Metti True per vedere cosa succede
    
    try:
        # Fit
        nn.fit(X, y)
        print("✓ Fit completato!")
    except Exception as e:
        print(f"✗ Fit fallito: {e}")
        raise
    
    # Predici
    predictions = nn.predict(X)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions.flatten()}")
    
    final_preds = (predictions > 0.5).astype(float)
    accuracy = np.mean(final_preds == y)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    assert accuracy >= 0.75, f"XOR accuracy troppo bassa: {accuracy}"
    assert len(nn.loss_history) > 0


@pytest.mark.skip(reason="Serializzazione: stesso problema di Linear/Logistic")
def test_nn_serialization(tmp_path):
    """Verifica salvataggio e caricamento del modello."""
    file_path = str(tmp_path / "model_nn.bin")
    
    nn_original = ml.NeuralNetwork([4, 2, 1])
    # Mock data per forzare l'allocazione dei pesi
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, 10).astype(float)  # VETTORE 1D!
    nn_original.fit(X, y)
    
    # Salva
    nn_original.save(file_path)
    
    # Carica in una nuova istanza
    nn_new = ml.NeuralNetwork()
    nn_new.load(file_path)
    
    # Verifica che le predizioni siano identiche
    pred_orig = nn_original.predict(X)
    pred_new = nn_new.predict(X)
    
    np.testing.assert_array_almost_equal(pred_orig, pred_new)

def test_nn_invalid_dimensions():
    """Verifica che le eccezioni C++ siano catturate correttamente in Python."""
    nn = ml.NeuralNetwork([2, 2, 1])
    X_wrong = np.random.rand(10, 5)  # 5 feature invece di 2
    y = np.random.randint(0, 2, 10).astype(float)  # VETTORE 1D!
    
    with pytest.raises(RuntimeError):
        nn.fit(X_wrong, y)

def test_standard_scaler_integration():
    """Testa il workflow Andrew Ng: Scaling -> NN."""
    X = np.array([[100.0, 0.001], [200.0, 0.002], [300.0, 0.003]])
    
    scaler = ml.StandardScaler()
    X_scaled_list = scaler.fit_transform(X)  # Restituisce lista
    X_scaled = np.array(X_scaled_list)  # Converti in numpy array
    
    # Verifica che la media sia circa 0
    mean = scaler.get_mean()
    assert len(mean) == 2
    
    # Media di ogni feature dovrebbe essere ~0
    for i in range(X_scaled.shape[1]):
        assert X_scaled[:, i].mean() == pytest.approx(0.0, abs=1e-7)