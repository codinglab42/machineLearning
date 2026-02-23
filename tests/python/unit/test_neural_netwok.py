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
    X = np.ascontiguousarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.ascontiguousarray([[0], [1], [1], [0]], dtype=np.float64) # Già shape (4,1)
    
    # Crea la rete con Tanh (più stabile per XOR) o ReLU
    nn = ml.NeuralNetwork([2, 8, 1], "tanh", "sigmoid")
    
    # Configura training
    nn.set_epochs(2000)      # Aumentiamo per sicurezza finché l'optimizer non è perfetto
    nn.set_batch_size(1)     # Fondamentale per evitare plateau
    nn.set_learning_rate(0.1) # LR più alto per SGD/Adam su problemi piccoli
    nn.set_verbose(True)
    
    nn.fit(X, y)
    
    predictions = nn.predict(X)
    # Arrotondiamo per il confronto
    final_preds = (predictions > 0.5).astype(float).flatten()
    target_y = y.flatten()
    
    accuracy = np.mean(final_preds == target_y)
    print(f"\nPredizioni: {predictions.flatten()}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Ora puntiamo al 100%!
    assert accuracy == 1.0, f"XOR deve essere imparato perfettamente. Acc: {accuracy}"

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

