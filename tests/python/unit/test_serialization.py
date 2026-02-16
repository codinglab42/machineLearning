import pytest
import numpy as np
import machine_learning_module as ml

@pytest.fixture
def sample_regression_data():
    """Genera dati per Linear Regression (3 feature)"""
    np.random.seed(42)
    X = np.ascontiguousarray(np.random.rand(20, 3), dtype=np.float64)
    y = np.ascontiguousarray(np.dot(X, np.array([1.5, -2.0, 1.0])) + 0.5, dtype=np.float64)
    return X, y

@pytest.fixture
def sample_binary_classification_data():
    """Genera dati per Logistic Regression / NN (2 feature)"""
    np.random.seed(42)
    X = np.ascontiguousarray(np.random.rand(20, 2), dtype=np.float64)
    y = np.ascontiguousarray((X[:, 0] + X[:, 1] > 1).astype(float), dtype=np.float64)
    return X, y

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


def test_save_load_linear(sample_regression_data, tmp_path):
        """Test salvataggio e caricamento"""
        X, y = sample_regression_data
        model = ml.LinearRegression()
        model.fit(X, y)

        save_path = tmp_path / "model.bin"
        model.save(str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

        loaded_model = ml.LinearRegression()
        loaded_model.load(str(save_path))

        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

def test_save_load_logistic(sample_binary_classification_data, tmp_path):
        """Test salvataggio e caricamento"""
        X, y = sample_binary_classification_data
        model = ml.LogisticRegression()
        model.fit(X, y)
        
        save_path = tmp_path / "logistic_model.bin"
        model.save(str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        
        loaded_model = ml.LogisticRegression()
        loaded_model.load(str(save_path))
        
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    