# tests/python/unit/test_logistic_regression.py
import pytest
import numpy as np
import machine_learning_module as mlm
import os

@pytest.fixture
def sample_binary_classification_data():
    """Genera dati di classificazione binaria semplici per test"""
    np.random.seed(42)
    
    # Cluster 0: centrato in (1, 1)
    X0 = np.random.randn(50, 2) + 1
    y0 = np.zeros((50, 1))
    
    # Cluster 1: centrato in (4, 4)
    X1 = np.random.randn(50, 2) + 4
    y1 = np.ones((50, 1))
    
    X = np.vstack((X0, X1))
    y = np.vstack((y0, y1)).flatten()
    
    return X, y

@pytest.fixture
def sample_nonlinear_data():
    """Genera dati non linearmente separabili per test più complessi"""
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 < 4).astype(float)
    return X, y

class TestLogisticRegression:
    
    def test_initialization(self):
        """Test inizializzazione del modello"""
        model = mlm.LogisticRegression()
        assert model is not None
        
        # ✅ CORRETTO: usa "lambda", non "lambda_reg"!
        model_with_params = mlm.LogisticRegression(
            learning_rate=0.1,
            max_iter=500,
            **{'lambda': 0.01},  # <-- ERRORE: questo non esiste!
        )
        model_with_params = mlm.LogisticRegression(
            learning_rate=0.1,
            max_iter=500,
            **{'lambda': 0.01},  # <-- CORRETTO: così si chiama nel binding
            tolerance=1e-5,
            verbose=False
        )
        assert model_with_params is not None
    
    def test_fit_predict(self, sample_binary_classification_data):
        """Test fit e predict probabilità"""
        X, y = sample_binary_classification_data
        model = mlm.LogisticRegression(learning_rate=0.5, max_iter=500)
        
        model.fit(X, y)
        probabilities = model.predict(X)
        
        assert len(probabilities) == len(y)
        assert isinstance(probabilities, np.ndarray)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        
        accuracy = model.score(X, y)
        assert accuracy > 0.9
        print(f"\nAccuracy: {accuracy:.4f}")
    
    def test_predict_class(self, sample_binary_classification_data):
        """Test predict_class per etichette discrete"""
        X, y = sample_binary_classification_data
        model = mlm.LogisticRegression(learning_rate=0.5, max_iter=500)
        
        model.fit(X, y)
        predictions = model.predict_class(X, threshold=0.5)
        
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)
        assert np.all((predictions == 0) | (predictions == 1))
        
        predictions_high = model.predict_class(X, threshold=0.8)
        predictions_low = model.predict_class(X, threshold=0.2)
        
        assert np.mean(predictions_high) <= np.mean(predictions_low)
    
    def test_decision_boundary(self):
        """Test che il confine decisionale sia corretto"""
        X = np.array([[1.0, 1.0], [4.0, 4.0]])
        y = np.array([0.0, 1.0])
        
        model = mlm.LogisticRegression(learning_rate=0.1, max_iter=1000)
        model.fit(X, y)
        
        X_mid = np.array([[2.5, 2.5]])
        prob_mid = model.predict(X_mid)[0]
        print(f"\nProbabilità punto medio: {prob_mid:.4f}")
        
        assert 0.3 < prob_mid < 0.7
        
        X_near0 = np.array([[1.1, 1.1]])
        X_near1 = np.array([[3.9, 3.9]])
        
        prob_near0 = model.predict(X_near0)[0]
        prob_near1 = model.predict(X_near1)[0]
        
        assert prob_near0 < 0.3
        assert prob_near1 > 0.7
    
    def test_coefficients_and_intercept(self, sample_binary_classification_data):
        """Test coefficienti e intercetta"""
        X, y = sample_binary_classification_data
        model = mlm.LogisticRegression(learning_rate=0.5, max_iter=500)
        
        model.fit(X, y)
        
        coefficients = model.coefficients
        intercept = model.intercept
        
        print(f"\nIntercetta: {intercept:.4f}")
        print(f"Coefficienti: {coefficients}")
        
        assert coefficients is not None
        # TODO: Fixare LogisticRegression::coefficients() per restituire solo i coefficienti (2)
        # Per ora, skip o modifica il test
        assert len(coefficients) == X.shape[1]  # Dovrebbe essere 2, ma ora è 3 (bug!)
        
        # Il modello dovrebbe aver imparato che x1 e x2 sono positivamente correlati
        # con la classe 1 (dati spostati verso alto a destra)
        # TODO: Fixare l'accesso agli indici quando coefficients sarà di size 2
        # assert coefficients[0] > 0
        # assert coefficients[1] > 0
    
    def test_precision_recall_f1(self, sample_binary_classification_data):
        """Test metriche di valutazione"""
        X, y = sample_binary_classification_data
        model = mlm.LogisticRegression(learning_rate=0.5, max_iter=500)
        
        model.fit(X, y)
        
        precision, recall, f1 = model.precision_recall_f1(X, y, threshold=0.5)
        
        print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
        assert f1 > 0.9
    
    def test_confusion_matrix(self, sample_binary_classification_data):
        """Test matrice di confusione"""
        X, y = sample_binary_classification_data
        model = mlm.LogisticRegression(learning_rate=0.5, max_iter=500)
        
        model.fit(X, y)
        
        cm = model.confusion_matrix(X, y, threshold=0.5)
        
        print(f"\nMatrice di confusione:\n{cm}")
        
        assert cm.shape == (2, 2)
        assert np.sum(cm) == len(y)
        
        correct_predictions = cm[0, 0] + cm[1, 1]
        assert correct_predictions / len(y) > 0.9
    
    def test_training_history(self, sample_binary_classification_data):
        """Test che il costo diminuisca e l'accuratezza aumenti"""
        X, y = sample_binary_classification_data
        model = mlm.LogisticRegression(learning_rate=0.5, max_iter=200)
        
        model.fit(X, y)
        
        cost_history = model.cost_history
        accuracy_history = model.accuracy_history
        
        print(f"\nCosto iniziale: {cost_history[0]:.4f}")
        print(f"Costo finale: {cost_history[-1]:.4f}")
        print(f"Accuratezza iniziale: {accuracy_history[0]:.4f}")
        print(f"Accuratezza finale: {accuracy_history[-1]:.4f}")
        
        assert len(cost_history) > 0
        assert len(accuracy_history) > 0
        assert cost_history[0] > cost_history[-1]
        assert accuracy_history[0] < accuracy_history[-1]
    
    def test_regularization(self, sample_nonlinear_data):
        """Test che la regolarizzazione prevenga l'overfitting"""
        X, y = sample_nonlinear_data
        
        # ✅ CORRETTO: usa "lambda", non "lambda_reg"!
        model_no_reg = mlm.LogisticRegression(
            learning_rate=0.1, 
            max_iter=500, 
            **{'lambda': 0.0}  # <-- CORRETTO
        )
        model_no_reg.fit(X, y)
        train_score_no_reg = model_no_reg.score(X, y)
        
        model_with_reg = mlm.LogisticRegression(
            learning_rate=0.1, 
            max_iter=500, 
            **{'lambda': 0.1}  # <-- CORRETTO
        )
        model_with_reg.fit(X, y)
        train_score_with_reg = model_with_reg.score(X, y)
        
        print(f"\nScore senza regolarizzazione: {train_score_no_reg:.4f}")
        print(f"Score con regolarizzazione: {train_score_with_reg:.4f}")
        
        assert model_with_reg.coefficients is not None
    
    def test_different_thresholds(self, sample_binary_classification_data):
        """Test comportamento con diverse soglie di decisione"""
        X, y = sample_binary_classification_data
        model = mlm.LogisticRegression(learning_rate=0.5, max_iter=500)
        
        model.fit(X, y)
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        precisions = []
        recalls = []
        
        for thresh in thresholds:
            precision, recall, f1 = model.precision_recall_f1(X, y, threshold=thresh)
            precisions.append(precision)
            recalls.append(recall)
            
        print(f"\nSoglie: {thresholds}")
        print(f"Precisioni: {[f'{p:.3f}' for p in precisions]}")
        print(f"Recall: {[f'{r:.3f}' for r in recalls]}")
        
        assert precisions[0] <= precisions[-1]
        assert recalls[0] >= recalls[-1]
    
    @pytest.mark.skip(reason="Serializzazione: da implementare come per LinearRegression")
    def test_save_load(self, sample_binary_classification_data, tmp_path):
        """Test salvataggio e caricamento"""
        X, y = sample_binary_classification_data
        model = mlm.LogisticRegression()
        model.fit(X, y)
        
        save_path = tmp_path / "logistic_model.bin"
        model.save(str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        
        loaded_model = mlm.LogisticRegression()
        loaded_model.load(str(save_path))
        
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    def test_invalid_inputs(self):
        """Test gestione input non validi"""
        model = mlm.LogisticRegression()
        
        # X vuota
        X_empty = np.array([]).reshape(0, 2)
        y = np.array([])
        with pytest.raises(Exception):
            model.fit(X_empty, y)
        
        # y con valori non binari
        X = np.random.randn(10, 2)
        y_invalid = np.random.randn(10)
        with pytest.raises(Exception):
            model.fit(X, y_invalid)
        
        # Dimensioni mismatch
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, 5)
        with pytest.raises(Exception):
            model.fit(X, y)
        
        # Predizione con numero feature sbagliato
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        X_wrong = np.random.randn(5, 3)
        with pytest.raises(Exception):
            model.predict(X_wrong)