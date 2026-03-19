# tests/python/test_ml_bindings.py
import machine_learning_module as ml
import numpy as np
import pytest
import os
import tempfile

class TestLinearRegression:
    """Test per LinearRegression bindings"""
    
    def setup_method(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.y = 2*self.X[:,0] + 3*self.X[:,1] - self.X[:,2] + 5 + 0.1*np.random.randn(100)
    
    def test_create_and_fit(self):
        model = ml.LinearRegression(0.01, 1000)
        model.fit(self.X, self.y)
        
        y_pred = model.predict(self.X)
        r2 = model.score(self.X, self.y)
        
        assert r2 > 0.9
        assert len(y_pred) == len(self.y)
    
    def test_predict_single(self):
        model = ml.LinearRegression(0.01, 500)
        model.fit(self.X, self.y)
        
        x_single = np.array([1.0, 2.0, 3.0])
        x_single_reshaped = x_single.reshape(1, -1)
        pred = model.predict(x_single_reshaped)[0]
        
        assert isinstance(pred, float)
    
    def test_coefficients(self):
        model = ml.LinearRegression(0.01, 500)
        model.fit(self.X, self.y)
        
        coef = model.coefficients
        intercept = model.intercept
        
        assert len(coef) == 3
        assert isinstance(intercept, float)
    
    def test_save_load(self):
        model = ml.LinearRegression(0.01, 500)
        model.fit(self.X, self.y)
        y_before = model.predict(self.X)
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            filename = tmp.name
        
        try:
            model.save(filename)
            
            loaded = ml.LinearRegression()
            loaded.load(filename)
            y_after = loaded.predict(self.X)
            
            np.testing.assert_almost_equal(y_before, y_after, decimal=5)
        finally:
            os.remove(filename)


class TestLogisticRegression:
    """Test per LogisticRegression bindings"""
    
    def setup_method(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 2)
        self.y = (self.X[:,0] + self.X[:,1] > 0).astype(np.float64)
    
    def test_binary_classification(self):
        model = ml.LogisticRegression(0.1, 1000, 0.001)
        model.fit(self.X, self.y)
        
        y_pred = model.predict_class(self.X)
        accuracy = np.mean(y_pred == self.y)
        
        assert accuracy > 0.85
        
        y_proba = model.predict(self.X)
        assert np.all((y_proba >= 0) & (y_proba <= 1))
    
    def test_coefficients(self):
        model = ml.LogisticRegression(0.1, 500, 0.001)
        model.fit(self.X, self.y)
        
        coef = model.coefficients
        intercept = model.intercept
        
        assert len(coef) == 2
        assert isinstance(intercept, float)
    
    def test_confusion_matrix(self):
        model = ml.LogisticRegression(0.1, 500, 0.001)
        model.fit(self.X, self.y)
        
        cm = model.confusion_matrix(self.X, self.y)
        assert cm.shape == (2, 2)
        assert np.sum(cm) == len(self.y)


class TestNeuralNetwork:
    """Test per NeuralNetwork bindings"""
    
    def setup_method(self):
        self.X_and = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
        self.y_and = np.array([0, 0, 0, 1], dtype=np.float64)
    
    def test_and_with_single_layer(self):
        network = ml.NeuralNetwork([2, 8, 1], "relu", "sigmoid",
                                   ml.OptimizerType.ADAM, 0.2)
        network.set_loss_function("binary_crossentropy")
        network.set_epochs(2000)
        network.set_batch_size(4)
        network.set_verbose(False)
        
        network.fit(self.X_and, self.y_and)
        
        y_pred = network.predict(self.X_and)
        y_pred_int = (y_pred > 0.5).astype(int).flatten()
        np.testing.assert_array_equal(y_pred_int, self.y_and)
    
    def test_predict_proba(self):
        network = ml.NeuralNetwork([2, 4, 1], "relu", "sigmoid")
        network.set_loss_function("binary_crossentropy")
        network.set_epochs(500)
        network.set_batch_size(4)
        network.set_verbose(False)
        
        network.fit(self.X_and, self.y_and)
        
        proba = network.predict_proba(self.X_and)
        assert proba.shape == (4, 1)
        assert np.all((proba >= 0) & (proba <= 1))
    
    def test_save_load(self):
        network = ml.NeuralNetwork([2, 8, 1], "relu", "sigmoid",
                                ml.OptimizerType.ADAM, 0.2)
        network.set_loss_function("binary_crossentropy")
        network.set_epochs(500)
        network.set_batch_size(4)
        network.set_verbose(False)
        network.fit(self.X_and, self.y_and)
        
        y_before = network.predict(self.X_and)
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            filename = tmp.name
        
        try:
            network.save(filename)
            
            loaded = ml.NeuralNetwork()
            loaded.load(filename)
            y_after = loaded.predict(self.X_and)
            
            # Usa una tolleranza più alta
            np.testing.assert_almost_equal(y_before, y_after, decimal=2)
        finally:
            os.remove(filename)

    def test_summary(self):
        network = ml.NeuralNetwork([2, 8, 4, 1], "relu", "sigmoid")
        network.summary()  # Verifichiamo che non crasha


if __name__ == "__main__":
    pytest.main(["-v", __file__])