#!/usr/bin/env python3
"""
Edge cases, error handling, and memory layout tests for ML Library Python Bindings
"""

import unittest
import sys
import os
import numpy as np

# Path del modulo pybind11
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/pybinding'))

try:
    import machine_learning_modules as ml
except ImportError:
    print("❌ Failed to import machine_learning_modules")
    print("   Make sure the module is built: make -j$(nproc)")
    sys.exit(1)


class TestErrorHandlingAndShapes(unittest.TestCase):
    """Test per la gestione degli errori sulle dimensioni e parametri non validi"""

    def setUp(self):
        np.random.seed(42)

    def test_dimension_mismatch_fit(self):
        """Mismatched row count between X and y during fit"""
        X = np.random.randn(100, 5)
        y = np.random.randn(80)  # 80 invece di 100
        
        model = ml.LinearRegression()
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(X, y)

    def test_feature_mismatch_predict(self):
        """Predicting on data with a different number of features than trained on"""
        X_train = np.random.randn(100, 5)
        y_train = (X_train[:, 0] > 0).astype(float)
        
        X_test_wrong = np.random.randn(20, 3)  # 3 feature invece di 5
        
        model = ml.LogisticRegression(max_iter=50)
        model.fit(X_train, y_train)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.predict(X_test_wrong)

    def test_predict_before_fit(self):
        """Calling predict or score before model is fitted"""
        nn = ml.NeuralNetwork([5, 8, 1], "relu", "sigmoid", ml.OptimizerType.ADAM, 0.001)
        nn.build(5, 2)
        X = np.random.randn(10, 5)
        
        # Un modello non addestrato deve sollevare un'eccezione o gestire la predizione senza crashare
        with self.assertRaises((ValueError, RuntimeError)):
            nn.predict(X)

    def test_empty_dataset(self):
        """Passing empty arrays to fit"""
        X_empty = np.empty((0, 5))
        y_empty = np.empty((0,))
        
        model = ml.LinearRegression()
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(X_empty, y_empty)


class TestMemoryAndNumPyInterop(unittest.TestCase):
    """Test per il layout di memoria (C vs Fortran) e lo slicing NumPy"""

    def setUp(self):
        np.random.seed(42)

    def test_strided_numpy_arrays(self):
        """Array NumPy non contigui in memoria (es. slicing con passo > 1)"""
        X_full = np.random.randn(200, 4)
        y_full = (X_full[:, 0] + X_full[:, 1] > 0).astype(float)
        
        # Slicing con passo 2 -> memoria non contigua
        X_strided = X_full[::2, :]
        y_strided = y_full[::2]
        
        model = ml.LogisticRegression(max_iter=50, verbose=False)
        # Non deve andare in SegFault ma gestire la memoria in modo sicuro
        model.fit(X_strided, y_strided)
        preds = model.predict(X_strided)
        self.assertEqual(len(preds), 100)

    def test_c_vs_fortran_layout_consistency(self):
        """Verifica che C-Contiguous e Fortran-Contiguous diano lo stesso risultato"""
        X_base = np.random.randn(100, 3)
        y = 2.0 * X_base[:, 0] + 0.5 * X_base[:, 1]
        
        X_c = np.ascontiguousarray(X_base)
        X_f = np.asfortranarray(X_base)
        
        model_c = ml.LinearRegression(learning_rate=0.01, max_iter=100)
        model_c.fit(X_c, y)
        pred_c = model_c.predict(X_c)
        
        model_f = ml.LinearRegression(learning_rate=0.01, max_iter=100)
        model_f.fit(X_f, y)
        pred_f = model_f.predict(X_f)
        
        # Le predizioni devono coincidere perfettamente
        np.testing.assert_allclose(pred_c, pred_f, rtol=1e-5)

    def test_float32_input_casting(self):
        """Verifica la gestione di array in float32 invece di float64"""
        X_f32 = np.random.randn(100, 4).astype(np.float32)
        y_f32 = (X_f32[:, 0] > 0).astype(np.float32)
        
        model = ml.LogisticRegression(max_iter=30)
        # Deve convertire/gestire il tipo senza sollevare errori di tipo inattesi
        model.fit(X_f32, y_f32)
        preds = model.predict(X_f32)
        self.assertEqual(len(preds), 100)


class TestCornerCasesData(unittest.TestCase):
    """Test per casi limite sui dati (batch piccolissimi, varianza zero)"""

    def setUp(self):
        np.random.seed(42)

    def test_small_batch_size(self):
        """Batch size maggiore del numero totale di campioni"""
        X = np.random.randn(10, 4)
        y = (X[:, 0] > 0).astype(float)
        
        nn = ml.NeuralNetwork([4, 8, 1], "relu", "sigmoid", ml.OptimizerType.ADAM, 0.01)
        nn.set_epochs(10)
        nn.set_batch_size(64)  # 64 > 10 campioni totali
        nn.set_verbose(False)
        nn.build(4, 2)
        
        # Deve completare l'addestramento senza crashare su sotto-batch vuoti
        nn.fit(X, y)
        score = nn.score(X, y)
        self.assertGreaterEqual(score, 0.0)

    def test_zero_variance_scaler(self):
        """StandardScaler con una colonna costante (varianza zero)"""
        X = np.random.randn(100, 3)
        X[:, 1] = 7.0  # Colonna con valore fisso -> std = 0
        
        scaler = ml.StandardScaler()
        X_scaled = np.array(scaler.fit_transform(X))
        
        # Non devono esserci valori NaN o Inf derivanti da divisioni per zero
        self.assertFalse(np.isnan(X_scaled).any(), "StandardScaler produced NaNs on zero-variance feature")
        self.assertFalse(np.isinf(X_scaled).any(), "StandardScaler produced Infs on zero-variance feature")


if __name__ == "__main__":
    unittest.main()