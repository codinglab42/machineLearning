#!/usr/bin/env python3
"""
Integration tests for ML Library Python Bindings
"""

import unittest
import sys
import os
import numpy as np
import time

# Aggiungi il path del modulo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/pybinding'))

try:
    import machine_learning_module as ml
except ImportError:
    print("❌ Failed to import machine_learning_module")
    print("   Make sure the module is built: make -j$(nproc)")
    sys.exit(1)


class TestFullPipeline(unittest.TestCase):
    """Integration tests for full ML pipeline"""
    
    def setUp(self):
        np.random.seed(42)
        self.X_classification = np.random.randn(300, 10)
        self.y_classification = (np.sum(self.X_classification[:, :5], axis=1) > 0).astype(float)
        
        self.X_regression = np.random.randn(300, 5)
        self.y_regression = np.sin(self.X_regression[:, 0]) + np.cos(self.X_regression[:, 1]) + \
                            np.random.randn(300) * 0.05
    
    def test_neural_network_with_scaler(self):
        """Test NeuralNetwork with StandardScaler"""
        # Scale data
        scaler = ml.StandardScaler()
        X_scaled = scaler.fit_transform(self.X_classification.tolist())
        X_scaled = np.array(X_scaled)
        
        # Create network
        nn = ml.NeuralNetwork()
        nn.add_dense_layer(32, activation="relu")
        nn.add_batch_norm_layer()
        nn.add_dense_layer(16, activation="relu")
        nn.add_dense_layer(1, activation="sigmoid")
        
        nn.set_loss_function("binary_crossentropy")
        nn.set_optimizer(ml.OptimizerType.ADAM, learning_rate=0.001)
        nn.set_epochs(50)
        nn.set_batch_size(32)
        nn.set_verbose(False)
        nn.build(10, 1)
        
        # Train
        nn.fit(X_scaled, self.y_classification)
        score = nn.score(X_scaled, self.y_classification)
        self.assertGreater(score, 0.75)
    
    def test_neural_network_with_regularization(self):
        """Test NeuralNetwork with L2 regularization"""
        nn = ml.NeuralNetwork()
        nn.add_dense_layer(32, activation="relu")
        nn.add_dropout_layer(0.2)
        nn.add_dense_layer(16, activation="relu")
        nn.add_dense_layer(1, activation="sigmoid")
        
        nn.set_loss_function("binary_crossentropy")
        nn.set_regularizer(ml.RegularizerType.L2, strength=0.001)
        nn.set_optimizer(ml.OptimizerType.ADAM, learning_rate=0.001)
        nn.set_epochs(50)
        nn.set_batch_size(32)
        nn.set_verbose(False)
        nn.build(10, 1)
        
        nn.fit(self.X_classification, self.y_classification)
        score = nn.score(self.X_classification, self.y_classification)
        self.assertGreater(score, 0.75)
    
    def test_neural_network_regression_with_scaler(self):
        """Test NeuralNetwork regression with StandardScaler"""
        # Scale data
        scaler = ml.StandardScaler()
        X_scaled = scaler.fit_transform(self.X_regression.tolist())
        X_scaled = np.array(X_scaled)
        
        # Scale target
        y_mean = np.mean(self.y_regression)
        y_std = np.std(self.y_regression)
        y_scaled = (self.y_regression - y_mean) / y_std
        
        nn = ml.NeuralNetwork()
        nn.add_dense_layer(32, activation="relu")
        nn.add_batch_norm_layer()
        nn.add_dense_layer(16, activation="relu")
        nn.add_dense_layer(1, activation="linear")
        
        nn.set_loss_function("mse")
        nn.set_optimizer(ml.OptimizerType.ADAM, learning_rate=0.001)
        nn.set_epochs(50)
        nn.set_batch_size(32)
        nn.set_verbose(False)
        nn.build(5, 1)
        
        nn.fit(X_scaled, y_scaled)
        r2 = nn.score(X_scaled, y_scaled)
        self.assertGreater(r2, 0.8)
    
    def test_logistic_regression_with_scaler(self):
        """Test LogisticRegression with StandardScaler"""
        scaler = ml.StandardScaler()
        X_scaled = scaler.fit_transform(self.X_classification.tolist())
        X_scaled = np.array(X_scaled)
        
        model = ml.LogisticRegression(learning_rate=0.1, max_iter=500, verbose=False)
        model.fit(X_scaled, self.y_classification)
        acc = model.score(X_scaled, self.y_classification)
        self.assertGreater(acc, 0.75)
    
    def test_cross_validation(self):
        """Test LinearRegression cross-validation"""
        X = np.random.randn(150, 3)
        y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 0.5 + np.random.randn(150) * 0.1
        
        scores = ml.LinearRegression.cross_val_score(X, y, cv=5)
        self.assertEqual(len(scores), 5)
        self.assertGreater(np.mean(scores), 0.85)


class TestSerialization(unittest.TestCase):
    """Integration tests for serialization"""
    
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = (np.sum(self.X[:, :3], axis=1) > 0).astype(float)
        self.filename = "test_model.bin"
    
    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
    
    def test_neural_network_save_load(self):
        """Test saving and loading NeuralNetwork"""
        # Create and train model
        nn = ml.NeuralNetwork()
        nn.add_dense_layer(16, activation="relu")
        nn.add_dense_layer(8, activation="relu")
        nn.add_dense_layer(1, activation="sigmoid")
        nn.set_epochs(20)
        nn.set_verbose(False)
        nn.build(5, 1)
        nn.fit(self.X, self.y)
        
        # Save
        nn.save(self.filename)
        self.assertTrue(os.path.exists(self.filename))
        
        # Load
        nn_loaded = ml.NeuralNetwork()
        nn_loaded.load(self.filename)
        
        # Compare predictions
        y_pred_original = nn.predict(self.X)
        y_pred_loaded = nn_loaded.predict(self.X)
        diff = np.linalg.norm(y_pred_original - y_pred_loaded)
        self.assertLess(diff, 1e-6)
    
    def test_linear_regression_save_load(self):
        """Test saving and loading LinearRegression"""
        X = np.random.randn(100, 3)
        y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 0.5 + np.random.randn(100) * 0.1
        
        model = ml.LinearRegression(learning_rate=0.01, max_iter=200)
        model.fit(X, y)
        
        # Save
        model.save(self.filename)
        self.assertTrue(os.path.exists(self.filename))
        
        # Load
        model_loaded = ml.LinearRegression()
        model_loaded.load(self.filename)
        
        # Compare predictions
        y_pred_original = model.predict(X)
        y_pred_loaded = model_loaded.predict(X)
        diff = np.linalg.norm(y_pred_original - y_pred_loaded)
        self.assertLess(diff, 1e-6)


class TestOptimizerIntegration(unittest.TestCase):
    """Integration tests for optimizers with layers"""
    
    def test_adam_training_convergence(self):
        """Test Adam optimizer converges"""
        # Simple linear problem
        X = np.random.randn(100, 3)
        y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 0.5
        
        nn = ml.NeuralNetwork()
        nn.add_dense_layer(1, activation="linear", use_bias=True)
        nn.set_loss_function("mse")
        nn.set_optimizer(ml.OptimizerType.ADAM, learning_rate=0.01)
        nn.set_epochs(200)
        nn.set_batch_size(32)
        nn.set_verbose(False)
        nn.build(3, 1)
        nn.fit(X, y)
        
        r2 = nn.score(X, y)
        self.assertGreater(r2, 0.95)
    
    def test_sgd_training_convergence(self):
        """Test SGD optimizer converges"""
        # Simple classification
        X = np.random.randn(100, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        
        nn = ml.NeuralNetwork()
        nn.add_dense_layer(8, activation="relu")
        nn.add_dense_layer(1, activation="sigmoid")
        nn.set_loss_function("binary_crossentropy")
        nn.set_optimizer(ml.OptimizerType.SGD, learning_rate=0.1)
        nn.set_epochs(300)
        nn.set_batch_size(32)
        nn.set_verbose(False)
        nn.build(3, 1)
        nn.fit(X, y)
        
        score = nn.score(X, y)
        self.assertGreater(score, 0.75)


class TestLossIntegration(unittest.TestCase):
    """Integration tests for loss functions with neural network"""
    
    def test_bce_with_neural_network(self):
        """Test binary cross entropy loss"""
        X = np.random.randn(200, 5)
        y = (np.sum(X[:, :3], axis=1) > 0).astype(float)
        
        nn = ml.NeuralNetwork()
        nn.add_dense_layer(16, activation="relu")
        nn.add_dense_layer(8, activation="relu")
        nn.add_dense_layer(1, activation="sigmoid")
        nn.set_loss_function("binary_crossentropy")
        nn.set_epochs(100)
        nn.set_verbose(False)
        nn.build(5, 1)
        nn.fit(X, y)
        
        loss, val_loss, acc = nn.get_training_history()
        self.assertGreater(loss[0], loss[-1])
        self.assertGreater(acc[-1], 0.7)
    
    def test_mse_with_neural_network(self):
        """Test MSE loss"""
        X = np.random.randn(200, 5)
        y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.randn(200) * 0.05
        
        nn = ml.NeuralNetwork()
        nn.add_dense_layer(32, activation="relu")
        nn.add_dense_layer(16, activation="relu")
        nn.add_dense_layer(1, activation="linear")
        nn.set_loss_function("mse")
        nn.set_epochs(100)
        nn.set_verbose(False)
        nn.build(5, 1)
        nn.fit(X, y)
        
        loss, val_loss, acc = nn.get_training_history()
        self.assertGreater(loss[0], loss[-1])
        r2 = nn.score(X, y)
        self.assertGreater(r2, 0.8)


if __name__ == "__main__":
    unittest.main()