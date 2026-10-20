#!/usr/bin/env python3
"""
Unit tests for ML Library Python Bindings
"""

import unittest
import numpy as np

import ml_core as ml



class TestEnums(unittest.TestCase):
    def test_optimizer_type(self):
        self.assertIsNotNone(ml.OptimizerType.SGD)
        self.assertIsNotNone(ml.OptimizerType.MOMENTUM)
        self.assertIsNotNone(ml.OptimizerType.ADAM)
        self.assertIsNotNone(ml.OptimizerType.RMSPROP)
        self.assertIsNotNone(ml.OptimizerType.ADAGRAD)
    
    def test_regularizer_type(self):
        self.assertIsNotNone(ml.RegularizerType.NONE)
        self.assertIsNotNone(ml.RegularizerType.L1)
        self.assertIsNotNone(ml.RegularizerType.L2)
        self.assertIsNotNone(ml.RegularizerType.ELASTIC_NET)
    
    def test_layer_type(self):
        self.assertIsNotNone(ml.LayerType.DENSE)
        self.assertIsNotNone(ml.LayerType.CONV2D)
        self.assertIsNotNone(ml.LayerType.LSTM)
        self.assertIsNotNone(ml.LayerType.GRU)
    
    def test_pool_type(self):
        self.assertIsNotNone(ml.PoolType.MAX)
        self.assertIsNotNone(ml.PoolType.AVG)


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 2)
        self.y = 2.0 * self.X[:, 0] + 1.5 * self.X[:, 1] + 0.5 + np.random.randn(100) * 0.1
    
    def test_construction(self):
        model = ml.LinearRegression()
        self.assertIsNotNone(model)
    
    def test_fit_and_predict(self):
        model = ml.LinearRegression(learning_rate=0.01, max_iter=200)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        self.assertEqual(y_pred.shape, (100,))
    
    def test_score(self):
        model = ml.LinearRegression(learning_rate=0.01, max_iter=200)
        model.fit(self.X, self.y)
        r2 = model.score(self.X, self.y)
        self.assertGreater(r2, 0.85)
    
    def test_coefficients(self):
        model = ml.LinearRegression(learning_rate=0.01, max_iter=200)
        model.fit(self.X, self.y)
        coef = model.coefficients
        self.assertEqual(len(coef), 2)
    
    def test_metrics(self):
        model = ml.LinearRegression(learning_rate=0.01, max_iter=200)
        model.fit(self.X, self.y)
        mse = model.mse(self.X, self.y)
        mae = model.mae(self.X, self.y)
        self.assertGreaterEqual(mse, 0)
        self.assertGreaterEqual(mae, 0)


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(200, 5)
        self.y = (np.sum(self.X[:, :3], axis=1) > 0).astype(float)

    def _check_is_fitted(self, nn):
        """Helper robusto per verificare is_fitted sia come metodo sia come proprietà."""
        if callable(getattr(nn, 'is_fitted', None)):
            return nn.is_fitted()
        return bool(nn.is_fitted)

    def test_add_layers_with_build(self):
        nn = ml.NeuralNetwork([5, 16, 8, 1], "relu", "sigmoid", ml.OptimizerType.ADAM, 0.001)
        self.assertEqual(nn.num_layers, 3)

    def test_build(self):
        # Per evitare il warning n_classes, passiamo n_classes = 2 se l'ultimo layer ha 1 neurone (binary)
        nn = ml.NeuralNetwork([5, 16, 8, 1], "relu", "sigmoid", ml.OptimizerType.ADAM, 0.001)
        nn.build(5, 2)
        # build() inizializza la rete ma NON imposta fitted_=true (quello lo fa fit())
        # Verifichiamo invece che i layer siano allocati correttamente
        self.assertEqual(nn.num_layers, 3)

    def test_fit(self):
        nn = ml.NeuralNetwork([5, 16, 8, 1], "relu", "sigmoid", ml.OptimizerType.ADAM, 0.001)
        nn.set_loss_function("binary_crossentropy")
        nn.set_epochs(100)
        nn.set_batch_size(32)
        nn.set_verbose(True)
        nn.fit(self.X, self.y)
        self.assertTrue(self._check_is_fitted(nn))

    def test_predict(self):
        nn = ml.NeuralNetwork([5, 16, 8, 1], "relu", "sigmoid", ml.OptimizerType.ADAM, 0.001)
        nn.set_epochs(100)
        nn.set_verbose(False)
        nn.fit(self.X, self.y)
        y_pred = nn.predict(self.X)
        self.assertEqual(y_pred.shape, (200,))

    def test_predict_proba(self):
        nn = ml.NeuralNetwork([5, 16, 8, 1], "relu", "sigmoid", ml.OptimizerType.ADAM, 0.001)
        nn.set_epochs(100)
        nn.set_verbose(False)
        nn.fit(self.X, self.y)
        proba = nn.predict_proba(self.X)
        self.assertEqual(proba.shape, (200, 1))

    def test_score(self):
        nn = ml.NeuralNetwork([5, 16, 8, 1], "relu", "sigmoid", ml.OptimizerType.ADAM, 0.001)
        nn.set_loss_function("binary_crossentropy")
        nn.set_epochs(200)
        nn.set_batch_size(32)
        nn.set_verbose(True)
        nn.fit(self.X, self.y)
        score = nn.score(self.X, self.y)
        print(f"Score: {score}")
        self.assertGreater(score, 0.65)

    def test_training_history(self):
        nn = ml.NeuralNetwork([5, 16, 8, 1], "relu", "sigmoid", ml.OptimizerType.ADAM, 0.001)
        nn.set_loss_function("binary_crossentropy")
        nn.set_epochs(50)
        nn.set_batch_size(32)
        nn.set_verbose(False)
        nn.fit(self.X, self.y)
        loss, val_loss, acc = nn.get_training_history()
        self.assertGreater(len(loss), 0)
        if len(loss) > 1:
            self.assertGreater(loss[0], loss[-1])


class TestOptimizers(unittest.TestCase):
    def test_sgd(self):
        optimizer = ml.SGDOptimizer(learning_rate=0.1)
        w = np.asfortranarray(np.random.randn(5, 5) * 5)
        g = np.asfortranarray(np.ones((5, 5))) # Gradiente non nullo
        w_original = w.copy()

        for _ in range(10):
            optimizer.update_weights(w, g)

        diff = np.linalg.norm(w - w_original)
        self.assertGreater(diff, 1e-6)

    def test_momentum(self):
        optimizer = ml.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
        w = np.asfortranarray(np.random.randn(5, 5) * 5)
        g = np.asfortranarray(np.ones((5, 5)))
        w_original = w.copy()

        for _ in range(10):
            optimizer.update_weights(w, g)

        diff = np.linalg.norm(w - w_original)
        self.assertGreater(diff, 1e-6)

    def test_adam(self):
        optimizer = ml.AdamOptimizer(learning_rate=0.001)
        w = np.asfortranarray(np.random.randn(5, 5) * 5)
        g = np.asfortranarray(np.ones((5, 5)))
        w_original = w.copy()

        for _ in range(10):
            optimizer.update_weights(w, g)

        diff = np.linalg.norm(w - w_original)
        self.assertGreater(diff, 1e-6)

    def test_rmsprop(self):
        optimizer = ml.RMSpropOptimizer(learning_rate=0.001, rho=0.9)
        w = np.asfortranarray(np.random.randn(5, 5) * 5)
        g = np.asfortranarray(np.ones((5, 5)))
        w_original = w.copy()

        optimizer.update_weights(w, g)

        self.assertGreater(np.linalg.norm(w - w_original), 1e-6)
        self.assertEqual(optimizer.get_type_str(), "rmsprop")
        self.assertEqual(optimizer.clone().get_type_str(), "rmsprop")

    def test_adagrad(self):
        optimizer = ml.AdagradOptimizer(learning_rate=0.01)
        w = np.asfortranarray(np.random.randn(5, 5) * 5)
        g = np.asfortranarray(np.ones((5, 5)))
        w_original = w.copy()

        optimizer.update_weights(w, g)

        self.assertGreater(np.linalg.norm(w - w_original), 1e-6)
        self.assertEqual(optimizer.get_type_str(), "adagrad")
        self.assertEqual(optimizer.clone().get_type_str(), "adagrad")

    def test_optimizer_factory_includes_adaptive_optimizers(self):
        self.assertIn("rmsprop", ml.OptimizerFactory.list_optimizers())
        self.assertIn("adagrad", ml.OptimizerFactory.list_optimizers())
        rmsprop = ml.OptimizerFactory.create_by_name("rmsprop", 0.01)
        adagrad = ml.OptimizerFactory.create_by_type(ml.OptimizerType.ADAGRAD, 0.01)
        self.assertEqual(rmsprop.get_type_str(), "rmsprop")
        self.assertEqual(rmsprop.clone().get_type_str(), "rmsprop")
        self.assertEqual(adagrad.get_type_str(), "adagrad")
        self.assertEqual(adagrad.clone().get_type_str(), "adagrad")


class TestBindingFactories(unittest.TestCase):
    def test_loss_factory_creates_each_registered_loss(self):
        loss_names = ml.LossFactory.list_losses()
        self.assertEqual(
            set(loss_names),
            {
                "mse",
                "mae",
                "binary_crossentropy",
                "categorical_crossentropy",
                "huber",
            },
        )

        for name in loss_names:
            with self.subTest(name=name):
                loss = ml.LossFactory.create(name)
                self.assertTrue(loss.name())

    def test_regularizer_factory_maps_name_and_type(self):
        regularizer = ml.RegularizerFactory.create_by_type(
            ml.RegularizerType.L2, strength=0.5
        )
        self.assertEqual(regularizer.get_type_str(), "l2")
        self.assertEqual(regularizer.get_strength(), 0.5)
        self.assertEqual(
            ml.RegularizerFactory.string_to_type("l2"), ml.RegularizerType.L2
        )
        self.assertEqual(
            ml.RegularizerFactory.type_to_string(ml.RegularizerType.L2), "l2"
        )
        regularizer_by_name = ml.RegularizerFactory.create_by_name(
            "l1", strength=0.25
        )
        self.assertEqual(regularizer_by_name.get_type_str(), "l1")
        self.assertEqual(regularizer_by_name.get_strength(), 0.25)

    def test_layer_factory_creates_by_type(self):
        layer = ml.LayerFactory.create(ml.LayerType.DENSE)
        self.assertEqual(layer.get_type(), "DenseLayer")


class TestScalers(unittest.TestCase):
    def test_min_max_scaler_round_trip(self):
        X = np.array([[1.0, -2.0], [3.0, 2.0], [5.0, 6.0]])
        scaler = ml.MinMaxScaler(feature_range_min=-1.0, feature_range_max=1.0)

        X_scaled = np.asarray(scaler.fit_transform(X))
        X_restored = np.asarray(scaler.inverse_transform(X_scaled))

        np.testing.assert_allclose(X_scaled.min(axis=0), [-1.0, -1.0])
        np.testing.assert_allclose(X_scaled.max(axis=0), [1.0, 1.0])
        np.testing.assert_allclose(X_restored, X)


if __name__ == "__main__":
    unittest.main()