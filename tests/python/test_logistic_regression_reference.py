import tempfile
import unittest
from pathlib import Path

import numpy as np

import ml_core as ml


def _reference_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    regularization: float,
) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), features))
    sample_count = features.shape[0]
    penalty = np.diag(
        np.concatenate(([0.0], np.full(features.shape[1], regularization / sample_count)))
    )
    parameters = np.zeros(design.shape[1])

    for _ in range(100):
        probabilities = 1.0 / (1.0 + np.exp(-(design @ parameters)))
        gradient = design.T @ (probabilities - labels) / sample_count
        gradient += penalty @ parameters
        weights = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * weights[:, None]) / sample_count + penalty
        update = np.linalg.solve(hessian, gradient)
        parameters -= update
        if np.linalg.norm(gradient) < 1e-12:
            break
    else:
        raise AssertionError("reference Newton solver did not converge")

    return parameters


class TestLogisticRegressionReference(unittest.TestCase):
    def setUp(self) -> None:
        random = np.random.default_rng(20261029)
        self.features = random.normal(size=(240, 2))
        logits = -0.35 + 1.1 * self.features[:, 0] - 0.8 * self.features[:, 1]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        self.labels = random.binomial(1, probabilities).astype(np.float64)
        self.regularization = 0.3
        self.parameters = _reference_logistic_regression(
            self.features,
            self.labels,
            self.regularization,
        )

    def _fit_model(self) -> ml.LogisticRegression:
        model = ml.LogisticRegression(
            learning_rate=0.1,
            max_iter=50000,
            **{"lambda": self.regularization},
            tolerance=1e-12,
            verbose=False,
        )
        model.fit(self.features, self.labels)
        return model

    def test_fit_matches_newton_reference(self) -> None:
        model = self._fit_model()
        fitted_parameters = np.concatenate(([model.intercept], model.coefficients))
        reference_probabilities = 1.0 / (
            1.0 + np.exp(-(np.column_stack((np.ones(len(self.features)), self.features)) @ self.parameters))
        )
        expected_cost = np.mean(
            np.logaddexp(
                0.0,
                np.column_stack((np.ones(len(self.features)), self.features)) @ self.parameters,
            )
            - self.labels
            * (
                np.column_stack((np.ones(len(self.features)), self.features)) @ self.parameters
            )
        )
        expected_cost += self.regularization / (2.0 * len(self.labels)) * np.dot(
            self.parameters[1:],
            self.parameters[1:],
        )

        np.testing.assert_allclose(fitted_parameters, self.parameters, atol=1e-5, rtol=0.0)
        np.testing.assert_allclose(
            model.predict(self.features),
            reference_probabilities,
            atol=1e-5,
            rtol=0.0,
        )
        self.assertAlmostEqual(model.cost_history[-1], expected_cost, delta=1e-9)
        self.assertGreater(model.cost_history[0], model.cost_history[-1])

    def test_save_load_preserves_parameters_and_predictions(self) -> None:
        model = self._fit_model()
        predictions = model.predict(self.features)

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "logistic_regression.bin"
            model.save(str(model_path))
            self.assertTrue(model_path.is_file())

            restored = ml.LogisticRegression()
            restored.load(str(model_path))

        np.testing.assert_array_equal(restored.predict(self.features), predictions)
        np.testing.assert_array_equal(restored.coefficients, model.coefficients)
        self.assertEqual(restored.intercept, model.intercept)
        self.assertEqual(restored.n_iter, model.n_iter)
        np.testing.assert_array_equal(restored.cost_history, model.cost_history)
        np.testing.assert_array_equal(restored.accuracy_history, model.accuracy_history)


if __name__ == "__main__":
    unittest.main()
