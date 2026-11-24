import numpy as np
import pytest

import ml_core as ml


@pytest.mark.parametrize("layer_type", [ml.SimpleRNNLayer, ml.LSTMLayer, ml.GRULayer])
@pytest.mark.parametrize("return_sequences", [False, True])
def test_recurrent_sequence_gradients_and_reset(layer_type, return_sequences):
    layer = layer_type(2, 2)
    layer.set_input_shape(2)
    weights = layer.get_weights()
    weights[:] = np.arange(weights.size).reshape(weights.shape) * 0.02 + 0.03
    layer.set_weights(weights)
    biases = np.linspace(0.01, 0.04, layer.get_biases().size)
    layer.set_biases(biases)
    layer.set_return_sequences(return_sequences)
    assert f"return_sequences={'true' if return_sequences else 'false'}" in layer.get_config()
    x = np.arange(12, dtype=np.float64).reshape(2, 3, 2) / 20 - 0.2
    y = layer.forward(x, training=True)
    assert y.shape == ((2, 3, 2) if return_sequences else (2, 2))
    upstream = np.linspace(-0.3, 0.5, y.size).reshape(y.shape)
    dx = layer.backward(upstream)
    dw = layer.get_weights_gradient().copy()
    db = layer.get_bias_gradient().copy()
    assert dx.shape == x.shape

    def objective(values):
        return np.sum(layer.forward(values) * upstream)

    epsilon = 1e-6
    for index in np.ndindex(weights.shape):
        changed = weights.copy()
        changed[index] += epsilon
        layer.set_weights(changed)
        plus = objective(x)
        changed[index] -= 2 * epsilon
        layer.set_weights(changed)
        minus = objective(x)
        assert dw[index] == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-7)
    layer.set_weights(weights)
    for index in range(biases.size):
        changed = biases.copy()
        changed[index] += epsilon
        layer.set_biases(changed)
        plus = objective(x)
        changed[index] -= 2 * epsilon
        layer.set_biases(changed)
        minus = objective(x)
        assert db[index] == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-7)
    layer.set_biases(biases)
    for index in np.ndindex(x.shape):
        changed = x.copy()
        changed[index] += epsilon
        plus = objective(changed)
        changed[index] -= 2 * epsilon
        minus = objective(changed)
        assert dx[index] == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-7)
    layer.forward(np.ones((2, 2)))
    np.testing.assert_array_equal(layer.forward(x), y)


def test_stacked_network_sequence_training_and_save(tmp_path):
    network = ml.NeuralNetwork()
    network.add_recurrent_layer(ml.RecurrentType.SIMPLE_RNN, 3, return_sequences=True)
    network.add_recurrent_layer(ml.RecurrentType.GRU, 2)
    network.add_dense_layer(1, "linear")
    x = np.arange(24, dtype=np.float64).reshape(4, 3, 2)[:, :, ::-1] / 24
    y = np.array([0.1, 0.2, 0.3, 0.4])
    network.fit_advanced(x, y, epochs=2, batch_size=2)
    prediction = network.predict(x)
    assert prediction.shape == (4, 1)
    np.testing.assert_array_equal(network.predict(x), prediction)
    path = str(tmp_path / "sequence-model.bin")
    network.save(path)
    restored = ml.NeuralNetwork()
    restored.load(path)
    np.testing.assert_array_equal(restored.predict(x), prediction)


def test_network_sequence_output_and_target_shapes():
    network = ml.NeuralNetwork()
    network.add_recurrent_layer(ml.RecurrentType.LSTM, 2, return_sequences=True)
    network.add_recurrent_layer(ml.RecurrentType.GRU, 1, return_sequences=True)
    x = np.arange(12, dtype=np.float64).reshape(4, 3, 1) / 12
    y = x * 0.3
    network.fit_advanced(x, y, epochs=2, batch_size=3)
    assert network.predict(x).shape == y.shape
    with pytest.raises(ValueError, match="shape"):
        network.fit_advanced(x, y[:, -1, :], epochs=1, batch_size=4)


def test_recurrent_layer_still_accepts_stateful_2d_input():
    layer = ml.SimpleRNNLayer(1, 1)
    layer.set_input_shape(1)
    layer.set_weights(np.array([[0.5], [0.8]]))
    layer.set_biases(np.array([0.0]))
    x = np.ones((1, 1))
    first = layer.forward(x)
    second = layer.forward(x)
    assert not np.array_equal(first, second)
    layer.reset_state()
    np.testing.assert_array_equal(first, layer.forward(x))


def test_sequence_backward_requires_training_cache_and_matching_shape():
    layer = ml.GRULayer(2, 1)
    with pytest.raises(ValueError, match="weights"):
        layer.set_weights(np.zeros((1, 1)))
    x = np.ones((2, 3, 1))
    output = layer.forward(x)
    with pytest.raises(ValueError, match="training forward"):
        layer.backward(np.ones_like(output))
    layer.forward(x, training=True)
    with pytest.raises(ValueError, match="gradient"):
        layer.backward(np.ones((2, 3, 2)))
    with pytest.raises(ValueError, match="dimensions"):
        layer.forward(np.empty((2, 0, 1)))
