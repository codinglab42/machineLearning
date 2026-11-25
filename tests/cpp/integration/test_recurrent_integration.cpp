#include <gtest/gtest.h>
#include <cstdint>
#include "components/layers/simple_rnn_layer.h"
#include "components/layers/lstm_layer.h"
#include "components/layers/gru_layer.h"
#include "models/neural_network.h"
#include <Eigen/Dense>
#include <memory>
#include <sstream>
#include <vector>

namespace {

std::vector<std::unique_ptr<layers::RecurrentLayer>> make_recurrent_layers() {
    std::vector<std::unique_ptr<layers::RecurrentLayer>> result;
    result.push_back(std::make_unique<layers::SimpleRNNLayer>(2, 2));
    result.push_back(std::make_unique<layers::LSTMLayer>(2, 2));
    result.push_back(std::make_unique<layers::GRULayer>(2, 2));
    return result;
}

void set_test_parameters(layers::RecurrentLayer& layer) {
    layer.set_input_shape(2);
    Eigen::MatrixXd weights = layer.get_weights();
    for (Eigen::Index row = 0; row < weights.rows(); ++row) {
        for (Eigen::Index col = 0; col < weights.cols(); ++col) {
            weights(row, col) = 0.04 * (row + 1) - 0.07 * col;
        }
    }
    layer.set_weights(weights);
    if (layer.get_use_bias()) {
        Eigen::VectorXd biases = layer.get_biases();
        for (Eigen::Index i = 0; i < biases.size(); ++i) {
            biases(i) = 0.03 * (i + 1);
        }
        layer.set_biases(biases);
    }
}

} // namespace

TEST(RecurrentIntegrationTest, SingleStepGradientsWithPreviousState) {
    Eigen::MatrixXd previous_input(2, 2);
    previous_input << 0.3, -0.2, 0.1, 0.4;
    Eigen::MatrixXd input(2, 2);
    input << -0.4, 0.6, 0.2, -0.1;
    Eigen::MatrixXd upstream(2, 2);
    upstream << 0.7, -0.3, -0.5, 0.8;
    constexpr double epsilon = 1e-6;

    auto recurrent_layers = make_recurrent_layers();
    recurrent_layers.push_back(std::make_unique<layers::SimpleRNNLayer>(2, 2, "sigmoid"));
    recurrent_layers.push_back(std::make_unique<layers::LSTMLayer>(2, 2, "relu", "sigmoid"));
    for (auto& layer : recurrent_layers) {
        SCOPED_TRACE(layer->get_config());
        set_test_parameters(*layer);
        const Eigen::MatrixXd weights = layer->get_weights();
        const Eigen::VectorXd biases = layer->get_biases();
        layer->reset_state();
        layer->forward(previous_input, false);
        ASSERT_GT(layer->get_hidden_state().norm(), 0.0);
        layer->forward(input, true);
        const Eigen::MatrixXd input_gradient = layer->backward(upstream);
        const Eigen::MatrixXd weight_gradient = layer->get_weights_gradient();
        const Eigen::VectorXd bias_gradient = layer->get_bias_gradient();

        auto evaluate = [&](const Eigen::MatrixXd& current_weights,
                            const Eigen::VectorXd& current_biases,
                            const Eigen::MatrixXd& current_input) {
            // Hold the previous state fixed while perturbing the current step.
            layer->reset_state();
            layer->set_weights(weights);
            layer->set_biases(biases);
            layer->forward(previous_input, false);
            layer->set_weights(current_weights);
            layer->set_biases(current_biases);
            return (layer->forward(current_input, true).array() * upstream.array()).sum();
        };

        for (Eigen::Index row = 0; row < weights.rows(); ++row) {
            for (Eigen::Index col = 0; col < weights.cols(); ++col) {
                Eigen::MatrixXd changed = weights;
                changed(row, col) += epsilon;
                const double plus = evaluate(changed, biases, input);
                changed(row, col) -= 2 * epsilon;
                const double minus = evaluate(changed, biases, input);
                EXPECT_NEAR(weight_gradient(row, col), (plus - minus) / (2 * epsilon), 1e-7)
                    << "weight (" << row << ", " << col << ")";
            }
        }
        for (Eigen::Index i = 0; i < biases.size(); ++i) {
            Eigen::VectorXd changed = biases;
            changed(i) += epsilon;
            const double plus = evaluate(weights, changed, input);
            changed(i) -= 2 * epsilon;
            const double minus = evaluate(weights, changed, input);
            EXPECT_NEAR(bias_gradient(i), (plus - minus) / (2 * epsilon), 1e-7)
                << "bias " << i;
        }
        for (Eigen::Index row = 0; row < input.rows(); ++row) {
            for (Eigen::Index col = 0; col < input.cols(); ++col) {
                Eigen::MatrixXd changed = input;
                changed(row, col) += epsilon;
                const double plus = evaluate(weights, biases, changed);
                changed(row, col) -= 2 * epsilon;
                const double minus = evaluate(weights, biases, changed);
                EXPECT_NEAR(input_gradient(row, col), (plus - minus) / (2 * epsilon), 1e-7)
                    << "input (" << row << ", " << col << ")";
            }
        }
    }
}

TEST(RecurrentIntegrationTest, ResetAndBatchSizeChangeClearPreviousState) {
    for (auto& layer : make_recurrent_layers()) {
        SCOPED_TRACE(layer->get_type());
        set_test_parameters(*layer);
        const Eigen::MatrixXd input = Eigen::MatrixXd::Constant(2, 2, 0.4);
        layer->forward(input, false);
        ASSERT_GT(layer->get_hidden_state().norm(), 0.0);

        layer->reset_state();
        EXPECT_FALSE(layer->get_cache()->is_valid());
        layer->forward(input, true);
        layer->backward(Eigen::MatrixXd::Ones(2, 2));
        Eigen::MatrixXd gradient = layer->get_weights_gradient();
        EXPECT_DOUBLE_EQ(gradient.bottomRows(gradient.rows() / 2).norm(), 0.0);

        layer->forward(Eigen::MatrixXd::Constant(3, 2, 0.4), true);
        layer->backward(Eigen::MatrixXd::Ones(3, 2));
        gradient = layer->get_weights_gradient();
        EXPECT_DOUBLE_EQ(gradient.bottomRows(gradient.rows() / 2).norm(), 0.0);
    }
}

TEST(RecurrentIntegrationTest, ClippedActivationsHaveZeroGradientOutsideRange) {
    for (auto& layer : make_recurrent_layers()) {
        SCOPED_TRACE(layer->get_type());
        set_test_parameters(*layer);
        const Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(
            layer->get_weights().rows(), layer->get_weights().cols());
        Eigen::VectorXd biases = Eigen::VectorXd::Constant(layer->get_biases().size(), 15.0);
        layer->set_weights(weights);
        layer->set_biases(biases);
        const Eigen::MatrixXd input = Eigen::MatrixXd::Ones(1, 2);
        layer->forward(input, true);
        layer->backward(Eigen::MatrixXd::Ones(1, 2));
        const Eigen::VectorXd gradient = layer->get_bias_gradient();
        for (Eigen::Index i = 0; i < biases.size(); ++i) {
            layer->reset_state();
            biases(i) += 1e-5;
            layer->set_biases(biases);
            const double plus = layer->forward(input, false).sum();
            layer->reset_state();
            biases(i) -= 2e-5;
            layer->set_biases(biases);
            const double minus = layer->forward(input, false).sum();
            biases(i) += 1e-5;
            EXPECT_NEAR(gradient(i), (plus - minus) / 2e-5, 1e-10) << "bias " << i;
        }
    }
}

TEST(RecurrentIntegrationTest, SequenceBpttMatchesNumericalGradients) {
    constexpr double epsilon = 1e-6;
    auto recurrent_layers = make_recurrent_layers();
    recurrent_layers.push_back(std::make_unique<layers::SimpleRNNLayer>(2, 2, "sigmoid"));
    recurrent_layers.push_back(std::make_unique<layers::LSTMLayer>(2, 2, "relu", "sigmoid"));
    recurrent_layers.push_back(std::make_unique<layers::SimpleRNNLayer>(2, 2, "tanh", false));
    recurrent_layers.push_back(std::make_unique<layers::LSTMLayer>(2, 2, "tanh", "sigmoid", false));
    recurrent_layers.push_back(std::make_unique<layers::GRULayer>(2, 2, "tanh", "sigmoid", false));
    for (auto& layer : recurrent_layers) {
        SCOPED_TRACE(layer->get_config());
        set_test_parameters(*layer);
        const Eigen::MatrixXd weights = layer->get_weights();
        const Eigen::VectorXd biases = layer->get_use_bias() ? layer->get_biases() : Eigen::VectorXd();
        layers::SequenceTensor input{{0.3, -0.2, 0.6, 0.1, -0.4, 0.5,
                                      -0.1, 0.4, 0.2, -0.3, 0.7, 0.1},
                                     2, 3, 2, true};
        for (bool return_sequences : {false, true}) {
            SCOPED_TRACE(return_sequences);
            layer->set_return_sequences(return_sequences);
            const auto output = layer->forward_sequence(input, true);
            EXPECT_EQ(output.batch_size, 2);
            EXPECT_EQ(output.time_steps, return_sequences ? 3 : 1);
            EXPECT_EQ(output.features, 2);
            EXPECT_EQ(output.is_sequence, return_sequences);
            layers::SequenceTensor upstream = output;
            for (std::size_t i = 0; i < upstream.data.size(); ++i) {
                upstream.data[i] = 0.1 * (static_cast<double>(i) - 2.0);
            }
            const auto input_gradient = layer->backward_sequence(upstream);
            const Eigen::MatrixXd weight_gradient = layer->get_weights_gradient();
            const Eigen::VectorXd bias_gradient = layer->get_bias_gradient();
            auto evaluate = [&](const layers::SequenceTensor& x) {
                const auto y = layer->forward_sequence(x, false);
                double loss = 0.0;
                for (std::size_t i = 0; i < y.data.size(); ++i) {
                    loss += y.data[i] * upstream.data[i];
                }
                return loss;
            };
            for (Eigen::Index row = 0; row < weights.rows(); ++row) {
                for (Eigen::Index col = 0; col < weights.cols(); ++col) {
                    Eigen::MatrixXd changed = weights;
                    changed(row, col) += epsilon;
                    layer->set_weights(changed);
                    const double plus = evaluate(input);
                    changed(row, col) -= 2 * epsilon;
                    layer->set_weights(changed);
                    const double minus = evaluate(input);
                    EXPECT_NEAR(weight_gradient(row, col), (plus - minus) / (2 * epsilon), 1e-7)
                        << "weight (" << row << ", " << col << ")";
                }
            }
            layer->set_weights(weights);
            for (Eigen::Index i = 0; i < biases.size(); ++i) {
                Eigen::VectorXd changed = biases;
                changed(i) += epsilon;
                layer->set_biases(changed);
                const double plus = evaluate(input);
                changed(i) -= 2 * epsilon;
                layer->set_biases(changed);
                const double minus = evaluate(input);
                EXPECT_NEAR(bias_gradient(i), (plus - minus) / (2 * epsilon), 1e-7)
                    << "bias " << i;
            }
            if (layer->get_use_bias()) layer->set_biases(biases);
            for (std::size_t i = 0; i < input.data.size(); ++i) {
                auto changed = input;
                changed.data[i] += epsilon;
                const double plus = evaluate(changed);
                changed.data[i] -= 2 * epsilon;
                const double minus = evaluate(changed);
                EXPECT_NEAR(input_gradient.data[i], (plus - minus) / (2 * epsilon), 1e-7)
                    << "input " << i;
            }
            layer->forward(Eigen::MatrixXd::Constant(2, 2, 3.0), false);
            const auto repeated = layer->forward_sequence(input, false);
            ASSERT_EQ(repeated.data.size(), output.data.size());
            for (std::size_t i = 0; i < output.data.size(); ++i) {
                EXPECT_DOUBLE_EQ(repeated.data[i], output.data[i]);
            }
        }
    }
}

TEST(RecurrentIntegrationTest, NetworkTrainsStackedSequencesAndRestoresConfiguration) {
    models::NeuralNetwork network;
    network.add_recurrent_layer(models::RecurrentType::SIMPLE_RNN, 2, true);
    network.add_recurrent_layer(models::RecurrentType::GRU, 2, false);
    network.add_dense_layer(1, "linear");
    layers::SequenceTensor input{{0.1, 0.3, 0.2, 0.4, -0.5, 0.2,
                                  0.2, 0.5, -0.1, 0.3, 0.4, -0.2,
                                  0.3, 0.1, -0.2, 0.4, 0.1, 0.5},
                                 3, 3, 2, true};
    layers::SequenceTensor target{{0.3, -0.2, 0.6}, 3, 1, 1, false};
    network.fit_sequences(input, target, 1, 3, false);
    const Eigen::MatrixXd first_weights = network.get_layers()[0]->get_weights();
    const Eigen::MatrixXd second_weights = network.get_layers()[1]->get_weights();
    network.fit_sequences(input, target, 1, 3, false);
    EXPECT_GT((network.get_layers()[0]->get_weights() - first_weights).norm(), 1e-12);
    EXPECT_GT((network.get_layers()[1]->get_weights() - second_weights).norm(), 1e-12);
    const auto prediction = network.predict_sequences(input);
    EXPECT_FALSE(prediction.is_sequence);
    EXPECT_EQ(prediction.batch_size, 3);
    EXPECT_EQ(prediction.features, 1);
    EXPECT_EQ(prediction.data, network.predict_sequences(input).data);

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    network.serialize_binary(stream);
    models::NeuralNetwork restored;
    restored.deserialize_binary(stream);
    const auto* first = dynamic_cast<const layers::RecurrentLayer*>(restored.get_layers()[0].get());
    const auto* second = dynamic_cast<const layers::RecurrentLayer*>(restored.get_layers()[1].get());
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_TRUE(first->get_return_sequences());
    EXPECT_FALSE(second->get_return_sequences());
    EXPECT_EQ(prediction.data, restored.predict_sequences(input).data);
    network.fit_sequences(input, target, 1, 3, false);
    restored.fit_sequences(input, target, 1, 3, false);
    for (std::size_t i = 0; i < network.get_layers().size(); ++i) {
        EXPECT_TRUE(network.get_layers()[i]->get_weights().isApprox(
            restored.get_layers()[i]->get_weights(), 1e-10));
    }
}

TEST(RecurrentIntegrationTest, NetworkTrainsSequenceOutputsAndRejectsMismatchedTargets) {
    models::NeuralNetwork network;
    network.add_recurrent_layer(models::RecurrentType::LSTM, 2, true);
    network.add_recurrent_layer(models::RecurrentType::GRU, 1, true);
    layers::SequenceTensor input{{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3, 1, true};
    layers::SequenceTensor target{{0.2, 0.3, 0.4, -0.1, -0.2, -0.3}, 2, 3, 1, true};
    network.fit_sequences(input, target, 2, 2, false);
    const auto prediction = network.predict_sequences(input);
    EXPECT_TRUE(prediction.is_sequence);
    EXPECT_EQ(prediction.batch_size, 2);
    EXPECT_EQ(prediction.time_steps, 3);
    EXPECT_EQ(prediction.features, 1);
    auto invalid_target = target;
    invalid_target.is_sequence = false;
    invalid_target.time_steps = 1;
    invalid_target.data.resize(2);
    EXPECT_THROW(network.fit_sequences(input, invalid_target, 1, 2, false),
                 ml_exception::InvalidParameterException);

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    network.serialize_binary(stream);
    models::NeuralNetwork restored;
    restored.deserialize_binary(stream);
    EXPECT_EQ(prediction.data, restored.predict_sequences(input).data);
}

TEST(RecurrentIntegrationTest, NetworkLearnsFromEarlierSequenceSteps) {
    models::NeuralNetwork network;
    network.add_recurrent_layer(models::RecurrentType::SIMPLE_RNN, 1);
    network.set_optimizer(models::OptimizerType::SGD, 0.2);
    layers::SequenceTensor input{{1.0, 0.0, -1.0, 0.0}, 2, 2, 1, true};
    layers::SequenceTensor target{{0.5, -0.5}, 2, 1, 1, false};
    network.fit_sequences(input, target, 1, 2, false);
    Eigen::MatrixXd weights(2, 1);
    weights << 0.2, 0.2;
    network.get_layers()[0]->set_weights(weights);
    network.get_layers()[0]->set_biases(Eigen::VectorXd::Zero(1));
    const auto initial = network.predict_sequences(input);
    network.fit_sequences(input, target, 250, 2, false);
    const auto trained = network.predict_sequences(input);
    const double initial_error = std::abs(initial.data[0] - target.data[0]);
    EXPECT_LT(std::abs(trained.data[0] - target.data[0]), initial_error * 0.1);
    EXPECT_LT(std::abs(trained.data[1] - target.data[1]), initial_error * 0.1);
}

TEST(RecurrentIntegrationTest, SequenceCacheIsInvalidatedByResetAndMatrixForward) {
    layers::SequenceTensor input{{0.1, 0.2, 0.3}, 1, 3, 1, true};
    layers::SimpleRNNLayer layer(1, 1);
    layer.set_input_shape(1);
    auto output = layer.forward_sequence(input, true);
    layer.reset_state();
    EXPECT_THROW(layer.backward_sequence(output), std::invalid_argument);
    output = layer.forward_sequence(input, true);
    layer.forward(Eigen::MatrixXd::Ones(1, 1), true);
    EXPECT_THROW(layer.backward_sequence(output), std::invalid_argument);
}

TEST(RecurrentIntegrationTest, SequenceLayoutMatchesIndependentSingleStepCalls) {
    layers::SequenceTensor input{{0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                  0.7, 0.8, 0.9, 1.0, 1.1, 1.2},
                                 2, 3, 2, true};
    for (auto& layer : make_recurrent_layers()) {
        SCOPED_TRACE(layer->get_type());
        set_test_parameters(*layer);
        layer->set_return_sequences(true);
        const auto all = layer->forward_sequence(input, false);
        layer->reset_state();
        Eigen::MatrixXd step(2, 2);
        for (int t = 0; t < 3; ++t) {
            for (int b = 0; b < 2; ++b) {
                for (int f = 0; f < 2; ++f) {
                    step(b, f) = input.data[(b * 3 + t) * 2 + f];
                }
            }
            const Eigen::MatrixXd expected = layer->forward(step, false);
            for (int b = 0; b < 2; ++b) {
                for (int f = 0; f < 2; ++f) {
                    EXPECT_DOUBLE_EQ(all.data[(b * 3 + t) * 2 + f], expected(b, f));
                }
            }
        }
        layer->set_return_sequences(false);
        const auto last = layer->forward_sequence(input, false);
        for (int b = 0; b < 2; ++b) {
            for (int f = 0; f < 2; ++f) {
                EXPECT_DOUBLE_EQ(last.data[b * 2 + f], all.data[(b * 3 + 2) * 2 + f]);
            }
        }
    }
}

TEST(RecurrentIntegrationTest, StackedSequenceGradientsReachFirstLayer) {
    layers::SimpleRNNLayer first(2, 2);
    layers::LSTMLayer second(2, 2);
    set_test_parameters(first);
    set_test_parameters(second);
    first.set_return_sequences(true);
    layers::SequenceTensor input{{0.2, 0.1, -0.3, 0.4, 0.5, 0.2}, 1, 3, 2, true};
    const auto middle = first.forward_sequence(input, true);
    auto gradient = second.forward_sequence(middle, true);
    gradient.data = {0.4, -0.7};
    first.backward_sequence(second.backward_sequence(gradient));
    const Eigen::MatrixXd analytical = first.get_weights_gradient();
    const Eigen::MatrixXd weights = first.get_weights();
    auto evaluate = [&]() {
        const auto output = second.forward_sequence(first.forward_sequence(input, false), false);
        return output.data[0] * gradient.data[0] + output.data[1] * gradient.data[1];
    };
    for (Eigen::Index row = 0; row < weights.rows(); ++row) {
        for (Eigen::Index col = 0; col < weights.cols(); ++col) {
            Eigen::MatrixXd changed = weights;
            changed(row, col) += 1e-6;
            first.set_weights(changed);
            const double plus = evaluate();
            changed(row, col) -= 2e-6;
            first.set_weights(changed);
            const double minus = evaluate();
            EXPECT_NEAR(analytical(row, col), (plus - minus) / 2e-6, 1e-7);
        }
    }
}

TEST(RecurrentIntegrationTest, RejectsInvalidSequenceShapesAndArchitecture) {
    layers::SimpleRNNLayer layer(2, 2);
    layer.set_input_shape(2);
    layers::SequenceTensor wrong_size{{0.1}, 1, 2, 2, true};
    EXPECT_THROW(layer.forward_sequence(wrong_size), std::invalid_argument);
    layers::SequenceTensor empty{{}, 1, 0, 2, true};
    EXPECT_THROW(layer.forward_sequence(empty), std::invalid_argument);

    models::NeuralNetwork network;
    network.add_recurrent_layer(models::RecurrentType::GRU, 2, true);
    network.add_dense_layer(1, "linear");
    layers::SequenceTensor input{{0.1, 0.2, 0.3}, 1, 3, 1, true};
    layers::SequenceTensor target{{0.1}, 1, 1, 1, false};
    EXPECT_THROW(network.fit_sequences(input, target, 1, 1, false),
                 ml_exception::InvalidParameterException);
}

TEST(RecurrentIntegrationTest, LegacyGruLayerWithoutSequenceFlagCanBeLoaded) {
    layers::GRULayer layer(2, 2);
    layer.set_input_shape(2);
    layer.initialize_weights();
    std::ostringstream payload(std::ios::binary);
    layer.serialize(payload);
    std::string legacy = payload.str();
    const auto hidden_state = layer.get_hidden_state();
    legacy.resize(legacy.size() - 2 * sizeof(std::int32_t) -
                  static_cast<std::size_t>(hidden_state.size()) * sizeof(double) -
                  sizeof(std::int8_t));
    std::istringstream raw_legacy(legacy, std::ios::binary);
    layers::GRULayer raw_restored;
    raw_restored.deserialize_versioned(raw_legacy, 1);
    EXPECT_TRUE(raw_restored.get_weights().isApprox(layer.get_weights()));
    EXPECT_FALSE(raw_restored.get_return_sequences());
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    layers::LayerFactory::serialize_type(stream, layers::LayerType::GRU);
    utils::write_scalar(stream, std::uint32_t{1});
    stream.write(legacy.data(), static_cast<std::streamsize>(legacy.size()));
    utils::write_scalar(stream, std::uint32_t{0x12345678});
    auto restored = layers::deserialize_layer(stream);
    auto* recurrent = dynamic_cast<layers::GRULayer*>(restored.get());
    ASSERT_NE(recurrent, nullptr);
    EXPECT_FALSE(recurrent->get_return_sequences());
    EXPECT_TRUE(recurrent->get_weights().isApprox(layer.get_weights()));
    std::uint32_t sentinel = 0;
    utils::read_scalar(stream, sentinel);
    EXPECT_EQ(sentinel, 0x12345678u);
}

// ============================================================================
// TEST INTEGRAZIONE: SIMPLE RNN LAYER
// ============================================================================
TEST(RecurrentIntegrationTest, SimpleRNNForwardBackwardPipeline) {
    int units = 3;
    int input_size = 2;
    int batch_size = 1;
    
    auto rnn = std::make_unique<layers::SimpleRNNLayer>(units, input_size, "tanh", true);
    rnn->set_input_shape(input_size);
    
    // Generiamo un input controllato per 1 step temporale
    Eigen::MatrixXd input = Eigen::MatrixXd::Constant(batch_size, input_size, 0.5);
    
    // 1. Forward Pass (Training = true per salvare la cache)
    Eigen::MatrixXd output = rnn->forward(input, true);
    
    ASSERT_EQ(output.rows(), batch_size);
    ASSERT_EQ(output.cols(), units);
    
    // 2. Backward Pass
    Eigen::MatrixXd top_gradient = Eigen::MatrixXd::Constant(batch_size, units, 1.0);
    Eigen::MatrixXd bottom_gradient = rnn->backward(top_gradient);
    
    // Verifichiamo che il gradiente propagato all'input sia corretto geometricamente
    ASSERT_EQ(bottom_gradient.rows(), batch_size);
    ASSERT_EQ(bottom_gradient.cols(), input_size);
    
    // Verifichiamo l'estrazione e l'integrità dei gradienti dei pesi e dei bias isolati
    Eigen::MatrixXd dWeights = rnn->get_weights_gradient();
    Eigen::VectorXd dBias = rnn->get_bias_gradient();
    
    // I pesi estratti devono avere righe = (input_size + units) e colonne = units
    ASSERT_EQ(dWeights.rows(), input_size + units);
    ASSERT_EQ(dWeights.cols(), units);
    ASSERT_EQ(dBias.size(), units);
    
    // Verifichiamo che i gradienti non siano rimasti a zero (segno di attivazione del calcolo)
    EXPECT_GT(dWeights.norm(), 0.0);
    EXPECT_GT(dBias.norm(), 0.0);
}

// ============================================================================
// TEST INTEGRAZIONE: LSTM LAYER
// ============================================================================
TEST(RecurrentIntegrationTest, LSTMForwardBackwardPipeline) {
    int units = 4;
    int input_size = 3;
    int batch_size = 2;
    
    auto lstm = std::make_unique<layers::LSTMLayer>(units, input_size, "tanh", "sigmoid", true);
    lstm->set_input_shape(input_size);
    
    Eigen::MatrixXd input = Eigen::MatrixXd::Constant(batch_size, input_size, 0.1);
    
    // 1. Forward Pass
    Eigen::MatrixXd output = lstm->forward(input, true);
    
    ASSERT_EQ(output.rows(), batch_size);
    ASSERT_EQ(output.cols(), units);
    
    // 2. Backward Pass
    Eigen::MatrixXd top_gradient = Eigen::MatrixXd::Constant(batch_size, units, 0.5);
    Eigen::MatrixXd bottom_gradient = lstm->backward(top_gradient);
    
    ASSERT_EQ(bottom_gradient.rows(), batch_size);
    ASSERT_EQ(bottom_gradient.cols(), input_size);
    
    // Estrazione gradienti dopo la rifattorizzazione geometrica rigida delle 4 porte
    Eigen::MatrixXd dWeights = lstm->get_weights_gradient();
    Eigen::VectorXd dBias = lstm->get_bias_gradient();
    
    // LSTM concatena le 4 porte verticalmente lungo le righe
    ASSERT_EQ(dWeights.rows(), (input_size + units) * 4); // 4 porte sulle righe
    ASSERT_EQ(dWeights.cols(), units);                   // units sulle colonne
    ASSERT_EQ(dBias.size(), units * 4);
    
    EXPECT_GT(dWeights.norm(), 0.0);
    EXPECT_GT(dBias.norm(), 0.0);
}

// ============================================================================
// TEST INTEGRAZIONE: GRU LAYER
// ============================================================================
TEST(RecurrentIntegrationTest, GRUForwardBackwardPipeline) {
    int units = 3;
    int input_size = 2;
    int batch_size = 1;
    
    auto gru = std::make_unique<layers::GRULayer>(units, input_size, "tanh", "sigmoid", true);
    gru->set_input_shape(input_size);
    
    Eigen::MatrixXd input = Eigen::MatrixXd::Constant(batch_size, input_size, -0.2);
    
    // 1. Forward Pass
    Eigen::MatrixXd output = gru->forward(input, true);
    
    ASSERT_EQ(output.rows(), batch_size);
    ASSERT_EQ(output.cols(), units);
    
    // 2. Backward Pass
    Eigen::MatrixXd top_gradient = Eigen::MatrixXd::Constant(batch_size, units, 1.0);
    Eigen::MatrixXd bottom_gradient = gru->backward(top_gradient);
    
    ASSERT_EQ(bottom_gradient.rows(), batch_size);
    ASSERT_EQ(bottom_gradient.cols(), input_size);
    
    // Estrazione gradienti per le 3 porte di GRU (Update, Reset, Candidate)
    Eigen::MatrixXd dWeights = gru->get_weights_gradient();
    Eigen::VectorXd dBias = gru->get_bias_gradient();
    
    // GRU concatena le 3 porte verticalmente lungo le righe
    ASSERT_EQ(dWeights.rows(), (input_size + units) * 3); // 3 porte sulle righe
    ASSERT_EQ(dWeights.cols(), units);                   // units sulle colonne
    ASSERT_EQ(dBias.size(), units * 3);

    EXPECT_GT(dWeights.norm(), 0.0);
    EXPECT_GT(dBias.norm(), 0.0);
}