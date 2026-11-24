/**
 * @file test_neural_network_regularizer_integration.cpp
 * @brief Integration tests for Neural Network with Regularizers
 * 
 * Tests the integration between Neural Network and different regularizers:
 * - L1 regularization (Lasso)
 * - L2 regularization (Ridge)
 * - Elastic Net regularization
 * - Different regularization strengths
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <iostream>
#include <cmath>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#include "models/neural_network.h"
#include "components/regularizers/regularizer_factory.h"
#include "components/loss/loss_factory.h"

using namespace models;
using namespace Eigen;
using namespace testing;

/**
 * @brief Test fixture for Neural Network with Regularizer integration tests
 * 
 * Creates a binary classification dataset with 100 samples and 2 features.
 * The decision boundary is linear: x0 + x1 > 0
 */
class NeuralNetworkRegularizerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple binary classification dataset
        X_.resize(100, 2);
        X_.setRandom();
        
        // Decision boundary: x0 + x1 > 0
        y_ = (X_.col(0).array() + X_.col(1).array() > 0).cast<double>();
        
        // Ensure balanced classes
        int positive_count = y_.sum();
        int negative_count = y_.size() - positive_count;
        
        std::cout << "Dataset: " << y_.size() << " samples, "
                  << positive_count << " positive, "
                  << negative_count << " negative" << std::endl;
    }
    
    MatrixXd X_;
    VectorXd y_;

    static double weight_norm(const NeuralNetwork& nn) {
        double squared_norm = 0.0;
        for (const auto& layer : nn.get_layers()) {
            if (layer->has_weights()) {
                squared_norm += layer->get_weights().squaredNorm();
            }
        }
        return std::sqrt(squared_norm);
    }
};

// ============================================================================
// L2 Regularization (Ridge) Tests
// ============================================================================

/**
 * @brief Test Neural Network training with L2 regularization
 * 
 * Verifies that:
 * - The network can be trained with L2 regularization
 * - No exceptions are thrown during training
 * - Accuracy exceeds 85%
 */
TEST_F(NeuralNetworkRegularizerIntegrationTest, TrainWithL2Regularization) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.001);  // LR ridotto a 0.001
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L2, 0.0001);  // Regularization più piccola
    nn.set_epochs(500);  // Aumentato a 500
    nn.set_batch_size(32);
    nn.set_verbose(true);  // Metti verbose per vedere progresso
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "L2 Regularization accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}
/**
 * @brief Test L2 regularization with different strengths
 * 
 * Verifies that the network works with various regularization strengths
 */
TEST_F(NeuralNetworkRegularizerIntegrationTest, L2RegularizationWithDifferentStrengths) {
    std::vector<double> strengths = {0.0001, 0.001, 0.01, 0.1};
    
    for (double strength : strengths) {
        NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
        nn.set_loss_function("binary_crossentropy");
        nn.set_regularizer(RegularizerType::L2, strength);
        nn.set_epochs(80);
        nn.set_batch_size(32);
        nn.set_verbose(false);
        
        EXPECT_NO_THROW(nn.fit(X_, y_)) 
            << "Failed with L2 strength = " << strength;
        
        double score = nn.score(X_, y_);
        std::cout << "L2 strength=" << strength << " accuracy: " << score << std::endl;
        EXPECT_GT(score, 0.80) 
            << "Low accuracy with L2 strength = " << strength;
    }
}

// ============================================================================
// L1 Regularization (Lasso) Tests
// ============================================================================

/**
 * @brief Test Neural Network training with L1 regularization
 * 
 * Verifies that L1 regularization works correctly with neural networks
 */
TEST_F(NeuralNetworkRegularizerIntegrationTest, TrainWithL1Regularization) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.001);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L1, 0.0001);
    nn.set_epochs(500);
    nn.set_batch_size(32);
    nn.set_verbose(true);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "L1 Regularization accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}

/**
 * @brief Test L1 regularization with different strengths
 */
TEST_F(NeuralNetworkRegularizerIntegrationTest, L1RegularizationWithDifferentStrengths) {
    std::vector<double> strengths = {0.0001, 0.001, 0.01, 0.1};
    NeuralNetwork baseline({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    baseline.set_loss_function("binary_crossentropy");
    baseline.fit(X_, y_, 80, 32, false);
    
    for (double strength : strengths) {
        NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
        nn.set_loss_function("binary_crossentropy");
        nn.set_regularizer(RegularizerType::L1, strength);
        nn.set_epochs(80);
        nn.set_batch_size(32);
        nn.set_verbose(false);
        
        EXPECT_NO_THROW(nn.fit(X_, y_))
            << "Failed with L1 strength = " << strength;
        
        double score = nn.score(X_, y_);
        std::cout << "L1 strength=" << strength << " accuracy: " << score << std::endl;
        if (strength < 0.1) {
            EXPECT_GT(score, 0.80) << "Low accuracy with L1 strength = " << strength;
        } else {
            EXPECT_LT(weight_norm(nn), weight_norm(baseline));
        }
    }
}

// ============================================================================
// Elastic Net Regularization Tests
// ============================================================================

/**
 * @brief Test Neural Network training with Elastic Net regularization
 * 
 * Elastic Net combines L1 and L2 regularization
 */
TEST_F(NeuralNetworkRegularizerIntegrationTest, TrainWithElasticNetRegularization) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::ELASTIC_NET, 0.001);
    nn.set_epochs(100);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "Elastic Net Regularization accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}

/**
 * @brief Test Elastic Net with different strengths
 */
TEST_F(NeuralNetworkRegularizerIntegrationTest, ElasticNetWithDifferentStrengths) {
    std::vector<double> strengths = {0.0001, 0.001, 0.01, 0.1};
    
    for (double strength : strengths) {
        NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
        nn.set_loss_function("binary_crossentropy");
        nn.set_regularizer(RegularizerType::ELASTIC_NET, strength);
        nn.set_epochs(80);
        nn.set_batch_size(32);
        nn.set_verbose(false);
        
        EXPECT_NO_THROW(nn.fit(X_, y_))
            << "Failed with Elastic Net strength = " << strength;
        
        double score = nn.score(X_, y_);
        std::cout << "Elastic Net strength=" << strength << " accuracy: " << score << std::endl;
        EXPECT_GT(score, 0.80)
            << "Low accuracy with Elastic Net strength = " << strength;
    }
}

// ============================================================================
// Comparison Tests
// ============================================================================

/**
 * @brief Compare training with and without regularization
 * 
 * Verifies that:
 * - Model without regularization works
 * - Model with regularization works
 * - Both achieve reasonable accuracy
 */
TEST_F(NeuralNetworkRegularizerIntegrationTest, CompareWithAndWithoutRegularization) {
    // Without regularization
    NeuralNetwork nn_no_reg({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn_no_reg.set_loss_function("binary_crossentropy");
    nn_no_reg.set_epochs(100);
    nn_no_reg.set_batch_size(32);
    nn_no_reg.set_verbose(false);
    nn_no_reg.fit(X_, y_);
    
    // With L2 regularization
    NeuralNetwork nn_with_reg({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn_with_reg.set_loss_function("binary_crossentropy");
    nn_with_reg.set_regularizer(RegularizerType::L2, 0.01);
    nn_with_reg.set_epochs(100);
    nn_with_reg.set_batch_size(32);
    nn_with_reg.set_verbose(false);
    nn_with_reg.fit(X_, y_);
    
    double score_no_reg = nn_no_reg.score(X_, y_);
    double score_with_reg = nn_with_reg.score(X_, y_);
    
    std::cout << "Without regularization accuracy: " << score_no_reg << std::endl;
    std::cout << "With L2 regularization accuracy: " << score_with_reg << std::endl;
    
    // Both should achieve good accuracy
    EXPECT_GT(score_no_reg, 0.80);
    EXPECT_GT(score_with_reg, 0.80);
}

/**
 * @brief Compare different regularization types with same strength
 */
TEST_F(NeuralNetworkRegularizerIntegrationTest, CompareRegularizationTypes) {
    double strength = 0.001;
    
    struct RegularizerInfo {
        RegularizerType type;
        std::string name;
    };
    
    std::vector<RegularizerInfo> reg_types = {
        {RegularizerType::L1, "L1"},
        {RegularizerType::L2, "L2"},
        {RegularizerType::ELASTIC_NET, "ElasticNet"}
    };
    
    for (const auto& info : reg_types) {
        NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
        nn.set_loss_function("binary_crossentropy");
        nn.set_regularizer(info.type, strength);
        nn.set_epochs(100);
        nn.set_batch_size(32);
        nn.set_verbose(false);
        
        EXPECT_NO_THROW(nn.fit(X_, y_)) 
            << "Failed with " << info.name << " regularization";
        
        double score = nn.score(X_, y_);
        std::cout << info.name << " regularization accuracy: " << score << std::endl;
        EXPECT_GT(score, 0.85) 
            << "Low accuracy with " << info.name << " regularization";
    }
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/**
 * @brief Test with zero regularization strength (should behave like no regularization)
 */
TEST_F(NeuralNetworkRegularizerIntegrationTest, ZeroRegularizationStrength) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L2, 0.0);  // Zero strength
    nn.set_epochs(80);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "Zero regularization strength accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.80);
}

/**
 * @brief Test with very strong regularization
 * 
 * Very strong regularization may degrade performance but should not crash
 */
TEST_F(NeuralNetworkRegularizerIntegrationTest, VeryStrongRegularization) {
    NeuralNetwork baseline({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    baseline.set_loss_function("binary_crossentropy");
    baseline.fit(X_, y_, 80, 32, false);
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L2, 1.0);  // Very strong
    nn.set_epochs(80);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    // Should not throw even with very strong regularization
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "Very strong regularization (1.0) accuracy: " << score << std::endl;
    EXPECT_TRUE(std::isfinite(score));
    EXPECT_LT(weight_norm(nn), weight_norm(baseline));
}

TEST(NeuralNetworkRegularizerGradientTest, UpdatesWeightsButNotBiasForEachPenalty) {
    struct Case {
        RegularizerType type;
        double strength;
        double positive_gradient;
        double negative_gradient;
        double penalty;
    };
    const Case cases[] = {
        {RegularizerType::NONE, 0.2, 0.0, 0.0, 0.0},
        {RegularizerType::L1, 0.2, 0.2, -0.2, 0.15},
        {RegularizerType::L2, 0.2, 0.05, -0.1, 0.03125},
        {RegularizerType::ELASTIC_NET, 0.2, 0.0875, -0.125, 0.0609375},
        {RegularizerType::L2, 0.0, 0.0, 0.0, 0.0},
    };
    const MatrixXd X = MatrixXd::Zero(1, 3);
    const MatrixXd y = MatrixXd::Zero(1, 1);
    for (const auto& test : cases) {
        NeuralNetwork nn({3, 1}, "linear", "linear", OptimizerType::SGD, 0.1);
        nn.set_loss_function("mse");
        nn.fit(X, y, 1, 1, false);
        MatrixXd weights(3, 1);
        weights << 0.25, -0.5, 0.0;
        nn.get_layers()[0]->set_weights(weights);
        nn.get_layers()[0]->set_biases(VectorXd::Constant(1, 0.3));
        nn.set_regularizer(test.type, test.strength, {{"l1_ratio", 0.25}});

        nn.fit(X, y, 1, 1, false);
        const auto actual = nn.get_layers()[0]->get_weights();
        EXPECT_NEAR(actual(0, 0), 0.25 - 0.1 * test.positive_gradient, 1e-12);
        EXPECT_NEAR(actual(1, 0), -0.5 - 0.1 * test.negative_gradient, 1e-12);
        EXPECT_DOUBLE_EQ(actual(2, 0), 0.0);
        EXPECT_NEAR(nn.get_layers()[0]->get_biases()(0), 0.24, 1e-12);
        ASSERT_EQ(nn.get_loss_history().size(), 1u);
        EXPECT_NEAR(nn.get_loss_history()[0], 0.09 + test.penalty, 1e-12);
    }
}

TEST(NeuralNetworkRegularizerGradientTest, AddsPenaltyBeforeGradientClipping) {
    NeuralNetwork nn({2, 1}, "linear", "linear", OptimizerType::SGD, 0.1);
    nn.set_loss_function("mse");
    MatrixXd X(1, 2);
    X << 1.0, -2.0;
    const MatrixXd y = MatrixXd::Zero(1, 1);
    nn.fit(X, y, 1, 1, false);
    MatrixXd weights(2, 1);
    weights << 0.4, -0.2;
    nn.get_layers()[0]->set_weights(weights);
    nn.get_layers()[0]->set_biases(VectorXd::Constant(1, 0.1));
    nn.set_regularizer(RegularizerType::L2, 0.5);

    nn.fit(X, y, 1, 1, false);
    const double norm = std::sqrt(2.0 * 2.0 + 3.7 * 3.7);
    EXPECT_NEAR(nn.get_layers()[0]->get_weights()(0, 0), 0.4 - 0.1 * 2.0 / norm, 1e-12);
    EXPECT_NEAR(nn.get_layers()[0]->get_weights()(1, 0), -0.2 + 0.1 * 3.7 / norm, 1e-12);
    EXPECT_NEAR(nn.get_layers()[0]->get_biases()(0), -0.08, 1e-12);
    EXPECT_NEAR(nn.get_loss_history()[0], 0.86, 1e-12);
}

TEST(NeuralNetworkRegularizerGradientTest, UpdatesEveryTrainableLayer) {
    NeuralNetwork nn({2, 2, 1}, "linear", "linear", OptimizerType::SGD, 0.1);
    nn.set_loss_function("mse");
    const MatrixXd X = MatrixXd::Zero(1, 2);
    const MatrixXd y = MatrixXd::Zero(1, 1);
    nn.fit(X, y, 1, 1, false);
    MatrixXd first(2, 2);
    first << 0.2, -0.4, 0.3, -0.1;
    MatrixXd second(2, 1);
    second << 0.5, -0.25;
    nn.get_layers()[0]->set_weights(first);
    nn.get_layers()[1]->set_weights(second);
    nn.get_layers()[0]->set_biases(VectorXd::Zero(2));
    nn.get_layers()[1]->set_biases(VectorXd::Zero(1));
    nn.set_regularizer(RegularizerType::L2, 0.2);

    nn.fit(X, y, 1, 1, false);
    EXPECT_TRUE(nn.get_layers()[0]->get_weights().isApprox(0.98 * first, 1e-12));
    EXPECT_TRUE(nn.get_layers()[1]->get_weights().isApprox(0.98 * second, 1e-12));
}

TEST(NeuralNetworkRegularizerGradientTest, AppliesPenaltyForVectorTargets) {
    NeuralNetwork probe({2, 1}, "linear", "linear", OptimizerType::SGD, 0.1);
    NeuralNetwork nn({2, 1}, "linear", "linear", OptimizerType::SGD, 0.1);
    probe.build(2, 1);
    const MatrixXd initial = probe.get_layers()[0]->get_weights();
    nn.set_loss_function("mse");
    nn.set_regularizer(RegularizerType::L2, 0.01);
    const MatrixXd X = MatrixXd::Zero(1, 2);
    const VectorXd y = VectorXd::Zero(1);

    nn.fit(X, y, 1, 1, false);
    EXPECT_TRUE(nn.get_layers()[0]->get_weights().isApprox(0.999 * initial, 1e-12));
}

TEST(NeuralNetworkRegularizerGradientTest, UpdatesConvolutionalAndRecurrentWeights) {
    auto verify = [](NeuralNetwork& nn, const MatrixXd& X) {
        const MatrixXd y = MatrixXd::Zero(1, 1);
        nn.set_loss_function("mse");
        nn.fit(X, y, 1, 1, false);
        std::vector<MatrixXd> initial;
        for (const auto& layer : nn.get_layers()) {
            if (!layer->has_weights()) {
                continue;
            }
            MatrixXd weights = layer->get_weights();
            weights.setConstant(0.1);
            layer->set_weights(weights);
            if (layer->get_use_bias()) {
                layer->set_biases(VectorXd::Zero(layer->get_biases().size()));
            }
            if (auto* recurrent = dynamic_cast<layers::RecurrentLayer*>(layer.get())) {
                recurrent->reset_state();
            }
            initial.push_back(weights);
        }
        nn.set_regularizer(RegularizerType::L2, 0.2);
        nn.fit(X, y, 1, 1, false);
        size_t index = 0;
        for (const auto& layer : nn.get_layers()) {
            if (layer->has_weights()) {
                EXPECT_TRUE(layer->get_weights().isApprox(0.98 * initial[index], 1e-12));
                ++index;
            }
        }
    };

    NeuralNetwork conv({36, 1}, "linear", "linear", OptimizerType::SGD, 0.1);
    conv.reset();
    auto conv_layer = std::make_unique<layers::Conv2DLayer>(1, 3, 1, "valid", "linear");
    conv_layer->set_input_shape(36);
    conv.add_layer(std::move(conv_layer));
    conv.add_flatten_layer();
    conv.add_dense_layer(1, "linear");
    verify(conv, MatrixXd::Zero(1, 36));

    NeuralNetwork recurrent({2, 1}, "linear", "linear", OptimizerType::SGD, 0.1);
    recurrent.reset();
    recurrent.add_recurrent_layer(RecurrentType::SIMPLE_RNN, 2);
    recurrent.add_dense_layer(1, "linear");
    verify(recurrent, MatrixXd::Zero(1, 2));
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}