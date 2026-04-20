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
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L2, 0.001);
    nn.set_epochs(100);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    // Training should complete without exceptions
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    // Accuracy should be good
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
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L1, 0.001);
    nn.set_epochs(100);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
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
        EXPECT_GT(score, 0.80)
            << "Low accuracy with L1 strength = " << strength;
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
    // Accuracy may be lower but should still be better than random (0.5)
    EXPECT_GT(score, 0.55);
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}