/**
 * @file test_neural_network_regularizer_integration.cpp
 * @brief Integration tests for Neural Network with Regularizers
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <iostream>
#include <Eigen/Dense>

#include "models/neural_network.h"
#include "components/regularizers/regularizer_factory.h"

using namespace models;
using namespace Eigen;
using namespace testing;

// ============================================================================
// Utility function for feature normalization
// ============================================================================

/**
 * @brief Normalize features to have zero mean and unit variance
 * @param X Matrix to normalize (modified in place)
 */
void normalize_features(Eigen::MatrixXd& X) {
    for (int i = 0; i < X.cols(); ++i) {
        double mean = X.col(i).mean();
        double std = std::sqrt((X.col(i).array() - mean).square().mean());
        if (std < 1e-7) std = 1.0;
        X.col(i) = (X.col(i).array() - mean) / std;
    }
}

/**
 * @brief Test fixture for Neural Network with Regularizer integration tests
 */
class NeuralNetworkRegularizerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple binary classification dataset
        X_.resize(100, 2);
        X_.setRandom();
        
        // NORMALIZE FEATURES - CRITICAL for stable training!
        normalize_features(X_);
        
        // Decision boundary: x0 + x1 > 0 (after normalization)
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

TEST_F(NeuralNetworkRegularizerIntegrationTest, TrainWithL2Regularization) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L2, 0.001);
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "L2 Regularization accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}

TEST_F(NeuralNetworkRegularizerIntegrationTest, L2RegularizationWithDifferentStrengths) {
    std::vector<double> strengths = {0.0001, 0.001, 0.01};
    
    for (double strength : strengths) {
        NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
        nn.set_loss_function("binary_crossentropy");
        nn.set_regularizer(RegularizerType::L2, strength);
        nn.set_epochs(200);
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

TEST_F(NeuralNetworkRegularizerIntegrationTest, TrainWithL1Regularization) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L1, 0.001);
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "L1 Regularization accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}

TEST_F(NeuralNetworkRegularizerIntegrationTest, L1RegularizationWithDifferentStrengths) {
    std::vector<double> strengths = {0.0001, 0.001, 0.01};
    
    for (double strength : strengths) {
        NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
        nn.set_loss_function("binary_crossentropy");
        nn.set_regularizer(RegularizerType::L1, strength);
        nn.set_epochs(200);
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

TEST_F(NeuralNetworkRegularizerIntegrationTest, TrainWithElasticNetRegularization) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::ELASTIC_NET, 0.001);
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "Elastic Net Regularization accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}

TEST_F(NeuralNetworkRegularizerIntegrationTest, ElasticNetWithDifferentStrengths) {
    std::vector<double> strengths = {0.0001, 0.001, 0.01};
    
    for (double strength : strengths) {
        NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
        nn.set_loss_function("binary_crossentropy");
        nn.set_regularizer(RegularizerType::ELASTIC_NET, strength);
        nn.set_epochs(200);
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

TEST_F(NeuralNetworkRegularizerIntegrationTest, CompareWithAndWithoutRegularization) {
    // Without regularization
    NeuralNetwork nn_no_reg({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn_no_reg.set_loss_function("binary_crossentropy");
    nn_no_reg.set_epochs(200);
    nn_no_reg.set_batch_size(32);
    nn_no_reg.set_verbose(false);
    nn_no_reg.fit(X_, y_);
    
    // With L2 regularization
    NeuralNetwork nn_with_reg({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn_with_reg.set_loss_function("binary_crossentropy");
    nn_with_reg.set_regularizer(RegularizerType::L2, 0.01);
    nn_with_reg.set_epochs(200);
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
        nn.set_epochs(200);
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

TEST_F(NeuralNetworkRegularizerIntegrationTest, ZeroRegularizationStrength) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L2, 0.0);
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "Zero regularization strength accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.80);
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}