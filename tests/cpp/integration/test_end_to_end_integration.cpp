/**
 * @file test_end_to_end_integration.cpp
 * @brief End-to-end integration tests for the complete ML pipeline
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <iostream>
#include <Eigen/Dense>

#include "models/neural_network.h"
#include "components/loss/loss_factory.h"
#include "components/optimizers/optimizer_factory.h"
#include "components/regularizers/regularizer_factory.h"

using namespace models;
using namespace Eigen;
using namespace testing;

// ============================================================================
// Utility function for feature normalization
// ============================================================================

void normalize_features(Eigen::MatrixXd& X) {
    for (int i = 0; i < X.cols(); ++i) {
        double mean = X.col(i).mean();
        double std = std::sqrt((X.col(i).array() - mean).square().mean());
        if (std < 1e-7) std = 1.0;
        X.col(i) = (X.col(i).array() - mean) / std;
    }
}

/**
 * @brief Test fixture for end-to-end integration tests
 */
class EndToEndIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples_ = 500;
        n_features_ = 10;
        
        X_.resize(n_samples_, n_features_);
        X_.setRandom();
        
        // NORMALIZE FEATURES - CRITICAL for stable training!
        normalize_features(X_);
        
        // Create non-linear decision boundary
        // y = sign(x0^2 + sin(x1) + x2*x3 + x4)
        VectorXd y_raw = (X_.col(0).array().square() +
                          X_.col(1).array().sin() +
                          X_.col(2).array() * X_.col(3).array() +
                          X_.col(4).array()).matrix();
        
        y_ = (y_raw.array() > y_raw.mean()).cast<double>();
        
        // Ensure balanced classes
        int positive_count = y_.sum();
        int negative_count = y_.size() - positive_count;
        
        std::cout << "Dataset: " << y_.size() << " samples, "
                  << positive_count << " positive, "
                  << negative_count << " negative" << std::endl;
        std::cout << "Features: " << n_features_ << std::endl;
    }
    
    MatrixXd X_;
    VectorXd y_;
    int n_samples_;
    int n_features_;
};

TEST_F(EndToEndIntegrationTest, FullPipelineWithAllComponents) {
    NeuralNetwork nn({n_features_, 32, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    
    // Configure all components
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L2, 0.0001);
    nn.set_epochs(200);
    nn.set_batch_size(64);
    nn.set_validation_split(0.2);
    nn.set_verbose(true);
    
    // Training should complete without exceptions
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    // Verify training history
    auto [loss, val_loss, acc] = nn.get_training_history();
    EXPECT_GT(loss.size(), 0);
    if (loss.size() > 1) {
        EXPECT_LT(loss.back(), loss.front());  // Loss should decrease
    }
    
    // Verify performance
    double final_score = nn.score(X_, y_);
    std::cout << "Final accuracy: " << final_score << std::endl;
    EXPECT_GT(final_score, 0.85);
    
    // Verify predictions
    MatrixXd proba = nn.predict_proba(X_);
    for (int i = 0; i < std::min(10, (int)proba.rows()); ++i) {
        EXPECT_GE(proba(i, 0), 0.0);
        EXPECT_LE(proba(i, 0), 1.0);
    }
}

TEST_F(EndToEndIntegrationTest, CompareDifferentConfigurations) {
    struct Config {
        std::string name;
        OptimizerType optimizer;
        double lr;
        RegularizerType regularizer;
        double reg_strength;
    };
    
    std::vector<Config> configs = {
        {"SGD_no_reg", OptimizerType::SGD, 0.01, RegularizerType::NONE, 0.0},
        {"SGD_L2", OptimizerType::SGD, 0.01, RegularizerType::L2, 0.001},
        {"Adam_no_reg", OptimizerType::ADAM, 0.01, RegularizerType::NONE, 0.0},
        {"Adam_L2", OptimizerType::ADAM, 0.01, RegularizerType::L2, 0.001}
    };
    
    for (const auto& cfg : configs) {
        NeuralNetwork nn({n_features_, 32, 16, 1}, "relu", "sigmoid", cfg.optimizer, cfg.lr);
        nn.set_loss_function("binary_crossentropy");
        if (cfg.regularizer != RegularizerType::NONE) {
            nn.set_regularizer(cfg.regularizer, cfg.reg_strength);
        }
        nn.set_epochs(150);
        nn.set_batch_size(32);
        nn.set_verbose(false);
        
        EXPECT_NO_THROW(nn.fit(X_, y_)) << "Failed with config: " << cfg.name;
        
        double score = nn.score(X_, y_);
        std::cout << cfg.name << " accuracy: " << score << std::endl;
        EXPECT_GT(score, 0.80) << "Low accuracy for config: " << cfg.name;
    }
}

TEST_F(EndToEndIntegrationTest, TrainingHistoryCollector) {
    NeuralNetwork nn({n_features_, 32, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(100);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.fit(X_, y_);
    
    auto [loss, val_loss, acc] = nn.get_training_history();
    
    EXPECT_GT(loss.size(), 0);
    EXPECT_EQ(loss.size(), 100);  // Should match epochs
    
    // Check that loss values are finite
    for (double l : loss) {
        EXPECT_TRUE(std::isfinite(l));
    }
}

TEST_F(EndToEndIntegrationTest, ModelSerialization) {
    NeuralNetwork nn({n_features_, 32, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(50);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.fit(X_, y_);
    
    VectorXd y_pred_original = nn.predict(X_);
    
    std::string filename = "test_end_to_end_model.bin";
    EXPECT_NO_THROW(nn.save(filename));
    
    NeuralNetwork nn_loaded;
    EXPECT_NO_THROW(nn_loaded.load(filename));
    
    VectorXd y_pred_loaded = nn_loaded.predict(X_);
    
    double diff = (y_pred_original - y_pred_loaded).norm();
    std::cout << "Prediction difference: " << diff << std::endl;
    EXPECT_LT(diff, 0.01);  // Allow small differences
    
    std::remove(filename.c_str());
}

TEST_F(EndToEndIntegrationTest, PredictProbaRange) {
    NeuralNetwork nn({n_features_, 32, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(100);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.fit(X_, y_);
    
    MatrixXd proba = nn.predict_proba(X_);
    
    EXPECT_EQ(proba.rows(), n_samples_);
    EXPECT_EQ(proba.cols(), 1);
    
    // Check probability range
    for (int i = 0; i < n_samples_; ++i) {
        EXPECT_GE(proba(i, 0), 0.0);
        EXPECT_LE(proba(i, 0), 1.0);
    }
}

TEST_F(EndToEndIntegrationTest, ScoreMethod) {
    NeuralNetwork nn({n_features_, 32, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(100);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.fit(X_, y_);
    
    double score = nn.score(X_, y_);
    
    EXPECT_GE(score, 0.0);
    EXPECT_LE(score, 1.0);
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