/**
 * @file test_convergence_integration.cpp
 * @brief Integration tests for convergence behavior of neural networks
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "models/neural_network.h"
#include "components/optimizers/optimizer_factory.h"

using namespace models;
using namespace Eigen;
using namespace testing;

class ConvergenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Simple dataset for convergence test
        n_samples_ = 100;
        X_.resize(n_samples_, 1);
        X_ = VectorXd::LinSpaced(n_samples_, -3, 3);
        y_ = (X_.array().tanh() + 0.5).matrix();
        
        std::cout << "Convergence test: " << n_samples_ << " samples" << std::endl;
    }
    
    MatrixXd X_;
    VectorXd y_;
    int n_samples_;
};

TEST_F(ConvergenceTest, LossDecreasesMonotonically) {
    NeuralNetwork nn({1, 32, 1}, "tanh", "linear", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("mse");
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.fit(X_, y_);
    
    auto [loss, val_loss, acc] = nn.get_training_history();
    
    EXPECT_GT(loss.size(), 0);
    
    // Check that loss generally decreases (allow small increases up to 10%)
    for (size_t i = 1; i < loss.size(); ++i) {
        // Non deve aumentare troppo (più del 10%)
        EXPECT_LT(loss[i], loss[i-1] * 1.1) 
            << "Loss increased too much at epoch " << i 
            << ": " << loss[i-1] << " -> " << loss[i];
    }
    
    // Final loss should be lower than initial loss
    if (loss.size() > 1) {
        EXPECT_LT(loss.back(), loss.front());
        std::cout << "Loss decreased from " << loss.front() 
                  << " to " << loss.back() << std::endl;
    }
}

TEST_F(ConvergenceTest, DifferentLearningRates) {
    std::vector<double> lrs = {0.001, 0.01, 0.1, 0.5};
    
    for (double lr : lrs) {
        NeuralNetwork nn({1, 32, 1}, "tanh", "linear", OptimizerType::ADAM, lr);
        nn.set_loss_function("mse");
        nn.set_epochs(150);
        nn.set_batch_size(32);
        nn.set_verbose(false);
        
        EXPECT_NO_THROW(nn.fit(X_, y_)) << "Failed with LR=" << lr;
        
        auto [loss, val_loss, acc] = nn.get_training_history();
        double final_loss = loss.back();
        
        std::cout << "LR=" << lr << " final loss=" << final_loss << std::endl;
        
        // All learning rates should converge
        EXPECT_LT(final_loss, 0.1) << "Failed to converge with LR=" << lr;
        EXPECT_TRUE(std::isfinite(final_loss)) << "NaN/Inf loss with LR=" << lr;
    }
}

TEST_F(ConvergenceTest, ConvergenceWithSGD) {
    NeuralNetwork nn({1, 32, 1}, "tanh", "linear", OptimizerType::SGD, 0.01);
    nn.set_loss_function("mse");
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.fit(X_, y_);
    
    auto [loss, val_loss, acc] = nn.get_training_history();
    double final_loss = loss.back();
    
    std::cout << "SGD final loss: " << final_loss << std::endl;
    EXPECT_LT(final_loss, 0.15);
}

TEST_F(ConvergenceTest, FinalLossValue) {
    NeuralNetwork nn({1, 32, 1}, "tanh", "linear", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("mse");
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.fit(X_, y_);
    
    auto [loss, val_loss, acc] = nn.get_training_history();
    double final_loss = loss.back();
    
    std::cout << "Final loss: " << final_loss << std::endl;
    EXPECT_LT(final_loss, 0.05);
    
    // Check R^2 score
    VectorXd y_pred = nn.predict(X_);
    double ss_res = (y_ - y_pred).array().square().sum();
    double ss_tot = (y_.array() - y_.mean()).square().sum();
    double r2 = 1.0 - (ss_res / (ss_tot + 1e-7));
    
    std::cout << "R² score: " << r2 << std::endl;
    EXPECT_GT(r2, 0.95);
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}