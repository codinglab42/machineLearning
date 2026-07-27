/**
 * @file test_loss_optimizer_integration.cpp
 * @brief Integration tests for Loss and Optimizer
 * 
 * Test design:
 * 1. Test that gradient is correct using finite differences
 * 2. Test that optimizer decreases loss
 */

#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "components/loss/loss_factory.h"
#include "components/optimizers/optimizer_factory.h"

using namespace loss;
using namespace models;
using namespace Eigen;

// ============================================================================
// Helper: Compute gradient using finite differences
// ============================================================================

double compute_loss_value(
    const std::unique_ptr<Loss>& loss,
    const MatrixXd& w,
    const MatrixXd& X,
    const MatrixXd& y_true) {
    
    MatrixXd logits = X * w;
    MatrixXd pred = (1.0 / (1.0 + (-logits).array().exp())).matrix();
    return loss->compute(y_true, pred);
}

MatrixXd finite_difference_gradient(
    const std::unique_ptr<Loss>& loss,
    const MatrixXd& w,
    const MatrixXd& X,
    const MatrixXd& y_true,
    double epsilon = 1e-6) {
    
    MatrixXd grad = MatrixXd::Zero(w.rows(), w.cols());
    
    for (int i = 0; i < w.rows(); ++i) {
        for (int j = 0; j < w.cols(); ++j) {
            MatrixXd w_plus = w;
            MatrixXd w_minus = w;
            w_plus(i, j) += epsilon;
            w_minus(i, j) -= epsilon;
            
            double loss_plus = compute_loss_value(loss, w_plus, X, y_true);
            double loss_minus = compute_loss_value(loss, w_minus, X, y_true);
            
            grad(i, j) = (loss_plus - loss_minus) / (2 * epsilon);
        }
    }
    
    return grad;
}

// ============================================================================
// TEST 1: Binary Cross Entropy gradient matches finite differences
// ============================================================================

TEST(LossOptimizerIntegrationTest, BinaryCrossEntropyGradientIsCorrect) {
    auto loss = LossFactory::create("binary_crossentropy");
    
    // Simple dataset
    MatrixXd X(4, 1);
    X << -2.0, -1.0, 1.0, 2.0;
    
    MatrixXd y_true(4, 1);
    y_true << 0.0, 0.0, 1.0, 1.0;
    
    // Test with different weights
    std::vector<double> test_weights = {-2.0, -1.0, 0.0, 1.0, 2.0};
    
    for (double w_val : test_weights) {
        MatrixXd w(1, 1);
        w(0, 0) = w_val;
        
        // Forward pass
        MatrixXd logits = X * w;
        MatrixXd pred = (1.0 / (1.0 + (-logits).array().exp())).matrix();
        
        // Analytical gradient from loss
        MatrixXd analytical_grad = loss->gradient(y_true, pred);
        // For weights: dL/dw = X^T * dL/dz, where dL/dz = dL/dp * p*(1-p)
        MatrixXd dL_dp = analytical_grad;  // already (p - y)/n
        MatrixXd dp_dz = (pred.array() * (1.0 - pred.array())).matrix();
        MatrixXd dL_dz = dL_dp.array() * dp_dz.array();
        MatrixXd dL_dw = (X.transpose() * (pred - y_true)).array() / X.rows();
        
        // Finite difference gradient
        MatrixXd fd_grad = finite_difference_gradient(loss, w, X, y_true);
        
        std::cout << "w = " << w_val 
                  << ", analytical dL/dw = " << dL_dw(0, 0)
                  << ", finite diff = " << fd_grad(0, 0)
                  << std::endl;
        
        EXPECT_NEAR(dL_dw(0, 0), fd_grad(0, 0), 1e-4);
    }
}

// ============================================================================
// TEST 2: SGD decreases loss over multiple steps
// ============================================================================

TEST(LossOptimizerIntegrationTest, SGDDecreasesLoss) {
    auto loss = LossFactory::create("binary_crossentropy");
    auto optimizer = OptimizerFactory::create(OptimizerType::SGD, 0.1);
    
    // Dataset: learn to predict sign of x
    MatrixXd X(4, 1);
    X << -2.0, -1.0, 1.0, 2.0;
    
    MatrixXd y_true(4, 1);
    y_true << 0.0, 0.0, 1.0, 1.0;
    
    // Initialize weight
    MatrixXd w(1, 1);
    w << 0.0;
    
    // Record initial loss
    MatrixXd logits = X * w;
    MatrixXd pred = (1.0 / (1.0 + (-logits).array().exp())).matrix();
    double initial_loss = loss->compute(y_true, pred);
    
    // Perform gradient descent steps
    for (int step = 0; step < 200; ++step) {
        // Forward
        MatrixXd logits_step = X * w;
        MatrixXd pred_step = (1.0 / (1.0 + (-logits_step).array().exp())).matrix();
        
        // Gradient
        MatrixXd grad = loss->gradient(y_true, pred_step);
        MatrixXd dp_dz = (pred_step.array() * (1.0 - pred_step.array())).matrix();
        MatrixXd dL_dz = grad.array() * dp_dz.array();
        MatrixXd dL_dw = X.transpose() * dL_dz;
        
        // Update
        optimizer->update_weights(w, dL_dw);
    }
    
    // Final loss
    MatrixXd final_logits = X * w;
    MatrixXd final_pred = (1.0 / (1.0 + (-final_logits).array().exp())).matrix();
    double final_loss = loss->compute(y_true, final_pred);
    
    std::cout << "\nSGD: initial loss = " << initial_loss 
              << ", final loss = " << final_loss 
              << ", final w = " << w(0, 0) << std::endl;
    
    // Loss should decrease
    EXPECT_LT(final_loss, initial_loss);
    
    // Weight should become positive (since positive x -> y=1)
    EXPECT_GT(w(0, 0), 0);
    
    // Predictions should be correct
    for (int i = 0; i < 4; ++i) {
        if (X(i, 0) > 0) {
            EXPECT_GT(final_pred(i, 0), 0.5);
        } else {
            EXPECT_LT(final_pred(i, 0), 0.5);
        }
    }
}

// ============================================================================
// TEST 3: Adam decreases loss over multiple steps
// ============================================================================

TEST(LossOptimizerIntegrationTest, AdamDecreasesLoss) {
    auto loss = LossFactory::create("binary_crossentropy");
    auto optimizer = OptimizerFactory::create(OptimizerType::ADAM, 0.1);
    
    // Dataset
    MatrixXd X(4, 1);
    X << -2.0, -1.0, 1.0, 2.0;
    
    MatrixXd y_true(4, 1);
    y_true << 0.0, 0.0, 1.0, 1.0;
    
    // Initialize weight
    MatrixXd w(1, 1);
    w << 0.0;
    
    // Initial loss
    MatrixXd logits = X * w;
    MatrixXd pred = (1.0 / (1.0 + (-logits).array().exp())).matrix();
    double initial_loss = loss->compute(y_true, pred);
    
    // Adam steps
    for (int step = 0; step < 200; ++step) {
        MatrixXd logits_step = X * w;
        MatrixXd pred_step = (1.0 / (1.0 + (-logits_step).array().exp())).matrix();
        
        MatrixXd grad = loss->gradient(y_true, pred_step);
        MatrixXd dp_dz = (pred_step.array() * (1.0 - pred_step.array())).matrix();
        MatrixXd dL_dz = grad.array() * dp_dz.array();
        MatrixXd dL_dw = X.transpose() * dL_dz;
        
        optimizer->update_weights(w, dL_dw);
    }
    
    // Final loss
    MatrixXd final_logits = X * w;
    MatrixXd final_pred = (1.0 / (1.0 + (-final_logits).array().exp())).matrix();
    double final_loss = loss->compute(y_true, final_pred);
    
    std::cout << "\nAdam: initial loss = " << initial_loss 
              << ", final loss = " << final_loss 
              << ", final w = " << w(0, 0) << std::endl;
    
    EXPECT_LT(final_loss, initial_loss);
    EXPECT_GT(w(0, 0), 0);
    
    for (int i = 0; i < 4; ++i) {
        if (X(i, 0) > 0) {
            EXPECT_GT(final_pred(i, 0), 0.5);
        } else {
            EXPECT_LT(final_pred(i, 0), 0.5);
        }
    }
}

// ============================================================================
// TEST 4: MSE loss with SGD
// ============================================================================

TEST(LossOptimizerIntegrationTest, MSEWithSGD) {
    auto loss = LossFactory::create("mse");
    auto optimizer = OptimizerFactory::create(OptimizerType::SGD, 0.01);
    
    // Linear regression: y = 2*x
    MatrixXd X(5, 1);
    X << 0.0, 1.0, 2.0, 3.0, 4.0;
    
    MatrixXd y_true(5, 1);
    y_true << 0.0, 2.0, 4.0, 6.0, 8.0;
    
    // Initialize weight
    MatrixXd w(1, 1);
    w << 0.0;
    
    // Initial loss
    MatrixXd pred = X * w;
    double initial_loss = loss->compute(y_true, pred);
    
    // Gradient descent
    for (int step = 0; step < 200; ++step) {
        MatrixXd pred_step = X * w;
        MatrixXd grad = loss->gradient(y_true, pred_step);
        // dL/dw = X^T * dL/dpred
        MatrixXd dL_dw = X.transpose() * grad;
        optimizer->update_weights(w, dL_dw);
    }
    
    // Final loss
    MatrixXd final_pred = X * w;
    double final_loss = loss->compute(y_true, final_pred);
    
    std::cout << "\nMSE with SGD: initial loss = " << initial_loss 
              << ", final loss = " << final_loss 
              << ", final w = " << w(0, 0) 
              << " (expected 2.0)" << std::endl;
    
    EXPECT_LT(final_loss, initial_loss);
    EXPECT_NEAR(w(0, 0), 2.0, 0.1);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}