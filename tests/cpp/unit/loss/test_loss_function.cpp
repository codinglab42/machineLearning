// tests/cpp/unit/test_loss_functions.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/loss/loss_factory.h"
#include "components/loss/mean_squared_error_loss.h"
#include "components/loss/mean_absolute_error_loss.h"
#include "components/loss/binary_cross_entropy_loss.h"
#include "components/loss/categorical_cross_entropy_loss.h"
#include "components/loss/huber_loss.h"

using namespace loss;
using namespace Eigen;

class LossFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup per classificazione binaria
        y_true_binary.resize(4, 1);
        y_pred_binary.resize(4, 1);
        y_true_binary << 0, 1, 0, 1;
        y_pred_binary << 0.1, 0.9, 0.2, 0.8;
        
        // Setup per classificazione multi-classe
        y_true_multi.resize(4, 3);
        y_pred_multi.resize(4, 3);
        y_true_multi << 1, 0, 0,
                        0, 1, 0,
                        0, 0, 1,
                        1, 0, 0;
        y_pred_multi << 0.7, 0.2, 0.1,
                        0.1, 0.8, 0.1,
                        0.2, 0.3, 0.5,
                        0.6, 0.3, 0.1;
        
        // Setup per regressione
        y_true_reg.resize(4, 1);
        y_pred_reg.resize(4, 1);
        y_true_reg << 1.0, 2.0, 3.0, 4.0;
        y_pred_reg << 1.1, 1.9, 3.2, 3.8;
    }
    
    MatrixXd y_true_binary;
    MatrixXd y_pred_binary;
    MatrixXd y_true_multi;
    MatrixXd y_pred_multi;
    MatrixXd y_true_reg;
    MatrixXd y_pred_reg;
};

// ============================================================================
// Mean Squared Error (MSE)
// ============================================================================

TEST_F(LossFunctionsTest, MSECompute) {
    MeanSquaredErrorLoss mse;
    double loss = mse.compute(y_true_reg, y_pred_reg);
    
    // MSE = mean((y_true - y_pred)^2)
    // (0.01 + 0.01 + 0.04 + 0.04) / 4 = 0.025
    EXPECT_NEAR(loss, 0.025, 1e-6);
    EXPECT_FALSE(std::isnan(loss));
}

TEST_F(LossFunctionsTest, MSEGradient) {
    MeanSquaredErrorLoss mse;
    MatrixXd grad = mse.gradient(y_true_reg, y_pred_reg);
    
    // Gradient = 2*(y_pred - y_true)/n
    // (0.2, -0.2, 0.4, -0.4) / 4 = (0.05, -0.05, 0.1, -0.1)
    EXPECT_NEAR(grad(0, 0), 0.05, 1e-6);
    EXPECT_NEAR(grad(1, 0), -0.05, 1e-6);
    EXPECT_NEAR(grad(2, 0), 0.1, 1e-6);
    EXPECT_NEAR(grad(3, 0), -0.1, 1e-6);
}

TEST_F(LossFunctionsTest, MSEPerfectPrediction) {
    MeanSquaredErrorLoss mse;
    MatrixXd perfect_pred = y_true_reg;
    double loss = mse.compute(y_true_reg, perfect_pred);
    EXPECT_NEAR(loss, 0.0, 1e-10);
}

// ============================================================================
// Mean Absolute Error (MAE)
// ============================================================================

TEST_F(LossFunctionsTest, MAECompute) {
    MeanAbsoluteErrorLoss mae;
    double loss = mae.compute(y_true_reg, y_pred_reg);
    
    // MAE = mean(|y_true - y_pred|)
    // (0.1 + 0.1 + 0.2 + 0.2) / 4 = 0.15
    EXPECT_NEAR(loss, 0.15, 1e-6);
}

TEST_F(LossFunctionsTest, MAEGradient) {
    MeanAbsoluteErrorLoss mae;
    
    MatrixXd y_true(4, 1);
    MatrixXd y_pred(4, 1);
    y_true << 1.0, 2.0, 3.0, 4.0;
    y_pred << 1.1, 1.9, 3.2, 3.8;
    
    MatrixXd grad = mae.gradient(y_true, y_pred);
    
    // MAE gradient = sign(y_pred - y_true) / n
    // sign(0.1) = 1, sign(-0.1) = -1, sign(0.2) = 1, sign(-0.2) = -1
    // / 4 = 0.25, -0.25, 0.25, -0.25
    EXPECT_NEAR(grad(0, 0), 0.25, 1e-6);
    EXPECT_NEAR(grad(1, 0), -0.25, 1e-6);
    EXPECT_NEAR(grad(2, 0), 0.25, 1e-6);
    EXPECT_NEAR(grad(3, 0), -0.25, 1e-6);
}

// ============================================================================
// Binary Cross Entropy
// ============================================================================

TEST_F(LossFunctionsTest, BinaryCrossEntropyCompute) {
    BinaryCrossEntropyLoss bce;
    double loss = bce.compute(y_true_binary, y_pred_binary);
    
    EXPECT_FALSE(std::isnan(loss));
    EXPECT_FALSE(std::isinf(loss));
    EXPECT_GT(loss, 0);
}

TEST_F(LossFunctionsTest, BinaryCrossEntropyPerfectPrediction) {
    BinaryCrossEntropyLoss bce;
    MatrixXd perfect_pred = y_true_binary;
    double loss = bce.compute(y_true_binary, perfect_pred);
    EXPECT_NEAR(loss, 0.0, 1e-6);
}

TEST_F(LossFunctionsTest, BinaryCrossEntropyGradient) {
    BinaryCrossEntropyLoss bce;
    MatrixXd grad = bce.gradient(y_true_binary, y_pred_binary);
    
    EXPECT_EQ(grad.rows(), 4);
    EXPECT_EQ(grad.cols(), 1);
    EXPECT_FALSE(grad.hasNaN());
}

TEST_F(LossFunctionsTest, BinaryCrossEntropyClipping) {
    BinaryCrossEntropyLoss bce;
    
    // Predizioni estreme (vicine a 0 o 1)
    MatrixXd extreme_pred(2, 1);
    extreme_pred << 1e-10, 1 - 1e-10;
    MatrixXd extreme_true(2, 1);
    extreme_true << 1, 0;
    
    // Non deve lanciare eccezioni per log(0)
    EXPECT_NO_THROW(bce.compute(extreme_true, extreme_pred));
}

// ============================================================================
// Categorical Cross Entropy
// ============================================================================

TEST_F(LossFunctionsTest, CategoricalCrossEntropyCompute) {
    CategoricalCrossEntropyLoss cce;
    double loss = cce.compute(y_true_multi, y_pred_multi);
    
    EXPECT_FALSE(std::isnan(loss));
    EXPECT_FALSE(std::isinf(loss));
    EXPECT_GT(loss, 0);
}

TEST_F(LossFunctionsTest, CategoricalCrossEntropyPerfectPrediction) {
    CategoricalCrossEntropyLoss cce;
    double loss = cce.compute(y_true_multi, y_true_multi);
    EXPECT_NEAR(loss, 0.0, 1e-6);
}

TEST_F(LossFunctionsTest, CategoricalCrossEntropyGradient) {
    CategoricalCrossEntropyLoss cce;
    MatrixXd grad = cce.gradient(y_true_multi, y_pred_multi);
    
    EXPECT_EQ(grad.rows(), 4);
    EXPECT_EQ(grad.cols(), 3);
    EXPECT_FALSE(grad.hasNaN());
}

TEST_F(LossFunctionsTest, CategoricalCrossEntropySumToOne) {
    CategoricalCrossEntropyLoss cce;
    
    // Le predizioni dovrebbero sommare a 1 per ogni riga
    for (int i = 0; i < y_pred_multi.rows(); ++i) {
        double row_sum = y_pred_multi.row(i).sum();
        EXPECT_NEAR(row_sum, 1.0, 1e-6);
    }
}

// ============================================================================
// Huber Loss
// ============================================================================

TEST_F(LossFunctionsTest, HuberLossCompute) {
    HuberLoss huber(1.0);
    double loss = huber.compute(y_true_reg, y_pred_reg);
    
    EXPECT_FALSE(std::isnan(loss));
    EXPECT_FALSE(std::isinf(loss));
    EXPECT_GT(loss, 0);
}

TEST_F(LossFunctionsTest, HuberLossPerfectPrediction) {
    HuberLoss huber(1.0);
    double loss = huber.compute(y_true_reg, y_true_reg);
    EXPECT_NEAR(loss, 0.0, 1e-10);
}

TEST_F(LossFunctionsTest, HuberLossQuadraticRegion) {
    HuberLoss huber(1.0);
    
    // Piccoli residui -> comportamento quadratico (MSE-like)
    MatrixXd small_residual(1, 1);
    small_residual << 0.5;
    MatrixXd target(1, 1);
    target << 0;
    
    double loss = huber.compute(target, small_residual);
    // Per delta=1.0, residuo=0.5 -> 0.5 * 0.5^2 = 0.125
    EXPECT_NEAR(loss, 0.125, 1e-6);
}

TEST_F(LossFunctionsTest, HuberLossLinearRegion) {
    HuberLoss huber(1.0);
    
    // Grandi residui -> comportamento lineare
    MatrixXd large_residual(1, 1);
    large_residual << 2.0;
    MatrixXd target(1, 1);
    target << 0;
    
    double loss = huber.compute(target, large_residual);
    // Per delta=1.0, residuo=2.0 -> 1.0*(2.0 - 0.5*1.0) = 1.5
    EXPECT_NEAR(loss, 1.5, 1e-6);
}

TEST_F(LossFunctionsTest, HuberLossGradient) {
    HuberLoss huber(1.0);
    MatrixXd grad = huber.gradient(y_true_reg, y_pred_reg);
    
    EXPECT_EQ(grad.rows(), 4);
    EXPECT_EQ(grad.cols(), 1);
    EXPECT_FALSE(grad.hasNaN());
}

TEST_F(LossFunctionsTest, HuberLossDeltaParameter) {
    HuberLoss huber_small(0.5);
    HuberLoss huber_large(2.0);
    
    MatrixXd residual(1, 1);
    residual << 1.0;
    MatrixXd target(1, 1);
    target << 0;
    
    double loss_small = huber_small.compute(target, residual);
    double loss_large = huber_large.compute(target, residual);
    
    // Con delta piccolo, il residuo è considerato "grande" -> comportamento lineare
    // Con delta grande, il residuo è considerato "piccolo" -> comportamento quadratico
    EXPECT_NE(loss_small, loss_large);
}

// ============================================================================
// Loss Factory Tests
// ============================================================================

TEST_F(LossFunctionsTest, LossFactoryCreate) {
    auto mse = LossFactory::create("mse");
    EXPECT_EQ(mse->name(), "mse");
    
    auto mae = LossFactory::create("mae");
    EXPECT_EQ(mae->name(), "mae");
    
    auto binary_cross = LossFactory::create("binary_crossentropy");
    EXPECT_EQ(binary_cross->name(), "binary_crossentropy");
    
    auto categorical_cross = LossFactory::create("categorical_crossentropy");
    EXPECT_EQ(categorical_cross->name(), "categorical_crossentropy");
    
    auto huber = LossFactory::create("huber");
    EXPECT_EQ(huber->name(), "huber");
}

TEST_F(LossFunctionsTest, LossFactoryUnknownLossThrows) {
    EXPECT_THROW(LossFactory::create("unknown_loss"), ml_exception::InvalidParameterException);
}

// ============================================================================
// Vector vs Matrix Compute Tests
// ============================================================================

TEST_F(LossFunctionsTest, MSEVectorVsMatrixCompute) {
    MeanSquaredErrorLoss mse;
    
    // Versione VectorXd
    VectorXd y_true_vec = y_true_reg.col(0);
    VectorXd y_pred_vec = y_pred_reg.col(0);
    double loss_vec = mse.compute(y_true_vec, y_pred_vec);
    
    // Versione MatrixXd
    double loss_mat = mse.compute(y_true_reg, y_pred_reg);
    
    EXPECT_NEAR(loss_vec, loss_mat, 1e-10);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(LossFunctionsTest, NumericalStability) {
    BinaryCrossEntropyLoss bce;
    
    // Predizioni estreme
    MatrixXd y_true(2, 1);
    MatrixXd y_pred(2, 1);
    y_true << 1, 0;
    y_pred << 1e-15, 1 - 1e-15;  // Molto vicino a 0 e 1
    
    // Non deve produrre inf o nan
    double loss = bce.compute(y_true, y_pred);
    EXPECT_FALSE(std::isnan(loss));
    EXPECT_FALSE(std::isinf(loss));
}