/**
 * @file test_huber_loss.cpp
 * @brief Unit tests for Huber Loss function and its gradient
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/loss/huber_loss.h"
#include <memory>

using namespace loss;

class HuberLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Delta standard (zona quadratica per |r| <= 1.0)
        huber_std = std::make_unique<HuberLoss>(1.0);
        // Delta piccolo per forzare la transizione lineare velocemente
        huber_small = std::make_unique<HuberLoss>(0.5);
    }

    std::unique_ptr<HuberLoss> huber_std;
    std::unique_ptr<HuberLoss> huber_small;
};

// ============================================================================
// 1. TEST PER IL CALCOLO DELLA LOSS (compute)
// ============================================================================

TEST_F(HuberLossTest, VectorPerfectPrediction) {
    Eigen::VectorXd y_true(3); y_true << 1.0, 2.0, 3.0;
    Eigen::VectorXd y_pred(3); y_pred << 1.0, 2.0, 3.0;
    
    EXPECT_NEAR(huber_std->compute(y_true, y_pred), 0.0, 1e-9);
}

TEST_F(HuberLossTest, VectorQuadraticRegion) {
    Eigen::VectorXd y_true(2); y_true << 1.0, 2.0;
    Eigen::VectorXd y_pred(2); y_pred << 1.5, 1.8; // residui: 0.5, -0.2 (assoluti <= delta=1.0)
    
    // Loss attesa = (0.5 * 0.5^2 + 0.5 * (-0.2)^2) / 2 = (0.125 + 0.02) / 2 = 0.0725
    double expected = 0.0725;
    EXPECT_NEAR(huber_std->compute(y_true, y_pred), expected, 1e-6);
}

TEST_F(HuberLossTest, VectorLinearRegion) {
    Eigen::VectorXd y_true(1); y_true << 1.0;
    Eigen::VectorXd y_pred(1); y_pred << 4.0; // residuo = 3.0 ( > delta=1.0)
    
    // Loss attesa = delta * (|residual| - 0.5 * delta) = 1.0 * (3.0 - 0.5) = 2.5
    double expected = 2.5;
    EXPECT_NEAR(huber_std->compute(y_true, y_pred), expected, 1e-6);
}

TEST_F(HuberLossTest, MatrixBidimensionalHandling) {
    // Verifica la robustezza sulle matrici reali (evita bug di indicizzazione)
    Eigen::MatrixXd y_true(2, 2);
    y_true << 1.0, 2.0,
              3.0, 4.0;
              
    Eigen::MatrixXd y_pred(2, 2);
    y_pred << 1.2, 4.0,  // residui assoluti: 0.2 (quadratico), 2.0 (lineare per delta=0.5)
              3.1, 4.0;  // residui assoluti: 0.1 (quadratico), 0.0 (quadratico)

    // Calcolo manuale con delta = 0.5 su 4 elementi totali:
    // r(0,0) = 0.2 <= 0.5 -> 0.5 * 0.2^2 = 0.02
    // r(0,1) = 2.0 >  0.5 -> 0.5 * (2.0 - 0.25) = 0.875
    // r(1,0) = 0.1 <= 0.5 -> 0.5 * 0.1^2 = 0.005
    // r(1,1) = 0.0 <= 0.5 -> 0.0
    // Totale = (0.02 + 0.875 + 0.005 + 0.0) / 4 = 0.9 / 4 = 0.225
    double expected = 0.225;
    EXPECT_NEAR(huber_small->compute(y_true, y_pred), expected, 1e-6);
}

// ============================================================================
// 2. TEST PER IL CALCOLO DEL GRADIENTE (gradient)
// ============================================================================

TEST_F(HuberLossTest, GradientComputationStandardDelta) {
    Eigen::MatrixXd y_true(2, 2);
    y_true << 1.0, 2.0,
              3.0, 4.0;
              
    Eigen::MatrixXd y_pred(2, 2);
    y_pred << 1.5, 4.0,  // residuo (0,0) =  0.5  (<= delta, zona quadratica)
                         // residuo (0,1) =  2.0  (>  delta, zona lineare)
              2.2, 3.0;  // residuo (1,0) = -0.8  (<= delta, zona quadratica)
                         // residuo (1,1) = -1.0  (<= delta, zona quadratica)

    // Gradiente atteso prima della normalizzazione:
    // grad(0,0) = residuo = 0.5
    // grad(0,1) = delta * sign(residuo) = 1.0 * 1.0 = 1.0
    // grad(1,0) = residuo = -0.8
    // grad(1,1) = residuo = -1.0
    //
    // Normalizzazione: divisione per y_true.size() ovvero 4.0 elementi totali.
    Eigen::MatrixXd expected_grad(2, 2);
    expected_grad <<  0.5 / 4.0,  1.0 / 4.0,
                     -0.8 / 4.0, -1.0 / 4.0;
                     
    Eigen::MatrixXd actual_grad = huber_std->gradient(y_true, y_pred);
    
    EXPECT_TRUE(actual_grad.isApprox(expected_grad, 1e-6));
}

TEST_F(HuberLossTest, GradientComputationSmallDelta) {
    // Testiamo il gradiente con un delta più stretto (0.5) per forzare la zona lineare
    Eigen::MatrixXd y_true(1, 2);
    y_true << 1.0, 1.0;
    
    Eigen::MatrixXd y_pred(1, 2);
    y_pred << 2.0, 0.0; // residuo (0,0) =  1.0 (>  0.5) -> lineare positivo
                        // residuo (0,1) = -1.0 (< -0.5) -> lineare negativo
                        
    // Gradiente atteso prima della normalizzazione:
    // grad(0,0) = delta * sign(1.0) = 0.5 * 1.0 = 0.5
    // grad(0,1) = delta * sign(-1.0) = 0.5 * -1.0 = -0.5
    // Normalizzazione: 2 elementi totali, quindi diviso 2.0
    Eigen::MatrixXd expected_grad(1, 2);
    expected_grad << 0.5 / 2.0, -0.5 / 2.0;
    
    Eigen::MatrixXd actual_grad = huber_small->gradient(y_true, y_pred);
    
    EXPECT_TRUE(actual_grad.isApprox(expected_grad, 1e-6));
}

// ============================================================================
// MAIN EXECUTIVE
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}