#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/loss/mean_squared_error_loss.h"

using namespace loss;
using namespace Eigen;

class MeanSquaredErrorLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        loss = std::make_unique<MeanSquaredErrorLoss>();
    }
    
    std::unique_ptr<MeanSquaredErrorLoss> loss;
};

TEST_F(MeanSquaredErrorLossTest, PerfectPredictions) {
    VectorXd y_true(3);
    VectorXd y_pred(3);
    y_true << 1, 2, 3;
    y_pred << 1, 2, 3;
    
    double loss_value = loss->compute(y_true, y_pred);
    
    EXPECT_NEAR(loss_value, 0.0, 1e-6);
}

TEST_F(MeanSquaredErrorLossTest, SimpleCalculation) {
    VectorXd y_true(2);
    VectorXd y_pred(2);
    y_true << 1, 2;
    y_pred << 2, 4;
    
    double loss_value = loss->compute(y_true, y_pred);
    
    // ((2-1)^2 + (4-2)^2) / 2 = (1 + 4) / 2 = 2.5
    EXPECT_NEAR(loss_value, 2.5, 1e-6);
}

TEST_F(MeanSquaredErrorLossTest, MatrixInput) {
    MatrixXd y_true(3, 2);
    MatrixXd y_pred(3, 2);
    y_true << 1, 2, 3, 4, 5, 6;
    y_pred << 1, 2, 3, 4, 5, 6;
    
    double loss_value = loss->compute(y_true, y_pred);
    
    EXPECT_NEAR(loss_value, 0.0, 1e-6);
}

TEST_F(MeanSquaredErrorLossTest, Gradient) {
    MatrixXd y_true(2, 1);
    MatrixXd y_pred(2, 1);
    y_true << 1, 2;
    y_pred << 3, 5;
    
    MatrixXd grad = loss->gradient(y_true, y_pred);
    
    // MSE gradient: 2*(y_pred - y_true)/n
    MatrixXd expected = 2.0 * (y_pred - y_true) / 2.0;
    expected << 2.0, 3.0;
    
    EXPECT_NEAR(grad(0, 0), expected(0, 0), 1e-6);
    EXPECT_NEAR(grad(1, 0), expected(1, 0), 1e-6);
}

TEST_F(MeanSquaredErrorLossTest, Name) {
    EXPECT_EQ(loss->name(), "mse");
}