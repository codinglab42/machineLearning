#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/loss/mean_absolute_error_loss.h"

using namespace loss;
using namespace Eigen;

class MeanAbsoluteErrorLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        loss = std::make_unique<MeanAbsoluteErrorLoss>();
    }
    
    std::unique_ptr<MeanAbsoluteErrorLoss> loss;
};

TEST_F(MeanAbsoluteErrorLossTest, PerfectPredictions) {
    VectorXd y_true(3);
    VectorXd y_pred(3);
    y_true << 1, 2, 3;
    y_pred << 1, 2, 3;
    
    double loss_value = loss->compute(y_true, y_pred);
    EXPECT_NEAR(loss_value, 0.0, 1e-6);
}

TEST_F(MeanAbsoluteErrorLossTest, SimpleCalculation) {
    VectorXd y_true(2);
    VectorXd y_pred(2);
    y_true << 1, 2;
    y_pred << 3, 5;
    
    double loss_value = loss->compute(y_true, y_pred);
    // (|3-1| + |5-2|) / 2 = (2 + 3) / 2 = 2.5
    EXPECT_NEAR(loss_value, 2.5, 1e-6);
}

TEST_F(MeanAbsoluteErrorLossTest, Name) {
    EXPECT_EQ(loss->name(), "mae");
}