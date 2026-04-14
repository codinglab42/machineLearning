#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/loss/categorical_cross_entropy_loss.h"

using namespace loss;
using namespace Eigen;

class CategoricalCrossEntropyLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        loss = std::make_unique<CategoricalCrossEntropyLoss>();
    }
    
    std::unique_ptr<CategoricalCrossEntropyLoss> loss;
};

TEST_F(CategoricalCrossEntropyLossTest, PerfectPredictions) {
    MatrixXd y_true(2, 3);
    MatrixXd y_pred(2, 3);
    y_true << 1, 0, 0,
              0, 1, 0;
    y_pred << 0.9, 0.05, 0.05,
              0.1, 0.8, 0.1;
    
    double loss_value = loss->compute(y_true, y_pred);
    EXPECT_GT(loss_value, 0);
    EXPECT_LT(loss_value, 1);
}

TEST_F(CategoricalCrossEntropyLossTest, Name) {
    EXPECT_EQ(loss->name(), "categorical_crossentropy");
}