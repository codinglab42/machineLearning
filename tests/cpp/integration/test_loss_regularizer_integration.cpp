#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <Eigen/Dense>

#include "components/loss/loss_factory.h"
#include "components/loss/loss.h"
#include "components/regularizers/regularizer_factory.h"
#include "components/regularizers/regularizer.h"

using namespace loss;
using namespace models;
using namespace Eigen;
using namespace testing;

class LossRegularizerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        loss_ = LossFactory::create("mse");
        l1_ = RegularizerFactory::create(RegularizerType::L1, 0.01);
        l2_ = RegularizerFactory::create(RegularizerType::L2, 0.01);
        elastic_ = RegularizerFactory::create(RegularizerType::ELASTIC_NET, 0.01);
    }
    
    std::unique_ptr<Loss> loss_;
    std::unique_ptr<Regularizer> l1_;
    std::unique_ptr<Regularizer> l2_;
    std::unique_ptr<Regularizer> elastic_;
};

TEST_F(LossRegularizerIntegrationTest, MSEWithL2Regularization) {
    VectorXd y_true(3);
    VectorXd y_pred(3);
    y_true << 1, 2, 3;
    y_pred << 1.1, 1.9, 3.2;
    
    MatrixXd weights = MatrixXd::Random(5, 5);
    
    double data_loss = loss_->compute(y_true, y_pred);
    double reg_loss = l2_->compute_loss(weights);
    double total_loss = data_loss + reg_loss;
    
    EXPECT_GT(total_loss, data_loss);  // Regularization aumenta la loss
}

TEST_F(LossRegularizerIntegrationTest, GradientWithL1Regularization) {
    MatrixXd weights = MatrixXd::Random(10, 10);
    MatrixXd data_grad = MatrixXd::Random(10, 10);
    
    MatrixXd reg_grad = l1_->compute_gradient(weights);
    MatrixXd total_grad = data_grad + reg_grad;
    
    // L1 gradient should be ±lambda per elemento
    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            if (weights(i, j) > 0) {
                EXPECT_NEAR(reg_grad(i, j), 0.01, 1e-6);
            } else if (weights(i, j) < 0) {
                EXPECT_NEAR(reg_grad(i, j), -0.01, 1e-6);
            }
        }
    }
}

TEST_F(LossRegularizerIntegrationTest, MSEWithL1Regularization) {
    VectorXd y_true(3);
    VectorXd y_pred(3);
    y_true << 1, 2, 3;
    y_pred << 1.1, 1.9, 3.2;
    
    MatrixXd weights = MatrixXd::Random(5, 5);
    
    double data_loss = loss_->compute(y_true, y_pred);
    double reg_loss = l1_->compute_loss(weights);
    double total_loss = data_loss + reg_loss;
    
    EXPECT_GT(total_loss, data_loss);
}

TEST_F(LossRegularizerIntegrationTest, MSEWithElasticNetRegularization) {
    VectorXd y_true(3);
    VectorXd y_pred(3);
    y_true << 1, 2, 3;
    y_pred << 1.1, 1.9, 3.2;
    
    MatrixXd weights = MatrixXd::Random(5, 5);
    
    double data_loss = loss_->compute(y_true, y_pred);
    double reg_loss = elastic_->compute_loss(weights);
    double total_loss = data_loss + reg_loss;
    
    EXPECT_GT(total_loss, data_loss);
}

TEST_F(LossRegularizerIntegrationTest, GradientWithL2Regularization) {
    MatrixXd weights = MatrixXd::Random(10, 10);
    MatrixXd data_grad = MatrixXd::Random(10, 10);
    
    MatrixXd reg_grad = l2_->compute_gradient(weights);
    MatrixXd total_grad = data_grad + reg_grad;
    
    // L2 gradient should be λ * w
    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            EXPECT_NEAR(reg_grad(i, j), 0.01 * weights(i, j), 1e-6);
        }
    }
}

TEST_F(LossRegularizerIntegrationTest, RegularizerClone) {
    auto cloned = l2_->clone();
    
    EXPECT_NE(cloned.get(), l2_.get());
    EXPECT_EQ(cloned->get_type(), l2_->get_type());
    EXPECT_EQ(cloned->get_strength(), l2_->get_strength());
}