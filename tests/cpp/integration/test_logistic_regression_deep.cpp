#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "models/logistic_regression.h"
#include "utils/math_utils.h"
#include "exceptions/ml_exception.h"

using namespace models;
using namespace utils;
//using namespace ml;
using namespace testing;

class LogisticRegressionIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Crea dati linearmente separabili
        X.resize(100, 2);
        y.resize(100);
        
        std::random_device rd;
        std::mt19937 gen(42);
        
        for (int i = 0; i < 100; ++i) {
            if (i < 50) {
                X(i, 0) = 1.0 + std::normal_distribution<>(0, 0.2)(gen);
                X(i, 1) = 1.0 + std::normal_distribution<>(0, 0.2)(gen);
                y(i) = 0;
            } else {
                X(i, 0) = 3.0 + std::normal_distribution<>(0, 0.2)(gen);
                X(i, 1) = 3.0 + std::normal_distribution<>(0, 0.2)(gen);
                y(i) = 1;
            }
        }
    }
    
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
};

TEST_F(LogisticRegressionIntegrationTest, TrainWithDefaultParams) {
    LogisticRegression model(0.1, 1000, 0.001);
    model.fit(X, y);
    
    Eigen::VectorXd y_pred_proba = model.predict(X);
    Eigen::VectorXi y_pred = model.predict_class(X);
    
    double acc = MathUtils::accuracy_score(y.cast<int>(), y_pred);
    EXPECT_GT(acc, 0.95);
    
    EXPECT_GT(model.coefficients().norm(), 0.1);
}

TEST_F(LogisticRegressionIntegrationTest, Regularization) {
    LogisticRegression model_no_reg(0.1, 500, 0.0);
    model_no_reg.fit(X, y);
    
    LogisticRegression model_reg(0.1, 500, 1.0);
    model_reg.fit(X, y);
    
    EXPECT_GT(model_no_reg.score(X, y), 0.9);
    EXPECT_GT(model_reg.score(X, y), 0.9);
}

TEST_F(LogisticRegressionIntegrationTest, ConfusionMatrix) {
    LogisticRegression model(0.1, 500, 0.001);
    model.fit(X, y);
    
    Eigen::MatrixXd cm = model.confusion_matrix(X, y);
    EXPECT_EQ(cm.rows(), 2);
    EXPECT_EQ(cm.cols(), 2);
    
    EXPECT_GT(cm(0,0), 40);
    EXPECT_GT(cm(1,1), 40);
}

TEST_F(LogisticRegressionIntegrationTest, PrecisionRecallF1) {
    LogisticRegression model(0.1, 500, 0.001);
    model.fit(X, y);
    
    Eigen::Vector3d metrics = model.precision_recall_f1(X, y);
    EXPECT_GT(metrics(0), 0.8);
    EXPECT_GT(metrics(1), 0.8);
    EXPECT_GT(metrics(2), 0.8);
}

TEST_F(LogisticRegressionIntegrationTest, PredictProba) {
    LogisticRegression model(0.1, 500, 0.001);
    model.fit(X, y);
    
    Eigen::VectorXd proba = model.predict(X);
    for (int i = 0; i < proba.size(); ++i) {
        EXPECT_GE(proba(i), 0.0);
        EXPECT_LE(proba(i), 1.0);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}