#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "models/linear_regression.h"
#include "utils/math_utils.h"
#include "utils/standard_scaler.h"
#include "exceptions/ml_exception.h"

using namespace models;
using namespace utils;
using namespace ml;
using namespace testing;

class LinearRegressionIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Crea dati sintetici: y = 2*x1 + 3*x2 + 5
        X.resize(100, 2);
        y.resize(100);
        
        std::random_device rd;
        std::mt19937 gen(42);
        std::normal_distribution<> noise(0.0, 0.1);
        
        for (int i = 0; i < 100; ++i) {
            double x1 = i / 10.0;
            double x2 = i / 5.0;
            X(i, 0) = x1;
            X(i, 1) = x2;
            y(i) = 2*x1 + 3*x2 + 5 + noise(gen);
        }
    }
    
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
};

TEST_F(LinearRegressionIntegrationTest, TrainWithScaler) {
    Eigen::VectorXd mean, std;
    Eigen::MatrixXd X_scaled = X;
    MathUtils::standardize_features(X_scaled, mean, std);
    
    LinearRegression model(0.01, 1000, 0.0, LinearRegression::GRADIENT_DESCENT);
    model.fit(X_scaled, y);
    
    Eigen::VectorXd y_pred = model.predict(X_scaled);
    double mse = (y - y_pred).array().square().mean();
    EXPECT_LT(mse, 0.1);
    
    double r2 = model.score(X_scaled, y);
    EXPECT_GT(r2, 0.95);
}

TEST_F(LinearRegressionIntegrationTest, CompareSolvers) {
    LinearRegression gd_model(0.01, 1000, 0.0, LinearRegression::GRADIENT_DESCENT);
    gd_model.fit(X, y);
    
    LinearRegression ne_model(0.01, 1000, 0.0, LinearRegression::NORMAL_EQUATION);
    ne_model.fit(X, y);
    
    LinearRegression svd_model(0.01, 1000, 0.0, LinearRegression::SVD);
    svd_model.fit(X, y);
    
    double gd_score = gd_model.score(X, y);
    double ne_score = ne_model.score(X, y);
    double svd_score = svd_model.score(X, y);
    
    EXPECT_GT(gd_score, 0.9);
    EXPECT_GT(ne_score, 0.9);
    EXPECT_GT(svd_score, 0.9);
}

TEST_F(LinearRegressionIntegrationTest, PredictSingle) {
    LinearRegression model(0.01, 1000, 0.0, LinearRegression::GRADIENT_DESCENT);
    model.fit(X, y);
    
    Eigen::VectorXd x_sample(2);
    x_sample << 2.5, 3.5;
    
    double pred = model.predict(x_sample);
    EXPECT_GT(pred, 0);
}

TEST_F(LinearRegressionIntegrationTest, Score) {
    LinearRegression model(0.01, 1000, 0.0, LinearRegression::GRADIENT_DESCENT);
    model.fit(X, y);
    
    double r2 = model.score(X, y);
    EXPECT_GT(r2, 0.9);
    
    double mse = model.mse(X, y);
    EXPECT_LT(mse, 0.2);
    
    double mae = model.mae(X, y);
    EXPECT_LT(mae, 0.4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}