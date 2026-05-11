// tests/cpp/unit/models/test_linear_regression.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "models/linear_regression.h"
#include <cmath>

using namespace models;
using namespace Eigen;

class LinearRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Crea dati lineari: y = 2*x1 + 3*x2 + 5
        X.resize(100, 2);
        y.resize(100);
        for (int i = 0; i < 100; ++i) {
            X(i, 0) = i * 0.1;
            X(i, 1) = i * 0.2;
            y(i) = 2.0 * X(i, 0) + 3.0 * X(i, 1) + 5.0 + (rand() % 100) / 1000.0;  // piccolo rumore
        }
    }
    
    MatrixXd X;
    VectorXd y;
};

TEST_F(LinearRegressionTest, DefaultConstructor) {
    LinearRegression lr;
    EXPECT_EQ(lr.get_model_type(), "LinearRegression");
}

TEST_F(LinearRegressionTest, GradientDescentFit) {
    LinearRegression lr(0.01, 1000, 0.0, LinearRegression::GRADIENT_DESCENT);
    lr.fit(X, y);
    
    VectorXd pred = lr.predict(X);
    double r2 = lr.score(X, y);
    
    EXPECT_GT(r2, 0.95);
    EXPECT_NEAR(lr.intercept(), 5.0, 0.5);
    EXPECT_NEAR(lr.coefficients()(0), 2.0, 0.5);
    EXPECT_NEAR(lr.coefficients()(1), 3.0, 0.5);
}

TEST_F(LinearRegressionTest, NormalEquationFit) {
    LinearRegression lr(0.01, 1000, 0.0, LinearRegression::NORMAL_EQUATION);
    lr.fit(X, y);
    
    double r2 = lr.score(X, y);
    EXPECT_GT(r2, 0.95);
}

TEST_F(LinearRegressionTest, SVDFit) {
    LinearRegression lr(0.01, 1000, 0.0, LinearRegression::SVD);
    lr.fit(X, y);
    
    double r2 = lr.score(X, y);
    EXPECT_GT(r2, 0.95);
}

TEST_F(LinearRegressionTest, PredictSingleSample) {
    LinearRegression lr;
    lr.fit(X, y);
    
    VectorXd x_single(2);
    x_single << 1.0, 2.0;
    double pred = lr.predict(x_single);
    
    EXPECT_FALSE(std::isnan(pred));
    EXPECT_FALSE(std::isinf(pred));
}

TEST_F(LinearRegressionTest, MSEAndMAE) {
    LinearRegression lr;
    lr.fit(X, y);
    
    double mse = lr.mse(X, y);
    double mae = lr.mae(X, y);
    double r2 = lr.r2_score(X, y);
    
    EXPECT_GT(mse, 0);
    EXPECT_GT(mae, 0);
    EXPECT_GT(r2, 0.95);
    EXPECT_LT(mse, 1.0);
}

TEST_F(LinearRegressionTest, RegularizationEffect) {
    // Con regolarizzazione L2
    LinearRegression lr_reg(0.01, 1000, 0.1);
    lr_reg.fit(X, y);
    
    // Senza regolarizzazione
    LinearRegression lr_no_reg(0.01, 1000, 0.0);
    lr_no_reg.fit(X, y);
    
    // Entrambi dovrebbero funzionare
    EXPECT_GT(lr_reg.score(X, y), 0.9);
    EXPECT_GT(lr_no_reg.score(X, y), 0.9);
}

TEST_F(LinearRegressionTest, CrossValidation) {
    // Crea un dataset dedicato per cross-validation
    MatrixXd X_large(500, 2);
    VectorXd y_large(500);
    for (int i = 0; i < 500; ++i) {
        X_large(i, 0) = i * 0.02;
        X_large(i, 1) = i * 0.03;
        y_large(i) = 2.0 * X_large(i, 0) + 3.0 * X_large(i, 1) + 5.0;
    }
    
    VectorXd scores = LinearRegression::cross_val_score(X_large, y_large, 5);
    
    EXPECT_EQ(scores.size(), 5);
    for (int i = 0; i < scores.size(); ++i) {
        EXPECT_GT(scores(i), 0.9);
    }
}

TEST_F(LinearRegressionTest, Serialization) {
    LinearRegression lr;
    lr.fit(X, y);
    
    std::string filename = "test_lr_model.bin";
    lr.save(filename);
    
    LinearRegression loaded_lr;
    loaded_lr.load(filename);
    
    VectorXd pred_original = lr.predict(X);
    VectorXd pred_loaded = loaded_lr.predict(X);
    
    EXPECT_TRUE(pred_original.isApprox(pred_loaded, 1e-6));
    
    std::remove(filename.c_str());
}