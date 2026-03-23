/**
 * @file linear_regression_example.cpp
 * @brief Example of Linear Regression usage
 * 
 * Demonstrates:
 * - Generating synthetic data
 * - Training a linear regression model
 * - Making predictions
 * - Evaluating performance
 */

#include <iostream>
#include <iomanip>
#include "models/linear_regression.h"
#include "utils/math_utils.h"

using namespace models;
using namespace utils;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  LINEAR REGRESSION EXAMPLE" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Step 1: Generate synthetic data
    // y = 2*x1 + 3*x2 + 5 + noise
    std::cout << "1. Generating synthetic data..." << std::endl;
    
    int n_samples = 1000;
    int n_features = 2;
    
    Eigen::MatrixXd X(n_samples, n_features);
    Eigen::VectorXd y(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    std::normal_distribution<> noise(0.0, 0.1);
    
    for (int i = 0; i < n_samples; ++i) {
        X(i, 0) = dist(gen);
        X(i, 1) = dist(gen);
        y(i) = 2.0 * X(i, 0) + 3.0 * X(i, 1) + 5.0 + noise(gen);
    }
    
    std::cout << "   Generated " << n_samples << " samples with 2 features" << std::endl;
    std::cout << "   True relationship: y = 2*x1 + 3*x2 + 5 + noise\n" << std::endl;
    
    // Step 2: Split data into train and test
    std::cout << "2. Splitting data (80% train, 20% test)..." << std::endl;
    auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
    const auto& train = splits[0];
    const auto& test = splits[1];
    
    std::cout << "   Training samples: " << train.first.rows() << std::endl;
    std::cout << "   Test samples: " << test.first.rows() << "\n" << std::endl;
    
    // Step 3: Train model with different solvers
    std::cout << "3. Training models..." << std::endl;
    
    // Gradient Descent
    LinearRegression gd_model(0.01, 1000, 0.0, LinearRegression::GRADIENT_DESCENT);
    gd_model.fit(train.first, train.second);
    
    // Normal Equation
    LinearRegression ne_model(0.01, 1000, 0.0, LinearRegression::NORMAL_EQUATION);
    ne_model.fit(train.first, train.second);
    
    // SVD
    LinearRegression svd_model(0.01, 1000, 0.0, LinearRegression::SVD);
    svd_model.fit(train.first, train.second);
    
    // Step 4: Evaluate models
    std::cout << "\n4. Evaluation on test set:" << std::endl;
    std::cout << "   " << std::setw(20) << "Solver"
              << std::setw(15) << "R² Score"
              << std::setw(15) << "MSE"
              << std::setw(15) << "MAE" << std::endl;
    std::cout << "   " << std::string(65, '-') << std::endl;
    
    auto evaluate = [&](const LinearRegression& model, const std::string& name) {
        double r2 = model.score(test.first, test.second);
        double mse = model.mse(test.first, test.second);
        double mae = model.mae(test.first, test.second);
        
        std::cout << "   " << std::setw(20) << name
                  << std::setw(15) << std::fixed << std::setprecision(4) << r2
                  << std::setw(15) << mse
                  << std::setw(15) << mae << std::endl;
    };
    
    evaluate(gd_model, "Gradient Descent");
    evaluate(ne_model, "Normal Equation");
    evaluate(svd_model, "SVD");
    
    // Step 5: Print coefficients
    std::cout << "\n5. Learned coefficients (Normal Equation):" << std::endl;
    Eigen::VectorXd coef = ne_model.coefficients();
    double intercept = ne_model.intercept();
    
    std::cout << "   Intercept: " << intercept << std::endl;
    for (int i = 0; i < coef.size(); ++i) {
        std::cout << "   Coefficient x" << i+1 << ": " << coef(i) << std::endl;
    }
    std::cout << "   True: intercept = 5.00, coef1 = 2.00, coef2 = 3.00\n" << std::endl;
    
    // Step 6: Make a single prediction
    std::cout << "6. Making a single prediction:" << std::endl;
    Eigen::VectorXd x_new(2);
    x_new << 1.5, 2.5;
    double y_pred = ne_model.predict(x_new);
    double y_true = 2.0 * 1.5 + 3.0 * 2.5 + 5.0;
    
    std::cout << "   Input: x1 = 1.5, x2 = 2.5" << std::endl;
    std::cout << "   Predicted: " << y_pred << std::endl;
    std::cout << "   True value: " << y_true << std::endl;
    std::cout << "   Error: " << std::abs(y_pred - y_true) << "\n" << std::endl;
    
    // Step 7: Show training history
    std::cout << "7. Training history (Gradient Descent):" << std::endl;
    const auto& history = gd_model.cost_history();
    std::cout << "   First 5 iterations:" << std::endl;
    for (size_t i = 0; i < std::min(5UL, history.size()); ++i) {
        std::cout << "     Iteration " << i << ": cost = " << history[i] << std::endl;
    }
    std::cout << "   Final cost: " << history.back() << std::endl;
    
    std::cout << "\n✅ Linear Regression example completed!\n" << std::endl;
    
    return 0;
}