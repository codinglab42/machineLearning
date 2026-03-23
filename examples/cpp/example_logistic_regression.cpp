/**
 * @file logistic_regression_example.cpp
 * @brief Example of Logistic Regression usage
 * 
 * Demonstrates:
 * - Binary classification
 * - Training with regularization
 * - Confusion matrix and metrics
 */

#include <iostream>
#include <iomanip>
#include "models/logistic_regression.h"
#include "utils/math_utils.h"

using namespace models;
using namespace utils;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  LOGISTIC REGRESSION EXAMPLE" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Step 1: Generate synthetic binary classification data
    std::cout << "1. Generating synthetic classification data..." << std::endl;
    
    int n_samples = 1000;
    int n_features = 2;
    
    Eigen::MatrixXd X(n_samples, n_features);
    Eigen::VectorXd y(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    
    // Class 0: centered at (1, 1)
    // Class 1: centered at (3, 3)
    for (int i = 0; i < n_samples; ++i) {
        if (i < n_samples / 2) {
            X(i, 0) = 1.0 + dist(gen);
            X(i, 1) = 1.0 + dist(gen);
            y(i) = 0;
        } else {
            X(i, 0) = 3.0 + dist(gen);
            X(i, 1) = 3.0 + dist(gen);
            y(i) = 1;
        }
    }
    
    std::cout << "   Generated " << n_samples << " samples with 2 features" << std::endl;
    std::cout << "   Class 0: centered at (1,1)" << std::endl;
    std::cout << "   Class 1: centered at (3,3)\n" << std::endl;
    
    // Step 2: Split data
    std::cout << "2. Splitting data (80% train, 20% test)..." << std::endl;
    auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
    const auto& train = splits[0];
    const auto& test = splits[1];
    
    std::cout << "   Training samples: " << train.first.rows() << std::endl;
    std::cout << "   Test samples: " << test.first.rows() << "\n" << std::endl;
    
    // Step 3: Train model with different regularization
    std::cout << "3. Training models with different regularization..." << std::endl;
    
    std::vector<double> lambdas = {0.0, 0.001, 0.01, 0.1};
    
    std::cout << "   " << std::setw(15) << "Lambda"
              << std::setw(15) << "Accuracy"
              << std::setw(15) << "Precision"
              << std::setw(15) << "Recall"
              << std::setw(15) << "F1" << std::endl;
    std::cout << "   " << std::string(75, '-') << std::endl;
    
    for (double lambda : lambdas) {
        LogisticRegression model(0.1, 1000, lambda);
        model.fit(train.first, train.second);
        
        double accuracy = model.score(test.first, test.second);
        Eigen::Vector3d metrics = model.precision_recall_f1(test.first, test.second);
        
        std::cout << "   " << std::setw(15) << lambda
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << metrics(0)
                  << std::setw(15) << metrics(1)
                  << std::setw(15) << metrics(2) << std::endl;
    }
    
    // Step 4: Train best model and show details
    std::cout << "\n4. Training best model (lambda = 0.001)..." << std::endl;
    LogisticRegression model(0.1, 1000, 0.001);
    model.fit(train.first, train.second);
    
    // Step 5: Confusion matrix
    std::cout << "\n5. Confusion Matrix on test set:" << std::endl;
    Eigen::MatrixXd cm = model.confusion_matrix(test.first, test.second);
    
    std::cout << "   " << std::setw(10) << ""
              << std::setw(10) << "Pred 0"
              << std::setw(10) << "Pred 1" << std::endl;
    std::cout << "   " << std::string(30, '-') << std::endl;
    std::cout << "   " << std::setw(10) << "Actual 0"
              << std::setw(10) << cm(0,0)
              << std::setw(10) << cm(0,1) << std::endl;
    std::cout << "   " << std::setw(10) << "Actual 1"
              << std::setw(10) << cm(1,0)
              << std::setw(10) << cm(1,1) << std::endl;
    
    // Step 6: Coefficients
    std::cout << "\n6. Learned coefficients:" << std::endl;
    Eigen::VectorXd coef = model.coefficients();
    double intercept = model.intercept();
    
    std::cout << "   Intercept: " << intercept << std::endl;
    for (int i = 0; i < coef.size(); ++i) {
        std::cout << "   Coefficient x" << i+1 << ": " << coef(i) << std::endl;
    }
    
    // Step 7: Make predictions with different thresholds
    std::cout << "\n7. Making predictions with different thresholds:" << std::endl;

    // CORRETTO: usa una matrice 2D (1 riga, 2 colonne)
    Eigen::MatrixXd x_new(1, 2);
    x_new << 2.0, 2.0;
    double proba = model.predict(x_new)(0);

    std::cout << "   Input: x1 = 2.0, x2 = 2.0" << std::endl;
    std::cout << "   Probability: " << proba << std::endl;
    std::cout << "   Class (threshold=0.5): " << (proba > 0.5 ? 1 : 0) << std::endl;
    std::cout << "   Class (threshold=0.7): " << (proba > 0.7 ? 1 : 0) << std::endl;
    std::cout << "   Class (threshold=0.3): " << (proba > 0.3 ? 1 : 0) << std::endl;
    
    return 0;
}