#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "models/linear_regression.h"
#include "utils/math_utils.h"

using namespace models;
using namespace utils;
using namespace std::chrono;

// Funzione per generare dati sintetici
void generate_dataset(int n_samples, int n_features, 
                     Eigen::MatrixXd& X, Eigen::VectorXd& y) {
    X.resize(n_samples, n_features);
    y.resize(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    std::normal_distribution<> noise(0.0, 0.1);
    
    // Genera pesi casuali
    Eigen::VectorXd true_weights(n_features);
    for (int i = 0; i < n_features; ++i) {
        true_weights(i) = dist(gen);
    }
    
    // Genera dati: y = X * true_weights + rumore
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = dist(gen);
        }
        y(i) = X.row(i).dot(true_weights) + noise(gen);
    }
}

// Benchmark 1: Scaling con numero di campioni
void benchmark_sample_scaling() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "BENCHMARK 1: Scaling con numero di campioni" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Campioni" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "R²" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::vector<int> sample_sizes = {100, 1000, 10000, 100000, 500000};
    int n_features = 10;
    
    for (int n_samples : sample_sizes) {
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        generate_dataset(n_samples, n_features, X, y);
        
        auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
        
        auto start = high_resolution_clock::now();
        
        LinearRegression model(0.01, 1000, 0.0, LinearRegression::GRADIENT_DESCENT);
        model.fit(splits[0].first, splits[0].second);
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double score = model.score(splits[1].first, splits[1].second);
        
        std::cout << std::left << std::setw(15) << n_samples 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << score << std::endl;
    }
}

// Benchmark 2: Scaling con numero di feature
void benchmark_feature_scaling() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "BENCHMARK 2: Scaling con numero di feature" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Features" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "R²" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::vector<int> feature_sizes = {5, 10, 50, 100, 500};
    int n_samples = 10000;
    
    for (int n_features : feature_sizes) {
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        generate_dataset(n_samples, n_features, X, y);
        
        auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
        
        auto start = high_resolution_clock::now();
        
        LinearRegression model(0.01, 1000, 0.0, LinearRegression::GRADIENT_DESCENT);
        model.fit(splits[0].first, splits[0].second);
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double score = model.score(splits[1].first, splits[1].second);
        
        std::cout << std::left << std::setw(15) << n_features 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << score << std::endl;
    }
}

// Benchmark 3: Confronto tra solver
void benchmark_solver_comparison() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "BENCHMARK 3: Confronto tra solver" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::left << std::setw(20) << "Solver" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "R²" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    int n_samples = 10000;
    int n_features = 10;
    
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    generate_dataset(n_samples, n_features, X, y);
    
    auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
    
    // Gradient Descent
    {
        auto start = high_resolution_clock::now();
        LinearRegression model(0.01, 1000, 0.0, LinearRegression::GRADIENT_DESCENT);
        model.fit(splits[0].first, splits[0].second);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        double score = model.score(splits[1].first, splits[1].second);
        std::cout << std::left << std::setw(20) << "Gradient Descent" 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << score << std::endl;
    }
    
    // Normal Equation
    {
        auto start = high_resolution_clock::now();
        LinearRegression model(0.01, 1000, 0.0, LinearRegression::NORMAL_EQUATION);
        model.fit(splits[0].first, splits[0].second);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        double score = model.score(splits[1].first, splits[1].second);
        std::cout << std::left << std::setw(20) << "Normal Equation" 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << score << std::endl;
    }
    
    // SVD
    {
        auto start = high_resolution_clock::now();
        LinearRegression model(0.01, 1000, 0.0, LinearRegression::SVD);
        model.fit(splits[0].first, splits[0].second);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        double score = model.score(splits[1].first, splits[1].second);
        std::cout << std::left << std::setw(20) << "SVD" 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << score << std::endl;
    }
}

// Benchmark 4: Impatto della regolarizzazione
void benchmark_regularization() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "BENCHMARK 4: Impatto della regolarizzazione" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Lambda" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "R²" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    int n_samples = 10000;
    int n_features = 50;  // Più feature per vedere l'effetto
    
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    generate_dataset(n_samples, n_features, X, y);
    
    auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
    
    std::vector<double> lambdas = {0.0, 0.001, 0.01, 0.1, 1.0};
    
    for (double lambda : lambdas) {
        auto start = high_resolution_clock::now();
        
        LinearRegression model(0.01, 1000, lambda, LinearRegression::GRADIENT_DESCENT);
        model.fit(splits[0].first, splits[0].second);
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double score = model.score(splits[1].first, splits[1].second);
        
        std::cout << std::left << std::setw(15) << lambda 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << score << std::endl;
    }
}

int main() {
    std::cout << "\n";
    std::cout << "================================================" << std::endl;
    std::cout << "      BENCHMARK LINEAR REGRESSION - v1.0        " << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Data: " << __DATE__ << " " << __TIME__ << std::endl;
    
    benchmark_sample_scaling();
    benchmark_feature_scaling();
    benchmark_solver_comparison();
    benchmark_regularization();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Benchmark completato!" << std::endl;
    
    return 0;
}