// tests/performance/benchmark_logistic.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "models/logistic_regression.h"
#include "utils/math_utils.h"

using namespace models;
using namespace utils;
using namespace std::chrono;

// Genera dataset di classificazione binaria
void generate_classification_dataset(int n_samples, int n_features,
                                     Eigen::MatrixXd& X, Eigen::VectorXd& y) {
    X.resize(n_samples, n_features);
    y.resize(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    
    // Genera pesi casuali
    Eigen::VectorXd true_weights(n_features);
    for (int i = 0; i < n_features; ++i) {
        true_weights(i) = dist(gen);
    }
    
    // Genera dati linearmente separabili con un po' di rumore
    for (int i = 0; i < n_samples; ++i) {
        double z = 0.0;
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = dist(gen);
            z += X(i, j) * true_weights(j);
        }
        // Aggiungi rumore
        z += std::normal_distribution<>(0, 0.5)(gen);
        y(i) = (z > 0) ? 1.0 : 0.0;
    }
}

void benchmark_logistic_sample_scaling() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "LOGISTIC REGRESSION - Scaling con numero di campioni" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Campioni" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "Accuracy" 
              << std::setw(15) << "Iterazioni" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<int> sample_sizes = {100, 1000, 10000, 50000, 100000};
    int n_features = 10;
    
    for (int n_samples : sample_sizes) {
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        generate_classification_dataset(n_samples, n_features, X, y);
        
        auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
        
        auto start = high_resolution_clock::now();
        
        LogisticRegression model(0.01, 1000, 0.001, 1e-4, false);
        model.fit(splits[0].first, splits[0].second);
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double accuracy = model.score(splits[1].first, splits[1].second);
        int iterations = model.get_n_iter();  // <-- CAMBIATO QUI
        
        std::cout << std::left << std::setw(15) << n_samples 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << iterations << std::endl;
    }
}

void benchmark_logistic_feature_scaling() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "LOGISTIC REGRESSION - Scaling con numero di feature" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Features" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "Accuracy" 
              << std::setw(15) << "Iterazioni" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<int> feature_sizes = {5, 10, 50, 100, 500};
    int n_samples = 10000;
    
    for (int n_features : feature_sizes) {
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        generate_classification_dataset(n_samples, n_features, X, y);
        
        auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
        
        auto start = high_resolution_clock::now();
        
        LogisticRegression model(0.01, 1000, 0.001, 1e-4, false);
        model.fit(splits[0].first, splits[0].second);
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double accuracy = model.score(splits[1].first, splits[1].second);
        int iterations = model.get_n_iter();  // <-- CAMBIATO QUI
        
        std::cout << std::left << std::setw(15) << n_features 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << iterations << std::endl;
    }
}

void benchmark_logistic_regularization() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "LOGISTIC REGRESSION - Impatto della regolarizzazione" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Lambda" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "Accuracy" 
              << std::setw(15) << "Iterazioni" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    int n_samples = 10000;
    int n_features = 50;
    
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    generate_classification_dataset(n_samples, n_features, X, y);
    
    auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
    
    std::vector<double> lambdas = {0.0, 0.001, 0.01, 0.1, 1.0};
    
    for (double lambda : lambdas) {
        auto start = high_resolution_clock::now();
        
        LogisticRegression model(0.01, 1000, lambda, 1e-4, false);
        model.fit(splits[0].first, splits[0].second);
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double accuracy = model.score(splits[1].first, splits[1].second);
        int iterations = model.get_n_iter();  // <-- CAMBIATO QUI
        
        std::cout << std::left << std::setw(15) << lambda 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << iterations << std::endl;
    }
}

int main() {
    std::cout << "\n";
    std::cout << "================================================" << std::endl;
    std::cout << "     BENCHMARK LOGISTIC REGRESSION - v1.0       " << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Data: " << __DATE__ << " " << __TIME__ << std::endl;
    
    benchmark_logistic_sample_scaling();
    benchmark_logistic_feature_scaling();
    benchmark_logistic_regularization();
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Benchmark completato!" << std::endl;
    
    return 0;
}