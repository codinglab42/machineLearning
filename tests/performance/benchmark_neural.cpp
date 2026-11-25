// tests/performance/benchmark_neural.cpp
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include "models/neural_network.h"
#include "utils/math_utils.h"

using namespace models;
using namespace utils;
using namespace std::chrono;

// Genera dataset di classificazione binaria
void generate_binary_dataset(int n_samples, int n_features,
                            Eigen::MatrixXd& X, Eigen::VectorXd& y) {
    X.resize(n_samples, n_features);
    y.resize(n_samples);
    
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    
    // Genera dati non linearmente separabili
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = dist(gen);
        }
        // The odd and interaction terms keep both classes represented as feature count changes.
        double z = std::sin(X(i, 0)) + 0.5 * X(i, 1) * X(i, 2)
                 + 0.3 * X(i, 3) + std::normal_distribution<>(0, 0.3)(gen);
        y(i) = (z > 0) ? 1.0 : 0.0;
    }
}

void validate_classes(const std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>& splits) {
    for (const auto& split : splits) {
        double fraction = split.second.mean();
        if (fraction < 0.2 || fraction > 0.8) {
            throw std::runtime_error("Neural benchmark split has insufficient class balance");
        }
    }
}

// Funzione per calcolare accuracy manualmente
double compute_accuracy(const NeuralNetwork& network, 
                        const Eigen::MatrixXd& X, 
                        const Eigen::VectorXd& y) {
    Eigen::VectorXd y_pred = network.predict(X);
    Eigen::VectorXi y_pred_int = (y_pred.array() > 0.5).cast<int>();
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) {
        if (y_pred_int(i) == static_cast<int>(y(i))) correct++;
    }
    return static_cast<double>(correct) / y.size();
}

// Benchmark 1: Scaling con numero di campioni
void benchmark_neural_sample_scaling() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "NEURAL NETWORK - Scaling con numero di campioni" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Campioni" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "Accuracy" 
              << std::setw(15) << "Loss"
              << std::setw(15) << "Baseline" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<int> sample_sizes = {100, 500, 1000, 5000, 10000};
    int n_features = 10;
    int epochs = 100;
    
    for (int n_samples : sample_sizes) {
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        generate_binary_dataset(n_samples, n_features, X, y);
        
        auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
        validate_classes(splits);
        
        NeuralNetwork network({n_features, 32, 16, 1}, "relu", "sigmoid",
                              OptimizerType::ADAM, 0.01);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(epochs);
        network.set_batch_size(32);
        network.set_verbose(false);
        
        auto start = steady_clock::now();
        network.fit(splits[0].first, splits[0].second);
        auto end = steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        double accuracy = compute_accuracy(network, splits[1].first, splits[1].second);
        auto [loss, val_loss, acc] = network.get_training_history();
        if (loss.empty()) throw std::runtime_error("Neural training history is empty");
        double final_loss = loss.back();
        
        std::cout << std::left << std::setw(15) << n_samples 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << std::fixed << std::setprecision(4) << final_loss
                  << std::setw(15) << std::max(splits[1].second.mean(), 1.0 - splits[1].second.mean()) << std::endl;
    }
}

// Benchmark 2: Scaling con numero di feature
void benchmark_neural_feature_scaling() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "NEURAL NETWORK - Scaling con numero di feature" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Features" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "Accuracy" 
              << std::setw(15) << "Loss"
              << std::setw(15) << "Baseline" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<int> feature_sizes = {5, 10, 20, 50, 100};
    int n_samples = 5000;
    int epochs = 100;
    
    for (int n_features : feature_sizes) {
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        generate_binary_dataset(n_samples, n_features, X, y);
        
        auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
        validate_classes(splits);
        
        NeuralNetwork network({n_features, 64, 32, 1}, "relu", "sigmoid",
                              OptimizerType::ADAM, 0.01);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(epochs);
        network.set_batch_size(32);
        network.set_verbose(false);
        
        auto start = steady_clock::now();
        network.fit(splits[0].first, splits[0].second);
        auto end = steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        double accuracy = compute_accuracy(network, splits[1].first, splits[1].second);
        auto [loss, val_loss, acc] = network.get_training_history();
        if (loss.empty()) throw std::runtime_error("Neural training history is empty");
        double final_loss = loss.back();
        
        std::cout << std::left << std::setw(15) << n_features 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << std::fixed << std::setprecision(4) << final_loss
                  << std::setw(15) << std::max(splits[1].second.mean(), 1.0 - splits[1].second.mean()) << std::endl;
    }
}

// Benchmark 3: Scaling con numero di neuroni
void benchmark_neuron_scaling() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "NEURAL NETWORK - Scaling con numero di neuroni" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::left << std::setw(20) << "Architettura" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "Accuracy" 
              << std::setw(15) << "Parametri"
              << std::setw(15) << "Baseline" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<std::vector<int>> architectures = {
        {10, 16, 1},
        {10, 32, 1},
        {10, 64, 1},
        {10, 128, 1},
        {10, 256, 1}
    };
    
    int n_samples = 5000;
    int epochs = 50;
    
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    generate_binary_dataset(n_samples, 10, X, y);
    
    auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
    validate_classes(splits);
    
    for (const auto& arch : architectures) {
        NeuralNetwork network(arch, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(epochs);
        network.set_batch_size(32);
        network.set_verbose(false);
        
        auto start = steady_clock::now();
        network.fit(splits[0].first, splits[0].second);
        auto end = steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        double accuracy = compute_accuracy(network, splits[1].first, splits[1].second);
        int params = network.get_num_parameters();
        
        std::string arch_str = std::to_string(arch[0]) + "->" + 
                               std::to_string(arch[1]) + "->" + 
                               std::to_string(arch[2]);
        
        std::cout << std::left << std::setw(20) << arch_str
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << params
                  << std::setw(15) << std::max(splits[1].second.mean(), 1.0 - splits[1].second.mean()) << std::endl;
    }
}

// Benchmark 4: Confronto ottimizzatori
void benchmark_optimizers() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "NEURAL NETWORK - Confronto ottimizzatori" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Ottimizzatore" 
              << std::setw(15) << "Tempo(ms)" 
              << std::setw(15) << "Accuracy" 
              << std::setw(15) << "Loss"
              << std::setw(15) << "Baseline" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    int n_samples = 5000;
    int n_features = 10;
    int epochs = 100;
    
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    generate_binary_dataset(n_samples, n_features, X, y);
    
    auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
    validate_classes(splits);
    const std::vector<std::pair<std::string, OptimizerType>> optimizers = {
        {"SGD", OptimizerType::SGD}, {"Momentum", OptimizerType::MOMENTUM},
        {"Adam", OptimizerType::ADAM}, {"RMSprop", OptimizerType::RMSPROP},
        {"Adagrad", OptimizerType::ADAGRAD}
    };
    for (const auto& optimizer : optimizers) {
        NeuralNetwork network({n_features, 32, 1}, "relu", "sigmoid", optimizer.second, 0.01);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(epochs);
        network.set_batch_size(32);
        network.set_verbose(false);
        
        auto start = steady_clock::now();
        network.fit(splits[0].first, splits[0].second);
        auto end = steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        double accuracy = compute_accuracy(network, splits[1].first, splits[1].second);
        auto [loss, val_loss, acc] = network.get_training_history();
        if (loss.empty()) throw std::runtime_error("Neural training history is empty");
        double final_loss = loss.back();
        
        std::cout << std::left << std::setw(15) << optimizer.first
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << std::fixed << std::setprecision(4) << final_loss
                  << std::setw(15) << std::max(splits[1].second.mean(), 1.0 - splits[1].second.mean()) << std::endl;
    }
}

void benchmark_regularizers() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "NEURAL NETWORK - Confronto regolarizzatori\n"
              << std::string(70, '=') << "\n"
              << std::left << std::setw(15) << "Regularizer" << std::setw(15) << "Tempo(ms)"
              << std::setw(15) << "Accuracy" << std::setw(15) << "Loss"
              << std::setw(15) << "Baseline" << "\n" << std::string(70, '-') << std::endl;
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    generate_binary_dataset(5000, 10, X, y);
    auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
    validate_classes(splits);
    const std::vector<std::pair<std::string, RegularizerType>> regularizers = {
        {"None", RegularizerType::NONE}, {"L1", RegularizerType::L1},
        {"L2", RegularizerType::L2}, {"ElasticNet", RegularizerType::ELASTIC_NET}
    };
    for (const auto& regularizer : regularizers) {
        NeuralNetwork network({10, 32, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01,
                              regularizer.second, 0.001);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(100);
        network.set_batch_size(32);
        auto start = steady_clock::now();
        network.fit(splits[0].first, splits[0].second);
        double time_ms = duration<double, std::milli>(steady_clock::now() - start).count();
        const auto& loss = network.get_loss_history();
        if (loss.empty()) throw std::runtime_error("Neural training history is empty");
        std::cout << std::left << std::setw(15) << regularizer.first << std::setw(15) << time_ms
                  << std::setw(15) << compute_accuracy(network, splits[1].first, splits[1].second)
                  << std::setw(15) << loss.back()
                  << std::setw(15) << std::max(splits[1].second.mean(), 1.0 - splits[1].second.mean()) << std::endl;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n";
    std::cout << "================================================" << std::endl;
    std::cout << "       BENCHMARK NEURAL NETWORK - v1.0          " << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Data: " << __DATE__ << " " << __TIME__ << std::endl;
    
    benchmark_neural_sample_scaling();
    benchmark_neural_feature_scaling();
    benchmark_neuron_scaling();
    benchmark_optimizers();
    benchmark_regularizers();
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Benchmark completato!" << std::endl;
    
    return 0;
}