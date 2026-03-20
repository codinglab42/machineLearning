// tests/performance/benchmark_neural.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
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
    
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    
    // Genera dati non linearmente separabili
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = dist(gen);
        }
        // Funzione non lineare: sin(x0) + cos(x1) + ...
        double z = 0.0;
        for (int j = 0; j < n_features; ++j) {
            z += std::sin(X(i, j)) + std::cos(X(i, j) * 0.5);
        }
        z += std::normal_distribution<>(0, 0.5)(gen);
        y(i) = (z > 0) ? 1.0 : 0.0;
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
              << std::setw(15) << "Loss" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<int> sample_sizes = {100, 500, 1000, 5000, 10000};
    int n_features = 10;
    int epochs = 100;
    
    for (int n_samples : sample_sizes) {
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        generate_binary_dataset(n_samples, n_features, X, y);
        
        auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
        
        NeuralNetwork network({n_features, 32, 16, 1}, "relu", "sigmoid",
                              OptimizerType::ADAM, 0.01);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(epochs);
        network.set_batch_size(32);
        network.set_verbose(false);
        
        auto start = high_resolution_clock::now();
        network.fit(splits[0].first, splits[0].second);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double accuracy = compute_accuracy(network, splits[1].first, splits[1].second);
        auto [loss, val_loss, acc] = network.get_training_history();
        double final_loss = loss.empty() ? 0.0 : loss.back();
        
        std::cout << std::left << std::setw(15) << n_samples 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << std::fixed << std::setprecision(4) << final_loss << std::endl;
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
              << std::setw(15) << "Loss" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<int> feature_sizes = {5, 10, 20, 50, 100};
    int n_samples = 5000;
    int epochs = 100;
    
    for (int n_features : feature_sizes) {
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        generate_binary_dataset(n_samples, n_features, X, y);
        
        auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
        
        NeuralNetwork network({n_features, 64, 32, 1}, "relu", "sigmoid",
                              OptimizerType::ADAM, 0.01);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(epochs);
        network.set_batch_size(32);
        network.set_verbose(false);
        
        auto start = high_resolution_clock::now();
        network.fit(splits[0].first, splits[0].second);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double accuracy = compute_accuracy(network, splits[1].first, splits[1].second);
        auto [loss, val_loss, acc] = network.get_training_history();
        double final_loss = loss.empty() ? 0.0 : loss.back();
        
        std::cout << std::left << std::setw(15) << n_features 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << std::fixed << std::setprecision(4) << final_loss << std::endl;
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
              << std::setw(15) << "Parametri" << std::endl;
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
    
    for (const auto& arch : architectures) {
        NeuralNetwork network(arch, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(epochs);
        network.set_batch_size(32);
        network.set_verbose(false);
        
        auto start = high_resolution_clock::now();
        network.fit(splits[0].first, splits[0].second);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double accuracy = compute_accuracy(network, splits[1].first, splits[1].second);
        int params = network.get_num_parameters();
        
        std::string arch_str = std::to_string(arch[0]) + "->" + 
                               std::to_string(arch[1]) + "->" + 
                               std::to_string(arch[2]);
        
        std::cout << std::left << std::setw(20) << arch_str
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << params << std::endl;
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
              << std::setw(15) << "Loss" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    int n_samples = 5000;
    int n_features = 10;
    int epochs = 100;
    
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    generate_binary_dataset(n_samples, n_features, X, y);
    
    auto splits = MathUtils::train_test_split(X, y, 0.2, 42);
    
    // SGD
    {
        NeuralNetwork network({n_features, 32, 1}, "relu", "sigmoid", OptimizerType::SGD, 0.01);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(epochs);
        network.set_batch_size(32);
        network.set_verbose(false);
        
        auto start = high_resolution_clock::now();
        network.fit(splits[0].first, splits[0].second);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double accuracy = compute_accuracy(network, splits[1].first, splits[1].second);
        auto [loss, val_loss, acc] = network.get_training_history();
        double final_loss = loss.empty() ? 0.0 : loss.back();
        
        std::cout << std::left << std::setw(15) << "SGD"
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << std::fixed << std::setprecision(4) << final_loss << std::endl;
    }
    
    // Momentum
    {
        NeuralNetwork network({n_features, 32, 1}, "relu", "sigmoid", OptimizerType::MOMENTUM, 0.01);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(epochs);
        network.set_batch_size(32);
        network.set_verbose(false);
        
        auto start = high_resolution_clock::now();
        network.fit(splits[0].first, splits[0].second);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double accuracy = compute_accuracy(network, splits[1].first, splits[1].second);
        auto [loss, val_loss, acc] = network.get_training_history();
        double final_loss = loss.empty() ? 0.0 : loss.back();
        
        std::cout << std::left << std::setw(15) << "Momentum"
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << std::fixed << std::setprecision(4) << final_loss << std::endl;
    }
    
    // Adam
    {
        NeuralNetwork network({n_features, 32, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(epochs);
        network.set_batch_size(32);
        network.set_verbose(false);
        
        auto start = high_resolution_clock::now();
        network.fit(splits[0].first, splits[0].second);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double accuracy = compute_accuracy(network, splits[1].first, splits[1].second);
        auto [loss, val_loss, acc] = network.get_training_history();
        double final_loss = loss.empty() ? 0.0 : loss.back();
        
        std::cout << std::left << std::setw(15) << "Adam"
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(15) << std::fixed << std::setprecision(4) << final_loss << std::endl;
    }
}

int main() {
    std::cout << "\n";
    std::cout << "================================================" << std::endl;
    std::cout << "       BENCHMARK NEURAL NETWORK - v1.0          " << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Data: " << __DATE__ << " " << __TIME__ << std::endl;
    
    benchmark_neural_sample_scaling();
    benchmark_neural_feature_scaling();
    benchmark_neuron_scaling();
    benchmark_optimizers();
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Benchmark completato!" << std::endl;
    
    return 0;
}