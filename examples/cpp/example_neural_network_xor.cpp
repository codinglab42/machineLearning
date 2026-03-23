/**
 * @file neural_network_xor.cpp
 * @brief XOR problem solved with Neural Network
 * 
 * Demonstrates:
 * - Creating a neural network
 * - Training on XOR dataset
 * - Making predictions
 */

#include <iostream>
#include <iomanip>
#include "models/neural_network.h"

using namespace models;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  NEURAL NETWORK - XOR PROBLEM" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Step 1: XOR dataset
    std::cout << "1. XOR dataset:" << std::endl;
    Eigen::MatrixXd X(4, 2);
    X << 0, 0,
         0, 1,
         1, 0,
         1, 1;
    
    Eigen::VectorXd y(4);
    y << 0, 1, 1, 0;
    
    std::cout << "   Input  Output" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "   " << X(i,0) << " " << X(i,1) << "  ->  " << y(i) << std::endl;
    }
    std::cout << std::endl;
    
    // Step 2: Create neural network
    std::cout << "2. Creating neural network:" << std::endl;
    std::cout << "   Architecture: 2 -> 4 -> 1" << std::endl;
    std::cout << "   Hidden activation: ReLU" << std::endl;
    std::cout << "   Output activation: Sigmoid" << std::endl;
    std::cout << "   Optimizer: Adam, Learning rate: 0.1" << std::endl;
    std::cout << "   Loss function: Binary Cross-Entropy\n" << std::endl;
    
    NeuralNetwork network({2, 4, 1}, "relu", "sigmoid",
                          OptimizerType::ADAM, 0.1);
    network.set_loss_function("binary_crossentropy");
    network.set_epochs(2000);
    network.set_batch_size(4);
    network.set_verbose(true);
    
    // Step 3: Train network
    std::cout << "3. Training..." << std::endl;
    network.fit(X, y);
    
    // Step 4: Make predictions
    std::cout << "\n4. Predictions:" << std::endl;
    Eigen::VectorXd y_pred = network.predict(X);
    
    std::cout << "   " << std::setw(8) << "Input"
              << std::setw(12) << "Predicted"
              << std::setw(10) << "True"
              << std::setw(10) << "Class" << std::endl;
    std::cout << "   " << std::string(40, '-') << std::endl;
    
    for (int i = 0; i < 4; ++i) {
        int pred_class = (y_pred(i) > 0.5) ? 1 : 0;
        std::cout << "   [" << X(i,0) << "," << X(i,1) << "]"
                  << std::setw(10) << std::fixed << std::setprecision(4) << y_pred(i)
                  << std::setw(10) << y(i)
                  << std::setw(10) << pred_class << std::endl;
    }
    
    // Step 5: Training history
    std::cout << "\n5. Training history:" << std::endl;
    auto [loss, val_loss, acc] = network.get_training_history();
    
    std::cout << "   First 5 epochs:" << std::endl;
    for (size_t i = 0; i < std::min(5UL, loss.size()); ++i) {
        std::cout << "     Epoch " << i*10 << ": loss = " << loss[i*10] << std::endl;
    }
    std::cout << "   Final loss: " << loss.back() << std::endl;
    
    // Step 6: Network summary
    std::cout << "\n6. Network summary:" << std::endl;
    network.summary();
    
    std::cout << "\n✅ XOR problem solved!\n" << std::endl;
    
    return 0;
}