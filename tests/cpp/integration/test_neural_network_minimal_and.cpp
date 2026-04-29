/**
 * @file test_minimal_and.cpp
 * @brief Minimal test to verify neural network can learn AND gate
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <Eigen/Dense>

#include "models/neural_network.h"

using namespace models;
using namespace Eigen;
using namespace testing;

class MinimalANDTest : public ::testing::Test {
protected:
    void SetUp() override {
        // AND dataset
        X_.resize(4, 2);
        X_ << 0, 0,
              0, 1,
              1, 0,
              1, 1;
        y_.resize(4);
        y_ << 0, 0, 0, 1;
    }
    
    MatrixXd X_;
    VectorXd y_;
};

TEST_F(MinimalANDTest, LearnANDWithSGD) {
    std::cout << "\n=== TEST: Learn AND with SGD ===" << std::endl;
    
    // Aumenta i neuroni nascosti e usa più epoche
    NeuralNetwork nn({2, 32, 1}, "tanh", "sigmoid", OptimizerType::SGD, 1.0);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(3000);
    nn.set_batch_size(4);
    nn.set_verbose(true);
    
    nn.fit(X_, y_);
    
    Eigen::VectorXd pred = nn.predict(X_);
    
    std::cout << "\n=== PREDICTIONS ===" << std::endl;
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        bool pred_class = pred(i) > 0.5;
        bool true_class = y_(i) > 0.5;
        std::cout << "Input: (" << X_(i,0) << ", " << X_(i,1) 
                  << ") -> pred=" << pred(i) 
                  << " (" << (pred_class ? "1" : "0") << ")"
                  << " true=" << y_(i)
                  << (pred_class == true_class ? " ✓" : " ✗") << std::endl;
        if (pred_class == true_class) correct++;
    }
    
    std::cout << "\nAccuracy: " << correct << "/4" << std::endl;
    EXPECT_EQ(correct, 4);
}

TEST_F(MinimalANDTest, LearnANDWithAdam) {
    std::cout << "\n=== TEST: Learn AND with Adam ===" << std::endl;
    
    NeuralNetwork nn({2, 4, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(500);
    nn.set_batch_size(4);
    nn.set_verbose(true);
    
    nn.fit(X_, y_);
    
    Eigen::VectorXd pred = nn.predict(X_);
    
    std::cout << "\n=== PREDICTIONS ===" << std::endl;
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        bool pred_class = pred(i) > 0.5;
        bool true_class = y_(i) > 0.5;
        std::cout << "Input: (" << X_(i,0) << ", " << X_(i,1) 
                  << ") -> pred=" << pred(i) 
                  << " (" << (pred_class ? "1" : "0") << ")"
                  << " true=" << y_(i)
                  << (pred_class == true_class ? " ✓" : " ✗") << std::endl;
        if (pred_class == true_class) correct++;
    }
    
    std::cout << "\nAccuracy: " << correct << "/4" << std::endl;
    EXPECT_EQ(correct, 4);
}

TEST_F(MinimalANDTest, DebugWeightUpdates) {
    std::cout << "\n=== TEST: Debug Weight Updates ===" << std::endl;
    
    NeuralNetwork nn({2, 4, 1}, "relu", "sigmoid", OptimizerType::SGD, 0.5);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(10);
    nn.set_batch_size(4);
    nn.set_verbose(true);
    
    // Stampa pesi prima del training
    std::cout << "\nWeights BEFORE training:" << std::endl;
    auto& layers = nn.get_layers();
    if (!layers.empty()) {
        Eigen::MatrixXd w = layers[0]->get_weights();
        std::cout << "Layer 0 weights (first 2x4):" << std::endl;
        std::cout << w.topRows(2) << std::endl;
    }
    
    nn.fit(X_, y_);
    
    // Stampa pesi dopo il training
    std::cout << "\nWeights AFTER training:" << std::endl;
    if (!layers.empty()) {
        Eigen::MatrixXd w = layers[0]->get_weights();
        std::cout << "Layer 0 weights (first 2x4):" << std::endl;
        std::cout << w.topRows(2) << std::endl;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}