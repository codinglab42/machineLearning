#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "models/neural_network.h"
#include "utils/math_utils.h"

using namespace models;
using namespace utils;
using namespace testing;

class NeuralNetworkIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Dataset AND
        X_and.resize(4, 2);
        X_and << 0, 0,
                 0, 1,
                 1, 0,
                 1, 1;
        y_and.resize(4);
        y_and << 0, 0, 0, 1;
        
        // Dataset OR
        X_or.resize(4, 2);
        X_or << 0, 0,
                0, 1,
                1, 0,
                1, 1;
        y_or.resize(4);
        y_or << 0, 1, 1, 1;
        
        // Dataset XOR (più difficile)
        X_xor.resize(4, 2);
        X_xor << 0, 0,
                 0, 1,
                 1, 0,
                 1, 1;
        y_xor.resize(4);
        y_xor << 0, 1, 1, 0;
        
        // Dataset binario semplice
        X_binary.resize(100, 2);
        y_binary.resize(100);
        
        std::random_device rd;
        std::mt19937 gen(42);
        
        for (int i = 0; i < 100; ++i) {
            if (i < 50) {
                X_binary(i, 0) = 1.0 + std::normal_distribution<>(0, 0.3)(gen);
                X_binary(i, 1) = 1.0 + std::normal_distribution<>(0, 0.3)(gen);
                y_binary(i) = 0;
            } else {
                X_binary(i, 0) = 3.0 + std::normal_distribution<>(0, 0.3)(gen);
                X_binary(i, 1) = 3.0 + std::normal_distribution<>(0, 0.3)(gen);
                y_binary(i) = 1;
            }
        }
    }
    
    Eigen::MatrixXd X_and, X_or, X_xor, X_binary;
    Eigen::VectorXd y_and, y_or, y_xor, y_binary;
};

TEST_F(NeuralNetworkIntegrationTest, AND) {
    NeuralNetwork network({2, 8, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.5);
    network.set_loss_function("binary_crossentropy");
    network.set_epochs(2000);
    network.set_batch_size(4);
    network.set_verbose(true);
    
    network.fit(X_and, y_and);
    
    Eigen::VectorXd y_pred = network.predict(X_and);
    
    // DEBUG: stampa predizioni
    std::cout << "Predictions for AND:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "  Input: (" << X_and(i,0) << ", " << X_and(i,1) 
                  << ") -> true=" << y_and(i) 
                  << ", pred=" << y_pred(i)
                  << " (" << (y_pred(i) > 0.5 ? "1" : "0") << ")" << std::endl;
    }
    
    Eigen::VectorXi y_pred_int = (y_pred.array() > 0.5).cast<int>();
    
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        if (y_pred_int(i) == static_cast<int>(y_and(i))) correct++;
    }
    EXPECT_GE(correct, 4);
}

TEST_F(NeuralNetworkIntegrationTest, OR) {
    // Aumenta learning rate a 0.5 come AND
    NeuralNetwork network({2, 8, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.5);
    network.set_loss_function("binary_crossentropy");
    network.set_epochs(1000);
    network.set_batch_size(4);
    network.set_verbose(true);
    
    network.fit(X_or, y_or);
    
    Eigen::VectorXd y_pred = network.predict(X_or);
    
    // Debug output
    std::cout << "Predictions for OR:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "  Input: (" << X_or(i,0) << ", " << X_or(i,1) 
                  << ") -> true=" << y_or(i) 
                  << ", pred=" << y_pred(i)
                  << " (" << (y_pred(i) > 0.5 ? "1" : "0") << ")" << std::endl;
    }
    
    Eigen::VectorXi y_pred_int = (y_pred.array() > 0.5).cast<int>();
    
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        if (y_pred_int(i) == static_cast<int>(y_or(i))) correct++;
    }
    EXPECT_GE(correct, 4);
}

TEST_F(NeuralNetworkIntegrationTest, BinaryClassification) {
    NeuralNetwork network({2, 32, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.2);
    network.set_loss_function("binary_crossentropy");
    network.set_epochs(2000);
    network.set_batch_size(16);
    network.set_verbose(true);
    
    network.fit(X_binary, y_binary);
    
    // Debug: mostra alcune predizioni
    Eigen::VectorXd y_pred = network.predict(X_binary);
    std::cout << "\nSample predictions:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "  Sample " << i << ": true=" << y_binary(i) 
                  << ", pred=" << y_pred(i) 
                  << " (" << (y_pred(i) > 0.5 ? "1" : "0") << ")" << std::endl;
    }

    double score = network.score(X_binary, y_binary);
    std::cout << "Final accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.95);
}

TEST_F(NeuralNetworkIntegrationTest, History) {
    NeuralNetwork network({2, 8, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.2);
    network.set_loss_function("binary_crossentropy");
    network.set_epochs(200);
    network.set_batch_size(4);
    network.set_verbose(false);
    
    network.fit(X_and, y_and);
    
    auto [loss, val_loss, acc] = network.get_training_history();
    EXPECT_GT(loss.size(), 0);
    EXPECT_LT(loss.back(), loss.front());
}

TEST_F(NeuralNetworkIntegrationTest, PredictProba) {
    NeuralNetwork network({2, 8, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.2);
    network.set_loss_function("binary_crossentropy");
    network.set_epochs(200);
    network.set_batch_size(4);
    network.set_verbose(false);
    
    network.fit(X_and, y_and);
    
    Eigen::MatrixXd proba = network.predict_proba(X_and);
    EXPECT_EQ(proba.rows(), 4);
    EXPECT_EQ(proba.cols(), 1);
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(proba(i,0), 0.0);
        EXPECT_LE(proba(i,0), 1.0);
    }
}

TEST_F(NeuralNetworkIntegrationTest, Summary) {
    NeuralNetwork network({2, 8, 4, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.1);
    
    EXPECT_NO_THROW(network.summary());
    EXPECT_EQ(network.get_num_layers(), 3);
    EXPECT_GT(network.get_num_parameters(), 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}