/**
 * @file test_neural_network_integration.cpp
 * @brief Integration tests for Neural Network
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <iostream>
#include <Eigen/Dense>

#include "models/neural_network.h"
#include "components/regularizers/regularizer_factory.h"

using namespace models;
using namespace Eigen;
using namespace testing;

// Funzione di normalizzazione
void normalize_features(MatrixXd& X) {
    for (int i = 0; i < X.cols(); ++i) {
        double mean = X.col(i).mean();
        double std = std::sqrt((X.col(i).array() - mean).square().mean());
        if (std < 1e-7) std = 1.0;
        X.col(i) = (X.col(i).array() - mean) / std;
    }
}

class NeuralNetworkIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Crea dataset binario semplice (classi linearmente separabili)
        X_.resize(200, 2);
        y_.resize(200);
        
        std::random_device rd;
        std::mt19937 gen(42);
        
        for (int i = 0; i < 200; ++i) {
            if (i < 100) {
                X_(i, 0) = 1.0 + std::normal_distribution<>(0, 0.3)(gen);
                X_(i, 1) = 1.0 + std::normal_distribution<>(0, 0.3)(gen);
                y_(i) = 0.0;
            } else {
                X_(i, 0) = 3.0 + std::normal_distribution<>(0, 0.3)(gen);
                X_(i, 1) = 3.0 + std::normal_distribution<>(0, 0.3)(gen);
                y_(i) = 1.0;
            }
        }
        
        // Normalizza features
        normalize_features(X_);
        
        std::cout << "Dataset: " << y_.size() << " samples, "
                  << "Class 0: " << (y_.array() == 0).count() << ", "
                  << "Class 1: " << (y_.array() == 1).count() << std::endl;
    }
    
    MatrixXd X_;
    VectorXd y_;
};

// ============================================================================
// TEST DI BASE
// ============================================================================

TEST_F(NeuralNetworkIntegrationTest, TrainWithSGD) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::SGD, 0.5);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(300);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    // Build prima del training
    nn.build(2, 1);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "SGD accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}

TEST_F(NeuralNetworkIntegrationTest, TrainWithAdam) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.build(2, 1);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "Adam accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}

// ============================================================================
// TEST CON REGOLARIZZAZIONE
// ============================================================================

TEST_F(NeuralNetworkIntegrationTest, TrainWithL2Regularization) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L2, 0.001);
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.build(2, 1);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "L2 Regularization accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}

TEST_F(NeuralNetworkIntegrationTest, TrainWithL1Regularization) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_regularizer(RegularizerType::L1, 0.001);
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.build(2, 1);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "L1 Regularization accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}

// ============================================================================
// TEST DI DEEPER NETWORK
// ============================================================================

TEST_F(NeuralNetworkIntegrationTest, DeeperNetwork) {
    NeuralNetwork nn({2, 32, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(200);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    nn.build(2, 1);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    std::cout << "Deeper network accuracy: " << score << std::endl;
    EXPECT_GT(score, 0.85);
}

// ============================================================================
// TEST DI SERIALIZZAZIONE
// ============================================================================

TEST_F(NeuralNetworkIntegrationTest, Serialization) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(100);
    nn.set_batch_size(32);
    nn.set_verbose(false);
    
    //nn.build(2, 1);
    //nn.fit(X_, y_);
    EXPECT_NO_THROW(nn.fit(X_, y_));

    VectorXd y_pred_original = nn.predict(X_);
    
    std::string filename = "test_nn_integration.bin";
    EXPECT_NO_THROW(nn.save(filename));
    
    NeuralNetwork nn_loaded;
    EXPECT_NO_THROW(nn_loaded.load(filename));
    
    VectorXd y_pred_loaded = nn_loaded.predict(X_);
    
    double diff = (y_pred_original - y_pred_loaded).norm();
    std::cout << "Prediction difference: " << diff << std::endl;
    EXPECT_LT(diff, 0.01);
    
    std::remove(filename.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}