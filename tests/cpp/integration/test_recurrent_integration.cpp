#include <gtest/gtest.h>
#include "components/layers/simple_rnn_layer.h"
#include "components/layers/lstm_layer.h"
#include "components/layers/gru_layer.h"
#include <Eigen/Dense>
#include <memory>

// ============================================================================
// TEST INTEGRAZIONE: SIMPLE RNN LAYER
// ============================================================================
TEST(RecurrentIntegrationTest, SimpleRNNForwardBackwardPipeline) {
    int units = 3;
    int input_size = 2;
    int batch_size = 1;
    
    auto rnn = std::make_unique<layers::SimpleRNNLayer>(units, input_size, "tanh", true);
    rnn->set_input_shape(input_size);
    
    // Generiamo un input controllato per 1 step temporale
    Eigen::MatrixXd input = Eigen::MatrixXd::Constant(batch_size, input_size, 0.5);
    
    // 1. Forward Pass (Training = true per salvare la cache)
    Eigen::MatrixXd output = rnn->forward(input, true);
    
    ASSERT_EQ(output.rows(), batch_size);
    ASSERT_EQ(output.cols(), units);
    
    // 2. Backward Pass
    Eigen::MatrixXd top_gradient = Eigen::MatrixXd::Constant(batch_size, units, 1.0);
    Eigen::MatrixXd bottom_gradient = rnn->backward(top_gradient);
    
    // Verifichiamo che il gradiente propagato all'input sia corretto geometricamente
    ASSERT_EQ(bottom_gradient.rows(), batch_size);
    ASSERT_EQ(bottom_gradient.cols(), input_size);
    
    // Verifichiamo l'estrazione e l'integrità dei gradienti dei pesi e dei bias isolati
    Eigen::MatrixXd dWeights = rnn->get_weights_gradient();
    Eigen::VectorXd dBias = rnn->get_bias_gradient();
    
    // I pesi estratti devono avere righe = (input_size + units) e colonne = units
    ASSERT_EQ(dWeights.rows(), input_size + units);
    ASSERT_EQ(dWeights.cols(), units);
    ASSERT_EQ(dBias.size(), units);
    
    // Verifichiamo che i gradienti non siano rimasti a zero (segno di attivazione del calcolo)
    EXPECT_GT(dWeights.norm(), 0.0);
    EXPECT_GT(dBias.norm(), 0.0);
}

// ============================================================================
// TEST INTEGRAZIONE: LSTM LAYER
// ============================================================================
TEST(RecurrentIntegrationTest, LSTMForwardBackwardPipeline) {
    int units = 4;
    int input_size = 3;
    int batch_size = 2;
    
    auto lstm = std::make_unique<layers::LSTMLayer>(units, input_size, "tanh", "sigmoid", true);
    lstm->set_input_shape(input_size);
    
    Eigen::MatrixXd input = Eigen::MatrixXd::Constant(batch_size, input_size, 0.1);
    
    // 1. Forward Pass
    Eigen::MatrixXd output = lstm->forward(input, true);
    
    ASSERT_EQ(output.rows(), batch_size);
    ASSERT_EQ(output.cols(), units);
    
    // 2. Backward Pass
    Eigen::MatrixXd top_gradient = Eigen::MatrixXd::Constant(batch_size, units, 0.5);
    Eigen::MatrixXd bottom_gradient = lstm->backward(top_gradient);
    
    ASSERT_EQ(bottom_gradient.rows(), batch_size);
    ASSERT_EQ(bottom_gradient.cols(), input_size);
    
    // Estrazione gradienti dopo la rifattorizzazione geometrica rigida delle 4 porte
    Eigen::MatrixXd dWeights = lstm->get_weights_gradient();
    Eigen::VectorXd dBias = lstm->get_bias_gradient();
    
    // LSTM concatena le 4 porte verticalmente lungo le righe
    ASSERT_EQ(dWeights.rows(), (input_size + units) * 4); // 4 porte sulle righe
    ASSERT_EQ(dWeights.cols(), units);                   // units sulle colonne
    ASSERT_EQ(dBias.size(), units * 4);
    
    EXPECT_GT(dWeights.norm(), 0.0);
    EXPECT_GT(dBias.norm(), 0.0);
}

// ============================================================================
// TEST INTEGRAZIONE: GRU LAYER
// ============================================================================
TEST(RecurrentIntegrationTest, GRUForwardBackwardPipeline) {
    int units = 3;
    int input_size = 2;
    int batch_size = 1;
    
    auto gru = std::make_unique<layers::GRULayer>(units, input_size, "tanh", "sigmoid", true);
    gru->set_input_shape(input_size);
    
    Eigen::MatrixXd input = Eigen::MatrixXd::Constant(batch_size, input_size, -0.2);
    
    // 1. Forward Pass
    Eigen::MatrixXd output = gru->forward(input, true);
    
    ASSERT_EQ(output.rows(), batch_size);
    ASSERT_EQ(output.cols(), units);
    
    // 2. Backward Pass
    Eigen::MatrixXd top_gradient = Eigen::MatrixXd::Constant(batch_size, units, 1.0);
    Eigen::MatrixXd bottom_gradient = gru->backward(top_gradient);
    
    ASSERT_EQ(bottom_gradient.rows(), batch_size);
    ASSERT_EQ(bottom_gradient.cols(), input_size);
    
    // Estrazione gradienti per le 3 porte di GRU (Update, Reset, Candidate)
    Eigen::MatrixXd dWeights = gru->get_weights_gradient();
    Eigen::VectorXd dBias = gru->get_bias_gradient();
    
    // GRU concatena le 3 porte verticalmente lungo le righe
    ASSERT_EQ(dWeights.rows(), (input_size + units) * 3); // 3 porte sulle righe
    ASSERT_EQ(dWeights.cols(), units);                   // units sulle colonne
    ASSERT_EQ(dBias.size(), units * 3);

    EXPECT_GT(dWeights.norm(), 0.0);
    EXPECT_GT(dBias.norm(), 0.0);
}