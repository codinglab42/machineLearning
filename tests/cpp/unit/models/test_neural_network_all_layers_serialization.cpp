#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cstdio>

// Include di tutti i layer della tua architettura
#include "models/neural_network.h"
#include "components/layers/layer.h"
#include "components/layers/dense_layer.h"
#include "components/layers/conv2d_layer.h"
#include "components/layers/pooling_layer.h" // Assumendo che gestisca Max e Average
#include "components/layers/flatten_layer.h"
#include "components/layers/dropout_layer.h"
#include "components/layers/batch_norm_layer.h"
#include "components/layers/simple_rnn_layer.h"
#include "components/layers/lstm_layer.h"
#include "components/layers/gru_layer.h"

using namespace models;

class NeuralNetworkFullArchitectureTest : public ::testing::Test {
protected:
    void SetUp() override {
        filename_ = "test_full_architecture.bin";
        nn_ = std::make_unique<NeuralNetwork>();
    }
    
    void TearDown() override {
        nn_.reset();
        std::remove(filename_.c_str());
    }
    
    std::string filename_;
    std::unique_ptr<NeuralNetwork> nn_;
};

TEST_F(NeuralNetworkFullArchitectureTest, VerifyAllLayerTypesSerialization) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST DI INTEGRITÀ: TUTTI I LAYER (" << 10 << " TIPI)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "1. INSERIMENTO DI OGNI TIPOLOGIA DI LAYER NELLA RETE:" << std::endl;
    
    try {
        // 1. DENSE (units, activation, use_bias)
        nn_->add_layer(std::make_unique<layers::DenseLayer>(64, "relu", true));
        std::cout << "   ✓ Aggiunto: DENSE" << std::endl;
        
        // 2. CONV2D (filters, kernel_rows, kernel_cols, activation)
        nn_->add_layer(std::make_unique<layers::Conv2DLayer>(16, 3, 3, "relu"));
        std::cout << "   ✓ Aggiunto: CONV2D" << std::endl;
        
        // 3. MAX_POOLING & 4. AVERAGE_POOLING 
        // Firma: (pool_size, stride, PoolType, channels)
        nn_->add_layer(std::make_unique<layers::PoolingLayer>(2, 2, layers::PoolingLayer::MAX, 1)); 
        nn_->add_layer(std::make_unique<layers::PoolingLayer>(2, 2, layers::PoolingLayer::AVG, 1));
        std::cout << "   ✓ Aggiunti: 2x PoolingLayer (MAX e AVG)" << std::endl;
        
        // 5. FLATTEN
        nn_->add_layer(std::make_unique<layers::FlattenLayer>());
        std::cout << "   ✓ Aggiunto: FLATTEN" << std::endl;
        
        // 6. DROPOUT (rate)
        nn_->add_layer(std::make_unique<layers::DropoutLayer>(0.25));
        std::cout << "   ✓ Aggiunto: DROPOUT" << std::endl;
        
        // 7. BATCH_NORM (features)
        nn_->add_layer(std::make_unique<layers::BatchNormLayer>(64));
        std::cout << "   ✓ Aggiunto: BATCH_NORM" << std::endl;
        
        // 8. SIMPLE_RNN (units, input_size, activation, use_bias)
        nn_->add_layer(std::make_unique<layers::SimpleRNNLayer>(32, 64, "tanh", true));
        std::cout << "   ✓ Aggiunto: SIMPLE_RNN" << std::endl;
        
        // 9. LSTM (units, input_size, activation, recurrent_activation, use_bias)
        nn_->add_layer(std::make_unique<layers::LSTMLayer>(32, 64, "tanh", "sigmoid", true));
        std::cout << "   ✓ Aggiunto: LSTM" << std::endl;
        
        // 10. GRU (Assumendo firma speculare a LSTM: units, input_size, act, rec_act, use_bias)
        nn_->add_layer(std::make_unique<layers::GRULayer>(32, 64, "tanh", "sigmoid", true));
        std::cout << "   ✓ Aggiunto: GRU" << std::endl;
        
        // Costruisci la rete definendo le feature di ingresso globali
        nn_->build(128, 10);
        
    } catch (const std::exception& e) {
        FAIL() << "Inizializzazione o build della rete fallita: " << e.what();
    }

    std::cout << "\n2. SERIALIZZAZIONE COMPLETA SUL FILE..." << std::endl;
    std::ofstream out(filename_, std::ios::binary);
    ASSERT_TRUE(out.is_open());
    nn_->serialize_binary(out);
    out.close();
    
    // Verifica preliminare dell'header tramite parsing manuale veloce
    std::ifstream in_check(filename_, std::ios::binary);
    ASSERT_TRUE(in_check.is_open());
    
    int n_features = 0, n_classes = 0;
    double lr = 0.0;
    bool fitted = false;
    int epochs = 0, batch_size = 0;
    
    in_check.read(reinterpret_cast<char*>(&n_features), sizeof(int));
    in_check.read(reinterpret_cast<char*>(&n_classes), sizeof(int));
    in_check.read(reinterpret_cast<char*>(&lr), sizeof(double));
    in_check.read(reinterpret_cast<char*>(&fitted), sizeof(bool));
    in_check.read(reinterpret_cast<char*>(&epochs), sizeof(int));
    in_check.read(reinterpret_cast<char*>(&batch_size), sizeof(int));
    
    size_t loss_len = 0;
    in_check.read(reinterpret_cast<char*>(&loss_len), sizeof(size_t));
    in_check.seekg(loss_len, std::ios::cur); // Salta il nome della loss
    
    size_t num_layers = 0;
    in_check.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    in_check.close();
    
    std::cout << "   Controllore Header -> Layer totali scritti: " << num_layers << std::endl;
    EXPECT_EQ(num_layers, 10);

    std::cout << "\n3. DESERIALIZZAZIONE POLIMORFICA (La prova del nove):" << std::endl;
    NeuralNetwork loaded_nn;
    try {
        std::ifstream in_official(filename_, std::ios::binary);
        ASSERT_TRUE(in_official.is_open());
        
        // Se uno qualsiasi dei 10 layer ha un disallineamento nei pesi, 
        // nei flag o nelle stringhe, questa chiamata lancerà un'eccezione.
        loaded_nn.deserialize_binary(in_official);
        in_official.close();
        
        std::cout << "   ✓ ECCELLENTE: Tutti e 10 i layer sono stati deserializzati correttamente!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "   ✗ CRASH RILEVATO: " << e.what() << std::endl;
        FAIL() << "La pipeline binarizzata si rompe quando incontra la combinazione di tutti i layer.";
    }
}