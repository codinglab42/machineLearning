#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cstdio>

// Include dei tuoi header ufficiali
#include "models/neural_network.h"
#include "components/layers/layer.h"
#include "components/layers/dense_layer.h"
#include "components/layers/dropout_layer.h"

using namespace models;

class NeuralNetworkSerializationDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        filename_ = "test_debug_network.bin";
        nn_ = std::make_unique<NeuralNetwork>();
    }
    
    void TearDown() override {
        nn_.reset();
        std::remove(filename_.c_str());
    }
    
    std::string filename_;
    std::unique_ptr<NeuralNetwork> nn_;
};

TEST_F(NeuralNetworkSerializationDebugTest, FullBinaryDumpAndAnalysis) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "DEBUG: ALLINEAMENTO HEADER REALE RETE" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "1. COSTRUZIONE RETE:" << std::endl;
    nn_->add_layer(std::make_unique<layers::DenseLayer>(128, "relu", true));
    nn_->add_layer(std::make_unique<layers::DenseLayer>(64, "relu", true));
    nn_->add_layer(std::make_unique<layers::DropoutLayer>(0.5));
    nn_->add_layer(std::make_unique<layers::DenseLayer>(32, "relu", true));
    nn_->add_layer(std::make_unique<layers::DenseLayer>(10, "softmax", false));
    
    // Configura parametri per riflettere i campi scritti nell'header
    nn_->build(128, 10);
    
    std::cout << "\n2. SERIALIZZAZIONE BINARIA CON METODO UFFICIALE:" << std::endl;
    std::ofstream out(filename_, std::ios::binary);
    ASSERT_TRUE(out.is_open());
    
    nn_->serialize_binary(out);
    out.close();
    
    std::cout << "   File salvato con successo." << std::endl;
    
    std::cout << "\n3. ANALISI PARSING SPERIMENTALE (Speculare al sorgente):" << std::endl;
    std::ifstream in(filename_, std::ios::binary);
    ASSERT_TRUE(in.is_open());
    
    // Variabili speculari per catturare i campi esatti
    int n_features = 0;
    int n_classes = 0;
    double learning_rate = 0.0; // Cambia in float se nella classe usi float
    bool fitted = false;
    int epochs = 0;
    int batch_size = 0;
    
    in.read(reinterpret_cast<char*>(&n_features), sizeof(int));
    in.read(reinterpret_cast<char*>(&n_classes), sizeof(int));
    in.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    in.read(reinterpret_cast<char*>(&fitted), sizeof(bool));
    in.read(reinterpret_cast<char*>(&epochs), sizeof(int));
    in.read(reinterpret_cast<char*>(&batch_size), sizeof(int));
    
    std::cout << "   [HEADER SCALARE]" << std::endl;
    std::cout << "     n_features:    " << n_features << " (Atteso: 128)" << std::endl;
    std::cout << "     n_classes:     " << n_classes << " (Atteso: 10)" << std::endl;
    std::cout << "     learning_rate: " << learning_rate << std::endl;
    std::cout << "     fitted:        " << (fitted ? "true" : "false") << " (Atteso: true)" << std::endl;
    std::cout << "     epochs:        " << epochs << std::endl;
    std::cout << "     batch_size:    " << batch_size << std::endl;
    
    // Verifica allineamento stringa loss
    size_t loss_len = 0;
    in.read(reinterpret_cast<char*>(&loss_len), sizeof(size_t));
    std::cout << "     loss_len:      " << loss_len << " bytes" << std::endl;
    
    // Evitiamo crash se la lettura iniziale è corrotta
    ASSERT_LT(loss_len, 1000) << "Errore critico: la lunghezza del nome della loss è assurda. Disallineamento tipi nell'header.";
    
    std::string loss_name;
    loss_name.resize(loss_len);
    if (loss_len > 0) {
        in.read(&loss_name[0], loss_len);
    }
    std::cout << "     loss_name:     \"" << loss_name << "\"" << std::endl;
    
    // Verifica il contatore dei layer (size_t)
    size_t num_layers = 0;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    std::cout << "     num_layers:    " << num_layers << " (Atteso: 5)" << std::endl;
    
    EXPECT_EQ(n_features, 128);
    EXPECT_EQ(n_classes, 10);
    EXPECT_EQ(num_layers, 5);
    
    in.close();
    
    std::cout << "\n4. VERIFICA DESERIALIZZAZIONE METODO METODO LIBRERIA:" << std::endl;
    NeuralNetwork loaded_nn;
    try {
        std::ifstream in_official(filename_, std::ios::binary);
        ASSERT_TRUE(in_official.is_open());
        loaded_nn.deserialize_binary(in_official);
        in_official.close();
        std::cout << "   ✓ Il metodo NeuralNetwork::deserialize_binary() è allineato e funziona!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "   ✗ Errore in deserialize_binary: " << e.what() << std::endl;
        FAIL() << "La deserializzazione fallisce a causa dell'architettura interna dei layer.";
    }
}