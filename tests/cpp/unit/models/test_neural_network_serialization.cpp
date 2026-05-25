#include <gtest/gtest.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <memory>
#include <Eigen/Core>

// Include dei tuoi header (adatta i path secondo il tuo progetto)
#include "models/neural_network.h"
#include "components/layers/layer.h"
#include "components/layers/dense_layer.h"
#include "components/layers/dropout_layer.h"
#include "components/layers/conv2d_layer.h"
#include "components/layers/flatten_layer.h"
#include "components/layers/batch_norm_layer.h"
#include "components/layers/simple_rnn_layer.h"
#include "components/layers/lstm_layer.h"
#include "components/layers/gru_layer.h"
#include "components/layers/pooling_layer.h"

// Assumendo che tutto sia nel namespace ml
using namespace models;
using namespace Eigen;

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
    
    void printHexDump(const char* data, size_t size, size_t offset = 0) {
        std::cout << "    Offset " << std::setw(4) << std::setfill('0') << offset << ": ";
        for (size_t i = 0; i < size; ++i) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') 
                      << (static_cast<unsigned int>(data[i]) & 0xFF) << " ";
            if ((i + 1) % 8 == 0 && i + 1 < size) {
                std::cout << " ";
            }
        }
        std::cout << std::dec << std::endl;
    }
    
    std::string filename_;
    std::unique_ptr<NeuralNetwork> nn_;
};

TEST_F(NeuralNetworkSerializationDebugTest, FullBinaryDumpAndAnalysis) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "DEBUG: ANALISI COMPLETA SERIALIZZAZIONE" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // 1. COSTRUISCI UNA RETE COMPLESSA
    std::cout << "1. COSTRUZIONE RETE:" << std::endl;
    std::cout << "   Aggiungo layer con dimensioni reali..." << std::endl;
    
    nn_->add_layer(std::make_unique<layers::DenseLayer>(128, "relu", true));
    nn_->add_layer(std::make_unique<layers::DenseLayer>(64, "relu", true));
    nn_->add_layer(std::make_unique<layers::DropoutLayer>(0.5));
    nn_->add_layer(std::make_unique<layers::DenseLayer>(32, "relu", true));
    nn_->add_layer(std::make_unique<layers::DenseLayer>(10, "softmax", false));
    
    std::cout << "   Layer aggiunti: 5 (dense, dense, dropout, dense, dense)" << std::endl;
    std::cout << "   Input features: 128, Output classes: 10" << std::endl;

    nn_->build(128,10);
    
    // Addestra fittiziamente
    std::cout << "\n2. ADDESTRAMENTO FITTIZIO:" << std::endl;
    Eigen::MatrixXd dummy_input = Eigen::MatrixXd::Random(32, 128);
    Eigen::MatrixXd dummy_target = Eigen::MatrixXd::Random(32, 10);
    
    // Nota: potrebbe servire una funzione initialize() invece di fit
    // nn_->initialize(); // se esiste
    std::cout << "   Pesi inizializzati randomicamente" << std::endl;
    
    // 2. SALVA LA RETE
    std::cout << "\n3. SERIALIZZAZIONE:" << std::endl;
    std::ofstream out(filename_, std::ios::binary);
    ASSERT_TRUE(out.is_open());
    
    // Serialize con tracking della posizione
    std::streampos pos_before = out.tellp();
    nn_->serialize_binary(out);
    std::streampos pos_after = out.tellp();
    out.close();
    
    size_t file_size = pos_after - pos_before;
    std::cout << "   File salvato: " << filename_ << std::endl;
    std::cout << "   Dimensione file: " << file_size << " bytes" << std::endl;
    
    // 3. LEGGI E ANALIZZA IL FILE BYTE PER BYTE
    std::cout << "\n4. ANALISI ESADECIMALE DEL FILE:" << std::endl;
    std::cout << "   Leggo il file byte per byte...\n" << std::endl;
    
    std::ifstream in(filename_, std::ios::binary);
    ASSERT_TRUE(in.is_open());
    
    // Leggi header
    int n_features, n_classes;
    bool fitted;
    
    in.read(reinterpret_cast<char*>(&n_features), sizeof(int));
    in.read(reinterpret_cast<char*>(&n_classes), sizeof(int));
    in.read(reinterpret_cast<char*>(&fitted), sizeof(bool));
    
    std::cout << "   [HEADER]" << std::endl;
    std::cout << "     n_features: " << n_features << " (expected: 128)" << std::endl;
    std::cout << "     n_classes:  " << n_classes << " (expected: 10)" << std::endl;
    std::cout << "     fitted:     " << (fitted ? "true" : "false") << std::endl;
    
    // Leggi loss function name
    size_t loss_name_len;
    in.read(reinterpret_cast<char*>(&loss_name_len), sizeof(size_t));
    std::vector<char> loss_name_buf(loss_name_len + 1, '\0');
    in.read(loss_name_buf.data(), loss_name_len);
    std::string loss_name(loss_name_buf.data());
    
    std::cout << "     loss_name_len: " << loss_name_len << std::endl;
    std::cout << "     loss_name:     \"" << loss_name << "\"" << std::endl;
    
    // Leggi numero di layer
    size_t num_layers;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    
    std::cout << "\n   [LAYER LIST] (total: " << num_layers << ")\n" << std::endl;
    
    // Per ogni layer, analizza
    for (size_t i = 0; i < num_layers; ++i) {
        std::streampos layer_start = in.tellg();
        
        // Leggi tipo layer
        size_t type_len;
        in.read(reinterpret_cast<char*>(&type_len), sizeof(size_t));
        std::vector<char> type_buf(type_len + 1, '\0');
        in.read(type_buf.data(), type_len);
        std::string layer_type(type_buf.data());
        
        std::cout << "   Layer " << i << ": " << layer_type << std::endl;
        std::cout << "     Posizione inizio: " << layer_start << " bytes" << std::endl;
        std::cout << "     Type length:      " << type_len << " bytes" << std::endl;
        
        // ANALISI SPECIFICA PER TIPO DI LAYER
        if (layer_type == "DenseLayer") {
            std::cout << "     [ANALISI DenseLayer]" << std::endl;
            
            // Prova a leggere i prossimi 16 byte
            char debug_buffer[16];
            std::streampos current_pos = in.tellg();
            in.read(debug_buffer, 16);
            
            std::cout << "     Primi 16 byte dopo il tipo:" << std::endl;
            printHexDump(debug_buffer, 16, current_pos);
            
            // Interpreta come interi
            int* as_ints = reinterpret_cast<int*>(debug_buffer);
            std::cout << "     Interpretati come int: ";
            for (int j = 0; j < 4; ++j) {
                std::cout << as_ints[j] << " ";
            }
            std::cout << std::endl;
            
            // Ripristina posizione
            in.seekg(current_pos);
        }
        else if (layer_type == "DropoutLayer") {
            std::cout << "     [ANALISI DropoutLayer]" << std::endl;
            
            // Dropout di solito salva solo il rate
            float rate;
            in.read(reinterpret_cast<char*>(&rate), sizeof(float));
            std::cout << "     Dropout rate: " << rate << std::endl;
            std::cout << "     Dropout rate OK? " << (rate == 0.5f ? "SI" : "NO") << std::endl;
            continue; // Skip la deserializzazione di test
        }
        
        // SKIP la deserializzazione di test per evitare errori
        // e spostati alla fine del layer
        std::cout << "     Saltata deserializzazione test" << std::endl;
        
        // Trova la fine del layer (metodo semplice: leggi fino a EOF del layer)
        // NOTA: questo è un placeholder, dovresti implementare la logica corretta
        std::cout << std::endl;
    }
    
    in.close();
    
    // 4. TENTA LOAD COMPLETO
    std::cout << "\n5. TEST LOAD COMPLETO:" << std::endl;
    NeuralNetwork loaded_nn;
    
    try {
        loaded_nn.load(filename_);
        std::cout << "   ✓ Load completato con successo!" << std::endl;
    } catch (const std::bad_alloc& e) {
        std::cout << "   ✗ Load fallito con bad_alloc: " << e.what() << std::endl;
        FAIL() << "bad_alloc durante load completo";
    } catch (const std::exception& e) {
        std::cout << "   ✗ Load fallito: " << e.what() << std::endl;
    }
    
    // 5. VERIFICA CONSISTENZA
    std::cout << "\n6. VERIFICA CONSISTENZA:" << std::endl;
    
    std::stringstream ss_loaded;
    loaded_nn.serialize_binary(ss_loaded);
    
    std::stringstream ss_original;
    nn_->serialize_binary(ss_original);
    
    std::string original_str = ss_original.str();
    std::string loaded_str = ss_loaded.str();
    
    if (original_str == loaded_str) {
        std::cout << "   ✓ Serializzazione IDENTICA dopo load!" << std::endl;
    } else {
        std::cout << "   ✗ Serializzazione DIVERSA dopo load!" << std::endl;
        std::cout << "     Original size: " << original_str.size() << " bytes" << std::endl;
        std::cout << "     Loaded size:   " << loaded_str.size() << " bytes" << std::endl;
        
        // Trova primo byte diverso
        for (size_t i = 0; i < std::min(original_str.size(), loaded_str.size()); ++i) {
            if (original_str[i] != loaded_str[i]) {
                std::cout << "     Primo mismatch a byte " << i << std::endl;
                std::cout << "       Original: 0x" << std::hex << (int)(unsigned char)original_str[i] << std::endl;
                std::cout << "       Loaded:   0x" << (int)(unsigned char)loaded_str[i] << std::dec << std::endl;
                break;
            }
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "DEBUG COMPLETATO" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// Test specifico per rilevare il problema della dimensione
TEST_F(NeuralNetworkSerializationDebugTest, DetectDimensionMismatch) {
    std::cout << "\n=== TEST DIMENSION MISMATCH DETECTOR ===\n" << std::endl;
    
    // Crea rete con dimensioni specifiche
    nn_->add_layer(std::make_unique<layers::DenseLayer>(128, "relu", true));
    nn_->add_layer(std::make_unique<layers::DenseLayer>(64, "relu", true));

    nn_->build(128,64);
    
    // Salva
    std::ofstream out(filename_, std::ios::binary);
    nn_->serialize_binary(out);
    out.close();
    
    // Leggi manualmente per verificare le dimensioni salvate
    std::ifstream in(filename_, std::ios::binary);
    ASSERT_TRUE(in.is_open());
    
    // Skip header (n_features, n_classes, fitted, loss_name_len, loss_name)
    in.seekg(sizeof(int) + sizeof(int) + sizeof(bool));
    
    // Leggi loss_name_len e loss_name
    size_t loss_name_len;
    in.read(reinterpret_cast<char*>(&loss_name_len), sizeof(size_t));
    in.seekg(loss_name_len, std::ios::cur);
    
    // Leggi num_layers
    size_t num_layers;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    
    std::cout << "Numero layer salvati: " << num_layers << std::endl;
    
    for (size_t i = 0; i < num_layers; ++i) {
        // Leggi tipo
        size_t type_len;
        in.read(reinterpret_cast<char*>(&type_len), sizeof(size_t));
        std::vector<char> type_buf(type_len + 1, '\0');
        in.read(type_buf.data(), type_len);
        std::string layer_type(type_buf.data());
        
        std::cout << "\nLayer " << i << ": " << layer_type << std::endl;
        
        if (layer_type == "DenseLayer") {
            // Prova a leggere le dimensioni (assumendo che DenseLayer le salvi)
            int input_size, output_size;
            in.read(reinterpret_cast<char*>(&input_size), sizeof(int));
            in.read(reinterpret_cast<char*>(&output_size), sizeof(int));
            
            std::cout << "  input_size:  " << input_size << std::endl;
            std::cout << "  output_size: " << output_size << std::endl;
            
            // Verifica se sono ragionevoli
            if (input_size <= 0 || input_size > 1000000) {
                std::cout << "  ⚠️  WARNING: input_size non valido!" << std::endl;
            }
            if (output_size <= 0 || output_size > 1000000) {
                std::cout << "  ⚠️  WARNING: output_size non valido!" << std::endl;
            }
            
            // Verifica sequenza logica
            if (i == 0 && input_size != 128) {
                std::cout << "  ✗ ERRORE: Primo layer dovrebbe avere input_size=128, ma ho " << input_size << std::endl;
            }
            if (i == 0 && output_size != 64) {
                std::cout << "  ✗ ERRORE: Primo layer dovrebbe avere output_size=64, ma ho " << output_size << std::endl;
            }
            if (i == 1 && input_size != 64) {
                std::cout << "  ✗ ERRORE: Secondo layer dovrebbe avere input_size=64, ma ho " << input_size << std::endl;
            }
        } else {
            std::cout << "  (tipo non DenseLayer, skip)" << std::endl;
            // Per altri layer, non possiamo interpretare facilmente
            break;
        }
    }
    
    in.close();
}