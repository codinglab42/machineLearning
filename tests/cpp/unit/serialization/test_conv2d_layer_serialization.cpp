#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include "components/layers/conv2d_layer.h" // Adegua il path se necessario

namespace fs = std::filesystem;

class Conv2DLayerSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        filename = "test_conv2d_serial.bin";
    }

    void TearDown() override {
        if (fs::exists(filename)) {
            fs::remove(filename);
        }
    }

    std::string filename;
};

TEST_F(Conv2DLayerSerializationTest, SaveAndLoadConsistency) {
    // 1. Parametri di configurazione
    int in_h = 28, in_w = 28, in_c = 1; // Classica immagine MNIST
    int filters = 4;
    int k_size = 3;
    int strides = 1;

    // 2. ISTANZIAZIONE (Usando il costruttore reale a 5 argomenti)
    auto layer_out = std::make_shared<layers::Conv2DLayer>(
        filters, k_size, strides, "valid", "relu"
    );

    // Inizializziamo le dimensioni dell'input
    layer_out->set_input_shape(in_h * in_w * in_c);

    // Impostiamo pesi controllati (solo filtri puri: filters x kernel_elements)
    int kernel_elements = k_size * k_size * in_c;
    Eigen::MatrixXd fixed_weights = Eigen::MatrixXd::Constant(filters, kernel_elements, 0.05);
    layer_out->set_weights(fixed_weights);

    // Impostiamo un bias controllato separatamente
    Eigen::VectorXd fixed_biases = Eigen::VectorXd::Constant(filters, 0.01);
    layer_out->set_biases(fixed_biases);
    
    // Creiamo un input di test (batch size 1)
    Eigen::MatrixXd input = Eigen::MatrixXd::Ones(1, in_h * in_w * in_c);
    
    // Calcoliamo l'output originale
    Eigen::MatrixXd original_output = layer_out->forward(input);
    double original_sum = original_output.sum();

    // 3. SERIALIZZAZIONE
    {
        std::ofstream ofs(filename, std::ios::binary);
        ASSERT_TRUE(ofs.is_open()) << "Impossibile creare il file per la serializzazione";
        layer_out->serialize(ofs);
        ofs.close();
    }

    // 4. DESERIALIZZAZIONE
    // Creiamo un layer con parametri diversi per verificare che vengano sovrascritti
    auto layer_in = std::make_shared<layers::Conv2DLayer>(1, 1, 1, "valid", "none");
    {
        std::ifstream ifs(filename, std::ios::binary);
        ASSERT_TRUE(ifs.is_open()) << "Impossibile aprire il file per la deserializzazione";
        layer_in->deserialize(ifs);
        ifs.close();
    }

    // 5. VERIFICHE FINALI
    
    // A. Verifica Metadati
    EXPECT_EQ(layer_in->get_input_size(), in_h * in_w * in_c);
    EXPECT_EQ(layer_in->get_output_size(), layer_out->get_output_size());

    // B. Verifica Pesi (Valori numerici nel file)
    double weight_diff = (layer_out->get_weights() - layer_in->get_weights()).norm();
    EXPECT_LT(weight_diff, 1e-8) << "Errore: I pesi caricati non coincidono con quelli salvati!";

    // C. Verifica Forward (Integrità del calcolo e della cache)
    Eigen::MatrixXd loaded_output = layer_in->forward(input);
    double loaded_sum = loaded_output.sum();
    
    double output_diff = (original_output - loaded_output).norm();
    
    std::cout << "--- Risultati Test ---" << std::endl;
    std::cout << "Original Sum: " << original_sum << std::endl;
    std::cout << "Loaded Sum:   " << loaded_sum << std::endl;
    std::cout << "Diff Norm:    " << output_diff << std::endl;

    EXPECT_LT(output_diff, 1e-7) << "L'output del forward differisce drasticamente!";
}