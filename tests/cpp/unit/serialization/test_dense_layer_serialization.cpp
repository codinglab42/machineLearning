// tests/cpp/unit/serialization/test_dense_layer_serialization.cpp
#include <gtest/gtest.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "components/layers/dense_layer.h"

using namespace layers;
using namespace Eigen;

class DenseLayerSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<DenseLayer>(3, "relu", true);
        layer->set_input_shape(4);
        
        input.resize(2, 4);
        input.setRandom();
        
        filename = "test_dense_layer.bin";
    }
    
    void TearDown() override {
        std::remove(filename.c_str());
    }
    
    std::unique_ptr<DenseLayer> layer;
    MatrixXd input;
    std::string filename;
};

TEST_F(DenseLayerSerializationTest, SaveAndLoadWithBias) {
    MatrixXd original_output = layer->forward(input, true);
    
    std::ofstream ofs(filename, std::ios::binary);
    layer->serialize(ofs);
    ofs.close();
    
    DenseLayer loaded_layer(3, "relu", true);
    loaded_layer.set_input_shape(4);  // ← AGGIUNGI QUESTA LINEA!
    std::ifstream ifs(filename, std::ios::binary);
    loaded_layer.deserialize(ifs);
    ifs.close();
    
    MatrixXd original_weights = layer->get_weights();
    MatrixXd loaded_weights = loaded_layer.get_weights();
    EXPECT_TRUE(original_weights.isApprox(loaded_weights));
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}

TEST_F(DenseLayerSerializationTest, SaveAndLoadWithoutBias) {
    auto no_bias_layer = std::make_unique<DenseLayer>(3, "relu", false);
    no_bias_layer->set_input_shape(4);
    
    MatrixXd no_bias_input(2, 4);
    no_bias_input.setRandom();
    MatrixXd original_output = no_bias_layer->forward(no_bias_input, true);
    
    // Salva
    std::ofstream ofs(filename, std::ios::binary);
    no_bias_layer->serialize(ofs);
    ofs.close();
    
    // Carica
    DenseLayer loaded_layer(3, "relu", false);
    std::ifstream ifs(filename, std::ios::binary);
    loaded_layer.deserialize(ifs);
    ifs.close();
    
    // Verifica
    MatrixXd original_weights = no_bias_layer->get_weights();
    MatrixXd loaded_weights = loaded_layer.get_weights();
    EXPECT_TRUE(original_weights.isApprox(loaded_weights));
    
    MatrixXd loaded_output = loaded_layer.forward(no_bias_input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}

TEST_F(DenseLayerSerializationTest, SerializeToStream) {
    MatrixXd original_output = layer->forward(input, true);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    DenseLayer loaded_layer(3, "relu", true);
    loaded_layer.deserialize(ss);
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}