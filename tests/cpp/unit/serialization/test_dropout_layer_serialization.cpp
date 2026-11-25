// tests/cpp/unit/serialization/test_dropout_layer_serialization.cpp
#include <gtest/gtest.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "components/layers/dropout_layer.h"

using namespace layers;
using namespace Eigen;

class DropoutLayerSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<DropoutLayer>(0.3);
        layer->set_input_shape(10);
        
        input.resize(4, 10);
        input.setRandom();
        
        filename = "test_dropout_layer.bin";
    }
    
    void TearDown() override {
        std::remove(filename.c_str());
    }
    
    std::unique_ptr<DropoutLayer> layer;
    MatrixXd input;
    std::string filename;
};

TEST_F(DropoutLayerSerializationTest, SaveAndLoadInferenceMode) {
    // Inference mode (training = false)
    MatrixXd original_output = layer->forward(input, false);
    
    std::ofstream ofs(filename, std::ios::binary);
    layer->serialize(ofs);
    ofs.close();
    
    DropoutLayer loaded_layer(0.3);
    std::ifstream ifs(filename, std::ios::binary);
    loaded_layer.deserialize(ifs);
    ifs.close();
    
    // Verifica che la configurazione sia la stessa
    EXPECT_EQ(layer->get_type(), loaded_layer.get_type());
    EXPECT_EQ(layer->get_config(), loaded_layer.get_config());
    
    // Verifica che l'output sia identico in inference mode
    MatrixXd loaded_output = loaded_layer.forward(input, false);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}

TEST_F(DropoutLayerSerializationTest, SaveAndLoadTrainingMode) {
    layer->forward(input, true);
    
    std::ofstream ofs(filename, std::ios::binary);
    layer->serialize(ofs);
    ofs.close();
    MatrixXd expected_output = layer->forward(input, true);
    
    DropoutLayer loaded_layer(0.3);
    std::ifstream ifs(filename, std::ios::binary);
    loaded_layer.deserialize(ifs);
    ifs.close();
    
    // Verifica che la configurazione sia la stessa
    EXPECT_EQ(layer->get_rate(), loaded_layer.get_rate());
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(expected_output.isApprox(loaded_output, 1e-12));
    EXPECT_FALSE(loaded_output.hasNaN());
}

TEST_F(DropoutLayerSerializationTest, SerializeToStream) {
    MatrixXd original_output = layer->forward(input, false);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    DropoutLayer loaded_layer(0.3);
    loaded_layer.deserialize(ss);
    
    MatrixXd loaded_output = loaded_layer.forward(input, false);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}

TEST_F(DropoutLayerSerializationTest, DifferentRateValues) {
    std::vector<double> rates = {0.1, 0.3, 0.5, 0.7, 0.9};
    
    for (double rate : rates) {
        auto test_layer = std::make_unique<DropoutLayer>(rate);
        test_layer->set_input_shape(10);
        
        std::stringstream ss;
        test_layer->serialize(ss);
        
        DropoutLayer loaded_layer(rate);
        loaded_layer.deserialize(ss);
        
        EXPECT_EQ(test_layer->get_rate(), loaded_layer.get_rate());
    }
}