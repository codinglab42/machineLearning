// tests/cpp/unit/serialization/test_flatten_layer_serialization.cpp
#include <gtest/gtest.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "components/layers/flatten_layer.h"

using namespace layers;
using namespace Eigen;

class FlattenLayerSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<FlattenLayer>();
        
        // Simula input da un layer convoluzionale
        batch_size = 2;
        channels = 3;
        height = 4;
        width = 4;
        input_size = channels * height * width;  // 48
        
        input.resize(batch_size, input_size);
        input.setRandom();
        layer->set_input_shape(input_size);
        
        filename = "test_flatten_layer.bin";
    }
    
    void TearDown() override {
        std::remove(filename.c_str());
    }
    
    std::unique_ptr<FlattenLayer> layer;
    MatrixXd input;
    int batch_size;
    int channels;
    int height;
    int width;
    int input_size;
    std::string filename;
};

TEST_F(FlattenLayerSerializationTest, SaveAndLoad) {
    MatrixXd original_output = layer->forward(input, false);
    
    std::ofstream ofs(filename, std::ios::binary);
    layer->serialize(ofs);
    ofs.close();
    
    FlattenLayer loaded_layer;
    std::ifstream ifs(filename, std::ios::binary);
    loaded_layer.deserialize(ifs);
    ifs.close();
    
    // Verifica la configurazione
    EXPECT_EQ(layer->get_type(), loaded_layer.get_type());
    
    // Imposta input shape per il layer caricato
    loaded_layer.set_input_shape(input_size);
    
    // Verifica che l'output sia identico
    MatrixXd loaded_output = loaded_layer.forward(input, false);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}

TEST_F(FlattenLayerSerializationTest, SerializeToStream) {
    MatrixXd original_output = layer->forward(input, false);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    FlattenLayer loaded_layer;
    loaded_layer.deserialize(ss);
    loaded_layer.set_input_shape(input_size);
    
    MatrixXd loaded_output = loaded_layer.forward(input, false);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}

TEST_F(FlattenLayerSerializationTest, DifferentInputSizes) {
    std::vector<int> input_sizes = {12, 24, 48, 96};
    
    for (int size : input_sizes) {
        auto test_layer = std::make_unique<FlattenLayer>();
        test_layer->set_input_shape(size);
        
        std::stringstream ss;
        test_layer->serialize(ss);
        
        FlattenLayer loaded_layer;
        loaded_layer.deserialize(ss);
        loaded_layer.set_input_shape(size);
        
        EXPECT_EQ(test_layer->get_input_size(), loaded_layer.get_input_size());
    }
}