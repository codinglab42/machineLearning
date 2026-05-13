// tests/cpp/unit/serialization/test_simple_rnn_layer_serialization.cpp
#include <gtest/gtest.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "components/layers/simple_rnn_layer.h"

using namespace layers;
using namespace Eigen;

class SimpleRNNLayerSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<SimpleRNNLayer>(16, 8, "tanh", true);
        layer->set_input_shape(8);
        
        input.resize(4, 8);
        input.setRandom();
        
        filename = "test_simple_rnn_layer.bin";
    }
    
    void TearDown() override {
        std::remove(filename.c_str());
    }
    
    std::unique_ptr<SimpleRNNLayer> layer;
    MatrixXd input;
    std::string filename;
};

TEST_F(SimpleRNNLayerSerializationTest, SaveAndLoad) {
    MatrixXd original_output = layer->forward(input, true);
    
    std::ofstream ofs(filename, std::ios::binary);
    layer->serialize(ofs);
    ofs.close();
    
    SimpleRNNLayer loaded_layer(16, 8, "tanh", true);
    std::ifstream ifs(filename, std::ios::binary);
    loaded_layer.deserialize(ifs);
    ifs.close();
    
    MatrixXd original_weights = layer->get_weights();
    MatrixXd loaded_weights = loaded_layer.get_weights();
    EXPECT_TRUE(original_weights.isApprox(loaded_weights));
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}

TEST_F(SimpleRNNLayerSerializationTest, WithoutBias) {
    auto no_bias_layer = std::make_unique<SimpleRNNLayer>(16, 8, "tanh", false);
    no_bias_layer->set_input_shape(8);
    MatrixXd original_output = no_bias_layer->forward(input, true);
    
    std::stringstream ss;
    no_bias_layer->serialize(ss);
    
    SimpleRNNLayer loaded_layer(16, 8, "tanh", false);
    loaded_layer.deserialize(ss);
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}