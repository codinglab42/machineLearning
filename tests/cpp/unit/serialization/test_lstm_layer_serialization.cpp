// tests/cpp/unit/serialization/test_lstm_layer_serialization.cpp
#include <gtest/gtest.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "components/layers/lstm_layer.h"

using namespace layers;
using namespace Eigen;

class LSTMLayerSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<LSTMLayer>(16, 8, "tanh", "sigmoid", true);
        layer->set_input_shape(8);
        
        input.resize(4, 8);
        input.setRandom();
        
        filename = "test_lstm_layer.bin";
    }
    
    void TearDown() override {
        std::remove(filename.c_str());
    }
    
    std::unique_ptr<LSTMLayer> layer;
    MatrixXd input;
    std::string filename;
};

TEST_F(LSTMLayerSerializationTest, SaveAndLoad) {
    MatrixXd original_output = layer->forward(input, true);
    
    std::ofstream ofs(filename, std::ios::binary);
    layer->serialize(ofs);
    ofs.close();
    
    LSTMLayer loaded_layer(16, 8, "tanh", "sigmoid", true);
    std::ifstream ifs(filename, std::ios::binary);
    loaded_layer.deserialize(ifs);
    ifs.close();
    
    MatrixXd original_weights = layer->get_weights();
    MatrixXd loaded_weights = loaded_layer.get_weights();
    EXPECT_TRUE(original_weights.isApprox(loaded_weights));
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}