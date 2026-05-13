// tests/cpp/unit/serialization/test_conv2d_layer_serialization.cpp
#include <gtest/gtest.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "components/layers/conv2d_layer.h"

using namespace layers;
using namespace Eigen;

class Conv2DLayerSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<Conv2DLayer>(4, 3, 1, "valid", "relu");
        layer->set_input_shape(784);  // 28x28 image
        
        input.resize(2, 784);
        input.setRandom();
        
        filename = "test_conv2d_layer.bin";
    }
    
    void TearDown() override {
        std::remove(filename.c_str());
    }
    
    std::unique_ptr<Conv2DLayer> layer;
    MatrixXd input;
    std::string filename;
};

TEST_F(Conv2DLayerSerializationTest, SaveAndLoad) {
    MatrixXd original_output = layer->forward(input, true);
    
    std::ofstream ofs(filename, std::ios::binary);
    layer->serialize(ofs);
    ofs.close();
    
    Conv2DLayer loaded_layer(4, 3, 1, "valid", "relu");
    std::ifstream ifs(filename, std::ios::binary);
    loaded_layer.deserialize(ifs);
    ifs.close();
    
    MatrixXd original_weights = layer->get_weights();
    MatrixXd loaded_weights = loaded_layer.get_weights();
    EXPECT_TRUE(original_weights.isApprox(loaded_weights));
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}

TEST_F(Conv2DLayerSerializationTest, SerializeToStream) {
    MatrixXd original_output = layer->forward(input, true);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    Conv2DLayer loaded_layer(4, 3, 1, "valid", "relu");
    loaded_layer.deserialize(ss);
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}