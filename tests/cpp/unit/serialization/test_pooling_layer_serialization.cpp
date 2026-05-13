// tests/cpp/unit/serialization/test_pooling_layer_serialization.cpp
#include <gtest/gtest.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "components/layers/pooling_layer.h"

using namespace layers;
using namespace Eigen;

class PoolingLayerSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<PoolingLayer>(2, 2, PoolingLayer::MAX, 1);
        layer->set_input_shape(784);  // 28x28 image
        
        input.resize(2, 784);
        input.setRandom();
        
        filename = "test_pooling_layer.bin";
    }
    
    void TearDown() override {
        std::remove(filename.c_str());
    }
    
    std::unique_ptr<PoolingLayer> layer;
    MatrixXd input;
    std::string filename;
};

TEST_F(PoolingLayerSerializationTest, SaveAndLoad) {
    MatrixXd original_output = layer->forward(input, true);
    
    std::ofstream ofs(filename, std::ios::binary);
    layer->serialize(ofs);
    ofs.close();
    
    PoolingLayer loaded_layer(2, 2, PoolingLayer::MAX, 1);
    std::ifstream ifs(filename, std::ios::binary);
    loaded_layer.deserialize(ifs);
    ifs.close();
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}

TEST_F(PoolingLayerSerializationTest, AveragePooling) {
    auto avg_layer = std::make_unique<PoolingLayer>(2, 2, PoolingLayer::AVG, 1);
    avg_layer->set_input_shape(784);
    MatrixXd original_output = avg_layer->forward(input, true);
    
    std::stringstream ss;
    avg_layer->serialize(ss);
    
    PoolingLayer loaded_layer(2, 2, PoolingLayer::AVG, 1);
    loaded_layer.deserialize(ss);
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}