// tests/cpp/unit/serialization/test_batch_norm_layer_serialization.cpp
#include <gtest/gtest.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "components/layers/batch_norm_layer.h"

using namespace layers;
using namespace Eigen;

class BatchNormLayerSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<BatchNormLayer>(1e-5, 0.9);
        layer->set_input_shape(8);
        
        input.resize(4, 8);
        input.setRandom();
        
        filename = "test_batchnorm_layer.bin";
    }
    
    void TearDown() override {
        std::remove(filename.c_str());
    }
    
    std::unique_ptr<BatchNormLayer> layer;
    MatrixXd input;
    std::string filename;
};

TEST_F(BatchNormLayerSerializationTest, SaveAndLoad) {
    MatrixXd original_output = layer->forward(input, true);
    
    std::ofstream ofs(filename, std::ios::binary);
    layer->serialize(ofs);
    ofs.close();
    
    BatchNormLayer loaded_layer(1e-5, 0.9);
    std::ifstream ifs(filename, std::ios::binary);
    loaded_layer.deserialize(ifs);
    ifs.close();
    
    MatrixXd original_weights = layer->get_weights();
    MatrixXd loaded_weights = loaded_layer.get_weights();
    EXPECT_TRUE(original_weights.isApprox(loaded_weights));
    
    MatrixXd loaded_output = loaded_layer.forward(input, true);
    EXPECT_TRUE(original_output.isApprox(loaded_output, 1e-6));
}