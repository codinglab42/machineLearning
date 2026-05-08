// tests/cpp/unit/layer/test_pooling_layer.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/layers/pooling_layer.h"

using namespace layers;
using namespace Eigen;

class PoolingLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<PoolingLayer>(2, 2, PoolingLayer::MAX, 1);
        
        // 4x4 image, batch=2
        input.resize(2, 16);
        input.setRandom();
        layer->set_input_shape(16);
    }
    
    std::unique_ptr<PoolingLayer> layer;
    MatrixXd input;
};

TEST_F(PoolingLayerTest, Construction) {
    EXPECT_FALSE(layer->has_weights());
    EXPECT_FALSE(layer->get_use_bias());
    EXPECT_EQ(layer->get_type(), "PoolingLayer");
    EXPECT_EQ(layer->get_layer_type(), LayerType::MAX_POOLING);
    EXPECT_EQ(layer->get_parameter_count(), 0);
}

TEST_F(PoolingLayerTest, ForwardShape) {
    MatrixXd output = layer->forward(input);
    
    // After 2x2 pooling, 4x4 -> 2x2, channels=1 -> 4 features per sample
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 4);
}

TEST_F(PoolingLayerTest, ForwardMaxPooling) {
    MatrixXd output = layer->forward(input);
    
    for (int i = 0; i < output.size(); ++i) {
        EXPECT_FALSE(std::isnan(output(i)));
        EXPECT_FALSE(std::isinf(output(i)));
    }
}

TEST_F(PoolingLayerTest, ForwardAveragePooling) {
    auto avg_layer = std::make_unique<PoolingLayer>(2, 2, PoolingLayer::AVG, 1);
    avg_layer->set_input_shape(16);
    
    MatrixXd output = avg_layer->forward(input);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 4);
}

TEST_F(PoolingLayerTest, BackwardShape) {
    MatrixXd output = layer->forward(input, true);
    MatrixXd gradient = MatrixXd::Ones(2, 4);
    
    MatrixXd dX = layer->backward(gradient);
    
    EXPECT_EQ(dX.rows(), 2);
    EXPECT_EQ(dX.cols(), 16);
}