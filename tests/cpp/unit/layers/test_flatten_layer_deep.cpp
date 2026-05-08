// tests/cpp/unit/layer/test_flatten_layer.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/layers/flatten_layer.h"

using namespace layers;
using namespace Eigen;

class FlattenLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<FlattenLayer>();
        
        input.resize(2, 12);
        input.setRandom();
    }
    
    std::unique_ptr<FlattenLayer> layer;
    MatrixXd input;
};

TEST_F(FlattenLayerTest, Construction) {
    EXPECT_FALSE(layer->has_weights());
    EXPECT_FALSE(layer->get_use_bias());
    EXPECT_EQ(layer->get_type(), "FlattenLayer");
    EXPECT_EQ(layer->get_layer_type(), LayerType::FLATTEN);
    EXPECT_EQ(layer->get_parameter_count(), 0);
}

TEST_F(FlattenLayerTest, ForwardPassThrough) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), input.rows());
    EXPECT_EQ(output.cols(), input.cols());
    EXPECT_TRUE(output.isApprox(input));
}

TEST_F(FlattenLayerTest, BackwardPassThrough) {
    layer->forward(input);
    MatrixXd gradient = MatrixXd::Random(2, 12);
    
    MatrixXd dX = layer->backward(gradient);
    
    EXPECT_TRUE(dX.isApprox(gradient));
}

TEST_F(FlattenLayerTest, InputSizeRetained) {
    layer->forward(input);
    EXPECT_EQ(layer->get_input_size(), 12);
    EXPECT_EQ(layer->get_output_size(), 12);
}