// tests/cpp/unit/layers/test_conv2d_layer_deep.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/layers/conv2d_layer.h"

using namespace layers;
using namespace Eigen;

class Conv2DLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        filters = 4;
        kernel_size = 3;
        strides = 1;
        
        layer = std::make_unique<Conv2DLayer>(filters, kernel_size, strides, "valid", "relu");
        
        // 28x28 image, 1 channel, batch=1 -> 784 features
        input_size = 784;
        
        // IMPORTANTE: set_input_shape DEVE essere chiamato PRIMA di forward
        layer->set_input_shape(input_size);
        
        // Per Conv2D, input è [batch_size, input_size]
        batch_size = 1;
        input.resize(batch_size, input_size);
        input.setRandom();
    }
    
    std::unique_ptr<Conv2DLayer> layer;
    int filters;
    int kernel_size;
    int strides;
    int input_size;
    int batch_size;
    MatrixXd input;
};

TEST_F(Conv2DLayerTest, Construction) {
    EXPECT_TRUE(layer->has_weights());
    EXPECT_TRUE(layer->get_use_bias());
    EXPECT_EQ(layer->get_type(), "Conv2DLayer");
    EXPECT_EQ(layer->get_layer_type(), LayerType::CONV2D);
    EXPECT_EQ(layer->get_input_size(), input_size);
}

TEST_F(Conv2DLayerTest, ForwardShape) {
    MatrixXd output = layer->forward(input);
    
    // After 3x3 conv on 28x28 with valid padding -> 26x26
    int output_height = 26;
    int output_width = 26;
    int expected_output_size = output_height * output_width * filters;
    
    EXPECT_EQ(output.rows(), batch_size);
    EXPECT_EQ(output.cols(), expected_output_size);
}

TEST_F(Conv2DLayerTest, ForwardOutputFinite) {
    MatrixXd output = layer->forward(input);
    
    for (int i = 0; i < output.size(); ++i) {
        EXPECT_FALSE(std::isnan(output(i)));
        EXPECT_FALSE(std::isinf(output(i)));
    }
}

TEST_F(Conv2DLayerTest, BackwardShape) {
    MatrixXd output = layer->forward(input);
    int output_size = output.cols();
    MatrixXd gradient = MatrixXd::Ones(batch_size, output_size);
    
    MatrixXd dX = layer->backward(gradient);
    
    EXPECT_EQ(dX.rows(), batch_size);
    EXPECT_EQ(dX.cols(), input_size);
}

TEST_F(Conv2DLayerTest, ParameterCount) {
    int params = layer->get_parameter_count();
    int expected = filters * kernel_size * kernel_size + filters;
    EXPECT_EQ(params, expected);
}