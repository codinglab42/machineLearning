// tests/cpp/unit/layers/test_gru_layer_deep.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/layers/gru_layer.h"

using namespace layers;
using namespace Eigen;

class GRULayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        units = 16;
        input_size = 8;
        batch_size = 4;
        
        layer = std::make_unique<GRULayer>(units, input_size, "tanh", "sigmoid", true);
        layer->set_input_shape(input_size);
        
        input.resize(batch_size, input_size);
        input.setRandom();
    }
    
    std::unique_ptr<GRULayer> layer;
    int units;
    int input_size;
    int batch_size;
    MatrixXd input;
};

TEST_F(GRULayerTest, Construction) {
    EXPECT_TRUE(layer->has_weights());
    EXPECT_TRUE(layer->get_use_bias());
    EXPECT_EQ(layer->get_type(), "GRULayer");
    EXPECT_EQ(layer->get_layer_type(), LayerType::GRU);
    EXPECT_EQ(layer->get_input_size(), input_size);
    EXPECT_EQ(layer->get_output_size(), units);
}

TEST_F(GRULayerTest, ForwardShape) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), batch_size);
    EXPECT_EQ(output.cols(), units);
}

TEST_F(GRULayerTest, ResetState) {
    layer->forward(input);
    layer->reset_state();
    
    MatrixXd hidden = layer->get_hidden_state();
    EXPECT_EQ(hidden.rows(), 0);
}

TEST_F(GRULayerTest, BackwardShape) {
    MatrixXd output = layer->forward(input, true);
    MatrixXd gradient = MatrixXd::Ones(batch_size, units);
    
    MatrixXd dX = layer->backward(gradient);
    
    EXPECT_EQ(dX.rows(), batch_size);
    EXPECT_EQ(dX.cols(), input_size);
}

TEST_F(GRULayerTest, ParameterCount) {
    int params = layer->get_parameter_count();
    int expected = 3 * (input_size * units + units * units + units);
    EXPECT_EQ(params, expected);
}