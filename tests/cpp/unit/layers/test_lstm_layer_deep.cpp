// tests/cpp/unit/layers/test_lstm_layer_deep.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/layers/lstm_layer.h"

using namespace layers;
using namespace Eigen;

class LSTMLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        units = 16;
        input_size = 8;
        batch_size = 4;
        
        layer = std::make_unique<LSTMLayer>(units, input_size, "tanh", "sigmoid", true);
        layer->set_input_shape(input_size);
        
        input.resize(batch_size, input_size);
        input.setRandom();
    }
    
    std::unique_ptr<LSTMLayer> layer;
    int units;
    int input_size;
    int batch_size;
    MatrixXd input;
};

TEST_F(LSTMLayerTest, Construction) {
    EXPECT_TRUE(layer->has_weights());
    EXPECT_TRUE(layer->get_use_bias());
    EXPECT_EQ(layer->get_type(), "LSTMLayer");
    EXPECT_EQ(layer->get_layer_type(), LayerType::LSTM);
    EXPECT_EQ(layer->get_input_size(), input_size);
    EXPECT_EQ(layer->get_output_size(), units);
}

TEST_F(LSTMLayerTest, ForwardShape) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), batch_size);
    EXPECT_EQ(output.cols(), units);
}

TEST_F(LSTMLayerTest, ResetState) {
    layer->forward(input);
    layer->reset_state();
    
    MatrixXd hidden = layer->get_hidden_state();
    EXPECT_EQ(hidden.rows(), 0);
}

TEST_F(LSTMLayerTest, BackwardShape) {
    MatrixXd output = layer->forward(input, true);
    MatrixXd gradient = MatrixXd::Ones(batch_size, units);
    
    MatrixXd dX = layer->backward(gradient);
    
    EXPECT_EQ(dX.rows(), batch_size);
    EXPECT_EQ(dX.cols(), input_size);
}

TEST_F(LSTMLayerTest, ParameterCount) {
    int params = layer->get_parameter_count();
    int expected = 4 * (input_size * units + units * units + units);
    EXPECT_EQ(params, expected);
}