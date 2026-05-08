// tests/cpp/unit/layers/test_simple_rnn_layer_deep.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/layers/simple_rnn_layer.h"

using namespace layers;
using namespace Eigen;

class SimpleRNNLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        units = 16;
        input_size = 8;
        batch_size = 4;
        
        layer = std::make_unique<SimpleRNNLayer>(units, input_size, "tanh", true);
        
        // IMPORTANTE: set_input_shape DEVE essere chiamato PRIMA di forward
        layer->set_input_shape(input_size);
        
        input.resize(batch_size, input_size);
        input.setRandom();
    }
    
    std::unique_ptr<SimpleRNNLayer> layer;
    int units;
    int input_size;
    int batch_size;
    MatrixXd input;
};

TEST_F(SimpleRNNLayerTest, Construction) {
    EXPECT_TRUE(layer->has_weights());
    EXPECT_TRUE(layer->get_use_bias());
    EXPECT_EQ(layer->get_type(), "SimpleRNNLayer");
    EXPECT_EQ(layer->get_layer_type(), LayerType::SIMPLE_RNN);
    EXPECT_EQ(layer->get_input_size(), input_size);
    EXPECT_EQ(layer->get_output_size(), units);
}

TEST_F(SimpleRNNLayerTest, ForwardShape) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), batch_size);
    EXPECT_EQ(output.cols(), units);
}

TEST_F(SimpleRNNLayerTest, ResetState) {
    layer->forward(input);
    layer->reset_state();
    
    MatrixXd hidden = layer->get_hidden_state();
    EXPECT_EQ(hidden.rows(), 0);
}

TEST_F(SimpleRNNLayerTest, BackwardShape) {
    // Forza il ridimensionamento corretto
    layer->set_input_shape(input_size);
    
    MatrixXd output = layer->forward(input, true);
    MatrixXd gradient = MatrixXd::Ones(batch_size, units);
    
    MatrixXd dX = layer->backward(gradient);
    
    EXPECT_EQ(dX.rows(), batch_size);
    EXPECT_EQ(dX.cols(), input_size);
}

TEST_F(SimpleRNNLayerTest, ParameterCount) {
    int params = layer->get_parameter_count();
    int expected = input_size * units + units * units + units;
    EXPECT_EQ(params, expected);
}