// test/layers/simple_rnn_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/simple_rnn_layer.h"

using namespace layers;

class SimpleRNNLayerTest : public ::testing::TestWithParam<std::tuple<std::string, bool>> {
protected:
    void SetUp() override {
        auto [activation, use_bias] = GetParam();
        layer = std::make_unique<SimpleRNNLayer>(4, 3, activation, use_bias);
        layer->set_input_shape(3);
    }
    
    std::unique_ptr<SimpleRNNLayer> layer;
};

TEST_P(SimpleRNNLayerTest, ConstructorValidation) {
    EXPECT_THROW(SimpleRNNLayer(0, 3, "tanh", true), std::invalid_argument);
    EXPECT_THROW(SimpleRNNLayer(4, 0, "tanh", true), std::invalid_argument);
    EXPECT_NO_THROW(SimpleRNNLayer(4, 3, "tanh", true));
}

TEST_P(SimpleRNNLayerTest, SetInputShape) {
    EXPECT_THROW(layer->set_input_shape(-1), std::invalid_argument);
    EXPECT_NO_THROW(layer->set_input_shape(5));
    EXPECT_EQ(layer->get_input_size(), 5);
}

TEST_P(SimpleRNNLayerTest, Forward) {
    Eigen::MatrixXd input(2, 3);
    input.setRandom();
    
    auto output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 4);
}

TEST_P(SimpleRNNLayerTest, StatePersistence) {
    Eigen::MatrixXd input1(1, 3);
    input1.setRandom();
    
    auto output1 = layer->forward(input1);
    auto state1 = layer->get_hidden_state();
    
    Eigen::MatrixXd input2(1, 3);
    input2.setRandom();
    
    auto output2 = layer->forward(input2);
    auto state2 = layer->get_hidden_state();
    
    // Lo stato dovrebbe essere diverso dopo secondo forward
    EXPECT_FALSE(state1.isApprox(state2));
}

TEST_P(SimpleRNNLayerTest, ResetState) {
    Eigen::MatrixXd input(1, 3);
    input.setRandom();
    
    layer->forward(input);
    EXPECT_FALSE(layer->get_hidden_state().isZero());
    
    layer->reset_state();
    EXPECT_TRUE(layer->get_hidden_state().size() == 0);
}

TEST_P(SimpleRNNLayerTest, Backward) {
    Eigen::MatrixXd input(3, 3);
    input.setRandom();
    
    auto output = layer->forward(input, true);
    
    Eigen::MatrixXd gradient(3, 4);
    gradient.setOnes();
    
    auto dX = layer->backward(gradient, 0.01);
    
    EXPECT_EQ(dX.rows(), input.rows());
    EXPECT_EQ(dX.cols(), input.cols());
    
    // Verifica aggiornamento pesi
    auto weights_before = layer->get_weights();
    layer->backward(gradient, 0.01);
    auto weights_after = layer->get_weights();
    
    EXPECT_FALSE(weights_before.isApprox(weights_after));
}

TEST_P(SimpleRNNLayerTest, BackwardInference) {
    Eigen::MatrixXd input(3, 3);
    input.setRandom();
    
    auto output = layer->forward(input, false); // inference
    
    Eigen::MatrixXd gradient(3, 4);
    gradient.setOnes();
    
    auto dX = layer->backward(gradient, 0.01);
    
    // In inference, il gradiente passa inalterato
    EXPECT_TRUE(dX.isApprox(gradient));
}

TEST_P(SimpleRNNLayerTest, ActivationFunctions) {
    Eigen::MatrixXd input(1, 3);
    input.setOnes();
    
    auto output = layer->forward(input);
    
    auto [activation, use_bias] = GetParam();
    if (activation == "tanh") {
        EXPECT_GE(output.minCoeff(), -1);
        EXPECT_LE(output.maxCoeff(), 1);
    } else if (activation == "relu") {
        EXPECT_GE(output.minCoeff(), 0);
    } else if (activation == "sigmoid") {
        EXPECT_GE(output.minCoeff(), 0);
        EXPECT_LE(output.maxCoeff(), 1);
    }
}

TEST_P(SimpleRNNLayerTest, Serialization) {
    Eigen::MatrixXd input(2, 3);
    input.setRandom();
    layer->forward(input);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    auto new_layer = std::make_unique<SimpleRNNLayer>(4, 3, "tanh", true);
    new_layer->deserialize(ss);
    
    EXPECT_EQ(new_layer->get_config(), layer->get_config());
    
    auto weights_orig = layer->get_weights();
    auto weights_new = new_layer->get_weights();
    EXPECT_TRUE(weights_orig.isApprox(weights_new));
}

TEST_P(SimpleRNNLayerTest, ParameterCount) {
    auto [activation, use_bias] = GetParam();
    int expected = 3*4 + 4*4; // kernel + recurrent
    if (use_bias) expected += 4; // bias
    
    EXPECT_EQ(layer->get_parameter_count(), expected);
}

TEST_P(SimpleRNNLayerTest, BiasManagement) {
    auto [activation, use_bias] = GetParam();
    
    if (use_bias) {
        auto biases = layer->get_biases();
        EXPECT_EQ(biases.size(), 4);
        
        Eigen::VectorXd new_biases(4);
        new_biases << 1.0, 2.0, 3.0, 4.0;
        layer->set_biases(new_biases);
        
        auto updated_biases = layer->get_biases();
        EXPECT_TRUE(updated_biases.isApprox(new_biases));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SimpleRNNVariants,
    SimpleRNNLayerTest,
    ::testing::Combine(
        ::testing::Values("tanh", "relu", "sigmoid"),
        ::testing::Bool()
    )
);