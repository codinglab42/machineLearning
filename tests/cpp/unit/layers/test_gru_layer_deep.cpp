// test/layers/gru_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/gru_layer.h"

using namespace layers;

class GRULayerTest : public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        use_bias = GetParam();
        layer = std::make_unique<GRULayer>(4, 3, "tanh", "sigmoid", use_bias);
        layer->set_input_shape(3);
    }
    
    bool use_bias;
    std::unique_ptr<GRULayer> layer;
};

TEST_P(GRULayerTest, ConstructorValidation) {
    EXPECT_THROW(GRULayer(0, 3, "tanh", "sigmoid", true), std::invalid_argument);
    EXPECT_THROW(GRULayer(4, 0, "tanh", "sigmoid", true), std::invalid_argument);
    EXPECT_NO_THROW(GRULayer(4, 3, "tanh", "sigmoid", true));
}

TEST_P(GRULayerTest, Forward) {
    Eigen::MatrixXd input(2, 3);
    input.setRandom();
    
    auto output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 4);
}

TEST_P(GRULayerTest, StatePersistence) {
    Eigen::MatrixXd input1(1, 3);
    input1.setRandom();
    
    auto output1 = layer->forward(input1);
    auto state1 = layer->get_hidden_state();
    
    Eigen::MatrixXd input2(1, 3);
    input2.setRandom();
    
    auto output2 = layer->forward(input2);
    auto state2 = layer->get_hidden_state();
    
    EXPECT_FALSE(state1.isApprox(state2));
}

TEST_P(GRULayerTest, ResetState) {
    Eigen::MatrixXd input(1, 3);
    input.setRandom();
    
    layer->forward(input);
    EXPECT_FALSE(layer->get_hidden_state().isZero());
    
    layer->reset_state();
    EXPECT_TRUE(layer->get_hidden_state().size() == 0);
}

TEST_P(GRULayerTest, Backward) {
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

TEST_P(GRULayerTest, GateComputations) {
    Eigen::MatrixXd input(1, 3);
    input << 1, 1, 1;
    
    auto output = layer->forward(input, true);
    
    // Verifica che i gate siano nel range [0,1] (sigmoid)
    auto cache = std::dynamic_pointer_cast<layers::GRUCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    
    EXPECT_GE(cache->reset_gates[0].minCoeff(), 0);
    EXPECT_LE(cache->reset_gates[0].maxCoeff(), 1);
    EXPECT_GE(cache->update_gates[0].minCoeff(), 0);
    EXPECT_LE(cache->update_gates[0].maxCoeff(), 1);
}

TEST_P(GRULayerTest, Serialization) {
    Eigen::MatrixXd input(2, 3);
    input.setRandom();
    layer->forward(input);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    auto new_layer = std::make_unique<GRULayer>(4, 3, "tanh", "sigmoid", true);
    new_layer->deserialize(ss);
    
    EXPECT_EQ(new_layer->get_config(), layer->get_config());
    
    auto weights_orig = layer->get_weights();
    auto weights_new = new_layer->get_weights();
    EXPECT_TRUE(weights_orig.isApprox(weights_new));
}

TEST_P(GRULayerTest, ParameterCount) {
    // kernel: 3 * (input_size * units) = 3*3*4 = 36
    // recurrent: 3 * (units * units) = 3*4*4 = 48
    // bias: 3 * units = 12 se use_bias
    int expected = 36 + 48;
    if (use_bias) expected += 12;
    
    EXPECT_EQ(layer->get_parameter_count(), expected);
}

TEST_P(GRULayerTest, BiasManagement) {
    if (use_bias) {
        auto biases = layer->get_biases();
        EXPECT_EQ(biases.size(), 12); // 3 * 4
        
        Eigen::VectorXd new_biases(12);
        new_biases.setRandom();
        layer->set_biases(new_biases);
        
        auto updated_biases = layer->get_biases();
        EXPECT_TRUE(updated_biases.isApprox(new_biases));
    }
}

INSTANTIATE_TEST_SUITE_P(
    GRUVariants,
    GRULayerTest,
    ::testing::Bool()
);