// test/layers/dense_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/dense_layer.h"

using namespace layers;

class DenseLayerTest : public ::testing::TestWithParam<std::tuple<std::string, bool>> {
protected:
    void SetUp() override {
        auto [activation, use_bias] = GetParam();
        layer = std::make_unique<DenseLayer>(3, activation, use_bias);
        layer->set_input_shape(5);
    }
    
    std::unique_ptr<DenseLayer> layer;
};

TEST_P(DenseLayerTest, ConstructorValidation) {
    EXPECT_THROW(DenseLayer(0, "relu", true), std::invalid_argument);
    EXPECT_NO_THROW(DenseLayer(1, "relu", true));
}

TEST_P(DenseLayerTest, SetInputShape) {
    EXPECT_THROW(layer->set_input_shape(-1), std::invalid_argument);
    EXPECT_NO_THROW(layer->set_input_shape(10));
    
    auto weights = layer->get_weights();
    if (std::get<1>(GetParam())) { // use_bias
        EXPECT_EQ(weights.cols(), 4); // 3 weights + 1 bias
    } else {
        EXPECT_EQ(weights.cols(), 3);
    }
}

TEST_P(DenseLayerTest, Forward) {
    Eigen::MatrixXd input(2, 5);
    input << 1, 2, 3, 4, 5,
             2, 3, 4, 5, 6;
    
    auto output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 3);
}

TEST_P(DenseLayerTest, ForwardWithTrainingFlag) {
    Eigen::MatrixXd input(2, 5);
    input.setRandom();
    
    auto output1 = layer->forward(input, true);
    auto output2 = layer->forward(input, false);
    
    EXPECT_EQ(output1.rows(), output2.rows());
    EXPECT_EQ(output1.cols(), output2.cols());
}

TEST_P(DenseLayerTest, Backward) {
    Eigen::MatrixXd input(3, 5);
    input.setRandom();
    
    auto output = layer->forward(input);
    
    Eigen::MatrixXd gradient(3, 3);
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

TEST_P(DenseLayerTest, ActivationFunctions) {
    Eigen::MatrixXd input(1, 5);
    input.setOnes();
    
    auto output = layer->forward(input);
    
    auto [activation, use_bias] = GetParam();
    if (activation == "relu") {
        EXPECT_GE(output.minCoeff(), 0);
    } else if (activation == "sigmoid") {
        EXPECT_GE(output.minCoeff(), 0);
        EXPECT_LE(output.maxCoeff(), 1);
    } else if (activation == "tanh") {
        EXPECT_GE(output.minCoeff(), -1);
        EXPECT_LE(output.maxCoeff(), 1);
    } else if (activation == "softmax") {
        auto sum = output.row(0).sum();
        EXPECT_NEAR(sum, 1.0, 1e-10);
    }
}

TEST_P(DenseLayerTest, Serialization) {
    Eigen::MatrixXd input(2, 5);
    input.setRandom();
    layer->forward(input);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    auto new_layer = std::make_unique<DenseLayer>(3, "relu", true);
    new_layer->deserialize(ss);
    
    EXPECT_EQ(new_layer->get_config(), layer->get_config());
    
    auto weights_orig = layer->get_weights();
    auto weights_new = new_layer->get_weights();
    EXPECT_TRUE(weights_orig.isApprox(weights_new));
}

TEST_P(DenseLayerTest, ParameterCount) {
    auto [activation, use_bias] = GetParam();
    int expected = 5 * 3; // weights
    if (use_bias) expected += 3; // bias
    
    EXPECT_EQ(layer->get_parameter_count(), expected);
}

TEST_P(DenseLayerTest, BiasManagement) {
    auto [activation, use_bias] = GetParam();
    
    if (use_bias) {
        auto biases = layer->get_biases();
        EXPECT_EQ(biases.size(), 3);
        
        Eigen::VectorXd new_biases(3);
        new_biases << 1.0, 2.0, 3.0;
        layer->set_biases(new_biases);
        
        auto updated_biases = layer->get_biases();
        EXPECT_TRUE(updated_biases.isApprox(new_biases));
    } else {
        EXPECT_TRUE(layer->get_biases().size() == 0);
    }
}

INSTANTIATE_TEST_SUITE_P(
    DenseLayerVariants,
    DenseLayerTest,
    ::testing::Combine(
        ::testing::Values("relu", "sigmoid", "tanh", "softmax", "linear"),
        ::testing::Bool()
    )
);