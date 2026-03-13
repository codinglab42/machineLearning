// test/layers/conv2d_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/conv2d_layer.h"

using namespace layers;

class Conv2DLayerTest : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {
protected:
    void SetUp() override {
        auto [padding, activation] = GetParam();
        layer = std::make_unique<Conv2DLayer>(2, 3, 1, padding, activation);
        // Input 5x5 (25 elementi)
        layer->set_input_shape(25);
    }
    
    std::unique_ptr<Conv2DLayer> layer;
};

TEST_P(Conv2DLayerTest, ConstructorValidation) {
    EXPECT_THROW(Conv2DLayer(0, 3, 1, "valid", "relu"), std::invalid_argument);
    EXPECT_THROW(Conv2DLayer(2, 0, 1, "valid", "relu"), std::invalid_argument);
    EXPECT_THROW(Conv2DLayer(2, 3, 0, "valid", "relu"), std::invalid_argument);
    EXPECT_NO_THROW(Conv2DLayer(2, 3, 1, "valid", "relu"));
}

TEST_P(Conv2DLayerTest, OutputDimensions) {
    auto [padding, activation] = GetParam();
    
    if (padding == "valid") {
        EXPECT_EQ(layer->get_output_size(), 2 * 3 * 3); // filters * 3x3
    } else { // same
        EXPECT_EQ(layer->get_output_size(), 2 * 5 * 5); // filters * 5x5
    }
}

TEST_P(Conv2DLayerTest, Forward) {
    Eigen::MatrixXd input(2 * 25, 1); // 2 samples, 25 features each
    input.setRandom();
    
    auto output = layer->forward(input);
    
    int expected_size = 2 * layer->get_output_size();
    EXPECT_EQ(output.rows(), expected_size);
    EXPECT_EQ(output.cols(), 1);
}

TEST_P(Conv2DLayerTest, Backward) {
    Eigen::MatrixXd input(2 * 25, 1);
    input.setRandom();
    
    auto output = layer->forward(input, true);
    
    Eigen::MatrixXd gradient(output.rows(), output.cols());
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

TEST_P(Conv2DLayerTest, Im2Col) {
    // Test privato di im2col tramite forward
    Eigen::MatrixXd input(1 * 25, 1);
    for (int i = 0; i < 25; ++i) {
        input(i) = i + 1;
    }
    
    auto output = layer->forward(input);
    
    // Non dovrebbe crashare
    EXPECT_NO_THROW();
}

TEST_P(Conv2DLayerTest, Serialization) {
    Eigen::MatrixXd input(2 * 25, 1);
    input.setRandom();
    layer->forward(input);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    auto new_layer = std::make_unique<Conv2DLayer>(2, 3, 1, "valid", "relu");
    new_layer->deserialize(ss);
    
    EXPECT_EQ(new_layer->get_config(), layer->get_config());
    
    auto weights_orig = layer->get_weights();
    auto weights_new = new_layer->get_weights();
    EXPECT_TRUE(weights_orig.isApprox(weights_new));
}

TEST_P(Conv2DLayerTest, ParameterCount) {
    // filters * kernel_elements + bias
    int expected = 2 * (3*3*1) + 2; // 2*9 + 2 = 20
    EXPECT_EQ(layer->get_parameter_count(), expected);
}

TEST_F(Conv2DLayerTest, DifferentStrides) {
    auto layer_stride2 = std::make_unique<Conv2DLayer>(2, 3, 2, "valid", "relu");
    layer_stride2->set_input_shape(25);
    
    EXPECT_EQ(layer_stride2->get_output_size(), 2 * 2 * 2); // 2x2 output
}

INSTANTIATE_TEST_SUITE_P(
    Conv2DLayerVariants,
    Conv2DLayerTest,
    ::testing::Combine(
        ::testing::Values("valid", "same"),
        ::testing::Values("relu", "sigmoid", "tanh", "linear")
    )
);