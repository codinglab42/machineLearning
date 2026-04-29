// test/layers/flatten_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/flatten_layer.h"

using namespace layers;

class FlattenLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<FlattenLayer>();
    }
    
    std::unique_ptr<FlattenLayer> layer;
};

TEST_F(FlattenLayerTest, Forward) {
    Eigen::MatrixXd input(3, 5);
    input.setRandom();
    
    auto output = layer->forward(input);
    
    // Flatten non cambia la forma
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 5);
    EXPECT_TRUE(output.isApprox(input));
}

TEST_F(FlattenLayerTest, ForwardWithTrainingFlag) {
    Eigen::MatrixXd input(3, 5);
    input.setRandom();
    
    auto output1 = layer->forward(input, true);
    auto output2 = layer->forward(input, false);
    
    EXPECT_TRUE(output1.isApprox(output2));
    EXPECT_TRUE(output1.isApprox(input));
}

TEST_F(FlattenLayerTest, Backward) {
    Eigen::MatrixXd input(2, 4);
    input.setRandom();
    
    layer->forward(input);
    
    Eigen::MatrixXd gradient(2, 4);
    gradient.setOnes();
    
    auto dX = layer->backward(gradient);
    
    // Gradiente passa inalterato
    EXPECT_TRUE(dX.isApprox(gradient));
}

TEST_F(FlattenLayerTest, BackwardWithoutForward) {
    Eigen::MatrixXd gradient(2, 4);
    gradient.setOnes();
    
    EXPECT_THROW(layer->backward(gradient), std::runtime_error);
}

TEST_F(FlattenLayerTest, Serialization) {
    Eigen::MatrixXd input(2, 5);
    input.setRandom();
    layer->forward(input);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    auto new_layer = std::make_unique<FlattenLayer>();
    new_layer->deserialize(ss);
    
    EXPECT_EQ(new_layer->get_config(), layer->get_config());
    EXPECT_EQ(new_layer->get_input_size(), 5);
}

TEST_F(FlattenLayerTest, NoParameters) {
    EXPECT_EQ(layer->get_parameter_count(), 0);
    EXPECT_FALSE(layer->has_weights());
    EXPECT_TRUE(layer->get_weights().size() == 0);
}

TEST_F(FlattenLayerTest, SetInputShape) {
    layer->set_input_shape(10);
    EXPECT_EQ(layer->get_input_size(), 10);
}

TEST_F(FlattenLayerTest, ClearCache) {
    Eigen::MatrixXd input(2, 5);
    input.setRandom();
    
    layer->forward(input);
    EXPECT_NE(layer->get_cache(), nullptr);
    
    layer->clear_cache();
    EXPECT_EQ(layer->get_cache(), nullptr);
}