// test/layers/dense_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/dense_layer.h"
#include "exceptions/ml_exception.h"

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
    EXPECT_THROW(DenseLayer(0, "relu", true), ml_exception::InvalidParameterException);
    EXPECT_NO_THROW(DenseLayer(1, "relu", true));
}

TEST_P(DenseLayerTest, SetInputShape) {
    EXPECT_THROW(layer->set_input_shape(-1), ml_exception::InvalidParameterException);
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
    EXPECT_TRUE(output.allFinite());
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
    
    auto output = layer->forward(input, true);
    
    Eigen::MatrixXd gradient(3, 3);
    gradient.setOnes();
    
    auto weights_before = layer->get_weights();
    auto dX = layer->backward(gradient);  // LR più alto
    
    EXPECT_EQ(dX.rows(), input.rows());
    EXPECT_EQ(dX.cols(), input.cols());
    EXPECT_TRUE(dX.allFinite());
    
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
    // linear non ha vincoli
}

TEST_P(DenseLayerTest, Serialization) {
    Eigen::MatrixXd input(2, 5);
    input.setRandom();
    
    // Forward per popolare cache e pesi
    auto output_before = layer->forward(input, true);
    auto config_before = layer->get_config();
    auto weights_before = layer->get_weights();
    
    std::stringstream ss;
    layer->serialize(ss);
    
    // Ottieni parametri correnti
    auto [activation, use_bias] = GetParam();
    
    // Crea nuovo layer con STESSI parametri
    auto new_layer = std::make_unique<DenseLayer>(3, activation, use_bias);
    new_layer->set_input_shape(5);
    new_layer->deserialize(ss);
    
    // Verifiche
    EXPECT_EQ(new_layer->get_config(), config_before);
    EXPECT_EQ(new_layer->get_input_size(), layer->get_input_size());
    EXPECT_EQ(new_layer->get_output_size(), layer->get_output_size());
    
    auto weights_new = new_layer->get_weights();
    EXPECT_TRUE(weights_before.isApprox(weights_new));
    
    // Verifica forward
    auto output_new = new_layer->forward(input, false);
    EXPECT_TRUE(output_before.isApprox(output_new));
}

TEST_P(DenseLayerTest, ParameterCount) {
    auto [activation, use_bias] = GetParam();
    int expected = 5 * 3; // weights: input_size * units
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

TEST_P(DenseLayerTest, ClearCache) {
    Eigen::MatrixXd input(2, 5);
    input.setRandom();
    
    layer->forward(input, true);
    
    auto cache = std::dynamic_pointer_cast<DenseCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_valid());
    EXPECT_GT(cache->input_cache.size(), 0);
    
    layer->clear_cache();
    
    auto cache_after = std::dynamic_pointer_cast<DenseCache>(layer->get_cache());
    ASSERT_NE(cache_after, nullptr);
    EXPECT_FALSE(cache_after->is_valid());
    EXPECT_EQ(cache_after->input_cache.size(), 0);
}

TEST_P(DenseLayerTest, EmptyInput) {
    Eigen::MatrixXd empty;
    EXPECT_THROW(layer->forward(empty), ml_exception::EmptyDatasetException);
}

INSTANTIATE_TEST_SUITE_P(
    DenseLayerVariants,
    DenseLayerTest,
    ::testing::Combine(
        ::testing::Values("relu", "sigmoid", "tanh", "softmax", "linear"),
        ::testing::Bool()
    )
);