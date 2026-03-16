// test/layers/conv2d_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/conv2d_layer.h"
#include "exceptions/ml_exception.h"  // Per le eccezioni personalizzate

using namespace layers;

// Suite parametrizzata
class Conv2DLayerTest : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {
protected:
    void SetUp() override {
        auto [padding, activation] = GetParam();
        layer = std::make_unique<Conv2DLayer>(2, 3, 1, padding, activation);
        layer->set_input_shape(25);
    }
    
    std::unique_ptr<Conv2DLayer> layer;
};

TEST_P(Conv2DLayerTest, ConstructorValidation) {
    // Usa le eccezioni personalizzate
    EXPECT_THROW(Conv2DLayer(0, 3, 1, "valid", "relu"), ml_exception::InvalidParameterException);
    EXPECT_THROW(Conv2DLayer(2, 0, 1, "valid", "relu"), ml_exception::InvalidParameterException);
    EXPECT_THROW(Conv2DLayer(2, 3, 0, "valid", "relu"), ml_exception::InvalidParameterException);
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
    Eigen::MatrixXd input(2 * 25, 1); // 2 samples, flattened
    input.setRandom();
    
    auto output = layer->forward(input);
    
    int expected_size = 2 * layer->get_output_size();
    EXPECT_EQ(output.rows(), expected_size);
    EXPECT_EQ(output.cols(), 1);
    EXPECT_TRUE(output.allFinite()); // Verifica che non ci siano NaN
}

TEST_P(Conv2DLayerTest, Backward) {
    Eigen::MatrixXd input(2 * 25, 1);
    input.setRandom();
    
    auto output = layer->forward(input, true);
    
    Eigen::MatrixXd gradient(output.rows(), output.cols());
    gradient.setOnes();
    
    auto weights_before = layer->get_weights();
    auto dX = layer->backward(gradient, 0.1); // LR più alto
    
    EXPECT_EQ(dX.rows(), input.rows());
    EXPECT_EQ(dX.cols(), input.cols());
    EXPECT_TRUE(dX.allFinite());
    
    // Verifica aggiornamento pesi
    auto weights_after = layer->get_weights();
    EXPECT_FALSE(weights_before.isApprox(weights_after));
}

TEST_P(Conv2DLayerTest, Im2Col) {
    Eigen::MatrixXd input(1 * 25, 1);
    for (int i = 0; i < 25; ++i) {
        input(i) = i + 1;
    }
    
    EXPECT_NO_THROW(layer->forward(input));
}

TEST_P(Conv2DLayerTest, Serialization) {
    // Prepara input
    Eigen::MatrixXd input(2 * 25, 1);
    input.setRandom();
    
    // Forward pass
    auto output_before = layer->forward(input, true);
    auto weights_before = layer->get_weights();
    auto config_before = layer->get_config();
    
    // Serializza
    std::stringstream ss;
    layer->serialize(ss);
    
    // Crea nuovo layer con gli STESSI parametri
    auto [padding, activation] = GetParam();
    auto new_layer = std::make_unique<Conv2DLayer>(2, 3, 1, padding, activation);
    
    // Deserializza
    new_layer->deserialize(ss);
    
    // Verifica configurazione
    EXPECT_EQ(new_layer->get_config(), config_before);
    
    // Verifica pesi
    auto weights_new = new_layer->get_weights();
    EXPECT_TRUE(weights_before.isApprox(weights_new));
    
    // Verifica output (inference mode)
    auto output_new = new_layer->forward(input, false);
    EXPECT_TRUE(output_before.isApprox(output_new));
}

TEST_P(Conv2DLayerTest, ParameterCount) {
    int expected = 2 * (3*3*1) + 2; // 2*9 + 2 = 20
    EXPECT_EQ(layer->get_parameter_count(), expected);
}

// Test non parametrizzati - fuori dalla suite
class Conv2DLayerStandaloneTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup comune se necessario
    }
};

TEST_F(Conv2DLayerStandaloneTest, DifferentStrides) {
    auto layer = std::make_unique<Conv2DLayer>(2, 3, 2, "valid", "relu");
    layer->set_input_shape(25);
    EXPECT_EQ(layer->get_output_size(), 2 * 2 * 2); // 2x2 output
}

TEST_F(Conv2DLayerStandaloneTest, ClearCache) {
    auto layer = std::make_unique<Conv2DLayer>(2, 3, 1, "valid", "relu");
    layer->set_input_shape(25);
    
    Eigen::MatrixXd input(2 * 25, 1);
    input.setRandom();
    
    layer->forward(input, true);
    EXPECT_NE(layer->get_cache(), nullptr);
    
    auto cache = std::dynamic_pointer_cast<ConvCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_valid());
    
    layer->clear_cache();
    
    // Dopo clear, la cache esiste ma è vuota
    auto cache_after = std::dynamic_pointer_cast<ConvCache>(layer->get_cache());
    ASSERT_NE(cache_after, nullptr);
    EXPECT_FALSE(cache_after->is_valid());
    EXPECT_EQ(cache_after->input_cache.size(), 0);
}

TEST_F(Conv2DLayerStandaloneTest, EmptyInput) {
    auto layer = std::make_unique<Conv2DLayer>(2, 3, 1, "valid", "relu");
    Eigen::MatrixXd empty;
    EXPECT_THROW(layer->forward(empty), ml_exception::EmptyDatasetException);
}

// Instantiate the test suite with parameters
INSTANTIATE_TEST_SUITE_P(
    Conv2DLayerVariants,
    Conv2DLayerTest,
    ::testing::Combine(
        ::testing::Values("valid", "same"),
        ::testing::Values("relu", "sigmoid", "tanh", "linear")
    )
);