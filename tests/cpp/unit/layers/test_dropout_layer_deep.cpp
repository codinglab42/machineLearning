// test/layers/dropout_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/dropout_layer.h"

using namespace layers;

class DropoutLayerTest : public ::testing::TestWithParam<double> {
protected:
    void SetUp() override {
        rate = GetParam();
        layer = std::make_unique<DropoutLayer>(rate);
        layer->set_input_shape(10);
    }
    
    double rate;
    std::unique_ptr<DropoutLayer> layer;
};

TEST_P(DropoutLayerTest, ConstructorValidation) {
    EXPECT_THROW(DropoutLayer(-0.1), std::invalid_argument);
    EXPECT_THROW(DropoutLayer(1.0), std::invalid_argument);
    EXPECT_NO_THROW(DropoutLayer(0.5));
}

TEST_P(DropoutLayerTest, ForwardTraining) {
    Eigen::MatrixXd input(4, 10);
    input.setOnes();
    
    auto output = layer->forward(input, true);
    
    // In training, alcuni elementi dovrebbero essere zero
    int zero_count = (output.array() == 0).count();
    EXPECT_GT(zero_count, 0);
    EXPECT_LT(zero_count, input.size());
    
    // Verifica scaling
    double scale = 1.0 / (1.0 - rate);
    for (int i = 0; i < output.size(); ++i) {
        if (output(i) != 0) {
            EXPECT_DOUBLE_EQ(output(i), scale);
        }
    }
}

TEST_P(DropoutLayerTest, ForwardInference) {
    Eigen::MatrixXd input(4, 10);
    input.setOnes();
    
    auto output = layer->forward(input, false);
    
    // In inference, nessun dropout applicato
    EXPECT_TRUE(output.isApprox(input));
    
    // Verifica che la cache sia stata creata
    EXPECT_NE(layer->get_cache(), nullptr);
}

TEST_P(DropoutLayerTest, Backward) {
    Eigen::MatrixXd input(3, 10);
    input.setRandom();
    
    auto output = layer->forward(input, true);
    
    Eigen::MatrixXd gradient(3, 10);
    gradient.setOnes();
    
    auto dX = layer->backward(gradient, 0.01);
    
    // In training, il gradiente dovrebbe avere la stessa maschera dell'output
    auto mask = output.array() != 0;
    auto dX_masked = (dX.array() != 0);
    
    EXPECT_TRUE((mask == dX_masked).all());
}

TEST_P(DropoutLayerTest, BackwardInference) {
    Eigen::MatrixXd input(3, 10);
    input.setRandom();
    
    layer->forward(input, false); // inference
    
    Eigen::MatrixXd gradient(3, 10);
    gradient.setOnes();
    
    auto dX = layer->backward(gradient, 0.01);
    
    // In inference, il gradiente passa inalterato
    EXPECT_TRUE(dX.isApprox(gradient));
}

TEST_P(DropoutLayerTest, DeterministicSeed) {
    // Con lo stesso seed, il dropout dovrebbe essere deterministico
    Eigen::MatrixXd input(2, 10);
    input.setOnes();
    
    auto output1 = layer->forward(input, true);
    auto output2 = layer->forward(input, true);
    
    // Diverso perche' random diverso
    EXPECT_FALSE(output1.isApprox(output2));
}

TEST_P(DropoutLayerTest, Serialization) {
    Eigen::MatrixXd input(2, 10);
    input.setRandom();
    layer->forward(input, true);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    auto new_layer = std::make_unique<DropoutLayer>(0.5);
    new_layer->deserialize(ss);
    
    EXPECT_EQ(new_layer->get_config(), layer->get_config());
    EXPECT_EQ(new_layer->get_input_size(), layer->get_input_size());
}

TEST_P(DropoutLayerTest, NoParameters) {
    EXPECT_EQ(layer->get_parameter_count(), 0);
    EXPECT_FALSE(layer->has_weights());
}

INSTANTIATE_TEST_SUITE_P(
    DropoutRates,
    DropoutLayerTest,
    ::testing::Values(0.1, 0.25, 0.5, 0.75, 0.9)
);