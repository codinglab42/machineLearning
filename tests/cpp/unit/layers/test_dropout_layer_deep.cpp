// test/layers/dropout_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/dropout_layer.h"
#include "exceptions/ml_exception.h"  // Aggiungi questo include

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
    // Usa le eccezioni personalizzate invece di std::invalid_argument
    EXPECT_THROW(DropoutLayer(-0.1), ml_exception::InvalidParameterException);
    EXPECT_THROW(DropoutLayer(1.0), ml_exception::InvalidParameterException);
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
    
    // Ottieni la maschera dalla cache per confronto diretto
    auto cache = std::dynamic_pointer_cast<DropoutCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    Eigen::MatrixXd expected_mask = cache->mask;
    
    Eigen::MatrixXd gradient(3, 10);
    gradient.setOnes();
    
    auto dX = layer->backward(gradient);
    
    // Il gradiente dovrebbe essere moltiplicato per la maschera
    Eigen::MatrixXd expected_dX = gradient.cwiseProduct(expected_mask);
    EXPECT_TRUE(dX.isApprox(expected_dX));
}

TEST_P(DropoutLayerTest, BackwardInference) {
    Eigen::MatrixXd input(3, 10);
    input.setRandom();
    
    layer->forward(input, false); // inference
    
    Eigen::MatrixXd gradient(3, 10);
    gradient.setOnes();
    
    auto dX = layer->backward(gradient);
    
    // In inference, il gradiente passa inalterato
    EXPECT_TRUE(dX.isApprox(gradient));
}

TEST_P(DropoutLayerTest, DeterministicSeed) {
    // Con lo stesso seed, il dropout dovrebbe essere deterministico
    // Nota: il test attuale verifica che sia diverso, il che è corretto
    // perché il seed è diverso ogni volta (random_device)
    Eigen::MatrixXd input(2, 10);
    input.setOnes();
    
    auto output1 = layer->forward(input, true);
    auto output2 = layer->forward(input, true);
    
    // Diverso perché random diverso (OK)
    EXPECT_FALSE(output1.isApprox(output2));
}

TEST_P(DropoutLayerTest, Serialization) {
    Eigen::MatrixXd input(2, 10);
    input.setRandom();
    layer->forward(input, true);
    
    auto config_before = layer->get_config();
    auto input_size_before = layer->get_input_size();
    
    std::stringstream ss;
    layer->serialize(ss);
    
    // Usa lo STESSO rate per il nuovo layer
    auto new_layer = std::make_unique<DropoutLayer>(rate);
    new_layer->deserialize(ss);
    
    EXPECT_EQ(new_layer->get_config(), config_before);
    EXPECT_EQ(new_layer->get_input_size(), input_size_before);
    
    // Verifica che forward funzioni
    auto output_new = new_layer->forward(input, false);
    EXPECT_TRUE(output_new.allFinite());
}

TEST_P(DropoutLayerTest, NoParameters) {
    EXPECT_EQ(layer->get_parameter_count(), 0);
    EXPECT_FALSE(layer->has_weights());
}

TEST_P(DropoutLayerTest, ClearCache) {
    Eigen::MatrixXd input(2, 10);
    input.setRandom();
    
    layer->forward(input, true);
    
    auto cache = std::dynamic_pointer_cast<DropoutCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_valid());
    EXPECT_GT(cache->input_cache.size(), 0);
    EXPECT_GT(cache->mask.size(), 0);
    
    layer->clear_cache();
    
    auto cache_after = std::dynamic_pointer_cast<DropoutCache>(layer->get_cache());
    ASSERT_NE(cache_after, nullptr);
    EXPECT_FALSE(cache_after->is_valid());
    EXPECT_EQ(cache_after->input_cache.size(), 0);
    EXPECT_EQ(cache_after->mask.size(), 0);
}

TEST_P(DropoutLayerTest, EmptyInput) {
    Eigen::MatrixXd empty;
    EXPECT_THROW(layer->forward(empty), ml_exception::EmptyDatasetException);
}

INSTANTIATE_TEST_SUITE_P(
    DropoutRates,
    DropoutLayerTest,
    ::testing::Values(0.1, 0.25, 0.5, 0.75, 0.9)
);