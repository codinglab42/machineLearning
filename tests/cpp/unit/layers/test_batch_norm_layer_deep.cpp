// test/layers/batch_norm_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/batch_norm_layer.h"
#include "exceptions/ml_exception.h"

using namespace layers;

class BatchNormLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<BatchNormLayer>(1e-5, 0.9);
        layer->set_input_shape(5);
    }
    
    std::unique_ptr<BatchNormLayer> layer;
};

TEST_F(BatchNormLayerTest, ConstructorValidation) {
    // Usa le eccezioni personalizzate invece di std::invalid_argument
    EXPECT_THROW(BatchNormLayer(-1.0, 0.9), ml_exception::InvalidParameterException);
    EXPECT_THROW(BatchNormLayer(1e-5, 1.5), ml_exception::InvalidParameterException);
    EXPECT_NO_THROW(BatchNormLayer(1e-5, 0.9));
}

TEST_F(BatchNormLayerTest, SetInputShape) {
    EXPECT_THROW(layer->set_input_shape(-1), ml_exception::InvalidParameterException);
    EXPECT_NO_THROW(layer->set_input_shape(10));
    
    auto weights = layer->get_weights();
    EXPECT_EQ(weights.rows(), 10);
    EXPECT_EQ(weights.cols(), 2);
    EXPECT_EQ(layer->get_input_size(), 10);
}

TEST_F(BatchNormLayerTest, ForwardTrainingMode) {
    Eigen::MatrixXd input(4, 5);
    input << 1, 2, 3, 4, 5,
             2, 3, 4, 5, 6,
             3, 4, 5, 6, 7,
             4, 5, 6, 7, 8;
    
    auto output = layer->forward(input, true);
    
    EXPECT_EQ(output.rows(), 4);
    EXPECT_EQ(output.cols(), 5);
    
    // Verifica che la media sia ~0 e varianza ~1
    Eigen::VectorXd mean = output.colwise().mean();
    for (int i = 0; i < mean.size(); ++i) {
        EXPECT_NEAR(mean(i), 0.0, 1e-10);
    }
    
    Eigen::VectorXd var = (output.rowwise() - mean.transpose())
                         .array().square().colwise().sum() / (output.rows() - 1);
    for (int i = 0; i < var.size(); ++i) {
        EXPECT_NEAR(var(i), 1.0, 1e-5);
    }
}

TEST_F(BatchNormLayerTest, ForwardInferenceMode) {
    Eigen::MatrixXd input(2, 5);
    input << 1, 2, 3, 4, 5,
             2, 3, 4, 5, 6;
    
    // Prima forward in training per aggiornare running stats
    layer->forward(input, true);
    
    // Poi inference
    auto output_inference = layer->forward(input, false);
    auto output_training = layer->forward(input, true);
    
    // Output dovrebbero essere diversi
    EXPECT_FALSE(output_inference.isApprox(output_training));
}

TEST_F(BatchNormLayerTest, Backward) {
    Eigen::MatrixXd input(4, 5);
    input.setRandom();
    
    auto output = layer->forward(input, true);
    
    Eigen::MatrixXd gradient(4, 5);
    gradient.setOnes();
    
    auto dX = layer->backward(gradient);
    
    EXPECT_EQ(dX.rows(), input.rows());
    EXPECT_EQ(dX.cols(), input.cols());
    
    // Verifica che i parametri siano stati aggiornati
    auto weights_before = layer->get_weights();
    layer->backward(gradient);
    auto weights_after = layer->get_weights();
    
    EXPECT_FALSE(weights_before.isApprox(weights_after));
}

TEST_F(BatchNormLayerTest, Serialization) {
    Eigen::MatrixXd input(2, 5);
    input.setRandom();
    layer->forward(input, true);
    
    std::stringstream ss;
    layer->serialize(ss);
    
    auto new_layer = std::make_unique<BatchNormLayer>();
    new_layer->deserialize(ss);
    
    EXPECT_EQ(new_layer->get_config(), layer->get_config());
    EXPECT_EQ(new_layer->get_input_size(), layer->get_input_size());
    
    auto weights_orig = layer->get_weights();
    auto weights_new = new_layer->get_weights();
    EXPECT_TRUE(weights_orig.isApprox(weights_new));
}

TEST_F(BatchNormLayerTest, ParameterCount) {
    layer->set_input_shape(10);
    EXPECT_EQ(layer->get_parameter_count(), 20); // gamma(10) + beta(10)
}

TEST_F(BatchNormLayerTest, ClearCache) {
    Eigen::MatrixXd input(2, 5);
    input.setRandom();
    
    layer->forward(input, true);
    EXPECT_NE(layer->get_cache(), nullptr);
    
    layer->clear_cache();
    EXPECT_EQ(layer->get_cache(), nullptr);
}

TEST_F(BatchNormLayerTest, EmptyInput) {
    Eigen::MatrixXd empty;
    EXPECT_THROW(layer->forward(empty), ml_exception::EmptyDatasetException);
}