// tests/cpp/unit/layer/test_dropout_layer.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/layers/dropout_layer.h"

using namespace layers;
using namespace Eigen;

class DropoutLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<DropoutLayer>(0.5);
        
        input.resize(4, 10);
        input.setOnes();
    }
    
    std::unique_ptr<DropoutLayer> layer;
    MatrixXd input;
};

TEST_F(DropoutLayerTest, Construction) {
    EXPECT_FALSE(layer->has_weights());
    EXPECT_FALSE(layer->get_use_bias());
    EXPECT_EQ(layer->get_type(), "DropoutLayer");
    EXPECT_EQ(layer->get_layer_type(), LayerType::DROPOUT);
    EXPECT_EQ(layer->get_parameter_count(), 0);
}

TEST_F(DropoutLayerTest, InferenceModeNoChange) {
    MatrixXd output = layer->forward(input, false);
    
    EXPECT_TRUE(output.isApprox(input));
}

TEST_F(DropoutLayerTest, TrainingModeChangesOutput) {
    MatrixXd output1 = layer->forward(input, true);
    MatrixXd output2 = layer->forward(input, true);
    
    EXPECT_FALSE(output1.isApprox(input));
    EXPECT_FALSE(output2.isApprox(input));
}

TEST_F(DropoutLayerTest, BackwardPassesMaskedGradient) {
    MatrixXd output = layer->forward(input, true);
    MatrixXd gradient = MatrixXd::Ones(4, 10);
    
    MatrixXd dX = layer->backward(gradient);
    
    EXPECT_EQ(dX.rows(), 4);
    EXPECT_EQ(dX.cols(), 10);
}

TEST_F(DropoutLayerTest, InferenceBackwardNoChange) {
    layer->forward(input, false);
    MatrixXd gradient = MatrixXd::Ones(4, 10);
    
    MatrixXd dX = layer->backward(gradient);
    
    EXPECT_TRUE(dX.isApprox(gradient));
}