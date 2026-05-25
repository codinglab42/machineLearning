/**
 * @file test_layer_optimizer_integration.cpp
 * @brief Integration tests for Layers with Optimizers
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <Eigen/Dense>

#include "components/layers/dense_layer.h"
#include "components/optimizers/optimizer_factory.h"
#include "components/optimizers/optimizer.h"

using namespace models;
using namespace layers;
using namespace Eigen;
using namespace testing;

class LayerOptimizerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer_ = std::make_unique<DenseLayer>(5, "relu", true);
        layer_->set_input_shape(3);
        sgd_ = OptimizerFactory::create(OptimizerType::SGD, 0.01);
        adam_ = OptimizerFactory::create(OptimizerType::ADAM, 0.001);
    }
    
    std::unique_ptr<DenseLayer> layer_;
    std::unique_ptr<Optimizer> sgd_;
    std::unique_ptr<Optimizer> adam_;
};

TEST_F(LayerOptimizerIntegrationTest, LayerUpdateWithSGD) {
    MatrixXd input = MatrixXd::Random(10, 3);
    MatrixXd grad = MatrixXd::Random(10, 5);
    
    // Forward pass
    MatrixXd output = layer_->forward(input);
    
    // Backward pass con SGD (learning rate è passato al metodo backward)
    MatrixXd dX = layer_->backward(grad);
    
    EXPECT_EQ(dX.rows(), input.rows());
    EXPECT_EQ(dX.cols(), input.cols());
}

TEST_F(LayerOptimizerIntegrationTest, LayerForwardPassDimensions) {
    MatrixXd input = MatrixXd::Random(10, 3);
    
    MatrixXd output = layer_->forward(input);
    
    EXPECT_EQ(output.rows(), 10);
    EXPECT_EQ(output.cols(), 5);
}

TEST_F(LayerOptimizerIntegrationTest, LayerBackwardPassDimensions) {
    MatrixXd input = MatrixXd::Random(10, 3);
    MatrixXd grad = MatrixXd::Random(10, 5);
    
    layer_->forward(input);
    MatrixXd dX = layer_->backward(grad);
    
    EXPECT_EQ(dX.rows(), input.rows());
    EXPECT_EQ(dX.cols(), input.cols());
}

TEST_F(LayerOptimizerIntegrationTest, WeightsChangeAfterBackward) {
    // Inizializza i pesi del layer
    layer_->initialize_weights();  // ← AGGIUNGI QUESTO!
    
    MatrixXd input = MatrixXd::Random(10, 3);
    MatrixXd grad = MatrixXd::Random(10, 5);
    
    // Forward e backward pass
    layer_->forward(input, true);  // training mode
    layer_->backward(grad);
    
    // I gradienti devono essere non-zero
    EXPECT_GT(layer_->get_weights_gradient().norm(), 0);
    EXPECT_FALSE(layer_->get_weights_gradient().isZero());
}

TEST_F(LayerOptimizerIntegrationTest, MultipleBackwardPasses) {
    MatrixXd input = MatrixXd::Random(10, 3);
    MatrixXd grad = MatrixXd::Random(10, 5);
    
    // First backward pass
    layer_->forward(input);
    MatrixXd dX1 = layer_->backward(grad);
    
    // Second backward pass
    layer_->forward(input);
    MatrixXd dX2 = layer_->backward(grad);
    
    // Both should have correct dimensions
    EXPECT_EQ(dX1.rows(), input.rows());
    EXPECT_EQ(dX1.cols(), input.cols());
    EXPECT_EQ(dX2.rows(), input.rows());
    EXPECT_EQ(dX2.cols(), input.cols());
}

TEST_F(LayerOptimizerIntegrationTest, DifferentBatchSizes) {
    std::vector<int> batch_sizes = {1, 8, 16, 32, 64};
    
    for (int batch_size : batch_sizes) {
        MatrixXd input = MatrixXd::Random(batch_size, 3);
        MatrixXd grad = MatrixXd::Random(batch_size, 5);
        
        // Forward pass
        MatrixXd output = layer_->forward(input);
        
        // Backward pass
        MatrixXd dX = layer_->backward(grad);
        
        EXPECT_EQ(dX.rows(), batch_size);
        EXPECT_EQ(dX.cols(), 3) << "Failed for batch size: " << batch_size;
    }
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}