#include <gtest/gtest.h>
#include "components/layers/conv2d_layer.h"
#include "components/layers/pooling_layer.h"
#include "components/layers/flatten_layer.h"
#include "components/layers/dense_layer.h"
#include "components/layers/batch_norm_layer.h"
#include "components/layers/dropout_layer.h"
#include "components/optimizers/sgd_optimizer.h" 
#include "components/loss/mean_squared_error_loss.h" 
#include "models/neural_network.h" 

#include <Eigen/Dense>
#include <memory>
#include <sstream>

// ============================================================================
// TEST: PIPELINE COMPLETA CNN -> POOL -> FLATTEN -> DENSE
// ============================================================================
TEST(HybridPipelineIntegrationTest, CNNToDenseForwardBackwardPipeline) {
    int batch_size = 1;
    int in_channels = 1;
    int in_height = 6;
    int in_width = 6;
    int input_size = in_channels * in_height * in_width; // 36

    auto conv = std::make_shared<layers::Conv2DLayer>(2, 3, 1, "valid", "linear");
    conv->set_input_shape(input_size);
    conv->initialize_weights(); 

    auto pool = std::make_shared<layers::PoolingLayer>(2, 2, layers::PoolingLayer::MAX, 2);
    pool->set_input_shape(conv->get_output_size());

    auto flatten = std::make_shared<layers::FlattenLayer>();
    flatten->set_input_shape(pool->get_output_size());

    auto dense = std::make_shared<layers::DenseLayer>(2, "linear");
    dense->set_input_shape(flatten->get_output_size());
    dense->initialize_weights(); 

    Eigen::MatrixXd input_data = Eigen::MatrixXd::Zero(batch_size, input_size);
    for (int i = 0; i < input_size; ++i) {
        input_data(0, i) = static_cast<double>(i + 1) * 0.1; 
    }

    // Pipeline Forward
    Eigen::MatrixXd x1 = conv->forward(input_data, true);
    ASSERT_EQ(x1.cols(), conv->get_output_size());

    Eigen::MatrixXd x2 = pool->forward(x1, true);
    ASSERT_EQ(x2.cols(), pool->get_output_size());

    Eigen::MatrixXd x3 = flatten->forward(x2, true);
    ASSERT_EQ(x3.cols(), flatten->get_output_size());
    ASSERT_EQ(x3.cols(), 8); 

    Eigen::MatrixXd final_output = dense->forward(x3, true);
    ASSERT_EQ(final_output.rows(), batch_size);
    ASSERT_EQ(final_output.cols(), 2);

    // Pipeline Backward
    Eigen::MatrixXd loss_gradient(batch_size, 2);
    loss_gradient(0, 0) = 0.25;
    loss_gradient(0, 1) = 0.75;

    Eigen::MatrixXd g3 = dense->backward(loss_gradient);
    ASSERT_EQ(g3.cols(), flatten->get_output_size());

    Eigen::MatrixXd g2 = flatten->backward(g3);
    ASSERT_EQ(g2.cols(), pool->get_output_size());

    Eigen::MatrixXd g1 = pool->backward(g2);
    ASSERT_EQ(g1.cols(), conv->get_output_size());

    Eigen::MatrixXd input_gradient = conv->backward(g1);
    
    ASSERT_EQ(input_gradient.rows(), batch_size);
    ASSERT_EQ(input_gradient.cols(), input_size);

    EXPECT_GT(input_gradient.norm(), 0.0);
    EXPECT_GT(conv->get_weights_gradient().norm(), 0.0);
    EXPECT_GT(dense->get_weights_gradient().norm(), 0.0);
}

// ============================================================================
// TEST: VARIABILITÀ DINAMICA DEL BATCH SIZE
// ============================================================================
TEST(HybridPipelineIntegrationTest, DynamicBatchSizeHandling) {
    int input_size = 4;
    int units = 2;

    auto dense = std::make_shared<layers::DenseLayer>(units, "relu");
    dense->set_input_shape(input_size);
    dense->initialize_weights();

    Eigen::MatrixXd input_batch3 = Eigen::MatrixXd::Random(3, input_size);
    Eigen::MatrixXd out_batch3 = dense->forward(input_batch3, true);
    ASSERT_EQ(out_batch3.rows(), 3);

    Eigen::MatrixXd grad_batch3 = Eigen::MatrixXd::Ones(3, units);
    Eigen::MatrixXd back_batch3 = dense->backward(grad_batch3);
    ASSERT_EQ(back_batch3.rows(), 3);

    Eigen::MatrixXd input_batch1 = Eigen::MatrixXd::Random(1, input_size);
    Eigen::MatrixXd out_batch1 = dense->forward(input_batch1, false); 
    ASSERT_EQ(out_batch1.rows(), 1);
    
    SUCCEED();
}

// ============================================================================
// TEST: INTEGRATION CON DROPOUT E BATCH NORMALIZATION
// ============================================================================
TEST(HybridPipelineIntegrationTest, BatchNormAndDropoutPipeline) {
    int batch_size = 4;  
    int input_size = 4;
    int units = 3;

    auto dense = std::make_shared<layers::DenseLayer>(units, "linear");
    dense->set_input_shape(input_size);
    dense->initialize_weights();

    auto bn = std::make_shared<layers::BatchNormLayer>();
    bn->set_input_shape(units);
    bn->initialize_weights(); 

    auto dropout = std::make_shared<layers::DropoutLayer>(0.2); 
    dropout->set_input_shape(units);

    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(batch_size, input_size);
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_size; ++i) {
            input(b, i) = static_cast<double>((b * input_size) + i + 1) * 0.25;
        }
    }
    
    Eigen::MatrixXd x1 = dense->forward(input, true);
    Eigen::MatrixXd x2 = bn->forward(x1, true);
    Eigen::MatrixXd final_output = dropout->forward(x2, true);

    ASSERT_EQ(final_output.rows(), batch_size);
    ASSERT_EQ(final_output.cols(), units);

    Eigen::MatrixXd loss_grad = Eigen::MatrixXd::Zero(batch_size, units);
    for (int b = 0; b < batch_size; ++b) {
        for (int u = 0; u < units; ++u) {
            loss_grad(b, u) = static_cast<double>(b + u + 1) * 0.5;
        }
    }
    
    Eigen::MatrixXd g3 = dropout->backward(loss_grad);
    Eigen::MatrixXd g2 = bn->backward(g3);
    Eigen::MatrixXd final_grad = dense->backward(g2);

    ASSERT_EQ(final_grad.rows(), batch_size);
    ASSERT_EQ(final_grad.cols(), input_size);
    
    EXPECT_GT(dense->get_weights_gradient().norm(), 0.0);
}

// ============================================================================
// TEST: INTEGRAZIONE OTTIMIZZATORE E AGGIORNAMENTO PESI REALISTICO
// ============================================================================
TEST(HybridPipelineIntegrationTest, OptimizerWeightUpdateAndConvergence) {
    int batch_size = 1;
    int input_size = 3;
    int units = 2;

    auto dense = std::make_shared<layers::DenseLayer>(units, "linear");
    dense->set_input_shape(input_size);
    dense->initialize_weights();

    Eigen::MatrixXd weights_before = dense->get_weights();
    Eigen::VectorXd bias_before = dense->get_biases();

    auto optimizer = std::make_shared<models::SGDOptimizer>(0.1);
    auto loss_fn = std::make_shared<loss::MeanSquaredErrorLoss>();

    Eigen::MatrixXd input(batch_size, input_size);
    input << 1.0, 0.5, -0.5;
    
    Eigen::MatrixXd target(batch_size, units);
    target << 0.0, 1.0;

    Eigen::MatrixXd output = dense->forward(input, true);

    double loss_value = loss_fn->compute(target, output);
    Eigen::MatrixXd loss_grad = loss_fn->gradient(target, output);

    dense->backward(loss_grad);

    Eigen::MatrixXd weights = dense->get_weights();
    Eigen::MatrixXd weights_gradient = dense->get_weights_gradient();
    
    Eigen::VectorXd bias = dense->get_biases();
    Eigen::VectorXd bias_gradient = dense->get_bias_gradient();

    optimizer->update(weights, weights_gradient);
    optimizer->update(bias, bias_gradient);

    dense->set_weights(weights);
    dense->set_biases(bias);

    Eigen::MatrixXd weights_after = dense->get_weights();

    double weight_diff = (weights_after - weights_before).norm();
    EXPECT_GT(weight_diff, 0.0);

    Eigen::MatrixXd new_output = dense->forward(input, false);
    double new_loss_value = loss_fn->compute(target, new_output);
    
    EXPECT_LT(new_loss_value, loss_value);
}

// ============================================================================
// TEST: INTEGRITÀ GEOMETRICA E NUMERICA POST-SERIALIZZAZIONE VIA STREAM
// ============================================================================
TEST(HybridPipelineIntegrationTest, SerializationAndNumericalConsistency) {
    int batch_size = 1;
    int input_size = 4;
    int units = 2;

    auto dense_original = std::make_shared<layers::DenseLayer>(units, "relu");
    dense_original->set_input_shape(input_size);
    dense_original->initialize_weights();

    Eigen::MatrixXd input = Eigen::MatrixXd::Random(batch_size, input_size);
    Eigen::MatrixXd output_original = dense_original->forward(input, false);

    std::stringstream stream;
    dense_original->serialize(stream);

    auto dense_cloned = std::make_shared<layers::DenseLayer>(units, "relu");
    dense_cloned->set_input_shape(input_size);
    
    dense_cloned->deserialize(stream);

    Eigen::MatrixXd output_cloned = dense_cloned->forward(input, false);

    ASSERT_EQ(output_cloned.rows(), output_original.rows());
    ASSERT_EQ(output_cloned.cols(), output_original.cols());

    for (int i = 0; i < output_original.size(); ++i) {
        EXPECT_NEAR(output_cloned(i), output_original(i), 1e-6);
    }
}