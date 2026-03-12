#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/dense_layer.h"
#include "components/layers/conv2d_layer.h"
#include "components/layers/pooling_layer.h"
#include "components/layers/flatten_layer.h"
#include "components/layers/dropout_layer.h"
#include "components/layers/batch_norm_layer.h"
#include "components/layers/simple_rnn_layer.h"
#include "components/layers/lstm_layer.h"
#include "components/layers/gru_layer.h"
#include "exceptions/exception_macros.h"
#include <Eigen/Dense>
#include <memory>
#include <sstream>

using namespace layers;
using Eigen::MatrixXd;
using Eigen::VectorXd;

//=============================================================================
// DENSE LAYER TESTS
//=============================================================================

class DenseLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<DenseLayer>(5, "relu", true);
        layer->set_input_shape(10);
    }
    
    std::unique_ptr<DenseLayer> layer;
    MatrixXd input = MatrixXd::Random(3, 10);  // batch 3, features 10
};

TEST_F(DenseLayerTest, Initialization) {
    EXPECT_EQ(layer->get_type(), "DenseLayer");
    EXPECT_EQ(layer->get_input_size(), 10);
    EXPECT_EQ(layer->get_output_size(), 5);
    EXPECT_TRUE(layer->has_weights());
    EXPECT_EQ(layer->get_parameter_count(), 10*5 + 5);  // weights + bias
}

TEST_F(DenseLayerTest, Forward) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 5);
    
    // Verifica che la cache sia stata creata
    auto cache = layer->get_cache();
    EXPECT_NE(cache, nullptr);
    EXPECT_EQ(cache->get_type(), "DenseCache");
}

TEST_F(DenseLayerTest, ForwardWithTraining) {
    MatrixXd output = layer->forward(input, true);
    
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 5);
}

TEST_F(DenseLayerTest, Backward) {
    // Forward pass prima
    MatrixXd output = layer->forward(input);
    
    // Gradiente fittizio
    MatrixXd gradient = MatrixXd::Random(3, 5);
    
    // Backward pass
    MatrixXd dX = layer->backward(gradient, 0.01);
    
    EXPECT_EQ(dX.rows(), 3);
    EXPECT_EQ(dX.cols(), 10);
}

TEST_F(DenseLayerTest, BackwardWithoutForward) {
    MatrixXd gradient = MatrixXd::Random(3, 5);
    
    // Dovrebbe lanciare eccezione perché cache non inizializzata
    EXPECT_THROW(layer->backward(gradient, 0.01), ml_exception::MLException);
}

TEST_F(DenseLayerTest, GetSetWeights) {
    MatrixXd weights = layer->get_weights();
    EXPECT_EQ(weights.rows(), 10);
    EXPECT_EQ(weights.cols(), 6);  // 5 weights + 1 bias
    
    // Modifica e reimposta
    weights.setRandom();
    layer->set_weights(weights);
    
    MatrixXd new_weights = layer->get_weights();
    EXPECT_EQ(new_weights.rows(), 10);
    EXPECT_EQ(new_weights.cols(), 6);
}

TEST_F(DenseLayerTest, GetSetBiases) {
    VectorXd biases = layer->get_biases();
    EXPECT_EQ(biases.size(), 5);
    
    biases.setConstant(2.0);
    layer->set_biases(biases);
    
    VectorXd new_biases = layer->get_biases();
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(new_biases(i), 2.0);
    }
}

TEST_F(DenseLayerTest, Serialization) {
    std::stringstream ss;
    
    // Serializza
    layer->serialize(ss);
    
    // Deserializza in un nuovo layer
    DenseLayer new_layer(1, "relu", true);
    new_layer.set_input_shape(10);
    new_layer.deserialize(ss);
    
    // Confronta configurazioni
    EXPECT_EQ(layer->get_config(), new_layer.get_config());
}

//=============================================================================
// CONV2D LAYER TESTS
//=============================================================================

class Conv2DLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<Conv2DLayer>(8, 3, 1, "valid", "relu");
        layer->set_input_shape(28*28);  // 28x28 immagine
    }
    
    std::unique_ptr<Conv2DLayer> layer;
    // Input: batch 2, 28x28x1 = 784 features
    MatrixXd input = MatrixXd::Random(2, 784);
};

TEST_F(Conv2DLayerTest, Initialization) {
    EXPECT_EQ(layer->get_type(), "Conv2DLayer");
    EXPECT_EQ(layer->get_input_size(), 784);
    // Output: 26x26x8 = 5408
    EXPECT_EQ(layer->get_output_size(), 26*26*8);
    EXPECT_TRUE(layer->has_weights());
}

TEST_F(Conv2DLayerTest, Forward) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 26*26*8);
    
    auto cache = layer->get_cache();
    EXPECT_NE(cache, nullptr);
    EXPECT_EQ(cache->get_type(), "ConvCache");
}

TEST_F(Conv2DLayerTest, Backward) {
    MatrixXd output = layer->forward(input);
    
    MatrixXd gradient = MatrixXd::Random(2, 26*26*8);
    MatrixXd dX = layer->backward(gradient, 0.001);
    
    EXPECT_EQ(dX.rows(), 2);
    EXPECT_EQ(dX.cols(), 784);
}

//=============================================================================
// POOLING LAYER TESTS
//=============================================================================

class PoolingLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<PoolingLayer>(2, 2, PoolingLayer::MAX, 1);
        layer->set_input_shape(16*16);  // 16x16 input
    }
    
    std::unique_ptr<PoolingLayer> layer;
    MatrixXd input = MatrixXd::Random(2, 256);  // batch 2, 16x16
};

TEST_F(PoolingLayerTest, Initialization) {
    EXPECT_EQ(layer->get_type(), "Pooling");
    EXPECT_EQ(layer->get_input_size(), 256);
    // Dopo pooling 2x2 con stride 2: 8x8 = 64
    EXPECT_EQ(layer->get_output_size(), 64);
    EXPECT_FALSE(layer->has_weights());
    EXPECT_EQ(layer->get_parameter_count(), 0);
}

TEST_F(PoolingLayerTest, Forward) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 64);
}

TEST_F(PoolingLayerTest, ForwardWithTraining) {
    MatrixXd output = layer->forward(input, true);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 64);
    
    auto cache = std::dynamic_pointer_cast<PoolingCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->get_training());
}

TEST_F(PoolingLayerTest, Backward) {
    MatrixXd output = layer->forward(input, true);
    
    MatrixXd gradient = MatrixXd::Random(2, 64);
    MatrixXd dX = layer->backward(gradient, 0.01);
    
    EXPECT_EQ(dX.rows(), 2);
    EXPECT_EQ(dX.cols(), 256);
}

TEST_F(PoolingLayerTest, AveragePooling) {
    auto avg_layer = std::make_unique<PoolingLayer>(2, 2, PoolingLayer::AVG, 1);
    avg_layer->set_input_shape(16*16);
    
    MatrixXd output = avg_layer->forward(input);
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 64);
}

//=============================================================================
// FLATTEN LAYER TESTS
//=============================================================================

class FlattenLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<FlattenLayer>();
        layer->set_input_shape(100);
    }
    
    std::unique_ptr<FlattenLayer> layer;
    MatrixXd input = MatrixXd::Random(3, 100);
};

TEST_F(FlattenLayerTest, Initialization) {
    EXPECT_EQ(layer->get_type(), "FlattenLayer");
    EXPECT_EQ(layer->get_input_size(), 100);
    EXPECT_EQ(layer->get_output_size(), 100);
    EXPECT_FALSE(layer->has_weights());
    EXPECT_EQ(layer->get_parameter_count(), 0);
}

TEST_F(FlattenLayerTest, Forward) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 100);
    EXPECT_EQ(output, input);  // Flatten non cambia l'input
}

TEST_F(FlattenLayerTest, Backward) {
    MatrixXd output = layer->forward(input);
    
    MatrixXd gradient = MatrixXd::Random(3, 100);
    MatrixXd dX = layer->backward(gradient, 0.01);
    
    EXPECT_EQ(dX.rows(), 3);
    EXPECT_EQ(dX.cols(), 100);
    EXPECT_EQ(dX, gradient);  // Il gradiente passa invariato
}

//=============================================================================
// DROPOUT LAYER TESTS
//=============================================================================

class DropoutLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<DropoutLayer>(0.3);
        layer->set_input_shape(20);
    }
    
    std::unique_ptr<DropoutLayer> layer;
    MatrixXd input = MatrixXd::Ones(5, 20);  // batch 5, features 20
};

TEST_F(DropoutLayerTest, Initialization) {
    EXPECT_EQ(layer->get_type(), "DropoutLayer");
    EXPECT_EQ(layer->get_input_size(), 20);
    EXPECT_EQ(layer->get_output_size(), 20);
    EXPECT_FALSE(layer->has_weights());
    EXPECT_EQ(layer->get_parameter_count(), 0);
}

TEST_F(DropoutLayerTest, InferenceForward) {
    MatrixXd output = layer->forward(input, false);  // inference mode
    
    EXPECT_EQ(output.rows(), 5);
    EXPECT_EQ(output.cols(), 20);
    EXPECT_EQ(output, input);  // In inference, nessun dropout
}

TEST_F(DropoutLayerTest, TrainingForward) {
    MatrixXd output = layer->forward(input, true);  // training mode
    
    EXPECT_EQ(output.rows(), 5);
    EXPECT_EQ(output.cols(), 20);
    
    // Alcuni elementi dovrebbero essere zero
    bool has_zeros = false;
    for (int i = 0; i < output.size(); ++i) {
        if (output(i) == 0.0) {
            has_zeros = true;
            break;
        }
    }
    EXPECT_TRUE(has_zeros);
    
    // Gli elementi non-zero dovrebbero essere scalati
    double scale = 1.0 / (1.0 - 0.3);
    for (int i = 0; i < output.size(); ++i) {
        if (output(i) != 0.0) {
            EXPECT_DOUBLE_EQ(output(i), scale);
        }
    }
}

TEST_F(DropoutLayerTest, Backward) {
    MatrixXd output = layer->forward(input, true);
    
    MatrixXd gradient = MatrixXd::Random(5, 20);
    MatrixXd dX = layer->backward(gradient, 0.01);
    
    EXPECT_EQ(dX.rows(), 5);
    EXPECT_EQ(dX.cols(), 20);
    
    // Il gradiente dovrebbe essere mascherato
    auto cache = std::dynamic_pointer_cast<DropoutCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    
    for (int i = 0; i < dX.size(); ++i) {
        if (cache->mask(i) == 0.0) {
            EXPECT_DOUBLE_EQ(dX(i), 0.0);
        }
    }
}

//=============================================================================
// BATCH NORM LAYER TESTS
//=============================================================================

class BatchNormLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<BatchNormLayer>(1e-5, 0.9);
        layer->set_input_shape(5);
    }
    
    std::unique_ptr<BatchNormLayer> layer;
    MatrixXd input = MatrixXd::Random(4, 5);  // batch 4, features 5
};

TEST_F(BatchNormLayerTest, Initialization) {
    EXPECT_EQ(layer->get_type(), "BatchNormLayer");
    EXPECT_EQ(layer->get_input_size(), 5);
    EXPECT_EQ(layer->get_output_size(), 5);
    EXPECT_TRUE(layer->has_weights());
    EXPECT_EQ(layer->get_parameter_count(), 10);  // gamma + beta
}

TEST_F(BatchNormLayerTest, InferenceForward) {
    // Prima forward in training per inizializzare running stats
    layer->forward(input, true);
    
    MatrixXd output = layer->forward(input, false);  // inference
    
    EXPECT_EQ(output.rows(), 4);
    EXPECT_EQ(output.cols(), 5);
}

TEST_F(BatchNormLayerTest, TrainingForward) {
    MatrixXd output = layer->forward(input, true);
    
    EXPECT_EQ(output.rows(), 4);
    EXPECT_EQ(output.cols(), 5);
    
    auto cache = std::dynamic_pointer_cast<BatchNormCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->training);
    EXPECT_EQ(cache->batch_mean.size(), 5);
    EXPECT_EQ(cache->batch_var.size(), 5);
}

TEST_F(BatchNormLayerTest, Backward) {
    MatrixXd output = layer->forward(input, true);
    
    MatrixXd gradient = MatrixXd::Random(4, 5);
    MatrixXd dX = layer->backward(gradient, 0.01);
    
    EXPECT_EQ(dX.rows(), 4);
    EXPECT_EQ(dX.cols(), 5);
}

//=============================================================================
// SIMPLE RNN LAYER TESTS
//=============================================================================

class SimpleRNNLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<SimpleRNNLayer>(8, 5, "tanh", true);
        layer->set_input_shape(5);
    }
    
    std::unique_ptr<SimpleRNNLayer> layer;
    MatrixXd input = MatrixXd::Random(3, 5);  // batch 3, features 5
};

TEST_F(SimpleRNNLayerTest, Initialization) {
    EXPECT_EQ(layer->get_type(), "SimpleRNNLayer");
    EXPECT_EQ(layer->get_input_size(), 5);
    EXPECT_EQ(layer->get_output_size(), 8);
    EXPECT_TRUE(layer->has_weights());
}

TEST_F(SimpleRNNLayerTest, Forward) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 8);
}

TEST_F(SimpleRNNLayerTest, ForwardWithTraining) {
    MatrixXd output = layer->forward(input, true);
    
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 8);
    
    auto cache = std::dynamic_pointer_cast<SimpleRNNCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->training);
}

TEST_F(SimpleRNNLayerTest, Backward) {
    MatrixXd output = layer->forward(input, true);
    
    MatrixXd gradient = MatrixXd::Random(3, 8);
    MatrixXd dX = layer->backward(gradient, 0.01);
    
    EXPECT_EQ(dX.rows(), 3);
    EXPECT_EQ(dX.cols(), 5);
}

TEST_F(SimpleRNNLayerTest, StateManagement) {
    layer->reset_state();
    
    MatrixXd state = layer->get_hidden_state();
    EXPECT_EQ(state.rows(), 0);  // Stato vuoto dopo reset
    
    layer->forward(input);
    state = layer->get_hidden_state();
    EXPECT_EQ(state.rows(), 3);
    EXPECT_EQ(state.cols(), 8);
}

//=============================================================================
// LSTM LAYER TESTS
//=============================================================================

class LSTMLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<LSTMLayer>(8, 5, "tanh", "sigmoid", true);
        layer->set_input_shape(5);
    }
    
    std::unique_ptr<LSTMLayer> layer;
    MatrixXd input = MatrixXd::Random(3, 5);  // batch 3, features 5
};

TEST_F(LSTMLayerTest, Initialization) {
    EXPECT_EQ(layer->get_type(), "LSTMLayer");
    EXPECT_EQ(layer->get_input_size(), 5);
    EXPECT_EQ(layer->get_output_size(), 8);
    EXPECT_TRUE(layer->has_weights());
}

TEST_F(LSTMLayerTest, Forward) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 8);
}

TEST_F(LSTMLayerTest, ForwardWithTraining) {
    MatrixXd output = layer->forward(input, true);
    
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 8);
    
    auto cache = std::dynamic_pointer_cast<LSTMCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->training);
}

TEST_F(LSTMLayerTest, Backward) {
    MatrixXd output = layer->forward(input, true);
    
    MatrixXd gradient = MatrixXd::Random(3, 8);
    MatrixXd dX = layer->backward(gradient, 0.01);
    
    EXPECT_EQ(dX.rows(), 3);
    EXPECT_EQ(dX.cols(), 5);
}

TEST_F(LSTMLayerTest, StateManagement) {
    layer->reset_state();
    
    MatrixXd hidden = layer->get_hidden_state();
    EXPECT_EQ(hidden.rows(), 0);
    
    layer->forward(input);
    hidden = layer->get_hidden_state();
    EXPECT_EQ(hidden.rows(), 3);
    EXPECT_EQ(hidden.cols(), 8);
}

//=============================================================================
// GRU LAYER TESTS
//=============================================================================

class GRULayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<GRULayer>(8, 5, "tanh", "sigmoid", true);
        layer->set_input_shape(5);
    }
    
    std::unique_ptr<GRULayer> layer;
    MatrixXd input = MatrixXd::Random(3, 5);  // batch 3, features 5
};

TEST_F(GRULayerTest, Initialization) {
    EXPECT_EQ(layer->get_type(), "GRULayer");
    EXPECT_EQ(layer->get_input_size(), 5);
    EXPECT_EQ(layer->get_output_size(), 8);
    EXPECT_TRUE(layer->has_weights());
}

TEST_F(GRULayerTest, Forward) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 8);
}

TEST_F(GRULayerTest, ForwardWithTraining) {
    MatrixXd output = layer->forward(input, true);
    
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 8);
    
    auto cache = std::dynamic_pointer_cast<GRUCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->training);
}

TEST_F(GRULayerTest, Backward) {
    MatrixXd output = layer->forward(input, true);
    
    MatrixXd gradient = MatrixXd::Random(3, 8);
    MatrixXd dX = layer->backward(gradient, 0.01);
    
    EXPECT_EQ(dX.rows(), 3);
    EXPECT_EQ(dX.cols(), 5);
}

TEST_F(GRULayerTest, StateManagement) {
    layer->reset_state();
    
    MatrixXd hidden = layer->get_hidden_state();
    EXPECT_EQ(hidden.rows(), 0);
    
    layer->forward(input);
    hidden = layer->get_hidden_state();
    EXPECT_EQ(hidden.rows(), 3);
    EXPECT_EQ(hidden.cols(), 8);
}

//=============================================================================
// LAYER INTERFACE TESTS
//=============================================================================

TEST(LayerInterfaceTest, AllLayersDeriveFromLayer) {
    DenseLayer dense(10);
    Conv2DLayer conv(8, 3);
    PoolingLayer pool(2, 2, PoolingLayer::MAX, 1);
    FlattenLayer flatten;
    DropoutLayer dropout(0.3);
    BatchNormLayer batchnorm;
    SimpleRNNLayer simple_rnn(8, 5);
    LSTMLayer lstm(8, 5);
    GRULayer gru(8, 5);
    
    std::vector<Layer*> layers = {
        &dense, &conv, &pool, &flatten, &dropout,
        &batchnorm, &simple_rnn, &lstm, &gru
    };
    
    for (auto* layer : layers) {
        EXPECT_NE(layer->get_type(), "");
        EXPECT_NO_THROW(layer->get_config());
        EXPECT_NO_THROW(layer->clear_cache());
    }
}

TEST(LayerInterfaceTest, CacheManagement) {
    DenseLayer layer(10);
    layer.set_input_shape(5);
    
    // Inizialmente cache nullptr
    EXPECT_EQ(layer.get_cache(), nullptr);
    
    // Forward crea la cache
    MatrixXd input = MatrixXd::Random(3, 5);
    layer.forward(input);
    
    auto cache = layer.get_cache();
    EXPECT_NE(cache, nullptr);
    EXPECT_EQ(cache->get_type(), "DenseCache");
    
    // Clear cache
    layer.clear_cache();
    EXPECT_EQ(layer.get_cache()->get_input().size(), 0);
}

//=============================================================================
// MAIN
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}