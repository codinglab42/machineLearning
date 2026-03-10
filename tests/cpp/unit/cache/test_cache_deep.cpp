#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/cache/layer_cache.h"
#include "components/cache/dense_cache.h"
#include "components/cache/conv_cache.h"
#include "components/cache/pooling_cache.h"
#include "components/cache/dropout_cache.h"
#include "components/cache/flatten_cache.h"
#include "components/cache/batchnorm_cache.h"
#include "components/cache/rnn_cache.h"
#include "components/cache/simple_rnn_cache.h"
#include "components/cache/lstm_cache.h"
#include "components/cache/gru_cache.h"
#include "components/cache/weighted_cache.h"
#include <Eigen/Dense>

using namespace layers;
using Eigen::MatrixXd;
using Eigen::VectorXd;

//=============================================================================
// DENSE CACHE TESTS
//=============================================================================

TEST(DenseCacheTest, InitialState) {
    DenseCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_EQ(cache.get_type(), "DenseCache");
    EXPECT_TRUE(cache.has_activation());
    
    // Verifica accesso modificabile
    EXPECT_EQ(cache.mutable_input().size(), 0);
    EXPECT_EQ(cache.mutable_z().size(), 0);
    EXPECT_EQ(cache.mutable_output().size(), 0);
}

TEST(DenseCacheTest, SetAndGet) {
    DenseCache cache;
    
    MatrixXd input = MatrixXd::Random(5, 10);
    MatrixXd z = MatrixXd::Random(5, 3);
    MatrixXd output = MatrixXd::Random(5, 3);
    
    // Usa i membri pubblici
    cache.input_cache = input;
    cache.z_cache = z;
    cache.output_cache = output;
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.get_input().rows(), 5);
    EXPECT_EQ(cache.get_input().cols(), 10);
    EXPECT_EQ(cache.get_output().rows(), 5);
    EXPECT_EQ(cache.get_output().cols(), 3);
    EXPECT_EQ(cache.z_cache.rows(), 5);
    EXPECT_EQ(cache.z_cache.cols(), 3);
    
    // Verifica mutable access
    EXPECT_EQ(cache.mutable_input().rows(), 5);
    EXPECT_EQ(cache.mutable_z().cols(), 3);
}

TEST(DenseCacheTest, Clear) {
    DenseCache cache;
    
    cache.input_cache = MatrixXd::Random(5, 10);
    cache.z_cache = MatrixXd::Random(5, 3);
    cache.output_cache = MatrixXd::Random(5, 3);
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_EQ(cache.z_cache.size(), 0);
}

//=============================================================================
// CONV CACHE TESTS
//=============================================================================

TEST(ConvCacheTest, InitialState) {
    ConvCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_EQ(cache.get_type(), "ConvCache");
    EXPECT_TRUE(cache.has_activation());
}

TEST(ConvCacheTest, SetDimensions) {
    ConvCache cache;
    
    cache.set_input_shape(32, 32, 3);
    cache.set_output_shape(30, 30, 16);
    cache.set_batch_size(4);
    cache.set_kernel_info(3, 1, "valid");
    
    EXPECT_EQ(cache.input_height, 32);
    EXPECT_EQ(cache.input_width, 32);
    EXPECT_EQ(cache.input_channels, 3);
    EXPECT_EQ(cache.output_height, 30);
    EXPECT_EQ(cache.output_width, 30);
    EXPECT_EQ(cache.filters, 16);
    EXPECT_EQ(cache.batch_size, 4);
    EXPECT_EQ(cache.kernel_size, 3);
    EXPECT_EQ(cache.strides, 1);
    EXPECT_EQ(cache.padding, "valid");
}

TEST(ConvCacheTest, SetData) {
    ConvCache cache;
    
    int batch = 2;
    int input_size = batch * 28 * 28 * 1;
    int output_size = batch * 26 * 26 * 8;
    
    cache.set_input_shape(28, 28, 1);
    cache.set_output_shape(26, 26, 8);
    cache.set_batch_size(batch);
    cache.set_kernel_info(3, 1, "valid");
    
    MatrixXd input = MatrixXd::Random(input_size, 1);
    MatrixXd z = MatrixXd::Random(output_size, 1);
    MatrixXd output = MatrixXd::Random(output_size, 1);
    MatrixXd col = MatrixXd::Random(batch, 26*26*9);
    
    cache.input_cache = input;
    cache.z_cache = z;
    cache.output_cache = output;
    cache.col_cache = col;
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.input_cache.rows(), input_size);
    EXPECT_EQ(cache.z_cache.rows(), output_size);
    EXPECT_EQ(cache.col_cache.rows(), batch);
}

TEST(ConvCacheTest, MutableAccess) {
    ConvCache cache;
    
    cache.mutable_input() = MatrixXd::Ones(100, 1);
    cache.mutable_z() = MatrixXd::Ones(50, 1);
    cache.mutable_output() = MatrixXd::Ones(50, 1);
    cache.mutable_col() = MatrixXd::Ones(2, 100);
    
    EXPECT_EQ(cache.mutable_input().rows(), 100);
    EXPECT_EQ(cache.mutable_z().rows(), 50);
    EXPECT_EQ(cache.mutable_col().cols(), 100);
}

TEST(ConvCacheTest, Clear) {
    ConvCache cache;
    
    cache.set_input_shape(28, 28, 1);
    cache.set_output_shape(26, 26, 8);
    cache.set_batch_size(2);
    cache.input_cache = MatrixXd::Random(100, 1);
    cache.z_cache = MatrixXd::Random(50, 1);
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.input_cache.size(), 0);
    EXPECT_EQ(cache.z_cache.size(), 0);
    EXPECT_EQ(cache.col_cache.size(), 0);
    EXPECT_EQ(cache.input_height, 0);
    EXPECT_EQ(cache.batch_size, 0);
}

//=============================================================================
// POOLING CACHE TESTS
//=============================================================================

TEST(PoolingCacheTest, InitialState) {
    PoolingCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_FALSE(cache.has_activation());
    EXPECT_EQ(cache.get_type(), "PoolingCache");
    EXPECT_FALSE(cache.get_training());
    EXPECT_TRUE(cache.get_max_indices().empty());
}

TEST(PoolingCacheTest, SetInputShape) {
    PoolingCache cache;
    
    cache.set_input_shape(32, 32, 3);
    
    // Non abbiamo getter pubblici per questi, ma possiamo verificarli indirettamente
    // attraverso il comportamento di is_valid() o altri metodi
    EXPECT_FALSE(cache.is_valid()); // Ancora senza input/output
}

TEST(PoolingCacheTest, SetData) {
    PoolingCache cache;
    
    MatrixXd input = MatrixXd::Random(2, 16);  // batch 2, 16 features
    MatrixXd output = MatrixXd::Random(2, 4);   // dopo pooling
    
    cache.set_input(input);
    cache.set_output(output);
    cache.set_training(true);
    
    EXPECT_TRUE(cache.get_training());
    EXPECT_EQ(cache.get_input().rows(), 2);
    EXPECT_EQ(cache.get_output().cols(), 4);
}

TEST(PoolingCacheTest, MaxIndices) {
    PoolingCache cache;
    
    cache.set_input_shape(4, 4, 1);
    cache.set_training(true);
    
    // Aggiungi indici per max pooling
    cache.add_max_index(0, 0, 0, 0, 0, 0);
    cache.add_max_index(0, 0, 0, 1, 0, 2);
    cache.add_max_index(1, 0, 1, 0, 2, 0);
    
    const auto& indices = cache.get_max_indices();
    ASSERT_EQ(indices.size(), 3);
    
    EXPECT_EQ(indices[0].batch, 0);
    EXPECT_EQ(indices[0].channel, 0);
    EXPECT_EQ(indices[0].output_row, 0);
    EXPECT_EQ(indices[0].output_col, 0);
    EXPECT_EQ(indices[0].input_h, 0);
    EXPECT_EQ(indices[0].input_w, 0);
    
    EXPECT_EQ(indices[2].batch, 1);
    EXPECT_EQ(indices[2].input_h, 2);
    EXPECT_EQ(indices[2].input_w, 0);
    
    // Test mutable access
    auto& mutable_indices = cache.mutable_max_indices();
    mutable_indices.clear();
    EXPECT_TRUE(cache.get_max_indices().empty());
}

TEST(PoolingCacheTest, Clear) {
    PoolingCache cache;
    
    cache.set_input(MatrixXd::Random(2, 16));
    cache.set_output(MatrixXd::Random(2, 4));
    cache.set_training(true);
    cache.add_max_index(0, 0, 0, 0, 0, 0);
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_FALSE(cache.get_training());
    EXPECT_TRUE(cache.get_max_indices().empty());
}

//=============================================================================
// DROPOUT CACHE TESTS
//=============================================================================

TEST(DropoutCacheTest, InitialState) {
    DropoutCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_FALSE(cache.has_activation());
    EXPECT_EQ(cache.get_type(), "DropoutCache");
    EXPECT_EQ(cache.mask.size(), 0);
    EXPECT_FALSE(cache.training);
}

TEST(DropoutCacheTest, SetData) {
    DropoutCache cache;
    
    MatrixXd input = MatrixXd::Random(3, 5);
    MatrixXd mask(3, 5);
    mask << 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 1.0;
    
    cache.input_cache = input;
    cache.mask = mask;
    cache.training = true;
    cache.output_cache = input.cwiseProduct(mask);
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.input_cache.rows(), 3);
    EXPECT_EQ(cache.mask.cols(), 5);
    EXPECT_TRUE(cache.training);
    EXPECT_DOUBLE_EQ(cache.mask(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(cache.mask(2, 2), 0.0);
}

TEST(DropoutCacheTest, MutableAccess) {
    DropoutCache cache;
    
    cache.mutable_input() = MatrixXd::Ones(3, 5);
    cache.mutable_mask() = MatrixXd::Zero(3, 5);
    cache.mutable_output() = MatrixXd::Ones(3, 5);
    
    EXPECT_EQ(cache.mutable_input()(0, 0), 1.0);
    EXPECT_EQ(cache.mutable_mask()(2, 2), 0.0);
}

TEST(DropoutCacheTest, Clear) {
    DropoutCache cache;
    
    cache.input_cache = MatrixXd::Random(3, 5);
    cache.mask = MatrixXd::Random(3, 5);
    cache.output_cache = MatrixXd::Random(3, 5);
    cache.training = true;
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.input_cache.size(), 0);
    EXPECT_EQ(cache.mask.size(), 0);
    EXPECT_FALSE(cache.training);
}

//=============================================================================
// FLATTEN CACHE TESTS
//=============================================================================

TEST(FlattenCacheTest, InitialState) {
    FlattenCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_FALSE(cache.has_activation());
    EXPECT_EQ(cache.get_type(), "FlattenCache");
    EXPECT_TRUE(cache.original_shape.empty());
}

TEST(FlattenCacheTest, SetData) {
    FlattenCache cache;
    
    MatrixXd input = MatrixXd::Random(10, 20);
    cache.input_cache = input;
    cache.output_cache = input;
    cache.original_shape = {10, 20};
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.get_input().rows(), 10);
    EXPECT_EQ(cache.get_output().cols(), 20);
    EXPECT_EQ(cache.original_shape[0], 10);
    EXPECT_EQ(cache.original_shape[1], 20);
}

TEST(FlattenCacheTest, MutableAccess) {
    FlattenCache cache;
    
    cache.mutable_input() = MatrixXd::Ones(5, 10);
    cache.mutable_output() = MatrixXd::Ones(5, 10);
    cache.mutable_shape() = {5, 10};
    
    EXPECT_EQ(cache.mutable_input()(0, 0), 1.0);
    EXPECT_EQ(cache.mutable_shape()[0], 5);
}

TEST(FlattenCacheTest, Clear) {
    FlattenCache cache;
    
    cache.input_cache = MatrixXd::Random(10, 20);
    cache.output_cache = MatrixXd::Random(10, 20);
    cache.original_shape = {10, 20};
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.input_cache.size(), 0);
    EXPECT_TRUE(cache.original_shape.empty());
}

//=============================================================================
// BATCHNORM CACHE TESTS
//=============================================================================

TEST(BatchNormCacheTest, InitialState) {
    BatchNormCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_FALSE(cache.has_activation());
    EXPECT_EQ(cache.get_type(), "BatchNormCache");
    EXPECT_FALSE(cache.training);
    EXPECT_EQ(cache.x_centered.size(), 0);
    EXPECT_EQ(cache.batch_mean.size(), 0);
}

TEST(BatchNormCacheTest, SetData) {
    BatchNormCache cache;
    
    MatrixXd input = MatrixXd::Random(4, 3);
    cache.input_cache = input;
    cache.training = true;
    
    // Simula calcolo statistiche
    cache.x_centered = input.rowwise() - input.colwise().mean();
    cache.batch_mean = input.colwise().mean();
    cache.batch_var = VectorXd::Ones(3);
    cache.inv_std = VectorXd::Ones(3);
    cache.x_norm = cache.x_centered;
    cache.output_cache = cache.x_norm;
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.input_cache.rows(), 4);
    EXPECT_EQ(cache.x_centered.cols(), 3);
    EXPECT_EQ(cache.batch_mean.size(), 3);
    EXPECT_TRUE(cache.training);
}

TEST(BatchNormCacheTest, MutableAccess) {
    BatchNormCache cache;
    
    cache.mutable_input() = MatrixXd::Ones(4, 3);
    cache.mutable_x_centered() = MatrixXd::Zero(4, 3);
    cache.mutable_x_norm() = MatrixXd::Ones(4, 3);
    cache.mutable_batch_mean() = VectorXd::Ones(3);
    cache.mutable_inv_std() = VectorXd::Ones(3);
    
    EXPECT_EQ(cache.mutable_input()(0, 0), 1.0);
    EXPECT_EQ(cache.mutable_batch_mean()[0], 1.0);
}

TEST(BatchNormCacheTest, Clear) {
    BatchNormCache cache;
    
    cache.input_cache = MatrixXd::Random(4, 3);
    cache.x_centered = MatrixXd::Random(4, 3);
    cache.batch_mean = VectorXd::Random(3);
    cache.training = true;
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.input_cache.size(), 0);
    EXPECT_EQ(cache.x_centered.size(), 0);
    EXPECT_EQ(cache.batch_mean.size(), 0);
    EXPECT_FALSE(cache.training);
}

//=============================================================================
// RNN CACHE TESTS
//=============================================================================

TEST(RNNCacheTest, InitialState) {
    RNNCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_TRUE(cache.has_activation());
    EXPECT_EQ(cache.get_type(), "RNNCache");
    EXPECT_TRUE(cache.hidden_states.empty());
    EXPECT_TRUE(cache.pre_activations.empty());
    EXPECT_EQ(cache.timesteps, 0);
    EXPECT_FALSE(cache.training);
}

TEST(RNNCacheTest, SetData) {
    RNNCache cache;
    
    cache.timesteps = 3;
    cache.batch_size = 2;
    cache.hidden_size = 4;
    cache.training = true;
    
    cache.input_cache = MatrixXd::Random(6, 10);  // batch*timesteps x features
    cache.output_cache = MatrixXd::Random(6, 4);  // batch*timesteps x hidden
    
    // Aggiungi stati
    for (int t = 0; t < 3; ++t) {
        cache.hidden_states.push_back(MatrixXd::Random(2, 4));
        cache.pre_activations.push_back(MatrixXd::Random(2, 4));
    }
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.hidden_states.size(), 3);
    EXPECT_EQ(cache.pre_activations.size(), 3);
    EXPECT_TRUE(cache.training);
}

TEST(RNNCacheTest, MutableAccess) {
    RNNCache cache;
    
    cache.mutable_input() = MatrixXd::Ones(6, 10);
    cache.mutable_output() = MatrixXd::Ones(6, 4);
    cache.mutable_hidden_states().push_back(MatrixXd::Ones(2, 4));
    
    EXPECT_EQ(cache.mutable_input()(0, 0), 1.0);
    EXPECT_EQ(cache.mutable_hidden_states()[0].rows(), 2);
}

TEST(RNNCacheTest, Clear) {
    RNNCache cache;
    
    cache.input_cache = MatrixXd::Random(6, 10);
    cache.output_cache = MatrixXd::Random(6, 4);
    cache.hidden_states.push_back(MatrixXd::Random(2, 4));
    cache.timesteps = 3;
    cache.training = true;
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.input_cache.size(), 0);
    EXPECT_TRUE(cache.hidden_states.empty());
    EXPECT_EQ(cache.timesteps, 0);
}

//=============================================================================
// SIMPLE RNN CACHE TESTS
//=============================================================================

TEST(SimpleRNNCacheTest, InitialState) {
    SimpleRNNCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_EQ(cache.get_type(), "SimpleRNNCache");
    EXPECT_TRUE(cache.z_values.empty());
}

TEST(SimpleRNNCacheTest, SetData) {
    SimpleRNNCache cache;
    
    cache.timesteps = 3;
    cache.batch_size = 2;
    cache.hidden_size = 4;
    cache.training = true;
    
    cache.input_cache = MatrixXd::Random(6, 10);
    cache.output_cache = MatrixXd::Random(6, 4);
    
    for (int t = 0; t < 3; ++t) {
        cache.hidden_states.push_back(MatrixXd::Random(2, 4));
        cache.z_values.push_back(MatrixXd::Random(2, 4));
    }
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.z_values.size(), 3);
}

//=============================================================================
// LSTM CACHE TESTS
//=============================================================================

TEST(LSTMCacheTest, InitialState) {
    LSTMCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_EQ(cache.get_type(), "LSTMCache");
    EXPECT_TRUE(cache.cell_states.empty());
    EXPECT_TRUE(cache.input_gates.empty());
    EXPECT_TRUE(cache.forget_gates.empty());
}

TEST(LSTMCacheTest, SetData) {
    LSTMCache cache;
    
    cache.timesteps = 3;
    cache.batch_size = 2;
    cache.hidden_size = 4;
    cache.training = true;
    
    cache.input_cache = MatrixXd::Random(6, 10);
    cache.output_cache = MatrixXd::Random(6, 4);
    
    for (int t = 0; t < 3; ++t) {
        cache.hidden_states.push_back(MatrixXd::Random(2, 4));
        cache.cell_states.push_back(MatrixXd::Random(2, 4));
        cache.input_gates.push_back(MatrixXd::Random(2, 4));
        cache.forget_gates.push_back(MatrixXd::Random(2, 4));
        cache.output_gates.push_back(MatrixXd::Random(2, 4));
        cache.cell_candidates.push_back(MatrixXd::Random(2, 4));
        cache.z_i.push_back(MatrixXd::Random(2, 4));
        cache.z_f.push_back(MatrixXd::Random(2, 4));
        cache.z_o.push_back(MatrixXd::Random(2, 4));
        cache.z_c.push_back(MatrixXd::Random(2, 4));
    }
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.cell_states.size(), 3);
    EXPECT_EQ(cache.input_gates.size(), 3);
    EXPECT_EQ(cache.z_i.size(), 3);
}

//=============================================================================
// GRU CACHE TESTS
//=============================================================================

TEST(GRUCacheTest, InitialState) {
    GRUCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_EQ(cache.get_type(), "GRUCache");
    EXPECT_TRUE(cache.reset_gates.empty());
    EXPECT_TRUE(cache.update_gates.empty());
}

TEST(GRUCacheTest, SetData) {
    GRUCache cache;
    
    cache.timesteps = 3;
    cache.batch_size = 2;
    cache.hidden_size = 4;
    cache.training = true;
    
    cache.input_cache = MatrixXd::Random(6, 10);
    cache.output_cache = MatrixXd::Random(6, 4);
    
    for (int t = 0; t < 3; ++t) {
        cache.hidden_states.push_back(MatrixXd::Random(2, 4));
        cache.reset_gates.push_back(MatrixXd::Random(2, 4));
        cache.update_gates.push_back(MatrixXd::Random(2, 4));
        cache.candidate_hidden.push_back(MatrixXd::Random(2, 4));
        cache.z_r.push_back(MatrixXd::Random(2, 4));
        cache.z_z.push_back(MatrixXd::Random(2, 4));
        cache.z_h.push_back(MatrixXd::Random(2, 4));
    }
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.reset_gates.size(), 3);
    EXPECT_EQ(cache.update_gates.size(), 3);
    EXPECT_EQ(cache.z_r.size(), 3);
}

//=============================================================================
// WEIGHTED CACHE TESTS
//=============================================================================

TEST(WeightedCacheTest, InitialState) {
    WeightedCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_TRUE(cache.has_activation());
    EXPECT_EQ(cache.get_type(), "WeightedCache");
    EXPECT_EQ(cache.get_weights().size(), 0);
    EXPECT_EQ(cache.get_biases().size(), 0);
    EXPECT_EQ(cache.get_weight_version(), 0);
}

TEST(WeightedCacheTest, SetInputOutput) {
    WeightedCache cache;
    
    MatrixXd input = MatrixXd::Random(5, 10);
    MatrixXd z = MatrixXd::Random(5, 3);
    MatrixXd output = MatrixXd::Random(5, 3);
    
    cache.set_input(input);
    cache.set_z(z);
    cache.set_output(output);
    
    EXPECT_EQ(cache.get_input().rows(), 5);
    EXPECT_EQ(cache.get_z().rows(), 5);
    EXPECT_EQ(cache.get_output().cols(), 3);
}

TEST(WeightedCacheTest, SetWeightsAndBiases) {
    WeightedCache cache;
    
    MatrixXd weights = MatrixXd::Random(5, 3);
    VectorXd biases = VectorXd::Random(3);
    
    cache.set_weights(weights);
    cache.set_biases(biases);
    
    EXPECT_EQ(cache.get_weights().rows(), 5);
    EXPECT_EQ(cache.get_weights().cols(), 3);
    EXPECT_EQ(cache.get_biases().size(), 3);
    
    // Test mutable access
    cache.mutable_weights() = MatrixXd::Ones(5, 3);
    cache.mutable_biases() = VectorXd::Ones(3);
    
    EXPECT_EQ(cache.get_weights()(0, 0), 1.0);
    EXPECT_EQ(cache.get_biases()(0), 1.0);
}

TEST(WeightedCacheTest, SetGradients) {
    WeightedCache cache;
    
    MatrixXd w_grad = MatrixXd::Random(5, 3);
    VectorXd b_grad = VectorXd::Random(3);
    
    cache.set_weight_gradient(w_grad);
    cache.set_bias_gradient(b_grad);
    
    EXPECT_EQ(cache.get_weight_gradient().rows(), 5);
    EXPECT_EQ(cache.get_weight_gradient().cols(), 3);
    EXPECT_EQ(cache.get_bias_gradient().size(), 3);
    
    // Test mutable access
    cache.mutable_weight_gradient() = MatrixXd::Ones(5, 3);
    cache.mutable_bias_gradient() = VectorXd::Ones(3);
    
    EXPECT_EQ(cache.get_weight_gradient()(0, 0), 1.0);
}

TEST(WeightedCacheTest, GradientHistory) {
    WeightedCache cache;
    
    for (int i = 0; i < 5; ++i) {
        cache.push_gradient_history(MatrixXd::Random(5, 3));
    }
    
    EXPECT_EQ(cache.get_gradient_history().size(), 5);
    
    cache.clear_gradient_history();
    EXPECT_TRUE(cache.get_gradient_history().empty());
}

TEST(WeightedCacheTest, OptimizerState) {
    WeightedCache cache;
    
    cache.set_weights(MatrixXd::Random(5, 3));
    
    auto& adam_state = cache.get_optimizer_state("adam");
    EXPECT_EQ(adam_state.timestep, 0);
    EXPECT_EQ(adam_state.first_moment.size(), 0);
    
    auto& state = cache.get_optimizer_state("momentum");
    state.timestep = 10;
    state.momentum = MatrixXd::Ones(5, 3);
    
    EXPECT_TRUE(cache.has_optimizer_state("momentum"));
    EXPECT_FALSE(cache.has_optimizer_state("unknown"));
    
    auto& retrieved = cache.get_optimizer_state("momentum");
    EXPECT_EQ(retrieved.timestep, 10);
    EXPECT_EQ(retrieved.momentum.rows(), 5);
}

TEST(WeightedCacheTest, Checkpointing) {
    WeightedCache cache;
    
    MatrixXd weights = MatrixXd::Random(5, 3);
    cache.set_weights(weights);
    
    cache.save_checkpoint(100);
    cache.save_checkpoint(200);
    
    MatrixXd loaded_weights;
    VectorXd loaded_biases;
    
    EXPECT_TRUE(cache.load_checkpoint(100, loaded_weights, loaded_biases));
    EXPECT_EQ(loaded_weights.rows(), 5);
    EXPECT_EQ(loaded_weights.cols(), 3);
    
    EXPECT_FALSE(cache.load_checkpoint(300, loaded_weights, loaded_biases));
    
    cache.clear_checkpoints();
    EXPECT_FALSE(cache.load_checkpoint(100, loaded_weights, loaded_biases));
}

TEST(WeightedCacheTest, VersionAndRegularization) {
    WeightedCache cache;
    
    cache.set_weight_version(42);
    EXPECT_EQ(cache.get_weight_version(), 42);
    
    cache.set_regularization_loss(0.123);
    EXPECT_DOUBLE_EQ(cache.get_regularization_loss(), 0.123);
}

TEST(WeightedCacheTest, Clear) {
    WeightedCache cache;
    
    cache.set_input(MatrixXd::Random(5, 10));
    cache.set_z(MatrixXd::Random(5, 3));
    cache.set_output(MatrixXd::Random(5, 3));
    cache.set_weights(MatrixXd::Random(5, 3));
    cache.set_biases(VectorXd::Random(3));
    cache.push_gradient_history(MatrixXd::Random(5, 3));
    cache.save_checkpoint(100);
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_weights().size(), 0);
    EXPECT_TRUE(cache.get_gradient_history().empty());
    EXPECT_EQ(cache.get_weight_version(), 0);
}

//=============================================================================
// INTEGRATION TESTS
//=============================================================================

TEST(CacheIntegrationTest, AllCachesDeriveFromLayerCache) {
    // Test che tutte le cache possano essere trattate come LayerCache*
    DenseCache dense;
    ConvCache conv;
    PoolingCache pooling;
    DropoutCache dropout;
    FlattenCache flatten;
    BatchNormCache batchnorm;
    RNNCache rnn;
    SimpleRNNCache simple_rnn;
    LSTMCache lstm;
    GRUCache gru;
    WeightedCache weighted;
    
    std::vector<LayerCache*> caches = {
        &dense, &conv, &pooling, &dropout, &flatten,
        &batchnorm, &rnn, &simple_rnn, &lstm, &gru, &weighted
    };
    
    for (auto* cache : caches) {
        EXPECT_NE(cache->get_type(), "");
        EXPECT_FALSE(cache->has_activation());  // Qualcuno può essere true, ma testiamo solo che esista
        cache->clear();
    }
}

//=============================================================================
// MAIN
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}