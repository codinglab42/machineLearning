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
    
    EXPECT_EQ(cache.mutable_input().size(), 0);
    EXPECT_EQ(cache.mutable_z().size(), 0);
    EXPECT_EQ(cache.mutable_output().size(), 0);
}

TEST(DenseCacheTest, SetAndGet) {
    DenseCache cache;
    
    MatrixXd input = MatrixXd::Random(5, 10);
    MatrixXd z = MatrixXd::Random(5, 3);
    MatrixXd output = MatrixXd::Random(5, 3);
    
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
    int input_height = 28;
    int input_width = 28;
    int input_channels = 1;
    int output_height = 26;
    int output_width = 26;
    int filters = 8;
    int kernel_size = 3;
    
    cache.set_input_shape(input_height, input_width, input_channels);
    cache.set_output_shape(output_height, output_width, filters);
    cache.set_batch_size(batch);
    cache.set_kernel_info(kernel_size, 1, "valid");
    
    int input_size = batch * input_height * input_width * input_channels;
    int output_size = batch * output_height * output_width * filters;
    int col_features = output_height * output_width * kernel_size * kernel_size * input_channels;
    
    MatrixXd input = MatrixXd::Random(input_size, 1);
    MatrixXd z = MatrixXd::Random(output_size, 1);
    MatrixXd output = MatrixXd::Random(output_size, 1);
    MatrixXd col = MatrixXd::Random(batch, col_features);
    
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
    
    int batch = 2;
    int input_height = 28;
    int input_width = 28;
    int input_channels = 1;
    int output_height = 26;
    int output_width = 26;
    int filters = 8;
    int kernel_size = 3;
    
    cache.set_input_shape(input_height, input_width, input_channels);
    cache.set_output_shape(output_height, output_width, filters);
    cache.set_batch_size(batch);
    cache.set_kernel_info(kernel_size, 1, "valid");
    
    int input_size = batch * input_height * input_width * input_channels;
    int output_size = batch * output_height * output_width * filters;
    int col_features = output_height * output_width * kernel_size * kernel_size * input_channels;
    
    cache.input_cache = MatrixXd::Random(input_size, 1);
    cache.z_cache = MatrixXd::Random(output_size, 1);        // <-- AGGIUNTO
    cache.output_cache = MatrixXd::Random(output_size, 1);
    cache.col_cache = MatrixXd::Random(batch, col_features);
    
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
    EXPECT_EQ(cache.get_type(), "PoolingCache");
    EXPECT_FALSE(cache.get_training());
    EXPECT_TRUE(cache.get_max_indices().empty());
}

TEST(PoolingCacheTest, SetInputShape) {
    PoolingCache cache;
    
    cache.set_input_shape(32, 32, 3);
    EXPECT_FALSE(cache.is_valid()); // Ancora senza input/output
}

TEST(PoolingCacheTest, SetData) {
    PoolingCache cache;
    
    MatrixXd input = MatrixXd::Random(2, 16);
    MatrixXd output = MatrixXd::Random(2, 4);
    
    cache.set_input(input);
    cache.set_output(output);
    cache.set_training(true);
    
    EXPECT_TRUE(cache.get_training());
    EXPECT_EQ(cache.get_input().rows(), 2);
    EXPECT_EQ(cache.get_output().cols(), 4);
    EXPECT_TRUE(cache.is_valid());
}

TEST(PoolingCacheTest, MaxIndices) {
    PoolingCache cache;
    
    cache.set_input_shape(4, 4, 1);
    cache.set_training(true);
    
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
    
    auto& mutable_indices = cache.mutable_max_indices();
    mutable_indices.clear();
    EXPECT_TRUE(cache.get_max_indices().empty());
}

TEST(PoolingCacheTest, Clear) {
    PoolingCache cache;
    
    MatrixXd input = MatrixXd::Random(2, 16);
    MatrixXd output = MatrixXd::Random(2, 4);
    
    cache.set_input(input);
    cache.set_output(output);
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
    
    // In training mode, servono tutti i dati
    cache.x_centered = input.rowwise() - input.colwise().mean();
    cache.x_norm = cache.x_centered;  // Semplificato per il test
    cache.batch_mean = input.colwise().mean();
    cache.batch_var = VectorXd::Ones(3);
    cache.inv_std = VectorXd::Ones(3);
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
    
    // Imposta tutti i dati necessari per essere valido in training mode
    MatrixXd input = MatrixXd::Random(4, 3);
    cache.input_cache = input;
    cache.output_cache = input;  // output_cache richiesto da is_valid()
    cache.x_centered = MatrixXd::Random(4, 3);
    cache.x_norm = MatrixXd::Random(4, 3);
    cache.batch_mean = VectorXd::Random(3);
    cache.batch_var = VectorXd::Random(3);
    cache.inv_std = VectorXd::Random(3);
    cache.training = true;
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.input_cache.size(), 0);
    EXPECT_EQ(cache.output_cache.size(), 0);
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
    EXPECT_EQ(cache.get_type(), "RNNCache");
    EXPECT_TRUE(cache.hidden_states.empty());
    EXPECT_TRUE(cache.pre_activations.empty());
    EXPECT_EQ(cache.timesteps, 0);
    EXPECT_FALSE(cache.training);
}

TEST(RNNCacheTest, SetData) {
    RNNCache cache;
    
    int timesteps = 3;
    int batch_size = 2;
    int hidden_size = 4;
    int input_size = 10;
    
    cache.timesteps = timesteps;
    cache.batch_size = batch_size;
    cache.hidden_size = hidden_size;
    cache.input_size = input_size;
    cache.training = true;
    
    cache.input_cache = MatrixXd::Random(batch_size * timesteps, input_size);
    cache.output_cache = MatrixXd::Random(batch_size * timesteps, hidden_size);
    
    // In training mode, hidden_states deve avere size timesteps + 1
    for (int t = 0; t < timesteps + 1; ++t) {
        cache.hidden_states.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    // pre_activations deve avere size timesteps
    for (int t = 0; t < timesteps; ++t) {
        cache.pre_activations.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.hidden_states.size(), timesteps + 1);
    EXPECT_EQ(cache.pre_activations.size(), timesteps);
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
    
    int timesteps = 3;
    int batch_size = 2;
    int hidden_size = 4;
    int input_size = 10;
    
    cache.timesteps = timesteps;
    cache.batch_size = batch_size;
    cache.hidden_size = hidden_size;
    cache.input_size = input_size;
    cache.input_cache = MatrixXd::Random(batch_size * timesteps, input_size);
    cache.output_cache = MatrixXd::Random(batch_size * timesteps, hidden_size);
    
    for (int t = 0; t < timesteps + 1; ++t) {
        cache.hidden_states.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    for (int t = 0; t < timesteps; ++t) {
        cache.pre_activations.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    cache.training = true;
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.input_cache.size(), 0);
    EXPECT_TRUE(cache.hidden_states.empty());
    EXPECT_TRUE(cache.pre_activations.empty());
    EXPECT_EQ(cache.timesteps, 0);
    EXPECT_EQ(cache.batch_size, 0);
    EXPECT_FALSE(cache.training);
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
    
    int timesteps = 3;
    int batch_size = 2;
    int hidden_size = 4;
    int input_size = 10;
    
    cache.timesteps = timesteps;
    cache.batch_size = batch_size;
    cache.hidden_size = hidden_size;
    cache.input_size = input_size;
    cache.training = true;
    
    cache.input_cache = MatrixXd::Random(batch_size * timesteps, input_size);
    cache.output_cache = MatrixXd::Random(batch_size * timesteps, hidden_size);
    
    // hidden_states deve avere size timesteps + 1
    for (int t = 0; t < timesteps + 1; ++t) {
        cache.hidden_states.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    
    // pre_activations deve avere size timesteps (questo mancava!)
    for (int t = 0; t < timesteps; ++t) {
        cache.pre_activations.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    
    // z_values è specifico di SimpleRNN
    for (int t = 0; t < timesteps; ++t) {
        cache.z_values.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.z_values.size(), timesteps);
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
    
    int timesteps = 3;
    int batch_size = 2;
    int hidden_size = 4;
    int input_size = 10;
    
    cache.timesteps = timesteps;
    cache.batch_size = batch_size;
    cache.hidden_size = hidden_size;
    cache.input_size = input_size;
    cache.training = true;
    
    cache.input_cache = MatrixXd::Random(batch_size * timesteps, input_size);
    cache.output_cache = MatrixXd::Random(batch_size * timesteps, hidden_size);
    
    // hidden_states e cell_states devono avere size timesteps + 1
    for (int t = 0; t < timesteps + 1; ++t) {
        cache.hidden_states.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.cell_states.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    
    // pre_activations deve avere size timesteps (questo mancava!)
    for (int t = 0; t < timesteps; ++t) {
        cache.pre_activations.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    
    // Dati specifici LSTM
    for (int t = 0; t < timesteps; ++t) {
        cache.input_gates.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.forget_gates.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.output_gates.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.cell_candidates.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.z_i.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.z_f.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.z_o.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.z_c.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.cell_states.size(), timesteps + 1);
    EXPECT_EQ(cache.input_gates.size(), timesteps);
    EXPECT_EQ(cache.z_i.size(), timesteps);
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
    
    int timesteps = 3;
    int batch_size = 2;
    int hidden_size = 4;
    int input_size = 10;
    
    cache.timesteps = timesteps;
    cache.batch_size = batch_size;
    cache.hidden_size = hidden_size;
    cache.input_size = input_size;
    cache.training = true;
    
    cache.input_cache = MatrixXd::Random(batch_size * timesteps, input_size);
    cache.output_cache = MatrixXd::Random(batch_size * timesteps, hidden_size);
    
    // hidden_states deve avere size timesteps + 1
    for (int t = 0; t < timesteps + 1; ++t) {
        cache.hidden_states.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    
    // pre_activations deve avere size timesteps (questo mancava!)
    for (int t = 0; t < timesteps; ++t) {
        cache.pre_activations.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    
    // Dati specifici GRU
    for (int t = 0; t < timesteps; ++t) {
        cache.reset_gates.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.update_gates.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.candidate_hidden.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.z_r.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.z_z.push_back(MatrixXd::Random(batch_size, hidden_size));
        cache.z_h.push_back(MatrixXd::Random(batch_size, hidden_size));
    }
    
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.reset_gates.size(), timesteps);
    EXPECT_EQ(cache.update_gates.size(), timesteps);
    EXPECT_EQ(cache.z_r.size(), timesteps);
}

//=============================================================================
// INTEGRATION TESTS
//=============================================================================

TEST(CacheIntegrationTest, AllCachesDeriveFromLayerCache) {
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

    
    std::vector<LayerCache*> caches = {
        &dense, &conv, &pooling, &dropout, &flatten,
        &batchnorm, &rnn, &simple_rnn, &lstm, &gru
    };
    
    for (auto* cache : caches) {
        EXPECT_NE(cache->get_type(), "");
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