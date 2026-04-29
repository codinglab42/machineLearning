#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/cache/batchnorm_cache.h"
#include "components/cache/conv_cache.h"
#include "components/cache/dense_cache.h"
#include "components/cache/dropout_cache.h"
#include "components/cache/flatten_cache.h"
#include "components/cache/gru_cache.h"
#include "components/cache/lstm_cache.h"
#include "components/cache/pooling_cache.h"
#include "components/cache/rnn_cache.h"
#include "components/cache/simple_rnn_cache.h"
#include <sys/resource.h>
#include <iostream>

long get_current_rss_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // Restituisce il picco di memoria in Kilobytes
}

using namespace layers;

// --- Helper Fixture per Cache semplici ---
template <typename T>
class CacheTest : public ::testing::Test {
protected:
    T cache;
};

// --- Registrazione dei test per classi base ---
using SimpleCacheTypes = ::testing::Types<DenseCache, BatchNormCache, DropoutCache, FlattenCache>;
TYPED_TEST_SUITE(CacheTest, SimpleCacheTypes);

TYPED_TEST(CacheTest, ClearResetsState) {
    this->cache.mutable_input() = Eigen::MatrixXd::Random(2, 2);
    this->cache.clear();
    EXPECT_EQ(this->cache.get_input().size(), 0);
}

// --- Test per Cache Complesse (RNN/LSTM/GRU) ---
TEST(RNNCacheTest, Initialization) {
    RNNCache cache;
    cache.timesteps = 5;
    cache.mutable_hidden_states().resize(5);
    EXPECT_EQ(cache.hidden_states.size(), 5);
}

TEST(LSTMCacheTest, LSTMStructure) {
    LSTMCache cache;
    cache.cell_states.push_back(Eigen::MatrixXd::Zero(1, 1));
    EXPECT_EQ(cache.cell_states.size(), 1);
    EXPECT_EQ(cache.get_type(), "LSTMCache");
}

TEST(GRUCacheTest, GRUStructure) {
    GRUCache cache;
    cache.reset_gates.push_back(Eigen::MatrixXd::Identity(2, 2));
    EXPECT_EQ(cache.reset_gates[0](0,0), 1.0);
}

// --- Test Specifici per Logiche di Dimensione ---
TEST(ConvCacheTest, ShapeValidation) {
    ConvCache cache;
    cache.set_input_shape(32, 32, 3);
    EXPECT_EQ(cache.input_height, 32);
    EXPECT_EQ(cache.input_channels, 3);
}

TEST(PoolingCacheTest, MaxIndexStorage) {
    PoolingCache cache;
    cache.add_max_index(0, 0, 1, 1, 10, 10);
    EXPECT_EQ(cache.get_max_indices().size(), 1);
    EXPECT_EQ(cache.get_max_indices()[0].batch, 0);
}


TEST(LayerCacheTest, DeepCopyConsistency) {
    DenseCache cache1;
    cache1.mutable_input() = Eigen::MatrixXd::Random(5, 5);
    
    // Simuliamo una copia (se hai definito un costruttore di copia)
    DenseCache cache2 = cache1; 
    
    // Modifichiamo cache2
    cache2.mutable_input() *= 2.0;
    
    // Verifichiamo che cache1 sia rimasta invariata
    EXPECT_FALSE(cache1.get_input().isApprox(cache2.get_input()));
}

TEST(ConvCacheTest, HandlesInvalidDimensionsGracefully) {
    ConvCache cache;
    // Impostiamo dimensioni che non hanno senso o causano mismatch
    cache.set_input_shape(0, 0, 0); 
    
    // Assicurati che is_valid() ritorni false invece di lanciare eccezioni non gestite
    EXPECT_FALSE(cache.is_valid());
}

TEST(LayerIntegrationTest, InputOutputFlow) {
    DenseCache dense;
    BatchNormCache bn;
    
    // Output di Dense deve diventare Input di BatchNorm
    Eigen::MatrixXd fake_output = Eigen::MatrixXd::Random(32, 10);
    dense.mutable_output() = fake_output;
    
    bn.mutable_input() = dense.get_output();
    
    EXPECT_TRUE(bn.get_input().isApprox(dense.get_output()));
}

