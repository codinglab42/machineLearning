#include <gtest/gtest.h>
#include "components/cache/basic_cache.h"
#include "components/cache/pooling_cache.h"
#include "components/cache/weighted_cache.h"
#include <Eigen/Dense>

using namespace layers;
using Eigen::MatrixXd;

// Test per BasicCache
TEST(BasicCacheTest, InitialState) {
    BasicCache cache;
    
    // Verifica stato iniziale
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_FALSE(cache.has_activation());
    EXPECT_EQ(cache.get_type(), "BasicCache");
}

TEST(BasicCacheTest, SetAndGet) {
    BasicCache cache;
    
    // Crea dati di test
    MatrixXd input = MatrixXd::Random(3, 4);
    MatrixXd output = MatrixXd::Random(3, 2);
    
    // Imposta i dati
    cache.set_input(input);
    cache.set_output(output);
    cache.set_has_activation(true);
    
    // Verifica
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.get_input().rows(), 3);
    EXPECT_EQ(cache.get_input().cols(), 4);
    EXPECT_EQ(cache.get_output().rows(), 3);
    EXPECT_EQ(cache.get_output().cols(), 2);
    EXPECT_TRUE(cache.has_activation());
    
    // Verifica che i dati siano corretti
    for (int i = 0; i < input.size(); ++i) {
        EXPECT_DOUBLE_EQ(cache.get_input()(i), input(i));
    }
}

TEST(BasicCacheTest, Clear) {
    BasicCache cache;
    
    // Popola la cache
    cache.set_input(MatrixXd::Random(3, 4));
    cache.set_output(MatrixXd::Random(3, 2));
    cache.set_has_activation(true);
    
    EXPECT_TRUE(cache.is_valid());
    
    // Pulisci
    cache.clear();
    
    // Verifica stato dopo clear
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input().size(), 0);
    EXPECT_EQ(cache.get_output().size(), 0);
    EXPECT_FALSE(cache.has_activation());
}

TEST(BasicCacheTest, MutableAccess) {
    BasicCache cache;
    
    // Usa mutable access per modificare
    cache.mutable_input() = MatrixXd::Ones(2, 2);
    cache.mutable_output() = MatrixXd::Zero(2, 3);
    cache.set_has_activation(false);
    
    // Verifica
    EXPECT_EQ(cache.get_input()(0, 0), 1.0);
    EXPECT_EQ(cache.get_output()(0, 0), 0.0);
}

// Test per PoolingCache
TEST(PoolingCacheTest, InitialState) {
    PoolingCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input_height(), 0);
    EXPECT_EQ(cache.get_input_width(), 0);
    EXPECT_EQ(cache.get_channels(), 0);
    EXPECT_TRUE(cache.get_max_indices().empty());
    EXPECT_EQ(cache.get_type(), "PoolingCache");
}

TEST(PoolingCacheTest, SetInputShape) {
    PoolingCache cache;
    
    cache.set_input_shape(32, 32, 3);
    
    EXPECT_EQ(cache.get_input_height(), 32);
    EXPECT_EQ(cache.get_input_width(), 32);
    EXPECT_EQ(cache.get_channels(), 3);
}

TEST(PoolingCacheTest, AddAndGetMaxIndices) {
    PoolingCache cache;
    cache.set_input_shape(4, 4, 1);  // Input 4x4, 1 canale
    
    // Aggiungi alcuni indici
    cache.add_max_index(PoolingCache::MaxIndex(0, 0, 0, 0, 5));   // batch 0, canale 0, output (0,0), input indice 5
    cache.add_max_index(PoolingCache::MaxIndex(0, 0, 0, 1, 7));   // batch 0, canale 0, output (0,1), input indice 7
    cache.add_max_index(PoolingCache::MaxIndex(0, 0, 1, 0, 13));  // batch 0, canale 0, output (1,0), input indice 13
    cache.add_max_index(PoolingCache::MaxIndex(0, 0, 1, 1, 15));  // batch 0, canale 0, output (1,1), input indice 15
    
    // Verifica
    const auto& indices = cache.get_max_indices();
    ASSERT_EQ(indices.size(), 4);
    
    EXPECT_EQ(indices[0].batch, 0);
    EXPECT_EQ(indices[0].channel, 0);
    EXPECT_EQ(indices[0].output_row, 0);
    EXPECT_EQ(indices[0].output_col, 0);
    EXPECT_EQ(indices[0].input_index, 5);
    
    EXPECT_EQ(indices[3].input_index, 15);
}

TEST(PoolingCacheTest, CalculateInputIndex) {
    PoolingCache cache;
    cache.set_input_shape(4, 4, 3);  // 3 canali, 4x4
    
    // Calcola indice per batch 1, canale 2, posizione (3, 2)
    // Formula: batch * (channels * height * width) + channel * (height * width) + h * width + w
    // 1 * (3 * 4 * 4) + 2 * (4 * 4) + 3 * 4 + 2
    // 1 * 48 + 2 * 16 + 12 + 2 = 48 + 32 + 14 = 94
    int idx = cache.calculate_input_index(1, 2, 3, 2);
    EXPECT_EQ(idx, 94);
}

TEST(PoolingCacheTest, FullPoolingCache) {
    PoolingCache cache;
    cache.set_input_shape(4, 4, 1);
    
    // Simula un forward con max pooling 2x2
    MatrixXd input(1, 16);
    for (int i = 0; i < 16; ++i) input(0, i) = i + 1;
    
    MatrixXd output(1, 4);
    output << 6, 8, 14, 16;
    
    // Popola la cache
    cache.set_input(input);
    cache.set_output(output);
    cache.set_has_activation(false);
    
    cache.add_max_index(PoolingCache::MaxIndex(0, 0, 0, 0, 5));   // indice del 6
    cache.add_max_index(PoolingCache::MaxIndex(0, 0, 0, 1, 7));   // indice dell'8
    cache.add_max_index(PoolingCache::MaxIndex(0, 0, 1, 0, 13));  // indice del 14
    cache.add_max_index(PoolingCache::MaxIndex(0, 0, 1, 1, 15));  // indice del 16
    
    // Verifica
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.get_input().cols(), 16);
    EXPECT_EQ(cache.get_output().cols(), 4);
    EXPECT_EQ(cache.get_max_indices().size(), 4);
    
    // Verifica clear
    cache.clear();
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_input_height(), 0);
    EXPECT_TRUE(cache.get_max_indices().empty());
}

// Test per WeightedCache
TEST(WeightedCacheTest, InitialState) {
    WeightedCache cache;
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_z().size(), 0);
    EXPECT_EQ(cache.get_type(), "WeightedCache");
}

TEST(WeightedCacheTest, SetAndGetZ) {
    WeightedCache cache;
    
    // Imposta input e output
    MatrixXd input = MatrixXd::Random(3, 4);
    MatrixXd z = MatrixXd::Random(3, 2);
    MatrixXd output = MatrixXd::Random(3, 2);
    
    cache.set_input(input);
    cache.set_z(z);
    cache.set_output(output);
    cache.set_has_activation(true);
    
    // Verifica
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.get_z().rows(), 3);
    EXPECT_EQ(cache.get_z().cols(), 2);
    
    // Confronta valori
    for (int i = 0; i < z.size(); ++i) {
        EXPECT_DOUBLE_EQ(cache.get_z()(i), z(i));
    }
}

TEST(WeightedCacheTest, Clear) {
    WeightedCache cache;
    
    cache.set_input(MatrixXd::Random(3, 4));
    cache.set_z(MatrixXd::Random(3, 2));
    cache.set_output(MatrixXd::Random(3, 2));
    cache.set_has_activation(true);
    
    EXPECT_TRUE(cache.is_valid());
    
    cache.clear();
    
    EXPECT_FALSE(cache.is_valid());
    EXPECT_EQ(cache.get_z().size(), 0);
}

TEST(WeightedCacheTest, MutableZ) {
    WeightedCache cache;
    
    cache.mutable_z() = MatrixXd::Ones(2, 3);
    
    EXPECT_EQ(cache.get_z()(0, 0), 1.0);
    EXPECT_EQ(cache.get_z()(1, 2), 1.0);
}

// Test di integrazione: come verrebbe usata in un layer
TEST(CacheIntegrationTest, PoolingLayerSimulation) {
    PoolingCache cache;
    
    // Simula forward di pooling
    int batch = 2;
    int channels = 3;
    int height = 4;
    int width = 4;
    
    cache.set_input_shape(height, width, channels);
    
    MatrixXd input = MatrixXd::Random(batch, channels * height * width);
    MatrixXd output = MatrixXd::Random(batch, channels * 2 * 2);  // dopo pooling 2x2
    
    cache.set_input(input);
    cache.set_output(output);
    
    // Simula salvataggio indici per max pooling
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < 2; ++oh) {
                for (int ow = 0; ow < 2; ++ow) {
                    int input_idx = cache.calculate_input_index(b, c, oh*2, ow*2);
                    cache.add_max_index(PoolingCache::MaxIndex(b, c, oh, ow, input_idx));
                }
            }
        }
    }
    
    // Verifica
    EXPECT_TRUE(cache.is_valid());
    EXPECT_EQ(cache.get_max_indices().size(), batch * channels * 4);
    
    // Simula backward: recupera gli indici
    const auto& indices = cache.get_max_indices();
    for (const auto& idx : indices) {
        EXPECT_GE(idx.input_index, 0);
        EXPECT_LT(idx.input_index, batch * channels * height * width);
    }
}